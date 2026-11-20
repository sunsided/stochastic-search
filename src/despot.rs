//! Determinized sparse partially observable trees (DESPOT; Somani, Ye, Hsu and Lee 2013;
//! Ye, Somani, Hsu and Lee 2017): anytime regularized search over a tree built from `K`
//! sampled scenarios, guided by upper and lower bounds.
//!
//! ```mermaid
//! graph TD
//!     A[sample K scenarios: start state and random stream] --> B{root gap above target?}
//!     B -- yes --> C[explore: action by upper bound, child by excess uncertainty]
//!     C --> D[expand leaves; prune blocked nodes; fix nodes beyond depth D]
//!     D --> E[back up bounds to the root]
//!     E --> B
//!     B -- no --> F[recommend the root action with the best lower bound]
//! ```

mod tree;

use core::mem;

use alloc::vec::Vec;

use crate::budget::{Budget, Limits, Stats};
use crate::despot::tree::{Branch, Node, ROOT, Tree, UNEXPANDED};
use crate::error::Error;
use crate::pomdp::{
    Plan, Pomdp, RolloutPolicy, below, check_horizon, check_reward, collect_actions,
};
use crate::rng::{SplitMix64, derive_seed};
use crate::solution::Status;

/// Settings of [`despot`].
///
/// ```
/// use stochastic_search::{DespotConfig, Limits};
///
/// let config = DespotConfig {
///     scenarios: 200,
///     regularization: 0.1,
///     ..DespotConfig::default()
/// }
/// .with_budget(Limits { max_trials: Some(5_000), ..DespotConfig::DEFAULT_LIMITS });
/// # let _ = config;
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DespotConfig<B = Limits> {
    /// The discount factor `gamma`, in `(0, 1]`. Default `0.95`.
    pub discount: f64,
    /// The number of sampled scenarios `K`, at least `1`. Default `500`, as in the paper.
    pub scenarios: usize,
    /// The regularization constant `lambda`, finite and non-negative: the utility charged
    /// per tree node, which trades tree size against overfitting to the scenarios. In
    /// reward units. Default `0`, as in the reference implementation.
    pub regularization: f64,
    /// The target factor `xi` of the excess uncertainty, in `[0, 1)`: each exploration aims
    /// to shrink the root gap to `xi` times its current size. At `1` the root itself would
    /// have no excess uncertainty and no exploration would leave it. Default `0.95`, as in the
    /// paper.
    pub xi: f64,
    /// The depth `D` of the tree: nodes deeper than `D` follow the default policy, and
    /// default policy simulations take their last step at depth `D`, so every scenario
    /// plays at most `D + 1` steps. At least `1`. Default `90`, as in the paper.
    pub max_depth: usize,
    /// Stop once the root gap `mu(b0) - ell(b0)` is at most this, finite and non-negative.
    /// Default `1e-6`.
    pub target_gap: f64,
    /// Seed of the scenarios. A seed reproduces a call exactly. Default `0`.
    pub seed: u64,
    /// Stops the search. Asked before every exploration, with [`Stats::trials`] counting
    /// explorations, [`Stats::states`] the tree nodes and [`Stats::backups`] the node
    /// backups. Default [`DespotConfig::DEFAULT_LIMITS`].
    pub budget: B,
}

impl DespotConfig {
    /// The default budget: `10^3` explorations, at most `10^6` tree nodes.
    pub const DEFAULT_LIMITS: Limits = Limits {
        max_trials: Some(1_000),
        ..Limits::DEFAULT
    };
}

impl Default for DespotConfig {
    fn default() -> Self {
        Self {
            discount: 0.95,
            scenarios: 500,
            regularization: 0.0,
            xi: 0.95,
            max_depth: 90,
            target_gap: 1e-6,
            seed: 0,
            budget: Self::DEFAULT_LIMITS,
        }
    }
}

impl<B> DespotConfig<B> {
    /// Returns the same configuration with another budget.
    #[must_use]
    pub fn with_budget<B2: Budget<Stats>>(self, budget: B2) -> DespotConfig<B2> {
        DespotConfig {
            discount: self.discount,
            scenarios: self.scenarios,
            regularization: self.regularization,
            xi: self.xi,
            max_depth: self.max_depth,
            target_gap: self.target_gap,
            seed: self.seed,
            budget,
        }
    }

    /// Checks the numeric settings.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfig`] for a setting outside its documented range.
    fn validate<S>(&self) -> Result<(), Error<S>> {
        check_horizon(self.discount, self.max_depth)?;
        let reason = if self.scenarios == 0 {
            "scenarios must be at least 1"
        } else if !(self.regularization.is_finite() && self.regularization >= 0.0) {
            "regularization must be finite and non-negative"
        } else if !(0.0..1.0).contains(&self.xi) {
            "xi must lie in [0, 1)"
        } else if !(self.target_gap.is_finite() && self.target_gap >= 0.0) {
            "target_gap must be finite and non-negative"
        } else {
            return Ok(());
        };

        Err(Error::InvalidConfig { reason })
    }
}

/// Plans one action from the particle belief `belief` with anytime regularized DESPOT.
///
/// Samples `K` scenarios ([`DespotConfig::scenarios`]): a start state drawn from `belief`
/// and a random stream per step, so every scenario plays out deterministically. The tree
/// branches on actions and on the observations the scenarios produce. Each node keeps an
/// upper bound (`upper`, seeded per scenario state) and a lower bound (the return of the
/// default policy `default`, simulated along each scenario) on its value, and their
/// regularized counterparts, which charge [`DespotConfig::regularization`] per node.
///
/// Each exploration (Ye et al. 2017, Algorithm 2) walks down from the root: it expands
/// leaves, takes the action with the best regularized upper bound, and follows the child
/// with the largest excess uncertainty, until that uncertainty is gone, the node is pruned
/// (an ancestor's gap no longer pays for the nodes below it) or the depth limit fixes the
/// node to the default policy. The bounds are then backed up to the root. The search stops
/// when the root gap reaches [`DespotConfig::target_gap`] ([`Status::Converged`]) or the
/// budget runs out ([`Status::BudgetExhausted`]).
///
/// The recommendation is the root action with the best regularized lower bound, or the
/// default policy's action (in the first scenario) if no action beats following the
/// default policy. [`Plan::value`] is the root's regularized lower bound `ell(b0)`: the
/// average discounted return of the best policy tree found, over the sampled scenarios,
/// minus `lambda` per tree node. It is an estimate from `K` scenarios, not a bound on the
/// true value.
///
/// # Bounds
///
/// `upper(s)` must bound the optimal discounted return from state `s` from above, and should
/// not loosen along a step (`upper(s) >= r + gamma * upper(s')`); otherwise the bounds lose
/// their meaning, though the search still returns. A node's averaged upper bound is raised
/// to its default-policy value if it falls below it. `R_max / (1 - gamma)` always works;
/// tighter bounds prune more.
///
/// # Errors
///
/// Returns [`Error::InvalidConfig`] for an invalid configuration, [`Error::EmptyBelief`] for
/// an empty `belief`, [`Error::NoActions`] if the POMDP has no actions,
/// [`Error::InvalidReward`] if a sampled reward is not finite, and
/// [`Error::InvalidHeuristic`] if `upper` returns NaN or an infinity.
///
/// # Panics
///
/// Panics if more than `u32::MAX` tree nodes or scenarios are stored.
///
/// # Example
///
/// The hidden coin from [`Pomdp`], with the bound `10`: peeking first is better than a blind
/// call.
///
/// ```
/// # use rand_core::Rng;
/// # use stochastic_search::{Pomdp, Step, unit_interval};
/// # struct Coin;
/// # impl Pomdp for Coin {
/// #     type State = bool;
/// #     type Action = &'static str;
/// #     type Observation = Option<bool>;
/// #     fn actions(&self, mut emit: impl FnMut(&'static str)) { emit("peek"); emit("heads"); emit("tails"); }
/// #     fn step<R: Rng + ?Sized>(&self, heads: &bool, action: &&'static str, rng: &mut R)
/// #         -> Step<bool, Option<bool>> {
/// #         if *action == "peek" {
/// #             let truthful = unit_interval(rng) < 0.9;
/// #             Step::new(*heads, Some(*heads == truthful), -1.0)
/// #         } else {
/// #             let right = *heads == (*action == "heads");
/// #             Step::new(*heads, None, if right { 10.0 } else { -10.0 }).terminal()
/// #         }
/// #     }
/// # }
/// use stochastic_search::{DespotConfig, UniformRandom, despot};
///
/// let config = DespotConfig { scenarios: 100, max_depth: 10, ..DespotConfig::default() };
/// let plan = despot(&Coin, &[true, false], UniformRandom, |_: &bool| 10.0, config)?;
///
/// assert_eq!(plan.action, "peek");
/// # Ok::<(), stochastic_search::Error<bool>>(())
/// ```
pub fn despot<P, R, U, B>(
    pomdp: &P,
    belief: &[P::State],
    default: R,
    upper: U,
    config: DespotConfig<B>,
) -> Result<Plan<P::Action>, Error<P::State>>
where
    P: Pomdp,
    P::State: Clone,
    P::Action: Clone,
    P::Observation: PartialEq,
    R: RolloutPolicy<P::State>,
    U: FnMut(&P::State) -> f64,
    B: Budget<Stats>,
{
    let mut search = Search::new(pomdp, belief, default, upper, config)?;
    let status = search.run()?;

    Ok(search.recommend(status))
}

/// The state of one [`despot`] call.
struct Search<'a, P: Pomdp, R, U, B> {
    /// The model.
    pomdp: &'a P,
    /// The actions, in emission order.
    actions: Vec<P::Action>,
    /// The tree.
    tree: Tree<P::State, P::Observation>,
    /// The default policy.
    default: R,
    /// The per-state upper bound.
    upper: U,
    /// The caller's budget.
    budget: B,
    /// The counters so far.
    stats: Stats,
    /// The seed the scenario streams derive from.
    seed: u64,
    /// The discount factor.
    discount: f64,
    /// `gamma^d` for `d` in `0..=max_depth + 1`.
    powers: Vec<f64>,
    /// `K` as a float.
    scenarios: f64,
    /// The regularization constant `lambda`.
    lambda: f64,
    /// The excess uncertainty factor `xi`.
    xi: f64,
    /// The depth `D`.
    max_depth: usize,
    /// The target root gap.
    target_gap: f64,
}

impl<'a, P, R, U, B> Search<'a, P, R, U, B>
where
    P: Pomdp,
    P::State: Clone,
    P::Action: Clone,
    P::Observation: PartialEq,
    R: RolloutPolicy<P::State>,
    U: FnMut(&P::State) -> f64,
    B: Budget<Stats>,
{
    /// Validates the input, samples the scenarios and initializes the root.
    ///
    /// # Errors
    ///
    /// See [`despot`].
    fn new(
        pomdp: &'a P,
        belief: &[P::State],
        default: R,
        upper: U,
        config: DespotConfig<B>,
    ) -> Result<Self, Error<P::State>> {
        config.validate()?;
        if belief.is_empty() {
            return Err(Error::EmptyBelief);
        }

        #[expect(
            clippy::cast_precision_loss,
            reason = "scenario counts beyond 2^53 are far outside any memory budget"
        )]
        let scenarios = config.scenarios as f64;
        let mut search = Self {
            pomdp,
            actions: collect_actions(pomdp)?,
            tree: Tree {
                nodes: Vec::new(),
                branches: Vec::new(),
            },
            default,
            upper,
            budget: config.budget,
            stats: Stats::default(),
            seed: config.seed,
            discount: config.discount,
            powers: powers(config.discount, config.max_depth + 1),
            scenarios,
            lambda: config.regularization,
            xi: config.xi,
            max_depth: config.max_depth,
            target_gap: config.target_gap,
        };

        let particles = search.sample_scenarios(belief, config.scenarios);
        search.add_node(None, 0, particles)?;

        Ok(search)
    }

    /// Draws the start states of `count` scenarios uniformly from `belief`.
    ///
    /// # Panics
    ///
    /// Panics if `count` exceeds `u32::MAX`.
    fn sample_scenarios(&self, belief: &[P::State], count: usize) -> Vec<(u32, P::State)> {
        let mut rng = SplitMix64::new(self.seed);

        (0..count)
            .map(|k| {
                let id = u32::try_from(k).expect("at most u32::MAX scenarios");
                (id, belief[below(&mut rng, belief.len())].clone())
            })
            .collect()
    }

    /// Explores and backs up until the root gap reaches the target or the budget runs out.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidReward`] or [`Error::InvalidHeuristic`] for bad model output.
    fn run(&mut self) -> Result<Status, Error<P::State>> {
        while self.gap(ROOT) > self.target_gap {
            self.stats.states = self.tree.nodes.len() as u64;
            if self.budget.exhausted(&self.stats) {
                return Ok(Status::BudgetExhausted);
            }

            self.stats.trials += 1;
            let leaf = self.explore()?;
            self.back_up_path(leaf);
        }

        self.stats.states = self.tree.nodes.len() as u64;

        Ok(Status::Converged)
    }

    /// One exploration from the root (Ye et al. 2017, Algorithm 2). Returns the node where
    /// it stopped.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidReward`] or [`Error::InvalidHeuristic`] for bad model output.
    fn explore(&mut self) -> Result<u32, Error<P::State>> {
        let mut node = ROOT;
        loop {
            if self.tree.node(node).depth > self.max_depth {
                self.tree.make_default(node);
                return Ok(node);
            }

            if self.tree.node(node).fixed
                || self.excess_uncertainty(node) <= 0.0
                || self.prune(node)
            {
                return Ok(node);
            }

            if self.tree.node(node).first_branch == UNEXPANDED {
                self.expand(node)?;
            }

            let Some(child) = self.most_uncertain_child(node) else {
                return Ok(node);
            };

            node = child;
        }
    }

    /// The child of `node`'s best action by regularized upper bound with the largest excess
    /// uncertainty; `None` if every scenario ended the episode under that action.
    fn most_uncertain_child(&self, node: u32) -> Option<u32> {
        let (action, _) = self.best_action(node, |n| n.mu);
        let branch = &self.tree.branches_of(node, self.actions.len())[action];

        branch
            .children
            .iter()
            .map(|&(_, child)| (child, self.excess_uncertainty(child)))
            .fold(None, |best: Option<(u32, f64)>, (child, e)| match best {
                Some((_, best_e)) if best_e >= e => best,
                _ => Some((child, e)),
            })
            .map(|(child, _)| child)
    }

    /// The gap `mu(b) - ell(b)` of a node.
    fn gap(&self, node: u32) -> f64 {
        let node = self.tree.node(node);

        node.mu - node.ell
    }

    /// The excess uncertainty `E(b) = eps(b) - |Phi_b| / K * xi * eps(b0)`.
    fn excess_uncertainty(&self, node: u32) -> f64 {
        self.gap(node) - self.weight(node) * self.xi * self.gap(ROOT)
    }

    /// The scenario fraction `|Phi_b| / K` of a node.
    fn weight(&self, node: u32) -> f64 {
        #[expect(
            clippy::cast_precision_loss,
            reason = "scenario counts beyond 2^53 are far outside any memory budget"
        )]
        let count = self.tree.node(node).scenarios.len() as f64;

        count / self.scenarios
    }

    /// Prunes the path ending at `node` (Ye et al. 2017, Algorithm 4): fixes every node to
    /// the default policy that an ancestor blocks, from `node` upwards until one is not
    /// blocked, backing up from each. Returns whether `node` was blocked.
    fn prune(&mut self, node: u32) -> bool {
        let mut current = node;
        let mut blocked = false;
        while self.is_blocked(current) {
            self.tree.make_default(current);
            self.back_up_path(current);
            blocked = true;
            match self.tree.node(current).parent {
                Some(parent) => current = parent,
                None => break,
            }
        }

        blocked
    }

    /// Returns `true` if a strict ancestor `b'` of `node` cannot pay for the nodes down to
    /// `node`: `|Phi_b'| / K * gamma^depth(b') * (U(b') - L0(b')) <= lambda * (depth(node) -
    /// depth(b'))` (Ye et al. 2017, eq. 12).
    fn is_blocked(&self, node: u32) -> bool {
        let depth = self.tree.node(node).depth;
        let mut ancestor = self.tree.node(node).parent;
        while let Some(id) = ancestor {
            let candidate = self.tree.node(id);
            let potential = self.weight(id)
                * self.powers[candidate.depth]
                * (candidate.upper - candidate.default_value);
            #[expect(
                clippy::cast_precision_loss,
                reason = "depths are bounded by max_depth, far below 2^53"
            )]
            let cost = self.lambda * (depth - candidate.depth) as f64;
            if potential <= cost {
                return true;
            }

            ancestor = candidate.parent;
        }

        false
    }

    /// Creates the branches of `node`: steps every scenario with every action, groups the
    /// successors by observation and initializes the children.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidReward`] or [`Error::InvalidHeuristic`] for bad model output.
    fn expand(&mut self, node: u32) -> Result<(), Error<P::State>> {
        let scenarios = mem::take(&mut self.tree.node_mut(node).scenarios);
        let depth = self.tree.node(node).depth;
        let first = u32::try_from(self.tree.branches.len()).expect("at most u32::MAX branches");

        let mut result = Ok(());
        for position in 0..self.actions.len() {
            result = self.add_branch(node, depth, &scenarios, position);
            if result.is_err() {
                break;
            }
        }

        let entry = self.tree.node_mut(node);
        entry.scenarios = scenarios;
        if result.is_ok() {
            entry.first_branch = first;
        } else {
            self.tree.branches.truncate(first as usize);
        }

        result
    }

    /// Appends the branch of the action at `position` below `node` and creates its children.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidReward`] or [`Error::InvalidHeuristic`] for bad model output.
    fn add_branch(
        &mut self,
        node: u32,
        depth: usize,
        scenarios: &[(u32, P::State)],
        position: usize,
    ) -> Result<(), Error<P::State>> {
        let mut reward_sum = 0.0;
        let mut groups: Vec<(P::Observation, Particles<P::State>)> = Vec::new();
        for (scenario, state) in scenarios {
            let mut rng = self.stream(*scenario, depth, false);
            let step = self.pomdp.step(state, &self.actions[position], &mut rng);
            check_reward(state, step.reward)?;
            reward_sum += step.reward;
            if step.terminal {
                continue;
            }

            match groups
                .iter_mut()
                .find(|(seen, _)| *seen == step.observation)
            {
                Some((_, group)) => group.push((*scenario, step.state)),
                None => groups.push((step.observation, alloc::vec![(*scenario, step.state)])),
            }
        }

        let mut children = Vec::with_capacity(groups.len());
        for (observation, group) in groups {
            children.push((observation, self.add_node(Some(node), depth + 1, group)?));
        }

        self.tree.branches.push(Branch {
            reward_sum,
            children,
        });

        Ok(())
    }

    /// Adds a node at `depth` with `scenarios` (never empty) and initializes its bounds
    /// (Ye et al. 2017, Section 4.2.1).
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidReward`] for a bad reward in a default policy simulation and
    /// [`Error::InvalidHeuristic`] for a NaN or infinite upper bound.
    fn add_node(
        &mut self,
        parent: Option<u32>,
        depth: usize,
        scenarios: Vec<(u32, P::State)>,
    ) -> Result<u32, Error<P::State>> {
        let mut default_sum = 0.0;
        let mut upper_sum = 0.0;
        for (scenario, state) in &scenarios {
            default_sum += self.simulate_default(*scenario, state.clone(), depth)?;
            let bound = (self.upper)(state);
            if !bound.is_finite() {
                return Err(Error::InvalidHeuristic {
                    state: state.clone(),
                });
            }

            upper_sum += bound;
        }

        #[expect(
            clippy::cast_precision_loss,
            reason = "scenario counts beyond 2^53 are far outside any memory budget"
        )]
        let count = scenarios.len() as f64;
        let default_value = default_sum / count;
        let upper = (upper_sum / count).max(default_value);
        let scale = count / self.scenarios * self.powers[depth.min(self.max_depth + 1)];
        let default_utility = scale * default_value;

        Ok(self.tree.push(Node {
            depth,
            parent,
            scenarios,
            upper,
            default_value,
            default_utility,
            mu: default_utility.max(scale * upper - self.lambda),
            ell: default_utility,
            first_branch: UNEXPANDED,
            fixed: false,
        }))
    }

    /// Simulates the default policy along `scenario` from `state` at `depth` until the
    /// episode ends or through depth `D` (the last depth the tree steps from), and returns
    /// the return discounted from `depth`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidReward`] for a non-finite reward.
    fn simulate_default(
        &mut self,
        scenario: u32,
        mut state: P::State,
        depth: usize,
    ) -> Result<f64, Error<P::State>> {
        let mut total = 0.0;
        let mut weight = 1.0;
        for step_depth in depth..=self.max_depth {
            let mut choice = self.stream(scenario, step_depth, true);
            let position = self.default.action(&state, self.actions.len(), &mut choice);
            let mut rng = self.stream(scenario, step_depth, false);
            let step = self.pomdp.step(&state, &self.actions[position], &mut rng);
            check_reward(&state, step.reward)?;
            total += weight * step.reward;
            if step.terminal {
                break;
            }

            weight *= self.discount;
            state = step.state;
        }

        Ok(total)
    }

    /// The random stream of `scenario` at `depth`: for the model's step, or for the default
    /// policy's choice (`policy`). Independent of the action and of the order of expansions,
    /// which is what makes a scenario deterministic.
    fn stream(&self, scenario: u32, depth: usize, policy: bool) -> SplitMix64 {
        let slot = 2 * depth as u64 + u64::from(policy);

        SplitMix64::new(derive_seed(self.seed, u64::from(scenario), slot))
    }

    /// Backs up the bounds of `node` and its ancestors (Ye et al. 2017, Algorithm 5).
    fn back_up_path(&mut self, node: u32) {
        let mut current = Some(node);
        while let Some(id) = current {
            self.back_up(id);
            self.stats.backups += 1;
            current = self.tree.node(id).parent;
        }
    }

    /// Recomputes `mu`, `ell` and `U` of an expanded, unfixed node from its branches.
    fn back_up(&mut self, id: u32) {
        let node = self.tree.node(id);
        if node.fixed || node.first_branch == UNEXPANDED {
            return;
        }

        let (mut mu, mut ell, mut upper) = (
            node.default_utility,
            node.default_utility,
            f64::NEG_INFINITY,
        );
        for branch in self.tree.branches_of(id, self.actions.len()) {
            let bounds = self.branch_bounds(id, branch);
            mu = mu.max(bounds.mu);
            ell = ell.max(bounds.ell);
            upper = upper.max(bounds.upper);
        }

        let node = self.tree.node_mut(id);
        node.mu = mu;
        node.ell = ell;
        node.upper = upper.max(node.default_value);
    }

    /// The bounds of taking `branch`'s action at `node`.
    fn branch_bounds(&self, node: u32, branch: &Branch<P::Observation>) -> Bounds {
        let entry = self.tree.node(node);
        #[expect(
            clippy::cast_precision_loss,
            reason = "scenario counts beyond 2^53 are far outside any memory budget"
        )]
        let count = entry.scenarios.len() as f64;
        let rho = self.powers[entry.depth] * branch.reward_sum / self.scenarios - self.lambda;
        let mut bounds = Bounds {
            mu: rho,
            ell: rho,
            upper: branch.reward_sum / count,
        };

        for &(_, child) in &branch.children {
            let child = self.tree.node(child);
            #[expect(
                clippy::cast_precision_loss,
                reason = "scenario counts beyond 2^53 are far outside any memory budget"
            )]
            let share = child.scenarios.len() as f64 / count;
            bounds.mu += child.mu;
            bounds.ell += child.ell;
            bounds.upper += self.discount * share * child.upper;
        }

        bounds
    }

    /// The position of the best action at the expanded `node` by `key` of its branch
    /// bounds, ties to the first emitted, and that bound.
    fn best_action(&self, node: u32, key: impl Fn(&Bounds) -> f64) -> (usize, f64) {
        self.tree
            .branches_of(node, self.actions.len())
            .iter()
            .enumerate()
            .map(|(position, branch)| (position, key(&self.branch_bounds(node, branch))))
            .fold((0, f64::NEG_INFINITY), |best, candidate| {
                if candidate.1 > best.1 {
                    candidate
                } else {
                    best
                }
            })
    }

    /// The recommendation (Ye et al. 2017, Algorithm 6): the root action with the best
    /// regularized lower bound, unless following the default policy is at least as good.
    fn recommend(mut self, status: Status) -> Plan<P::Action> {
        let root = self.tree.node(ROOT);
        let value = root.ell;
        let default_utility = root.default_utility;
        let position = if root.first_branch == UNEXPANDED {
            None
        } else {
            Some(self.best_action(ROOT, |b| b.ell))
                .filter(|&(_, ell)| ell >= default_utility)
                .map(|(position, _)| position)
        };

        let position = position.unwrap_or_else(|| self.default_action());

        Plan {
            action: self.actions[position].clone(),
            value,
            status,
            stats: self.stats,
        }
    }

    /// The default policy's choice at the root, in the first scenario.
    fn default_action(&mut self) -> usize {
        let (scenario, state) = self.tree.node(ROOT).scenarios[0].clone();
        let mut choice = self.stream(scenario, 0, true);

        self.default.action(&state, self.actions.len(), &mut choice)
    }
}

/// Scenarios with their states at one node.
type Particles<S> = Vec<(u32, S)>;

/// The bounds of one action at one node.
#[derive(Debug, Clone, Copy)]
struct Bounds {
    /// `mu(b, a)`: the regularized upper bound.
    mu: f64,
    /// `ell(b, a)`: the regularized lower bound.
    ell: f64,
    /// `U(b, a)`: the unregularized upper bound on the average return.
    upper: f64,
}

/// `gamma^d` for `d` in `0..=last`.
fn powers(gamma: f64, last: usize) -> Vec<f64> {
    let mut powers = Vec::with_capacity(last + 1);
    let mut power = 1.0;
    for _ in 0..=last {
        powers.push(power);
        power *= gamma;
    }

    powers
}

#[cfg(test)]
mod tests {
    use rand_core::Rng;

    use super::{DespotConfig, ROOT, Search};
    use crate::pomdp::{Pomdp, Step, UniformRandom};
    use crate::rng::unit_interval;

    /// A hidden coin: peek (cost 1, correct with probability 0.9), or call heads or tails
    /// (+10 if right, -10 if wrong, ends the episode).
    struct Coin;

    impl Pomdp for Coin {
        type State = bool;
        type Action = u8;
        type Observation = Option<bool>;

        fn actions(&self, mut emit: impl FnMut(u8)) {
            emit(0);
            emit(1);
            emit(2);
        }

        fn step<R: Rng + ?Sized>(
            &self,
            heads: &bool,
            action: &u8,
            rng: &mut R,
        ) -> Step<bool, Option<bool>> {
            if *action == 0 {
                let truthful = unit_interval(rng) < 0.9;
                return Step::new(*heads, Some(*heads == truthful), -1.0);
            }

            let right = *heads == (*action == 1);

            Step::new(*heads, None, if right { 10.0 } else { -10.0 }).terminal()
        }
    }

    /// Runs `explorations` explorations by hand and checks after each that `ell <= mu`
    /// holds at every node, that the root's upper bound never rises and, without
    /// regularization, that its lower bound never falls.
    fn check_bounds(regularization: f64, explorations: usize) {
        let config = DespotConfig {
            scenarios: 50,
            max_depth: 8,
            regularization,
            ..DespotConfig::default()
        };
        // 10 bounds every return and never loosens along a step (rewards are at most 10).
        let mut search = Search::new(
            &Coin,
            &[true, false],
            UniformRandom,
            |_: &bool| 10.0,
            config,
        )
        .expect("valid input");

        let (mut mu, mut ell) = (search.tree.node(ROOT).mu, search.tree.node(ROOT).ell);
        for _ in 0..explorations {
            let leaf = search.explore().expect("valid model");
            search.back_up_path(leaf);

            for node in &search.tree.nodes {
                assert!(node.ell <= node.mu + 1e-12, "ell above mu");
            }

            let root = search.tree.node(ROOT);
            assert!(
                root.mu <= mu + 1e-12,
                "root mu rose from {mu} to {}",
                root.mu
            );
            if regularization == 0.0 {
                assert!(
                    root.ell >= ell - 1e-12,
                    "root ell fell from {ell} to {}",
                    root.ell
                );
            }

            (mu, ell) = (root.mu, root.ell);
        }

        assert!(search.tree.nodes.len() > 1, "the tree never grew");
    }

    /// The bound invariants hold without regularization.
    #[test]
    fn bounds_stay_ordered_and_monotone() {
        check_bounds(0.0, 200);
    }

    /// The bound invariants hold with regularization and pruning.
    #[test]
    fn bounds_stay_ordered_with_regularization() {
        check_bounds(0.5, 200);
    }

    /// The same seed builds the same tree.
    #[test]
    fn same_seed_same_tree() {
        let build = || {
            let config = DespotConfig {
                scenarios: 30,
                max_depth: 6,
                ..DespotConfig::default()
            }
            .with_budget(crate::Limits {
                max_trials: Some(50),
                ..DespotConfig::DEFAULT_LIMITS
            });
            let mut search = Search::new(
                &Coin,
                &[true, false],
                UniformRandom,
                |_: &bool| 10.0,
                config,
            )
            .expect("valid input");
            search.run().expect("valid model");

            search
                .tree
                .nodes
                .iter()
                .map(|n| {
                    (
                        n.depth,
                        n.scenarios.clone(),
                        n.mu.to_bits(),
                        n.ell.to_bits(),
                    )
                })
                .collect::<alloc::vec::Vec<_>>()
        };

        assert_eq!(build(), build());
    }
}
