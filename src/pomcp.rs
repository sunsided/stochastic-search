//! Partially observable Monte-Carlo planning (POMCP, Silver and Veness 2010): UCT over
//! action-observation histories, started from a particle belief.
//!
//! ```mermaid
//! graph TD
//!     A[sample a particle from the belief] --> B[descend the tree: UCB action, sampled step]
//!     B --> C{history in the tree?}
//!     C -- yes --> B
//!     C -- no --> D[add it, roll out with the rollout policy]
//!     D --> E[back up the discounted return along the path]
//!     E --> A
//! ```

mod tree;

use alloc::vec::Vec;

use crate::budget::{Budget, Limits, TreeStats};
use crate::error::Error;
use crate::observer::{Event, Observer};
use crate::pomcp::tree::{ROOT, Tree};
use crate::pomdp::{
    Plan, Pomdp, RolloutPolicy, below, check_horizon, check_reward, collect_actions,
};
use crate::rng::SplitMix64;
use crate::solution::Status;

/// Settings of [`pomcp`].
///
/// ```
/// use stochastic_search::{Limits, PomcpConfig};
///
/// let config = PomcpConfig {
///     exploration: 110.0, // the reward range of the problem
///     ..PomcpConfig::default()
/// }
/// .with_budget(Limits { max_trials: Some(50_000), ..PomcpConfig::DEFAULT_LIMITS });
/// # let _ = config;
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PomcpConfig<B = Limits, O = ()> {
    /// The discount factor `gamma`, in `(0, 1]`. Default `0.95`.
    pub discount: f64,
    /// The UCB exploration constant `c`, finite and non-negative.
    ///
    /// Scale it to the range of returns: Silver and Veness use the difference between the
    /// largest and smallest return seen. Default `1.0`, which suits rewards in `[0, 1]`.
    pub exploration: f64,
    /// Simulations stop after this many steps from the belief, in the tree and in rollouts.
    /// At least `1`. Default `100`.
    pub max_depth: usize,
    /// Seed of the simulations. A seed reproduces a call exactly. Default `0`.
    pub seed: u64,
    /// Stops the search. Asked before every simulation with the [`TreeStats`] so far.
    /// Default [`PomcpConfig::DEFAULT_LIMITS`].
    pub budget: B,
    /// Receives [`Event::TreeProgress`] every [`PomcpConfig::progress_interval`]
    /// simulations. Default `()`, which ignores events at no cost.
    pub observer: O,
    /// Emit [`Event::TreeProgress`] every this many simulations; `None` never. At least
    /// `1`. Default `Some(1_000)`; ignored by the default observer.
    pub progress_interval: Option<u64>,
}

impl PomcpConfig {
    /// The default budget: `10^4` simulations, at most `10^6` history nodes.
    pub const DEFAULT_LIMITS: Limits = Limits {
        max_trials: Some(10_000),
        ..Limits::DEFAULT
    };
}

impl Default for PomcpConfig {
    fn default() -> Self {
        Self {
            discount: 0.95,
            exploration: 1.0,
            max_depth: 100,
            seed: 0,
            budget: Self::DEFAULT_LIMITS,
            observer: (),
            progress_interval: Some(1_000),
        }
    }
}

impl<B, O> PomcpConfig<B, O> {
    /// Returns the same configuration with the exploration constant set to the largest
    /// possible spread of discounted returns when every reward lies in `[min, max]`:
    /// `(max - min) * (1 + gamma + ... + gamma^(D - 1))` for discount `gamma` and depth `D`.
    ///
    /// The paper's rule, [`pomcp_exploration`], measures the spread instead and is usually
    /// smaller; this bound needs no simulations. A reversed or non-finite range gives an
    /// exploration constant that [`pomcp`] rejects.
    ///
    /// ```
    /// use stochastic_search::PomcpConfig;
    ///
    /// // Tiger: rewards from -100 to 10, two steps, no discounting.
    /// let config = PomcpConfig { discount: 1.0, max_depth: 2, ..PomcpConfig::default() }
    ///     .with_reward_range(-100.0, 10.0);
    /// assert_eq!(config.exploration, 220.0);
    /// ```
    #[must_use]
    pub fn with_reward_range(self, min: f64, max: f64) -> Self {
        let depth = u32::try_from(self.max_depth).map_or(f64::from(u32::MAX), f64::from);
        let horizon = if self.discount < 1.0 {
            (1.0 - libm::pow(self.discount, depth)) / (1.0 - self.discount)
        } else {
            depth
        };

        Self {
            exploration: (max - min) * horizon,
            ..self
        }
    }

    /// Returns the same configuration with another budget.
    #[must_use]
    pub fn with_budget<B2: Budget<TreeStats>>(self, budget: B2) -> PomcpConfig<B2, O> {
        PomcpConfig {
            discount: self.discount,
            exploration: self.exploration,
            max_depth: self.max_depth,
            seed: self.seed,
            budget,
            observer: self.observer,
            progress_interval: self.progress_interval,
        }
    }

    /// Returns the same configuration with another observer.
    #[must_use]
    pub fn with_observer<O2: Observer>(self, observer: O2) -> PomcpConfig<B, O2> {
        PomcpConfig {
            discount: self.discount,
            exploration: self.exploration,
            max_depth: self.max_depth,
            seed: self.seed,
            budget: self.budget,
            observer,
            progress_interval: self.progress_interval,
        }
    }

    /// Checks the numeric settings.
    ///
    /// # Events
    ///
    /// [`PomcpConfig::observer`] receives [`Event::TreeProgress`] every
    /// [`PomcpConfig::progress_interval`] simulations, with the best root action value and no
    /// gap. The default observer `()` costs nothing, and an observer never changes the plan.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfig`] for a discount outside `(0, 1]`, a negative or
    /// non-finite exploration constant, a zero depth or a zero progress interval.
    fn validate<S>(&self) -> Result<(), Error<S>> {
        check_horizon(self.discount, self.max_depth)?;
        if !(self.exploration.is_finite() && self.exploration >= 0.0) {
            return Err(Error::InvalidConfig {
                reason: "exploration must be finite and non-negative",
            });
        }

        if self.progress_interval == Some(0) {
            return Err(Error::InvalidConfig {
                reason: "progress_interval must be at least 1",
            });
        }

        Ok(())
    }
}

/// Plans one action from the particle belief `belief` with POMCP.
///
/// Every simulation starts from a particle drawn uniformly from `belief` and walks down a
/// tree of action-observation histories: in each history it takes the action with the best
/// UCB1 score `Q + c * sqrt(ln N / n)` (untried actions first, in emission order), samples
/// the step with [`Pomdp::step`] and follows the observation. On reaching a history not in
/// the tree, it adds it and estimates its value with a rollout of `rollout` (use
/// [`UniformRandom`](crate::UniformRandom) without domain knowledge). The discounted return
/// is then averaged into every action on the path.
///
/// The recommendation is the root action with the highest average return (not the UCB
/// score); [`Plan::value`] is that average, an estimate of the discounted return from
/// `belief`. POMCP has no stopping rule: it runs until the budget is exhausted and reports
/// [`Status::BudgetExhausted`].
///
/// This is the one-shot planning step. To plan a sequence of decisions, keep a [`Pomcp`]
/// planner, which reuses its tree, and track the belief with
/// [`update_belief`](crate::update_belief()).
///
/// # Errors
///
/// Returns [`Error::InvalidConfig`] for an invalid configuration, [`Error::EmptyBelief`] for
/// an empty `belief`, [`Error::NoActions`] if the POMDP has no actions and
/// [`Error::InvalidReward`] if a sampled reward is not finite.
///
/// # Panics
///
/// Panics if more than `u32::MAX` tree nodes are stored.
///
/// # Example
///
/// The hidden coin from [`Pomdp`]: a blind call has expected reward 0; peeking first and
/// calling the side seen has `-1 + 0.95 * (0.9 * 10 - 0.1 * 10) = 6.6`.
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
/// use stochastic_search::{PomcpConfig, UniformRandom, pomcp};
///
/// let config = PomcpConfig { exploration: 20.0, ..PomcpConfig::default() };
/// let plan = pomcp(&Coin, &[true, false], UniformRandom, config)?;
///
/// assert_eq!(plan.action, "peek");
/// # Ok::<(), stochastic_search::Error<bool>>(())
/// ```
pub fn pomcp<P, R, B, O>(
    pomdp: &P,
    belief: &[P::State],
    rollout: R,
    config: PomcpConfig<B, O>,
) -> Result<Plan<P::Action>, Error<P::State>>
where
    P: Pomdp,
    P::State: Clone,
    P::Action: Clone,
    P::Observation: PartialEq,
    R: RolloutPolicy<P::State>,
    B: Budget<TreeStats>,
    O: Observer,
{
    Pomcp::new(pomdp, rollout, config)?.plan(belief)
}

/// The paper's exploration constant `c = R_hi - R_lo` for [`PomcpConfig::exploration`]:
/// runs `simulations` simulations of POMCP from `belief` with `c = 0` and returns the highest
/// discounted return of a simulation minus the lowest discounted return of a rollout
/// (Silver and Veness 2010, Section 6: "`R_hi` was the highest return achieved during
/// sample runs of POMCP with `c = 0`, and `R_lo` was the lowest return achieved during
/// sample rollouts").
///
/// Uses `config` for everything but the exploration constant, budget and observer. If no
/// rollout ran (every simulation ended in the tree), `R_lo` is the lowest simulation
/// return instead. The result is finite and non-negative.
///
/// See also [`PomcpConfig::with_reward_range`], which needs no simulations but only bounds
/// the spread of returns from per-step rewards.
///
/// # Errors
///
/// As for [`pomcp`], and [`Error::InvalidConfig`] for zero `simulations`.
///
/// # Example
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
/// use stochastic_search::{PomcpConfig, UniformRandom, pomcp, pomcp_exploration};
///
/// let base = PomcpConfig::default();
/// let c = pomcp_exploration(&Coin, &[true, false], UniformRandom, &base, 1_000)?;
/// assert!(c > 0.0);
///
/// let plan = pomcp(&Coin, &[true, false], UniformRandom, PomcpConfig { exploration: c, ..base })?;
/// assert_eq!(plan.action, "peek");
/// # Ok::<(), stochastic_search::Error<bool>>(())
/// ```
pub fn pomcp_exploration<P, R, B, O>(
    pomdp: &P,
    belief: &[P::State],
    rollout: R,
    config: &PomcpConfig<B, O>,
    simulations: u64,
) -> Result<f64, Error<P::State>>
where
    P: Pomdp,
    P::State: Clone,
    P::Action: Clone,
    P::Observation: PartialEq,
    R: RolloutPolicy<P::State>,
{
    if simulations == 0 {
        return Err(Error::InvalidConfig {
            reason: "pomcp_exploration needs at least one simulation",
        });
    }

    let config = PomcpConfig {
        discount: config.discount,
        exploration: 0.0,
        max_depth: config.max_depth,
        seed: config.seed,
        budget: Limits {
            max_trials: Some(simulations),
            ..PomcpConfig::DEFAULT_LIMITS
        },
        observer: (),
        progress_interval: None,
    };
    let mut planner = Pomcp::new(pomdp, rollout, config)?;
    planner.plan(belief)?;
    let returns = planner.returns;
    let lowest = if returns.lowest_rollout.is_finite() {
        returns.lowest_rollout
    } else {
        returns.lowest
    };

    Ok((returns.highest - lowest).max(0.0))
}

/// A POMCP planner that keeps its search tree between decisions.
///
/// [`pomcp`] is one [`Pomcp::plan`] on a fresh planner. Keeping the planner lets
/// [`Pomcp::advance`] reuse the subtree of the action taken and the observation received,
/// as the paper does ("the node `T(h a o)` becomes the root of the new search tree"); the
/// next [`Pomcp::plan`] continues from its statistics. The simulation generator also
/// carries over, so a sequence of calls is reproducible from [`PomcpConfig::seed`].
///
/// The belief for the next call is the caller's: track it with
/// [`update_belief`](crate::update_belief()) (the paper's particle filter with
/// reinvigoration), from the same action and observation.
///
/// # Example
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
/// use stochastic_search::{Pomcp, PomcpConfig, UniformRandom};
///
/// let config = PomcpConfig { exploration: 20.0, ..PomcpConfig::default() };
/// let mut planner = Pomcp::new(&Coin, UniformRandom, config)?;
/// let plan = planner.plan(&[true, false])?;
/// assert_eq!(plan.action, "peek");
///
/// // Peeked and saw heads: keep the subtree below ("peek", Some(true)).
/// assert!(planner.advance(&"peek", &Some(true)));
/// let plan = planner.plan(&[true, true, true, false])?;
/// assert_eq!(plan.action, "heads");
/// # Ok::<(), stochastic_search::Error<bool>>(())
/// ```
pub struct Pomcp<'a, P: Pomdp, R, B = Limits, O = ()> {
    /// The model.
    pomdp: &'a P,
    /// The actions, in emission order.
    actions: Vec<P::Action>,
    /// The search tree.
    tree: Tree<P::Observation>,
    /// The rollout policy.
    rollout: R,
    /// The generator of all simulations.
    rng: SplitMix64,
    /// The discount factor.
    discount: f64,
    /// The UCB exploration constant.
    exploration: f64,
    /// The step limit of a simulation.
    max_depth: usize,
    /// The caller's budget.
    budget: B,
    /// The caller's observer.
    observer: O,
    /// Simulations between progress events.
    progress_interval: Option<u64>,
    /// The counters so far.
    stats: TreeStats,
    /// Reused buffer: the action nodes of the current simulation and their rewards.
    path: Vec<(u32, f64)>,
    /// The extreme returns seen, for [`pomcp_exploration`].
    returns: Returns,
}

/// The extreme returns of a planner's simulations: the paper's `R_hi` and `R_lo`.
#[derive(Debug, Clone, Copy)]
struct Returns {
    /// The highest discounted return of a whole simulation, from the root.
    highest: f64,
    /// The lowest discounted return of a rollout, from where it started.
    lowest_rollout: f64,
    /// The lowest discounted return of a whole simulation, the fallback for `R_lo` when
    /// no rollout ran.
    lowest: f64,
}

impl Default for Returns {
    fn default() -> Self {
        Self {
            highest: f64::NEG_INFINITY,
            lowest_rollout: f64::INFINITY,
            lowest: f64::INFINITY,
        }
    }
}

impl<'a, P, R, B, O> Pomcp<'a, P, R, B, O>
where
    P: Pomdp,
    P::State: Clone,
    P::Action: Clone,
    P::Observation: PartialEq,
    R: RolloutPolicy<P::State>,
    B: Budget<TreeStats>,
    O: Observer,
{
    /// Sets up a planner with an empty tree.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfig`] for an invalid configuration and
    /// [`Error::NoActions`] if the POMDP has no actions.
    pub fn new(
        pomdp: &'a P,
        rollout: R,
        config: PomcpConfig<B, O>,
    ) -> Result<Self, Error<P::State>> {
        config.validate()?;
        let actions = collect_actions(pomdp)?;

        Ok(Self {
            pomdp,
            tree: Tree::new(actions.len()),
            actions,
            rollout,
            rng: SplitMix64::new(config.seed),
            discount: config.discount,
            exploration: config.exploration,
            max_depth: config.max_depth,
            budget: config.budget,
            observer: config.observer,
            progress_interval: config.progress_interval,
            stats: TreeStats::default(),
            path: Vec::new(),
            returns: Returns::default(),
        })
    }

    /// Runs simulations from particles of `belief` until the budget is exhausted, and
    /// recommends an action as [`pomcp`] does. Continues the current tree: after
    /// [`Pomcp::advance`], from the reused subtree.
    ///
    /// The counters of the returned [`Plan`] count this call only, except
    /// [`TreeStats::nodes`], the size of the whole tree; the budget sees the same counters,
    /// so a [`Limits`] budget applies per call.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EmptyBelief`] for an empty `belief` and [`Error::InvalidReward`] if
    /// a sampled reward is not finite.
    ///
    /// # Panics
    ///
    /// Panics if more than `u32::MAX` tree nodes are stored.
    pub fn plan(&mut self, belief: &[P::State]) -> Result<Plan<P::Action>, Error<P::State>> {
        if belief.is_empty() {
            return Err(Error::EmptyBelief);
        }

        self.stats = TreeStats::default();
        while !self.exhausted() {
            self.stats.simulations += 1;
            let particle = belief[below(&mut self.rng, belief.len())].clone();
            self.simulate(particle)?;
            self.report_progress();
        }

        Ok(self.recommend())
    }

    /// Makes the history after `action` and `observation` the new root, keeping its subtree
    /// and dropping everything else. Returns `false` (and starts over with an empty tree) if
    /// the tree never saw that history, or `action` is not one of the POMDP's actions.
    pub fn advance(&mut self, action: &P::Action, observation: &P::Observation) -> bool
    where
        P::Action: PartialEq,
    {
        let position = self.actions.iter().position(|known| known == action);
        let tree = core::mem::replace(&mut self.tree, Tree::new(self.actions.len()));
        match position.and_then(|position| tree.child_of_root(position, observation)) {
            Some(child) => {
                self.tree = tree.into_subtree(child);
                true
            }
            None => false,
        }
    }

    /// The number of history nodes in the tree.
    #[must_use]
    pub fn tree_size(&self) -> usize {
        self.tree.len()
    }

    /// Emits [`Event::TreeProgress`] if one is due after the current simulation.
    fn report_progress(&mut self) {
        let due = self
            .progress_interval
            .is_some_and(|interval| self.stats.simulations % interval == 0);
        if O::ENABLED && due {
            self.stats.nodes = self.tree.len() as u64;
            let (_, value) = self.tree.best_root_action();
            self.observer.observe(Event::TreeProgress {
                stats: self.stats,
                value,
                gap: None,
            });
        }
    }

    /// Syncs the counters and asks the budget whether to stop.
    fn exhausted(&mut self) -> bool {
        self.stats.nodes = self.tree.len() as u64;

        self.budget.exhausted(&self.stats)
    }

    /// Runs one simulation from `state` and backs up its return.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidReward`] for a non-finite reward.
    fn simulate(&mut self, state: P::State) -> Result<(), Error<P::State>> {
        self.path.clear();
        let tail = self.descend(state)?;
        self.back_up(tail);

        Ok(())
    }

    /// Walks down the tree from the root, recording the path, until it adds a history (then
    /// rolls out from it), the episode ends or the depth limit. Returns the discounted
    /// return after the last recorded step.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidReward`] for a non-finite reward.
    fn descend(&mut self, mut state: P::State) -> Result<f64, Error<P::State>> {
        let mut node = ROOT;
        for depth in 0..self.max_depth {
            if !self.tree.is_expanded(node) {
                self.tree.expand(node);
                return self.roll_out(state, depth);
            }

            let action = self.tree.select(node, self.exploration);
            let label = &self.actions[self.tree.position(node, action)];
            let step = self.pomdp.step(&state, label, &mut self.rng);
            self.stats.steps += 1;
            check_reward(&state, step.reward)?;
            self.path.push((action, step.reward));
            if step.terminal {
                return Ok(0.0);
            }

            node = self.tree.child(action, step.observation);
            self.stats.depth = self.stats.depth.max(depth as u64 + 1);
            state = step.state;
        }

        Ok(0.0)
    }

    /// Simulates the rollout policy from `state`, `depth` steps below the root, until the
    /// episode ends or the depth limit, and returns the discounted return.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidReward`] for a non-finite reward.
    fn roll_out(&mut self, mut state: P::State, depth: usize) -> Result<f64, Error<P::State>> {
        let mut total = 0.0;
        let mut weight = 1.0;
        for _ in depth..self.max_depth {
            let index = self
                .rollout
                .action(&state, self.actions.len(), &mut self.rng);
            let step = self.pomdp.step(&state, &self.actions[index], &mut self.rng);
            self.stats.steps += 1;
            check_reward(&state, step.reward)?;
            total += weight * step.reward;
            if step.terminal {
                break;
            }

            weight *= self.discount;
            state = step.state;
        }

        self.returns.lowest_rollout = self.returns.lowest_rollout.min(total);

        Ok(total)
    }

    /// Averages the discounted returns into the action nodes of the path, deepest first,
    /// starting from `tail`, the return after the last step.
    fn back_up(&mut self, tail: f64) {
        let mut value = tail;
        for &(action, reward) in self.path.iter().rev() {
            value = reward + self.discount * value;
            self.tree.record(action, value);
            self.stats.backups += 1;
        }

        self.returns.highest = self.returns.highest.max(value);
        self.returns.lowest = self.returns.lowest.min(value);
    }

    /// The root action with the highest average return, ties to the first emitted.
    fn recommend(&mut self) -> Plan<P::Action> {
        let (position, value) = self.tree.best_root_action();
        self.stats.nodes = self.tree.len() as u64;

        Plan {
            action: self.actions[position].clone(),
            value,
            status: Status::BudgetExhausted,
            stats: self.stats,
        }
    }
}
