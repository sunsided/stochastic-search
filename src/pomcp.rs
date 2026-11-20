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

use crate::budget::{Budget, Limits, Stats};
use crate::error::Error;
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
pub struct PomcpConfig<B = Limits> {
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
    /// Stops the search. Asked before every simulation, with [`Stats::trials`] counting
    /// simulations, [`Stats::states`] the history nodes in the tree and [`Stats::backups`]
    /// the action-value updates. Default [`PomcpConfig::DEFAULT_LIMITS`].
    pub budget: B,
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
        }
    }
}

impl<B> PomcpConfig<B> {
    /// Returns the same configuration with another budget.
    #[must_use]
    pub fn with_budget<B2: Budget>(self, budget: B2) -> PomcpConfig<B2> {
        PomcpConfig {
            discount: self.discount,
            exploration: self.exploration,
            max_depth: self.max_depth,
            seed: self.seed,
            budget,
        }
    }

    /// Checks the numeric settings.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfig`] for a discount outside `(0, 1]`, a negative or
    /// non-finite exploration constant or a zero depth.
    fn validate<S>(&self) -> Result<(), Error<S>> {
        check_horizon(self.discount, self.max_depth)?;
        if !(self.exploration.is_finite() && self.exploration >= 0.0) {
            return Err(Error::InvalidConfig {
                reason: "exploration must be finite and non-negative",
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
/// This is the one-shot planning step. Belief tracking between decisions (POMCP's particle
/// filter in the tree) is up to the caller: pass the posterior particles to the next call.
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
pub fn pomcp<P, R, B>(
    pomdp: &P,
    belief: &[P::State],
    rollout: R,
    config: PomcpConfig<B>,
) -> Result<Plan<P::Action>, Error<P::State>>
where
    P: Pomdp,
    P::State: Clone,
    P::Action: Clone,
    P::Observation: PartialEq,
    R: RolloutPolicy<P::State>,
    B: Budget,
{
    config.validate()?;
    if belief.is_empty() {
        return Err(Error::EmptyBelief);
    }

    let actions = collect_actions(pomdp)?;
    let mut search = Search {
        pomdp,
        tree: Tree::new(actions.len()),
        actions,
        rollout,
        rng: SplitMix64::new(config.seed),
        discount: config.discount,
        exploration: config.exploration,
        max_depth: config.max_depth,
        budget: config.budget,
        stats: Stats::default(),
        path: Vec::new(),
    };

    while !search.exhausted() {
        search.stats.trials += 1;
        let particle = belief[below(&mut search.rng, belief.len())].clone();
        search.simulate(particle)?;
    }

    Ok(search.recommend())
}

/// The state of one [`pomcp`] call.
struct Search<'a, P: Pomdp, R, B> {
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
    /// The counters so far.
    stats: Stats,
    /// Reused buffer: the action nodes of the current simulation and their rewards.
    path: Vec<(u32, f64)>,
}

impl<P, R, B> Search<'_, P, R, B>
where
    P: Pomdp,
    P::State: Clone,
    P::Action: Clone,
    P::Observation: PartialEq,
    R: RolloutPolicy<P::State>,
    B: Budget,
{
    /// Syncs the counters and asks the budget whether to stop.
    fn exhausted(&mut self) -> bool {
        self.stats.states = self.tree.len() as u64;

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
            check_reward(&state, step.reward)?;
            self.path.push((action, step.reward));
            if step.terminal {
                return Ok(0.0);
            }

            node = self.tree.child(action, step.observation);
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

    /// Averages the discounted returns into the action nodes of the path, deepest first,
    /// starting from `tail`, the return after the last step.
    fn back_up(&mut self, tail: f64) {
        let mut value = tail;
        for &(action, reward) in self.path.iter().rev() {
            value = reward + self.discount * value;
            self.tree.record(action, value);
            self.stats.backups += 1;
        }
    }

    /// The root action with the highest average return, ties to the first emitted.
    fn recommend(mut self) -> Plan<P::Action> {
        let (position, value) = self.tree.best_root_action();
        self.stats.states = self.tree.len() as u64;

        Plan {
            action: self.actions[position].clone(),
            value,
            status: Status::BudgetExhausted,
            stats: self.stats,
        }
    }
}
