//! The generative POMDP model ([`Pomdp`], [`Step`]), rollout policies ([`RolloutPolicy`],
//! [`UniformRandom`]) and the result of an online planning call ([`Plan`]).
//!
//! The belief-space planners [`pomcp`](crate::pomcp()) and [`despot`](crate::despot())
//! never see transition or observation probabilities. They only sample: given a hidden
//! state and an action, the model draws a successor, an observation and a reward from a
//! caller-visible random generator. Beliefs are particle sets: a slice of states, each an
//! equally likely guess of the hidden state.

use rand_core::Rng;

use crate::budget::TreeStats;
use crate::error::Error;
use crate::solution::Status;

/// A partially observable Markov decision process, given as a generative model.
///
/// Unlike [`Mdp`](crate::Mdp), a POMDP maximizes discounted reward, as in the POMCP and
/// DESPOT papers: the planners maximize `E[sum_t gamma^t r_t]`. The action set is the same
/// in every state (the agent cannot tell states apart), so [`Pomdp::actions`] takes no
/// state.
///
/// # Determinism
///
/// [`Pomdp::step`] must draw all its randomness from `rng`. Then a seed reproduces a
/// planning call, and DESPOT's determinized scenarios work as intended: the same state,
/// action and generator state must give the same [`Step`].
///
/// # Example
///
/// A coin whose side is hidden. `peek` costs 1 and shows the side with probability 0.9;
/// `heads` and `tails` call a side and end the episode with reward 10 if right, -10 if
/// wrong.
///
/// ```
/// use rand_core::Rng;
/// use stochastic_search::{Pomdp, Step, unit_interval};
///
/// struct Coin;
///
/// impl Pomdp for Coin {
///     type State = bool; // heads?
///     type Action = &'static str;
///     type Observation = Option<bool>;
///
///     fn actions(&self, mut emit: impl FnMut(&'static str)) {
///         emit("peek");
///         emit("heads");
///         emit("tails");
///     }
///
///     fn step<R: Rng + ?Sized>(&self, heads: &bool, action: &&'static str, rng: &mut R)
///         -> Step<bool, Option<bool>>
///     {
///         if *action == "peek" {
///             let truthful = unit_interval(rng) < 0.9;
///             Step::new(*heads, Some(*heads == truthful), -1.0)
///         } else {
///             let right = *heads == (*action == "heads");
///             Step::new(*heads, None, if right { 10.0 } else { -10.0 }).terminal()
///         }
///     }
/// }
/// ```
pub trait Pomdp {
    /// A hidden state. Planners clone states into particle sets.
    type State;
    /// An action label, returned in [`Plan::action`].
    type Action;
    /// An observation. Planners compare observations with `==` to group histories.
    type Observation;

    /// Calls `emit` once for every action, in a fixed order. There must be at least one.
    ///
    /// Called once per planning call; planners refer to actions by their position in this
    /// order (as [`RolloutPolicy::action`] does).
    fn actions(&self, emit: impl FnMut(Self::Action));

    /// Samples the result of taking `action` in the hidden `state`: the successor state,
    /// the observation the agent receives, and the reward. The reward must be finite.
    ///
    /// Never called on a state that ended an episode ([`Step::terminal`]).
    fn step<R: Rng + ?Sized>(
        &self,
        state: &Self::State,
        action: &Self::Action,
        rng: &mut R,
    ) -> Step<Self::State, Self::Observation>;
}

/// One sampled transition of a [`Pomdp`].
#[derive(Debug, Clone, PartialEq)]
pub struct Step<S, O> {
    /// The successor state.
    pub state: S,
    /// The observation received after the transition.
    pub observation: O,
    /// The reward of the transition. Must be finite.
    pub reward: f64,
    /// Whether the episode ends with this transition. No reward follows a terminal step.
    pub terminal: bool,
}

impl<S, O> Step<S, O> {
    /// A non-terminal step.
    #[must_use]
    pub fn new(state: S, observation: O, reward: f64) -> Self {
        Self {
            state,
            observation,
            reward,
            terminal: false,
        }
    }

    /// The same step, marked as ending the episode.
    #[must_use]
    pub fn terminal(self) -> Self {
        Self {
            terminal: true,
            ..self
        }
    }
}

/// Picks actions for simulations below the search tree: POMCP's rollout policy and
/// DESPOT's default policy (whose simulated value is DESPOT's lower bound).
///
/// A good rollout policy is the main domain knowledge the planners can use; the default,
/// [`UniformRandom`], knows nothing. Implemented for `&mut P` of any policy `P`.
pub trait RolloutPolicy<S> {
    /// Returns the position of the action to take in `state`, below `actions` (the number
    /// of actions [`Pomdp::actions`] emits). Must draw all its randomness from `rng`.
    fn action<R: Rng + ?Sized>(&mut self, state: &S, actions: usize, rng: &mut R) -> usize;
}

impl<S, P: RolloutPolicy<S> + ?Sized> RolloutPolicy<S> for &mut P {
    fn action<R: Rng + ?Sized>(&mut self, state: &S, actions: usize, rng: &mut R) -> usize {
        (**self).action(state, actions, rng)
    }
}

/// The rollout policy that picks every action with equal probability.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UniformRandom;

impl<S> RolloutPolicy<S> for UniformRandom {
    fn action<R: Rng + ?Sized>(&mut self, _state: &S, actions: usize, rng: &mut R) -> usize {
        below(rng, actions)
    }
}

/// Returns a uniform sample from `0..bound` (`bound > 0`).
///
/// Uses the 64-bit multiply-shift reduction; its bias of at most `bound / 2^64` does not
/// matter for simulations.
pub(crate) fn below<R: Rng + ?Sized>(rng: &mut R, bound: usize) -> usize {
    let product = u128::from(rng.next_u64()) * bound as u128;

    usize::try_from(product >> 64).expect("the result is below bound, a usize")
}

/// The answer of an online POMDP planner for one decision.
#[derive(Debug, Clone, PartialEq)]
pub struct Plan<A> {
    /// The recommended action.
    pub action: A,
    /// The planner's estimate of the recommended action's value from the given belief. See
    /// [`pomcp`](crate::pomcp()) and [`despot`](crate::despot()) for what it means for each.
    pub value: f64,
    /// [`Status::Converged`] if the planner's own stopping rule fired (DESPOT's bound gap),
    /// [`Status::BudgetExhausted`] if the budget stopped it. POMCP has no stopping rule, so
    /// it always reports an exhausted budget.
    pub status: Status,
    /// Counters of the call; see each planner for what they count.
    pub stats: TreeStats,
}

/// Collects the actions of `pomdp` in emission order.
///
/// # Errors
///
/// Returns [`Error::NoActions`] if there are none.
pub(crate) fn collect_actions<P: Pomdp>(
    pomdp: &P,
) -> Result<alloc::vec::Vec<P::Action>, Error<P::State>> {
    let mut actions = alloc::vec::Vec::new();
    pomdp.actions(|action| actions.push(action));
    if actions.is_empty() {
        return Err(Error::NoActions);
    }

    Ok(actions)
}

/// Checks that a sampled reward is finite.
///
/// # Errors
///
/// Returns [`Error::InvalidReward`] with the state the step started from otherwise.
pub(crate) fn check_reward<S: Clone>(state: &S, reward: f64) -> Result<(), Error<S>> {
    if reward.is_finite() {
        Ok(())
    } else {
        Err(Error::InvalidReward {
            state: state.clone(),
            reward,
        })
    }
}

/// Checks the discount factor and the depth limit shared by the planners' configurations.
///
/// # Errors
///
/// Returns [`Error::InvalidConfig`] for a discount outside `(0, 1]` or a zero depth.
pub(crate) fn check_horizon<S>(discount: f64, max_depth: usize) -> Result<(), Error<S>> {
    if !(discount > 0.0 && discount <= 1.0) {
        return Err(Error::InvalidConfig {
            reason: "discount must lie in (0, 1]",
        });
    }

    if max_depth == 0 {
        return Err(Error::InvalidConfig {
            reason: "max_depth must be at least 1",
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::below;
    use crate::rng::SplitMix64;

    /// Every value below the bound shows up, and nothing at or above it.
    #[test]
    fn below_covers_the_range() {
        let mut rng = SplitMix64::new(3);
        let mut seen = [false; 5];
        for _ in 0..1000 {
            seen[below(&mut rng, 5)] = true;
        }

        assert!(seen.iter().all(|&s| s));
    }
}
