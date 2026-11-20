//! Plain real-time dynamic programming (RTDP, Barto, Bradtke and Singh 1995).

use core::hash::Hash;

use crate::budget::Budget;
use crate::config::Config;
use crate::lrtdp::Run;
use crate::meter::finish;
use crate::model::{Goal, Heuristic, Mdp};
use crate::observer::Observer;
use crate::solution::SolveResult;

/// Solves an SSP from `initial` with plain RTDP: greedy trials without solved labels.
///
/// Each trial starts at `initial`, backs up the current state, takes its greedy action and
/// samples a successor with a generator seeded from [`Config::seed`], until it reaches a
/// goal, a dead end or [`Config::max_trial_length`] steps. RTDP itself has no stopping
/// rule; this one checks between trials whether every state reachable from `initial` under
/// the greedy policy has a Bellman residual of at most [`Config::epsilon`] (the solved check
/// of [`lrtdp`](crate::lrtdp()), run only on `initial` and without backups). A check costs
/// about one evaluation per envelope state, so it runs only once as many backups happened
/// since the last check as that check took. When it passes, the policy is frozen and the
/// status is [`Status::Converged`](crate::Status::Converged).
///
/// Prefer [`lrtdp`](crate::lrtdp()): its labels stop trials at converged states, which makes
/// it much faster on states that trials rarely reach. Plain RTDP is here as the classic
/// baseline and for comparisons.
///
/// # Guarantee
///
/// A converged run has exactly the guarantee of [`lrtdp`](crate::lrtdp()) (section
/// "Guarantee" there): the passing check is the same solved check, on `initial`.
///
/// # Termination
///
/// As for [`lrtdp`](crate::lrtdp()), including the dead-end proofs under
/// [`DeadEnds::Infinite`](crate::DeadEnds::Infinite). Without labels, a run may take many
/// trials to push a rarely sampled outcome below epsilon; the default
/// [`Limits`](crate::Limits) budget still bounds it.
///
/// # Errors
///
/// Returns an [`Error`](crate::Error) for an invalid [`Config`], or for invalid MDP output or
/// heuristic estimates in the states the run touches.
///
/// # Panics
///
/// Panics if more than `u32::MAX` states, actions or outcomes are stored.
///
/// # Example
///
/// A sure route for cost 3, or a coin flip for cost 1 that retries on failure (expected
/// cost 2):
///
/// ```
/// use stochastic_search::{Config, Mdp, Zero, rtdp};
///
/// struct Routes;
///
/// impl Mdp for Routes {
///     type State = bool;
///     type Action = &'static str;
///
///     fn actions(&self, _done: &bool, mut emit: impl FnMut(&'static str, f64)) {
///         emit("sure", 3.0);
///         emit("flip", 1.0);
///     }
///
///     fn outcomes(&self, _done: &bool, action: &&'static str, mut emit: impl FnMut(bool, f64)) {
///         if *action == "sure" {
///             emit(true, 1.0);
///         } else {
///             emit(true, 0.5);
///             emit(false, 0.5);
///         }
///     }
/// }
///
/// let solution = rtdp(&Routes, false, &|done: &bool| *done, Zero, Config::default())?;
///
/// assert!(solution.is_converged());
/// assert!((solution.initial_value() - 2.0).abs() < 1e-5);
/// assert_eq!(solution.initial_action(), Some(&"flip"));
/// # Ok::<(), stochastic_search::Error<bool>>(())
/// ```
pub fn rtdp<M, G, H, B, O>(
    mdp: &M,
    initial: M::State,
    goal: &G,
    heuristic: H,
    config: Config<B, O>,
) -> SolveResult<M>
where
    M: Mdp,
    M::State: Clone + Eq + Hash,
    G: Goal<M::State>,
    H: Heuristic<M::State, f64>,
    B: Budget,
    O: Observer,
{
    let mut run = Run::new(mdp, initial, goal, heuristic, config)?;
    let status = finish(run.solve(false))?;

    Ok(run.into_solution(status))
}
