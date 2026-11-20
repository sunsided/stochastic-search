//! Focused real-time dynamic programming (FRTDP, Smith and Simmons 2006): priority-driven
//! successor choice and adaptive maximum depth.

use core::hash::Hash;

use alloc::vec::Vec;

use crate::bounds::{Explorer, Run, solve};
use crate::budget::{Budget, Limits, Stats};
use crate::config::Config;
use crate::error::Error;
use crate::model::{Goal, Heuristic, Mdp};
use crate::observer::Observer;
use crate::solution::SolveResult;
use crate::space::Id;

/// Settings of [`frtdp`]: the shared [`Config`] and FRTDP's adaptive maximum depth.
///
/// Build one with [`FrtdpConfig::default`], or with [`FrtdpConfig::new`] (or `into`) from a
/// [`Config`], and adjust it with [`FrtdpConfig::with_depth`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrtdpConfig<B = Limits, O = ()> {
    /// Epsilon (the gap that ends the run, and twice the gap below which a state counts as
    /// settled), dead ends, trial length, budget and observer. The seed only drives the
    /// fallback BRTDP trials (see [`frtdp`]).
    pub base: Config<B, O>,
    /// The initial maximum trial depth `D0`. Finite and at least `1`. Default `10`, as in
    /// the paper.
    pub initial_depth: f64,
    /// The factor `kD` by which the maximum depth grows after a trial whose late updates
    /// were at least as useful as its early ones. Finite and at least `1`; `1` keeps the
    /// depth fixed. Default `1.1`, as in the paper.
    pub depth_factor: f64,
}

impl Default for FrtdpConfig {
    fn default() -> Self {
        Self::new(Config::default())
    }
}

impl<B, O> FrtdpConfig<B, O> {
    /// Wraps `base` with the paper's depth settings, `D0 = 10` and `kD = 1.1`.
    #[must_use]
    pub fn new(base: Config<B, O>) -> Self {
        Self {
            base,
            initial_depth: 10.0,
            depth_factor: 1.1,
        }
    }

    /// Returns the same settings with another initial depth `D0` and growth factor `kD`.
    #[must_use]
    pub fn with_depth(self, initial_depth: f64, depth_factor: f64) -> Self {
        Self {
            initial_depth,
            depth_factor,
            ..self
        }
    }

    /// Checks the depth settings.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfig`] for an initial depth or depth factor that is not
    /// finite and at least 1.
    fn validate<S>(&self) -> Result<(), Error<S>> {
        let valid = |x: f64| x.is_finite() && x >= 1.0;
        if valid(self.initial_depth) && valid(self.depth_factor) {
            Ok(())
        } else {
            Err(Error::InvalidConfig {
                reason: "initial_depth and depth_factor must be finite and at least 1",
            })
        }
    }
}

impl<B, O> From<Config<B, O>> for FrtdpConfig<B, O> {
    fn from(base: Config<B, O>) -> Self {
        Self::new(base)
    }
}

/// Solves an SSP from `initial` with focused RTDP, keeping a lower and an upper bound on
/// every stored state's optimal value.
///
/// FRTDP (Smith and Simmons 2006) differs from [`brtdp`](crate::brtdp()) only in how trials
/// pick states; bounds, dead-end handling, guarantee, policy (greedy on the upper bound,
/// with the fill pass after convergence), events and errors are as documented there.
///
/// Every state carries a priority, initially its excess gap `gap(s) - epsilon / 2`, and after
/// each backup `min(gap(s) - epsilon / 2, max_s' P(s' | s, a) * priority(s'))` for the
/// lower-greedy action `a`: an estimate of how much the uncertainty below `s` contributes to
/// the gap at `initial`. A trial follows, deterministically, the successor of highest
/// probability times priority, and ends at a state whose excess gap is not positive or at
/// the current maximum depth `D` (the paper's adaptive maximum depth). `D` starts at
/// [`FrtdpConfig::initial_depth`] and is multiplied by [`FrtdpConfig::depth_factor`] after a
/// trial whose average update quality (the change of the lower bound times the path
/// probability) in its late part (depth above `D / kD`) is at least that of its early part;
/// an empty part averages 0. [`Config::max_trial_length`] still caps every trial.
///
/// The paper maximizes rewards; here its optimistic bound is the lower bound and its
/// pessimistic one the upper bound. Ties go to the first successor emitted.
///
/// # Deviation from the paper
///
/// The deterministic choice can starve a successor: on a random SSP, trials cycled through
/// states whose backups no longer changed anything while a low-probability successor that
/// held the gap up was never chosen, and the run stalled. After an FRTDP trial that did not
/// shrink the initial state's gap, the next trial therefore runs as a plain BRTDP trial
/// (with `tau = 10`), which samples successors by gap with the seed of
/// [`Config::seed`]. This at worst halves the speed of BRTDP and restores its convergence.
///
/// # Errors
///
/// As for [`brtdp`](crate::brtdp()), with invalid depth settings reported as
/// [`Error::InvalidConfig`].
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
/// use stochastic_search::{FrtdpConfig, Mdp, Zero, frtdp, heuristic_fn};
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
/// let upper = heuristic_fn(|_: &bool| Some(3.0));
/// let config = FrtdpConfig::default();
/// let solution = frtdp(&Routes, false, &|done: &bool| *done, Zero, upper, config)?;
///
/// assert!(solution.is_converged());
/// let (lower, upper) = (solution.initial_value(), solution.initial_upper_bound().unwrap());
/// assert!(lower <= 2.0 && 2.0 <= upper && upper - lower <= 1e-6);
/// assert_eq!(solution.initial_action(), Some(&"flip"));
/// # Ok::<(), stochastic_search::Error<bool>>(())
/// ```
pub fn frtdp<M, G, L, U, B, O>(
    mdp: &M,
    initial: M::State,
    goal: &G,
    lower: L,
    upper: U,
    config: FrtdpConfig<B, O>,
) -> SolveResult<M>
where
    M: Mdp,
    M::State: Clone + Eq + Hash,
    G: Goal<M::State>,
    L: Heuristic<M::State, f64>,
    U: Heuristic<M::State, f64>,
    B: Budget<Stats>,
    O: Observer,
{
    config.validate()?;
    let explorer = Explorer::Frtdp(Focus::new(config.initial_depth, config.depth_factor));

    solve((mdp, initial, goal), lower, upper, config.base, explorer)
}

/// FRTDP's per-run state: priorities and the adaptive maximum depth.
pub(crate) struct Focus {
    /// The priority of every stored state, indexed by [`Id`].
    pub(crate) priority: Vec<f64>,
    /// The current maximum trial depth `D`.
    depth_limit: f64,
    /// The growth factor `kD`.
    depth_factor: f64,
    /// The probability of the current trial's path so far (the occupancy estimate `W`).
    path_probability: f64,
    /// Summed update quality and update count in the early part of the current trial.
    early: (f64, u32),
    /// Summed update quality and update count in the late part of the current trial.
    late: (f64, u32),
}

impl Focus {
    /// Starts with maximum depth `initial_depth` and growth factor `depth_factor`.
    fn new(initial_depth: f64, depth_factor: f64) -> Self {
        Self {
            priority: Vec::new(),
            depth_limit: initial_depth,
            depth_factor,
            path_probability: 1.0,
            early: (0.0, 0),
            late: (0.0, 0),
        }
    }

    /// Resets the per-trial statistics.
    pub(crate) fn start_trial(&mut self) {
        self.path_probability = 1.0;
        self.early = (0.0, 0);
        self.late = (0.0, 0);
    }

    /// Grows the maximum depth if the late updates of the trial were at least as useful
    /// as the early ones.
    pub(crate) fn end_trial(&mut self) {
        let average = |(sum, count): (f64, u32)| {
            if count == 0 {
                0.0
            } else {
                sum / f64::from(count)
            }
        };

        if average(self.late) >= average(self.early) {
            self.depth_limit *= self.depth_factor;
        }
    }

    /// Records the quality of an update at trial depth `depth` in the late or early part.
    fn track(&mut self, quality: f64, depth: f64) {
        let part = if depth > self.depth_limit / self.depth_factor {
            &mut self.late
        } else {
            &mut self.early
        };

        part.0 += quality;
        part.1 = part.1.saturating_add(1);
    }
}

impl<M: Mdp, G, L, U, B, O> Run<'_, M, G, L, U, B, O> {
    /// The FRTDP state of this run.
    fn focus(&mut self) -> &mut Focus {
        match &mut self.explorer {
            Explorer::Frtdp(focus) => focus,
            _ => unreachable!("only FRTDP runs ask for their focus"),
        }
    }

    /// The excess gap `gap(s) - epsilon / 2`: positive while `state` is not settled.
    fn excess(&self, state: Id) -> f64 {
        self.gap(state) - self.epsilon / 2.0
    }

    /// Sets the priority of `state` after its backup, from its lower-greedy `action`.
    pub(crate) fn update_priority(&mut self, state: Id, action: Option<u32>) {
        let mut priority = self.excess(state);
        if let Some(action) = action {
            let (_, best) = self.best_successor(action);
            priority = priority.min(best);
        }

        self.focus().priority[state as usize] = priority;
    }

    /// The successor of the action at `index` with the highest probability times priority,
    /// and that product. Ties go to the first outcome.
    fn best_successor(&self, index: u32) -> (Id, f64) {
        let Explorer::Frtdp(focus) = &self.explorer else {
            unreachable!("only FRTDP runs have priorities");
        };

        let mut best = (0, f64::NEG_INFINITY);
        for outcome in self.space.outcomes(index) {
            let score = outcome.probability * focus.priority[outcome.target as usize];
            if score > best.1 {
                best = (outcome.target, score);
            }
        }

        best
    }

    /// FRTDP's successor choice at `state`, backed up just now, under its lower-greedy action
    /// at `index`: records the update quality, ends the trial at a settled state or at the
    /// depth limit, and otherwise follows the successor of highest priority.
    pub(crate) fn frtdp_next(&mut self, state: Id, index: u32, depth: usize) -> Option<Id> {
        let depth = f64::from(u32::try_from(depth).unwrap_or(u32::MAX));
        let change = self.last_change;
        let focus = self.focus();
        focus.track(change * focus.path_probability, depth);
        if self.excess(state) <= 0.0 || depth >= self.focus().depth_limit {
            return None;
        }

        let (next, _) = self.best_successor(index);
        let probability: f64 = self
            .space
            .outcomes(index)
            .iter()
            .filter(|outcome| outcome.target == next)
            .map(|outcome| outcome.probability)
            .sum();
        self.focus().path_probability *= probability;

        Some(next)
    }
}
