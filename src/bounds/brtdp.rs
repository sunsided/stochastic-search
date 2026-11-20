//! Bounded real-time dynamic programming (BRTDP, McMahan, Likhachev and Gordon 2005):
//! gap-weighted successor sampling.

use core::hash::Hash;

use crate::bounds::{Explorer, Run, solve};
use crate::budget::{Budget, Limits, Stats};
use crate::config::Config;
use crate::error::Error;
use crate::model::{Goal, Heuristic, Mdp};
use crate::observer::Observer;
use crate::solution::SolveResult;
use crate::space::Id;

/// Settings of [`brtdp`]: the shared [`Config`] and BRTDP's trial-end ratio.
///
/// Build one with [`BrtdpConfig::default`], or with [`BrtdpConfig::new`] (or `into`) from a
/// [`Config`], and adjust it with [`BrtdpConfig::with_tau`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BrtdpConfig<B = Limits, O = ()> {
    /// Epsilon (the gap that ends the run), dead ends, seed, trial length, budget and
    /// observer.
    pub base: Config<B, O>,
    /// The trial-end ratio `tau`: a trial ends once the expected gap of the next state
    /// falls below the initial state's gap divided by `tau`. Finite and greater than `1`;
    /// larger values make trials longer.
    ///
    /// Default [`BrtdpConfig::DEFAULT_TAU`].
    pub tau: f64,
}

impl BrtdpConfig {
    /// The default `tau`, `10`: the low end of the range the paper used ("we used `tau`
    /// between 10 and 100 in our experiments"), which favours short trials.
    pub const DEFAULT_TAU: f64 = 10.0;
}

impl Default for BrtdpConfig {
    fn default() -> Self {
        Self::new(Config::default())
    }
}

impl<B, O> BrtdpConfig<B, O> {
    /// Wraps `base` with the default `tau`.
    #[must_use]
    pub fn new(base: Config<B, O>) -> Self {
        Self {
            base,
            tau: BrtdpConfig::DEFAULT_TAU,
        }
    }

    /// Returns the same settings with another `tau`.
    #[must_use]
    pub fn with_tau(self, tau: f64) -> Self {
        Self { tau, ..self }
    }

    /// Checks `tau`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfig`] for a `tau` that is not finite and greater than 1.
    fn validate<S>(&self) -> Result<(), Error<S>> {
        if self.tau.is_finite() && self.tau > 1.0 {
            Ok(())
        } else {
            Err(Error::InvalidConfig {
                reason: "tau must be finite and greater than 1",
            })
        }
    }
}

impl<B, O> From<Config<B, O>> for BrtdpConfig<B, O> {
    fn from(base: Config<B, O>) -> Self {
        Self::new(base)
    }
}

/// Solves an SSP from `initial` with bounded RTDP, keeping a lower and an upper bound on
/// every stored state's optimal value.
///
/// `lower` estimates the lower bound of a new state as the heuristic of
/// [`lrtdp`](crate::lrtdp()) does (admissible for the guarantee, `None` for a proven dead
/// end). `upper` estimates its upper bound: at least the optimal value, `None` if no finite
/// bound is known. Each trial starts at `initial`; at every state it backs up both bounds,
/// takes the action that is greedy on the lower bound, and samples a successor with
/// probability proportional to `P(s' | s, a) * (U(s') - L(s'))`, so trials go where the
/// bounds are far apart. A trial ends when that weight summed over the successors falls
/// below `(U(initial) - L(initial)) / tau` ([`BrtdpConfig::tau`]), at a goal or dead end, or
/// after [`Config::max_trial_length`] steps; then it backs up its states again, last first.
/// The run stops when `U(initial) - L(initial) <= epsilon` ([`Config::epsilon`]). After
/// convergence, a fill pass backs up every state that the upper-greedy policy reaches from
/// `initial` but no trial backed up, so the returned policy has an action wherever one
/// exists in its envelope (unless the budget runs out during the pass).
///
/// The returned [`Solution`](crate::Solution) has the lower bounds as [`Solution::value`](crate::Solution::value) and the upper
/// bounds as [`Solution::upper_bound`](crate::Solution::upper_bound). Its policy is greedy on the upper bound, as of each
/// state's last backup, as the paper prescribes (`greedy(v_u)`): only an upper-greedy policy
/// has a cost guarantee. [`frtdp`](crate::frtdp()) and [`vpi_rtdp`](crate::vpi_rtdp())
/// share everything in this documentation except how trials pick states.
///
/// # Dead ends
///
/// An infinite upper bound on a state with a finite optimal value never becomes finite: on a
/// cycle such as "retry until it works", a backup computes `c + p * 0 + (1 - p) * inf`.
/// BRTDP therefore requires a finite upper bound on every state it stores, except where the
/// lower bound already proves a dead end (then both bounds are infinite and the gap counts as
/// `0`, so trials never sample it):
///
/// - Under [`DeadEnds::Penalty`](crate::DeadEnds::Penalty)`(p)` this always holds: an upper
///   estimate of `None` or above `p` becomes `p`, since giving up is always possible. With
///   `upper` returning `None` everywhere, `p` is the initial upper bound of every state.
/// - Under [`DeadEnds::Infinite`](crate::DeadEnds::Infinite), the run fails with
///   [`Error::UnboundedUpper`] at the first stored state whose upper estimate is `None` or
///   infinite while its lower estimate is finite. Such a problem needs a lower bound that
///   recognizes its dead ends, a finite upper bound, or a penalty.
///
/// Since every stored state that may have a finite value has a finite upper bound, BRTDP
/// needs no dead-end proof: a state with a finite upper bound has a proper policy if the
/// bound is valid.
///
/// # Guarantee
///
/// If `lower` is admissible and `upper` is a valid upper bound (at least the optimal value
/// `V*` under the configured dead-end semantics), then at any time, in particular whatever
/// the [`Status`](crate::Status), every stored state `s` satisfies
/// `L(s) <= V*(s) <= U(s)` (up to floating-point rounding): Bellman backups of a lower bound
/// stay below `V*`, those of an upper bound stay above it. A run with
/// [`Status::Converged`](crate::Status::Converged) also has
/// `U(initial) - L(initial) <= epsilon`.
///
/// If, in addition, the upper bound is monotone (`U(s) >= min_a Q_U(s, a)` everywhere, as
/// for the constant penalty), backups keep it monotone, and every stored state with a policy
/// action `pi(s)` satisfies `U(s) >= Q_U(s, pi(s))`. Then following the policy from a state
/// whose policy successors all have actions costs at most `U` in expectation. After a
/// converged run and its fill pass, that holds from `initial`: following the policy costs at
/// most `U(initial)`. A run stopped by the budget may leave envelope states without an
/// action; [`Solution::action`](crate::Solution::action) is `None` there.
///
/// # Termination
///
/// With the default [`Limits`](crate::Limits) budget the call always returns. With valid
/// bounds both converge to `V*` on the states trials keep visiting, so the gap at `initial`
/// shrinks to `epsilon` on finite problems.
///
/// # Events
///
/// [`Event::Progress`](crate::Event::Progress) as for [`lrtdp`](crate::lrtdp()), with the
/// lower bound as `initial_value` and the current gap `U(initial) - L(initial)` as
/// `residual`. No dead-end or divergence events.
///
/// # Cost
///
/// One backup updates both bounds and counts once in [`Stats::backups`](crate::Stats). The
/// upper bounds take 8 more bytes per stored state. One seed gives identical results on
/// every run.
///
/// # Errors
///
/// Returns an [`Error`] for an invalid [`Config`] or `tau`, for invalid MDP output or estimates in the
/// states the run touches (see [`Error::InvalidHeuristic`] for the upper bound), and
/// [`Error::UnboundedUpper`] as described in "Dead ends".
///
/// # Panics
///
/// Panics if more than `u32::MAX` states, actions or outcomes are stored.
///
/// # Example
///
/// A sure route for cost 3, or a coin flip for cost 1 that retries on failure (expected
/// cost 2). The sure route bounds the value from above:
///
/// ```
/// use stochastic_search::{BrtdpConfig, Mdp, Zero, brtdp, heuristic_fn};
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
/// let config = BrtdpConfig::default();
/// let solution = brtdp(&Routes, false, &|done: &bool| *done, Zero, upper, config)?;
///
/// assert!(solution.is_converged());
/// let (lower, upper) = (solution.initial_value(), solution.initial_upper_bound().unwrap());
/// assert!(lower <= 2.0 && 2.0 <= upper && upper - lower <= 1e-6);
/// assert_eq!(solution.initial_action(), Some(&"flip"));
/// # Ok::<(), stochastic_search::Error<bool>>(())
/// ```
pub fn brtdp<M, G, L, U, B, O>(
    mdp: &M,
    initial: M::State,
    goal: &G,
    lower: L,
    upper: U,
    config: BrtdpConfig<B, O>,
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
    let explorer = Explorer::Brtdp { tau: config.tau };

    solve((mdp, initial, goal), lower, upper, config.base, explorer)
}

impl<M: Mdp, G, L, U, B, O> Run<'_, M, G, L, U, B, O> {
    /// BRTDP's successor choice: samples a successor of the action at `index` with
    /// probability proportional to `P(s') * gap(s')`, or returns `None` if that weight
    /// summed over the successors is `0` or below the initial state's gap divided by `tau`.
    ///
    /// Called after the current state's backup, so at the initial state the summed weight is
    /// at least its gap (up to rounding) and a trial always moves on from there.
    pub(crate) fn brtdp_next(&mut self, index: u32, tau: f64) -> Option<Id> {
        let u = self.rng.next_unit();
        let (total, _) = self.gap_weights(index);
        if total <= 0.0 || total < self.gap(self.start) / tau {
            return None;
        }

        self.pick_weighted(index, total, u)
    }
}
