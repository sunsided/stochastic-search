//! Bounded real-time dynamic programming (BRTDP, McMahan, Likhachev and Gordon 2005).
//!
//! ```mermaid
//! graph TD
//!     A{"gap of s0 at most epsilon?"} -- no --> B[back up both bounds, act greedily on the lower bound]
//!     B --> C{"expected gap of successors below gap(s0) / tau?"}
//!     C -- no --> D[sample a successor weighted by probability times gap]
//!     D --> B
//!     C -- yes --> E[back up the trial in reverse]
//!     E --> A
//!     A -- yes --> F[return both bounds and the upper-greedy policy]
//! ```

use core::hash::Hash;
use core::mem;

use alloc::vec::Vec;

use crate::budget::Budget;
use crate::config::Config;
use crate::error::Error;
use crate::meter::{Halt, Meter, finish};
use crate::model::{Goal, Heuristic, Mdp};
use crate::observer::Observer;
use crate::rng::SplitMix64;
use crate::solution::{Solution, SolveResult, Status};
use crate::space::{Id, Kind, Problem, Space};

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
/// below `(U(initial) - L(initial)) / tau` ([`Config::tau`]), at a goal or dead end, or after
/// [`Config::max_trial_length`] steps; then it backs up its states again, last first. The run
/// stops when `U(initial) - L(initial) <= epsilon` ([`Config::epsilon`]).
///
/// The returned [`Solution`] has the lower bounds as [`Solution::value`] and the upper
/// bounds as [`Solution::upper_bound`]. Its policy is greedy on the upper bound, as of each
/// state's last backup.
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
/// whose policy successors all have actions costs at most `U` in expectation. A run that
/// stops early may still leave states without an action in the policy's envelope (states no
/// trial backed up); [`Solution::action`] is `None` there.
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
/// Returns an [`Error`] for an invalid [`Config`], for invalid MDP output or estimates in the
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
/// use stochastic_search::{Config, Mdp, Zero, brtdp, heuristic_fn};
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
/// let solution = brtdp(&Routes, false, &|done: &bool| *done, Zero, upper, Config::default())?;
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
    config: Config<B, O>,
) -> SolveResult<M>
where
    M: Mdp,
    M::State: Clone + Eq + Hash,
    G: Goal<M::State>,
    L: Heuristic<M::State, f64>,
    U: Heuristic<M::State, f64>,
    B: Budget,
    O: Observer,
{
    let mut run = Run::new(mdp, initial, goal, lower, upper, config)?;
    let status = finish(run.solve())?;

    Ok(run.into_solution(status))
}

/// The state of one BRTDP run.
struct Run<'a, M: Mdp, G, L, U, B, O> {
    /// The problem being solved; its heuristic is the lower bound.
    problem: Problem<'a, M, G, L>,
    /// The initial upper bound of newly stored states.
    upper_heuristic: U,
    /// The states touched so far; [`Node::value`](crate::space::Node) is the lower bound
    /// and [`Node::policy`](crate::space::Node) the upper-greedy action.
    space: Space<M::State, M::Action>,
    /// The upper bound of every stored state, indexed by [`Id`]. Infinite only where the
    /// lower bound is infinite too, so gaps are never infinite or NaN.
    upper: Vec<f64>,
    /// Budget, counters and observer.
    meter: Meter<B, O>,
    /// The successor sampler.
    rng: SplitMix64,
    /// The gap at the initial state that ends the run.
    epsilon: f64,
    /// The trial-end ratio.
    tau: f64,
    /// The step limit of one trial.
    max_trial_length: usize,
    /// The initial state.
    start: Id,
    /// Reused buffer: the states of the current trial, in visiting order.
    trail: Vec<Id>,
}

impl<'a, M, G, L, U, B, O> Run<'a, M, G, L, U, B, O>
where
    M: Mdp,
    M::State: Clone + Eq + Hash,
    G: Goal<M::State>,
    L: Heuristic<M::State, f64>,
    U: Heuristic<M::State, f64>,
    B: Budget,
    O: Observer,
{
    /// Validates `config` and sets up a run from `initial` with an empty space.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfig`] for an invalid configuration, and the estimate
    /// errors of [`Run::estimate_upper`] for `initial`.
    fn new(
        mdp: &'a M,
        initial: M::State,
        goal: &'a G,
        lower: L,
        upper_heuristic: U,
        config: Config<B, O>,
    ) -> Result<Self, Error<M::State>> {
        config.validate()?;
        let mut problem = Problem {
            mdp,
            goal,
            heuristic: lower,
            cap: config.dead_ends.cap(),
        };

        let mut space = Space::new(problem.cap);
        let start = space.intern(&mut problem, initial)?;
        let mut run = Self {
            problem,
            upper_heuristic,
            space,
            upper: Vec::new(),
            rng: SplitMix64::new(config.seed),
            epsilon: config.epsilon,
            tau: config.tau,
            max_trial_length: config.max_trial_length,
            meter: Meter::new(config),
            start,
            trail: Vec::new(),
        };
        run.estimate_new_states()?;

        Ok(run)
    }

    /// Ends the run with `status` and hands both bounds to a [`Solution`].
    fn into_solution(mut self, status: Status) -> Solution<M::State, M::Action> {
        self.meter.sync(&self.space);

        Solution::new(self.space, self.start, status, self.meter.stats).with_upper(self.upper)
    }

    /// Runs trials until the gap at the initial state is at most epsilon.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    fn solve(&mut self) -> Result<(), Halt<M::State>> {
        while self.gap(self.start) > self.epsilon {
            self.meter.check(&self.space)?;
            self.meter.stats.trials += 1;
            self.trial()?;
        }

        Ok(())
    }

    /// Runs one trial: a gap-guided walk from the initial state, then backups in reverse.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    fn trial(&mut self) -> Result<(), Halt<M::State>> {
        let mut trail = mem::take(&mut self.trail);
        trail.clear();

        let result = self.walk(&mut trail).and_then(|()| {
            while let Some(state) = trail.pop() {
                self.backup(state)?;
            }

            Ok(())
        });
        self.trail = trail;

        result
    }

    /// Walks from the initial state, backing up every state on the way and recording it in
    /// `trail`, until [`Run::next_state`] ends the trial or the length limit is reached.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    fn walk(&mut self, trail: &mut Vec<Id>) -> Result<(), Halt<M::State>> {
        let mut state = self.start;
        while trail.len() < self.max_trial_length {
            trail.push(state);
            let Some(action) = self.backup(state)? else {
                break;
            };

            let Some(next) = self.next_state(action) else {
                break;
            };

            state = next;
        }

        Ok(())
    }

    /// Samples a successor of the action at `index` with probability proportional to
    /// `P(s') * gap(s')`, or returns `None` if that weight summed over the successors is `0`
    /// or below the initial state's gap divided by `tau`.
    ///
    /// Called after the current state's backup, so at the initial state the summed weight is
    /// at least its gap (up to rounding) and a trial always moves on from there.
    fn next_state(&mut self, index: u32) -> Option<Id> {
        let u = self.rng.next_unit();
        let outcomes = self.space.outcomes(index);
        let total: f64 = outcomes
            .iter()
            .map(|outcome| outcome.probability * self.gap(outcome.target))
            .sum();
        if total <= 0.0 || total < self.gap(self.start) / self.tau {
            return None;
        }

        let mut remaining = u * total;
        let mut last = None;
        for outcome in outcomes {
            let weight = outcome.probability * self.gap(outcome.target);
            if weight > 0.0 {
                if remaining < weight {
                    return Some(outcome.target);
                }

                remaining -= weight;
                last = Some(outcome.target);
            }
        }

        last
    }

    /// Expands `state` if needed and backs up both bounds. Returns the action that is greedy
    /// on the lower bound, which the trial follows; stores the upper-greedy one as the
    /// policy.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    fn backup(&mut self, state: Id) -> Result<Option<u32>, Halt<M::State>> {
        self.space.expand(&mut self.problem, state)?;
        self.estimate_new_states()?;
        self.meter.charge_backup(&self.space)?;

        let lower = self.space.bellman(state);
        let upper = {
            let bounds = &self.upper;
            self.space
                .bellman_under(state, |target| bounds[target as usize])
        };
        let node = self.space.node_mut(state);
        node.value = lower.value;
        node.policy = upper.policy;
        self.upper[state as usize] = upper.value;

        if self.meter.progress_due() {
            let value = self.space.node(self.start).value;
            let residual = self.gap(self.start);
            self.meter.emit_progress(&self.space, value, residual);
        }

        Ok(lower.policy)
    }

    /// Estimates the upper bound of every state stored since the last call.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Run::estimate_upper`].
    fn estimate_new_states(&mut self) -> Result<(), Error<M::State>> {
        while self.upper.len() < self.space.len() {
            let id = Id::try_from(self.upper.len()).expect("interned indices fit an id");
            let bound = self.estimate_upper(id)?;
            self.upper.push(bound);
        }

        Ok(())
    }

    /// The initial upper bound of the stored state `id`: `0` at goals, the cap at proven dead
    /// ends, otherwise the upper estimate capped at the dead-end value.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidHeuristic`] for an estimate that is NaN, negative infinity or
    /// below the state's lower bound, and [`Error::UnboundedUpper`] for an infinite upper
    /// bound (`None` or infinity under an infinite cap) on a state with a finite lower bound.
    fn estimate_upper(&mut self, id: Id) -> Result<f64, Error<M::State>> {
        let node = self.space.node(id);
        let (kind, lower, cap) = (node.kind, node.value, self.problem.cap);
        if kind != Kind::Open {
            return Ok(lower);
        }

        let state = self.space.state(id);
        let bound = match self.upper_heuristic.estimate(state) {
            Some(u) if u.is_nan() || u == f64::NEG_INFINITY || u < lower => {
                return Err(Error::InvalidHeuristic {
                    state: state.clone(),
                });
            }
            Some(u) => u.min(cap),
            None => cap,
        };

        if bound.is_infinite() {
            return Err(Error::UnboundedUpper {
                state: state.clone(),
            });
        }

        Ok(bound)
    }

    /// The gap between the bounds of `id`.
    fn gap(&self, id: Id) -> f64 {
        gap(self.space.node(id).value, self.upper[id as usize])
    }
}

/// The gap `upper - lower`, or `0` if the bounds meet or cross (including both infinite).
fn gap(lower: f64, upper: f64) -> f64 {
    if upper > lower { upper - lower } else { 0.0 }
}
