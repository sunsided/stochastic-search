//! Labeled real-time dynamic programming (LRTDP, Bonet and Geffner 2003).
//!
//! ```mermaid
//! graph TD
//!     A[s0 solved?] -- no --> B[trial: back up, act greedily, sample]
//!     B --> C[unwind the trial through checkSolved]
//!     C --> A
//!     A -- yes --> D[return values and frozen policy]
//! ```

mod check_solved;
mod divergence;
mod monitoring;

use core::hash::Hash;
use core::mem;

use alloc::vec::Vec;

use crate::budget::{Budget, Stats};
use crate::config::Config;

use crate::error::Error;
use crate::lrtdp::divergence::Divergence;
use crate::meter::{Halt, Meter, finish};
use crate::model::{Goal, Heuristic, Mdp};
use crate::observer::Observer;
use crate::rng::SplitMix64;
use crate::solution::{Solution, SolveResult, Status};
use crate::space::{Id, Problem, Space};

/// Solves an SSP from `initial` with labeled RTDP.
///
/// Runs greedy trials from `initial`, sampling successors with a generator seeded from
/// [`Config::seed`], and labels states solved once every state reachable from them under the
/// greedy policy has a Bellman residual of at most [`Config::epsilon`]. Stops when `initial`
/// is solved or the [`Config::budget`] is exhausted.
///
/// The heuristic is taken by value; pass `&mut h` to keep it, or wrap a closure in
/// [`heuristic_fn`](crate::heuristic_fn). Use [`Zero`](crate::Zero) if there is none.
///
/// # Guarantee
///
/// When the status is [`Status::Converged`], with `V` the returned
/// values and `pi` the returned policy, on every state `s` reachable from `initial` under
/// `pi` (the *envelope*):
///
/// - `|V(s) - Q(s, pi(s))| <= epsilon`, for any heuristic. The policy is the greedy action
///   frozen when `s` was labeled solved, and solved states and their policy successors are
///   never updated again.
/// - `|V(s) - min_a Q(s, a)| <= epsilon` (the full Bellman residual) if the heuristic is
///   consistent, so that backups never lower a value.
/// - If the heuristic is admissible: `V(initial) <= V*(initial)`.
/// - If, in addition, every action costs at least `c_min > epsilon` (under
///   [`DeadEnds::Penalty`](crate::DeadEnds::Penalty), `c_min` also bounds the penalty):
///   `V*(initial) - V(initial) <= epsilon * V(initial) / (c_min - epsilon)`, and `pi`
///   reaches a goal, or a state where it gives up, with probability 1.
///
/// Epsilon-consistency alone is not `|V(initial) - V*(initial)| <= epsilon`; to get a
/// tolerance `t`, pick `epsilon <= t * c_min / (V + t)` for an upper estimate `V` of the
/// optimal value. Values outside the envelope are only bounds. See `docs/design.md`, section
/// "Convergence criterion".
///
/// # Termination
///
/// With the default [`Limits`](crate::Limits) budget the call always returns; an exhausted
/// budget yields [`Status::BudgetExhausted`], never a
/// verdict. Under [`DeadEnds::Infinite`](crate::DeadEnds::Infinite), a trap (a cycle that
/// never reaches a goal) shows up as cut-off trials or as values that solved checks keep
/// raising. LRTDP proves by graph analysis, after every cut-off trial and whenever the
/// backup count has doubled since the last analysis, which stored states have no proper
/// policy and makes them dead ends, so avoidable traps are avoided and an unavoidable one
/// yields [`Status::Unsolvable`].
/// With [`Limits::unlimited`](crate::Limits::unlimited), a run on an infinite state space, or
/// one whose MDP violates the SSP assumptions (zero-cost cycles), may not return.
///
/// # Events
///
/// [`Config::observer`] receives [`Event`](crate::Event)s: progress, finished dead-end
/// analyses and suspected divergence. They are informational; the status only reports
/// proven facts. The default observer `()` costs nothing.
///
/// Only the states the run touched are stored. One seed gives identical results on every
/// run.
///
/// # Errors
///
/// Returns an [`Error`] for an invalid [`Config`], or for invalid MDP output or heuristic
/// estimates in the states the run touches.
///
/// # Panics
///
/// Panics if more than `u32::MAX` states, actions or outcomes are stored.
///
/// # Example
///
/// Two routes to the goal: a sure one for cost 3, and a coin flip for cost 1 that retries on
/// failure (expected cost 2):
///
/// ```
/// use stochastic_search::{Config, Mdp, Zero, lrtdp};
///
/// struct Routes;
///
/// impl Mdp for Routes {
///     type State = &'static str;
///     type Action = &'static str;
///
///     fn actions(&self, _state: &&'static str, mut emit: impl FnMut(&'static str, f64)) {
///         emit("sure", 3.0);
///         emit("flip", 1.0);
///     }
///
///     fn outcomes(
///         &self,
///         state: &&'static str,
///         action: &&'static str,
///         mut emit: impl FnMut(&'static str, f64),
///     ) {
///         if *action == "sure" {
///             emit("goal", 1.0);
///         } else {
///             emit("goal", 0.5);
///             emit(state, 0.5);
///         }
///     }
/// }
///
/// let solution = lrtdp(
///     &Routes,
///     "start",
///     &|state: &&str| *state == "goal",
///     Zero,
///     Config::default(),
/// )?;
///
/// assert!(solution.is_converged());
/// assert!((solution.initial_value() - 2.0).abs() < 1e-5);
/// assert_eq!(solution.initial_action(), Some(&"flip"));
/// # Ok::<(), stochastic_search::Error<&str>>(())
/// ```
pub fn lrtdp<M, G, H, B, O>(
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
    B: Budget<Stats>,
    O: Observer,
{
    let mut run = Run::new(mdp, initial, goal, heuristic, config)?;
    let status = finish(run.solve(true))?;

    Ok(run.into_solution(status))
}

/// The state of one trial-based or depth-first run over a lazily built space: LRTDP, plain
/// RTDP ([`rtdp`](crate::rtdp())) and iLAO* ([`ilao_star`](crate::ilao_star())) share it,
/// including its solved check, dead-end proofs and events.
pub(crate) struct Run<'a, M: Mdp, G, H, B, O> {
    /// The problem being solved.
    pub(crate) problem: Problem<'a, M, G, H>,
    /// The states touched so far, with values, labels and policy.
    pub(crate) space: Space<M::State, M::Action>,
    /// The residual threshold for labeling.
    pub(crate) epsilon: f64,
    /// The step limit of one trial.
    pub(crate) max_trial_length: usize,
    /// Budget, counters and observer.
    pub(crate) meter: Meter<B, O>,
    /// The initial state.
    pub(crate) start: Id,
    /// The change of the initial state's value at its most recent backup, for progress
    /// events.
    pub(crate) start_residual: f64,
    /// Watches the initial value for steady growth, for divergence events.
    divergence: Divergence,
    /// The successor sampler.
    pub(crate) rng: SplitMix64,
    /// The epoch of the current solved check, compared against [`Node::mark`](crate::space::Node).
    pub(crate) epoch: u32,
    /// The backup count at the last dead-end analysis; see [`Run::maybe_prove_dead_ends`].
    pub(crate) analyzed_at: u64,
    /// Plain RTDP: the backup count when the last convergence check finished.
    pub(crate) checked_at: u64,
    /// Plain RTDP: the backups the last convergence check took.
    pub(crate) check_cost: u64,
    /// Reused buffer: the states of the current trial, in visiting order.
    pub(crate) trail: Vec<Id>,
    /// Reused buffer: the frontier of a solved check.
    pub(crate) open: Vec<Id>,
    /// Reused buffer: the states a solved check visited.
    pub(crate) closed: Vec<Id>,
}

impl<'a, M, G, H, B, O> Run<'a, M, G, H, B, O>
where
    M: Mdp,
    M::State: Clone + Eq + Hash,
    G: Goal<M::State>,
    H: Heuristic<M::State, f64>,
    B: Budget<Stats>,
    O: Observer,
{
    /// Validates `config` and sets up a run from `initial` with an empty space.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfig`] for an invalid
    /// configuration and [`Error::InvalidHeuristic`] for an
    /// invalid estimate of `initial`.
    pub(crate) fn new(
        mdp: &'a M,
        initial: M::State,
        goal: &'a G,
        heuristic: H,
        config: Config<B, O>,
    ) -> Result<Self, Error<M::State>> {
        config.validate()?;
        let mut problem = Problem {
            mdp,
            goal,
            heuristic,
            cap: config.dead_ends.cap(),
        };

        let mut space = Space::new(problem.cap);
        let start = space.intern(&mut problem, initial)?;

        Ok(Self {
            problem,
            epsilon: config.epsilon,
            max_trial_length: config.max_trial_length,
            rng: SplitMix64::new(config.seed),
            divergence: Divergence::new(config.divergence_window, space.node(start).value),
            space,
            meter: Meter::new(config),
            start,
            start_residual: f64::INFINITY,
            epoch: 0,
            analyzed_at: 0,
            checked_at: 0,
            check_cost: 0,
            trail: Vec::new(),
            open: Vec::new(),
            closed: Vec::new(),
        })
    }

    /// Ends the run with `status` and hands its space to a [`Solution`].
    pub(crate) fn into_solution(mut self, status: Status) -> Solution<M::State, M::Action> {
        self.meter.sync(&self.space);

        Solution::new(self.space, self.start, status, self.meter.stats)
    }

    /// Runs trials until the initial state is solved, proving dead ends and reporting
    /// events between trials.
    ///
    /// With `labeled` (LRTDP), every trial is unwound through the solved checks. Without
    /// (plain RTDP), nothing is labeled during trials; the initial state's envelope is
    /// checked on the schedule of [`Run::maybe_check_converged`] instead.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    pub(crate) fn solve(&mut self, labeled: bool) -> Result<(), Halt<M::State>> {
        let start = self.start;
        while !self.space.node(start).solved {
            self.meter.check(&self.space)?;
            self.meter.stats.trials += 1;
            self.trial(start, labeled)?;
            if !labeled {
                self.maybe_check_converged()?;
            }

            self.maybe_prove_dead_ends(false);
            self.track_divergence();
        }

        Ok(())
    }

    /// Runs one trial from `start` and, if `labeled`, unwinds it through the solved checks.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    fn trial(&mut self, start: Id, labeled: bool) -> Result<(), Halt<M::State>> {
        let mut trail = mem::take(&mut self.trail);
        trail.clear();

        let result = self.walk(start, &mut trail).and_then(|cut_off| {
            if cut_off {
                self.maybe_prove_dead_ends(true);
            }

            if labeled {
                self.unwind(&mut trail)
            } else {
                Ok(())
            }
        });
        self.trail = trail;

        result
    }

    /// Walks greedily from `start` until a solved state, a state without a finite action,
    /// or the trial length limit, backing up every state on the way and recording it in
    /// `trail`. Returns `true` if the trial was cut off at the length limit.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    fn walk(&mut self, start: Id, trail: &mut Vec<Id>) -> Result<bool, Halt<M::State>> {
        let mut state = start;
        while !self.space.node(state).solved {
            if trail.len() >= self.max_trial_length {
                return Ok(true);
            }

            trail.push(state);
            self.backup(state)?;

            let Some(action) = self.space.node(state).policy else {
                break;
            };

            state = self.space.sample(action, self.rng.next_unit());
        }

        Ok(false)
    }

    /// Checks the trail's states for solved labels, last visited first, and stops at the
    /// first one that is not solved yet.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    fn unwind(&mut self, trail: &mut Vec<Id>) -> Result<(), Halt<M::State>> {
        while let Some(state) = trail.pop() {
            if !self.check_solved(state, true)? {
                break;
            }
        }

        Ok(())
    }

    /// Expands `state` if needed, backs it up and returns its residual.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    pub(crate) fn backup(&mut self, state: Id) -> Result<f64, Halt<M::State>> {
        self.space.expand(&mut self.problem, state)?;
        self.meter.charge_backup(&self.space)?;
        let residual = self.space.update(state);
        if state == self.start {
            self.start_residual = residual;
        }

        self.report_progress();

        Ok(residual)
    }

    /// Plain RTDP: checks whether the initial state's greedy envelope is
    /// epsilon-consistent, and labels it solved (ending the run) if so.
    ///
    /// A check costs about one evaluation per envelope state, so it only runs once at least
    /// as many backups happened since the last check as that check took: checks take at
    /// most half the run's backups. Delaying a check never changes the values, only when the
    /// run notices convergence.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    fn maybe_check_converged(&mut self) -> Result<(), Halt<M::State>> {
        let before = self.meter.stats.backups;
        if before - self.checked_at < self.check_cost {
            return Ok(());
        }

        self.check_solved(self.start, false)?;
        self.checked_at = self.meter.stats.backups;
        self.check_cost = self.checked_at - before;

        Ok(())
    }
}
