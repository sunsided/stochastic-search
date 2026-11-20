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

use core::hash::Hash;
use core::mem;

use alloc::vec::Vec;

use crate::budget::Budget;
use crate::config::Config;

use crate::meter::{Halt, Meter, finish};
use crate::model::{Goal, Heuristic, Mdp};
use crate::rng::SplitMix64;
use crate::solution::{Solution, SolveResult};
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
/// When the status is [`Status::Converged`](crate::Status::Converged), with `V` the returned
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
/// budget yields [`Status::BudgetExhausted`](crate::Status::BudgetExhausted), never a
/// verdict. Under [`DeadEnds::Infinite`](crate::DeadEnds::Infinite), a trap (a cycle that
/// never reaches a goal) shows up as cut-off trials; LRTDP then proves by graph analysis
/// which stored states have no proper policy and makes them dead ends, so avoidable traps
/// are avoided and an unavoidable one yields [`Status::Unsolvable`](crate::Status::Unsolvable).
/// With [`Limits::unlimited`](crate::Limits::unlimited), a run on an infinite state space, or
/// one whose MDP violates the SSP assumptions (zero-cost cycles), may not return.
///
/// Only the states the run touched are stored. One seed gives identical results on every
/// run.
///
/// # Errors
///
/// Returns an [`Error`](crate::Error) for an invalid [`Config`], or for invalid MDP output or heuristic
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
pub fn lrtdp<M, G, H, B>(
    mdp: &M,
    initial: M::State,
    goal: &G,
    heuristic: H,
    config: Config<B>,
) -> SolveResult<M>
where
    M: Mdp,
    M::State: Clone + Eq + Hash,
    G: Goal<M::State>,
    H: Heuristic<M::State, f64>,
    B: Budget,
{
    config.validate()?;
    let mut problem = Problem {
        mdp,
        goal,
        heuristic,
        cap: config.dead_ends.cap(),
    };

    let mut space = Space::new(problem.cap);
    let start = space.intern(&mut problem, initial)?;
    let mut run = Run {
        problem,
        space,
        epsilon: config.epsilon,
        max_trial_length: config.max_trial_length,
        meter: Meter::new(config.budget),
        rng: SplitMix64::new(config.seed),
        epoch: 0,
        analyzed_at: 0,
        trail: Vec::new(),
        open: Vec::new(),
        closed: Vec::new(),
    };

    let status = finish(run.solve(start))?;
    run.meter.sync(&run.space);

    Ok(Solution::new(run.space, start, status, run.meter.stats))
}

/// The state of one LRTDP run.
struct Run<'a, M: Mdp, G, H, B> {
    /// The problem being solved.
    problem: Problem<'a, M, G, H>,
    /// The states touched so far, with values, labels and policy.
    space: Space<M::State, M::Action>,
    /// The residual threshold for labeling.
    epsilon: f64,
    /// The step limit of one trial.
    max_trial_length: usize,
    /// Budget and counters.
    meter: Meter<B>,
    /// The successor sampler.
    rng: SplitMix64,
    /// The epoch of the current solved check, compared against [`Node::mark`](crate::space::Node).
    epoch: u32,
    /// The backup count at the last dead-end analysis; see [`Run::prove_dead_ends`].
    analyzed_at: u64,
    /// Reused buffer: the states of the current trial, in visiting order.
    trail: Vec<Id>,
    /// Reused buffer: the frontier of a solved check.
    open: Vec<Id>,
    /// Reused buffer: the states a solved check visited.
    closed: Vec<Id>,
}

impl<M, G, H, B> Run<'_, M, G, H, B>
where
    M: Mdp,
    M::State: Clone + Eq + Hash,
    G: Goal<M::State>,
    H: Heuristic<M::State, f64>,
    B: Budget,
{
    /// Runs trials until `start` is solved.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    fn solve(&mut self, start: Id) -> Result<(), Halt<M::State>> {
        while !self.space.node(start).solved {
            self.meter.check(&self.space)?;
            self.meter.stats.trials += 1;
            self.trial(start)?;
        }

        Ok(())
    }

    /// Runs one trial from `start` and unwinds it through the solved checks.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    fn trial(&mut self, start: Id) -> Result<(), Halt<M::State>> {
        let mut trail = mem::take(&mut self.trail);
        trail.clear();

        let result = self.walk(start, &mut trail).and_then(|cut_off| {
            if cut_off {
                self.prove_dead_ends();
            }

            self.unwind(&mut trail)
        });
        self.trail = trail;

        result
    }

    /// After a cut-off trial under infinite dead-end cost, proves by graph analysis which
    /// stored states cannot reach a goal with probability 1 and turns them into dead ends.
    ///
    /// Cut-off trials are the symptom of a trap cycle, whose values would otherwise rise
    /// forever. The analysis is linear in the stored graph per round, so it only runs once at
    /// least that many backups happened since the previous one; its cost stays proportional
    /// to the search effort.
    fn prove_dead_ends(&mut self) {
        let work = (self.space.len() + self.space.outcome_count()) as u64;
        let backups = self.meter.stats.backups;
        if self.problem.cap.is_finite() || backups - self.analyzed_at < work {
            return;
        }

        self.analyzed_at = backups;
        self.space.mark_dead_ends();
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
            if !self.check_solved(state)? {
                break;
            }
        }

        Ok(())
    }

    /// Expands `state` if needed and backs it up.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    fn backup(&mut self, state: Id) -> Result<(), Halt<M::State>> {
        self.space.expand(&mut self.problem, state)?;
        self.meter.charge_backup(&self.space)?;
        self.space.update(state);

        Ok(())
    }
}
