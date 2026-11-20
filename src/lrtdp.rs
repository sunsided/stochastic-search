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
/// With an admissible `heuristic` (use [`Zero`](crate::Zero) if there is none) and the SSP
/// assumptions (a proper policy exists, every action costs more than zero), a converged
/// run returns `V(initial) <= V*(initial)`, and the returned policy reaches a goal with
/// probability 1. See `docs/design.md`, section "Convergence criterion", for the exact
/// guarantee and the bound on `V*(initial) - V(initial)`.
///
/// # Termination
///
/// Under [`DeadEnds::Infinite`](crate::DeadEnds::Infinite), if no proper policy exists from
/// `initial` and a cycle that never reaches a goal is reachable, the values rise forever: with
/// the default unlimited budget, the call does not return. Use
/// [`DeadEnds::Penalty`](crate::DeadEnds::Penalty) or a budget when that can happen. Zero-cost
/// cycles violate the SSP assumptions and are not detected.
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
/// use stochastic_search::{Config, Mdp, lrtdp};
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
///     &|_: &&str| Some(0.0),
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
    heuristic: &H,
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
    let problem = Problem {
        mdp,
        goal,
        heuristic,
        cap: config.dead_ends.cap(),
    };

    let mut space = Space::new(problem.cap);
    let start = space.intern(&problem, initial)?;
    let mut run = Run {
        problem,
        space,
        epsilon: config.epsilon,
        max_trial_length: config.max_trial_length,
        meter: Meter::new(config.budget),
        rng: SplitMix64::new(config.seed),
        epoch: 0,
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

        let result = self
            .walk(start, &mut trail)
            .and_then(|()| self.unwind(&mut trail));
        self.trail = trail;

        result
    }

    /// Walks greedily from `start` until a solved state, a state without a finite action,
    /// or the trial length limit, backing up every state on the way and recording it in
    /// `trail`.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    fn walk(&mut self, start: Id, trail: &mut Vec<Id>) -> Result<(), Halt<M::State>> {
        let mut state = start;
        while !self.space.node(state).solved && trail.len() < self.max_trial_length {
            trail.push(state);
            self.backup(state)?;

            let Some(action) = self.space.node(state).policy else {
                break;
            };

            state = self.space.sample(action, self.rng.next_unit());
        }

        Ok(())
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
        self.space.expand(&self.problem, state)?;
        self.meter.charge_backup(&self.space)?;
        self.space.update(state);

        Ok(())
    }
}
