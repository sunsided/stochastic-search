//! MOLRTDP (Algorithm 5 of Chen, Trevizan and Thiébaux 2023): labeled RTDP with the
//! multi-objective backup, random greedy actions and a solved check over the successors of
//! all greedy actions.
//!
//! ```mermaid
//! graph TD
//!     A[s0 solved?] -- no --> B[trial: back up, sample a greedy action with an unsolved successor, sample a successor]
//!     B --> C[unwind the trial through checkSolved]
//!     C --> A
//!     A -- yes --> D[extract one policy per vector of V of s0]
//! ```

use core::hash::Hash;

use alloc::vec::Vec;

use crate::budget::{Budget, Stats};
use crate::config::Config;
use crate::meter::{Halt, finish};
use crate::mo::run::Run;
use crate::mo::solution::MoSolveResult;
use crate::mo::{MoMdp, Objectives};
use crate::model::{Goal, Heuristic};
use crate::observer::Observer;
use crate::space::Id;

/// Solves a multi-objective SSP from `initial` with MOLRTDP.
///
/// Runs trials from `initial`: back up the current state, pick a greedy action uniformly
/// among those with an unsolved successor (the greedy actions are every action whose
/// Q-value contributes a vector, or the bound `b`, to the state's value), then sample one
/// of its successors in proportion to its probability (the paper samples unsolved ones
/// only; see `docs/design.md`). A trial ends at a solved
/// state, at a state whose value holds only `b` (a dead end, or a trap whose value climbed
/// to the bound), or after [`Config::max_trial_length`] steps, and is unwound through the solved
/// check: a state is labeled solved once every state reachable from it under all greedy
/// actions has a residual (the Hausdorff distance between its value before and after a
/// backup) of at most [`Config::epsilon`]. Stops when `initial` is solved.
///
/// Sampling is seeded from [`Config::seed`]; one seed gives identical results.
///
/// # Guarantee
///
/// With an admissible heuristic (in the scalarised sense of the [`mo`](crate::mo) module docs), a converged
/// run's policies have values epsilon-consistent with the convex coverage set of the
/// proper policies, up to the margin of `objectives`; every state a returned policy
/// reaches is solved. Termination as for [`molao_star`](crate::molao_star()).
///
/// # Errors
///
/// Returns an [`Error`](crate::Error) for invalid settings, invalid MDP output or invalid
/// heuristic estimates in the states the run touches.
///
/// # Panics
///
/// Panics if more than `u32::MAX` states, actions or outcomes are stored.
///
/// # Example
///
/// Three objectives: each action is free in one of them.
///
/// ```
/// use stochastic_search::{Config, MoMdp, Objectives, Zero, molrtdp};
///
/// struct Three;
///
/// impl MoMdp<3> for Three {
///     type State = bool;
///     type Action = usize;
///
///     fn actions(&self, _done: &bool, mut emit: impl FnMut(usize, [f64; 3])) {
///         emit(0, [0.0, 1.0, 1.0]);
///         emit(1, [1.0, 0.0, 1.0]);
///         emit(2, [1.0, 1.0, 0.0]);
///     }
///
///     fn outcomes(&self, _done: &bool, _action: &usize, mut emit: impl FnMut(bool, f64)) {
///         emit(true, 1.0);
///     }
/// }
///
/// let objectives = Objectives::new([10.0; 3]);
/// let solution = molrtdp(&Three, false, &|done: &bool| *done, Zero, objectives, Config::default())?;
///
/// assert_eq!(solution.policies().len(), 3);
/// for policy in solution.policies() {
///     assert_eq!(policy.action(&false).copied(), policy.value().iter().position(|&c| c == 0.0));
/// }
/// # Ok::<(), stochastic_search::Error<bool>>(())
/// ```
pub fn molrtdp<M, G, H, B, O, const N: usize>(
    mdp: &M,
    initial: M::State,
    goal: &G,
    heuristic: H,
    objectives: Objectives<N>,
    config: Config<B, O>,
) -> MoSolveResult<M, N>
where
    M: MoMdp<N>,
    M::State: Clone + Eq + Hash,
    G: Goal<M::State>,
    H: Heuristic<M::State, Vec<[f64; N]>>,
    B: Budget<Stats>,
    O: Observer,
{
    let mut run = Run::new(mdp, initial, goal, heuristic, objectives, config)?;
    let status = finish(run.solve_lrtdp())?;

    Ok(run.into_solution(status))
}

/// Reused buffers of one MOLRTDP run.
#[derive(Default)]
struct Buffers {
    /// The states of the current trial, in visiting order.
    trail: Vec<Id>,
    /// The frontier of a solved check.
    open: Vec<Id>,
    /// The states a solved check visited.
    closed: Vec<Id>,
    /// The greedy actions of one state.
    actions: Vec<u32>,
}

impl<M, G, H, B, O, const N: usize> Run<'_, M, G, H, B, O, N>
where
    M: MoMdp<N>,
    M::State: Clone + Eq + Hash,
    G: Goal<M::State>,
    H: Heuristic<M::State, Vec<[f64; N]>>,
    B: Budget<Stats>,
    O: Observer,
{
    /// Runs trials until the initial state is solved (lines 2 to 13 of Algorithm 5).
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    fn solve_lrtdp(&mut self) -> Result<(), Halt<M::State>> {
        let mut buffers = Buffers::default();
        while !self.space.node(self.start).solved {
            self.meter.check(&self.space)?;
            self.meter.stats.trials += 1;
            self.walk(&mut buffers)?;
            self.unwind(&mut buffers)?;
        }

        Ok(())
    }

    /// Walks one trial from the initial state into `buffers.trail` (lines 3 to 10).
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    fn walk(&mut self, buffers: &mut Buffers) -> Result<(), Halt<M::State>> {
        buffers.trail.clear();
        let mut state = self.start;
        while !self.space.node(state).solved && buffers.trail.len() < self.max_trial_length {
            buffers.trail.push(state);
            self.backup(state)?;
            if self.space.node(state).value.points.is_empty() {
                // Only `b` is left: a dead end or a trap that climbed to the bound.
                break;
            }

            let Some(next) = self.sample_successor(state, &mut buffers.actions) else {
                break;
            };

            state = next;
        }

        Ok(())
    }

    /// Runs the solved check on the trial's states, last first, until one fails (lines 11
    /// to 13).
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    fn unwind(&mut self, buffers: &mut Buffers) -> Result<(), Halt<M::State>> {
        while let Some(state) = buffers.trail.pop() {
            if !self.check_solved(state, buffers)? {
                break;
            }
        }

        Ok(())
    }

    /// Picks a greedy action of `state` with an unsolved successor uniformly, then one of
    /// its successors in proportion to its probability. `None` if every greedy successor
    /// is solved.
    ///
    /// The paper samples among the unsolved successors only (`sampleUnsolvedNextState`).
    /// Then a trial ends only where no greedy successor is unsolved, so one inside a cycle
    /// runs to the length cut-off. Sampling from the whole distribution, as LRTDP does,
    /// ends the trial at the first solved state, a goal included.
    fn sample_successor(&mut self, state: Id, actions: &mut Vec<u32>) -> Option<Id> {
        self.greedy_actions(state, actions);
        actions.retain(|&action| {
            self.space
                .outcomes(action)
                .iter()
                .any(|outcome| !self.space.node(outcome.target).solved)
        });
        if actions.is_empty() {
            return None;
        }

        let count = u64::try_from(actions.len()).expect("action counts fit u64");
        let pick = usize::try_from(self.rng.next_u64() % count).expect("below the action count");
        let outcomes = self.space.outcomes(actions[pick]);
        let mut point = self.rng.next_unit();
        for outcome in outcomes {
            point -= outcome.probability;
            if point < 0.0 {
                return Some(outcome.target);
            }
        }

        outcomes.last().map(|outcome| outcome.target)
    }

    /// The solved check (routine `checkSolved`): backs up every unsolved state reachable
    /// from `start` under all greedy actions, stopping below states whose residual exceeds
    /// epsilon. Labels them all solved if none did; otherwise backs them up once more in
    /// reverse visiting order. Returns whether the check passed.
    ///
    /// Unlike the paper, which computes residuals without storing them, the check stores
    /// the backup it computes, so the greedy actions it follows match the stored values.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    fn check_solved(&mut self, start: Id, buffers: &mut Buffers) -> Result<bool, Halt<M::State>> {
        let epoch = self.next_epoch();
        buffers.open.clear();
        buffers.closed.clear();
        if !self.space.node(start).solved {
            self.space.node_mut(start).mark = epoch;
            buffers.open.push(start);
        }

        let mut passed = true;
        while let Some(state) = buffers.open.pop() {
            buffers.closed.push(state);
            if self.backup(state)? > self.epsilon {
                passed = false;
                continue;
            }

            self.push_unsolved_successors(state, epoch, buffers);
        }

        if passed {
            for &state in &buffers.closed {
                self.space.node_mut(state).solved = true;
            }
        } else {
            for index in (0..buffers.closed.len()).rev() {
                self.backup(buffers.closed[index])?;
            }
        }

        Ok(passed)
    }

    /// Pushes the unsolved, unvisited successors of every greedy action of `state`.
    fn push_unsolved_successors(&mut self, state: Id, epoch: u32, buffers: &mut Buffers) {
        self.greedy_actions(state, &mut buffers.actions);
        for &action in &buffers.actions {
            for index in 0..self.space.outcomes(action).len() {
                let target = self.space.outcomes(action)[index].target;
                let node = self.space.node_mut(target);
                if !node.solved && node.mark != epoch {
                    node.mark = epoch;
                    buffers.open.push(target);
                }
            }
        }
    }
}
