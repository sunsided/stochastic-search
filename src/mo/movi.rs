//! Multi-objective value iteration (MOVI) under the weak improper policy assumption
//! (Algorithm 2 of Chen, Trevizan and Thiébaux 2023): the exact reference.

use core::hash::Hash;

use alloc::vec::Vec;

use crate::budget::{Budget, Stats};
use crate::config::Config;
use crate::meter::{Halt, finish};
use crate::mo::heuristics::ZeroVector;
use crate::mo::run::Run;
use crate::mo::solution::MoSolveResult;
use crate::mo::{MoMdp, Objectives};
use crate::model::Goal;
use crate::observer::Observer;
use crate::space::Id;

/// Solves a multi-objective SSP from `initial` with value iteration over every reachable
/// state.
///
/// First enumerates every state reachable from `initial` under any action, then sweeps
/// the multi-objective Bellman backup with the bound `b` of `objectives` (equations 7 to 9)
/// over all of them, starting from `{0}`, until a sweep's largest residual (the Hausdorff
/// distance between a state's old and new value) is below [`Config::epsilon`]. Each sweep
/// updates values in place (Gauss-Seidel); the paper's Algorithm 2 computes a new value
/// function per sweep. [`Config::seed`] and [`Config::max_trial_length`] are ignored.
///
/// The paper's Theorem 3.1: under reachability (no dead ends) and the weak improper policy
/// assumption for `b` ([`Objectives::bound`]), the values converge to the convex coverage
/// set of the proper policies. Here `b` is dropped from every set rather than at the end,
/// and a sum containing `b` counts as `b`, so policies that risk a dead end are excluded
/// instead of breaking the theorem (see `docs/design.md`).
///
/// # Errors
///
/// Returns an [`Error`](crate::Error) for invalid settings or for invalid MDP output.
///
/// # Panics
///
/// Panics if more than `u32::MAX` states, actions or outcomes are stored.
///
/// # Example
///
/// A choice between a sure but slow action and a fast coin flip:
///
/// ```
/// use stochastic_search::{Config, MoMdp, Objectives, mo_value_iteration};
///
/// struct Choice;
///
/// impl MoMdp<2> for Choice {
///     type State = u8;
///     type Action = u8;
///
///     fn actions(&self, _state: &u8, mut emit: impl FnMut(u8, [f64; 2])) {
///         emit(0, [3.0, 0.0]);
///         emit(1, [1.0, 1.0]);
///     }
///
///     fn outcomes(&self, state: &u8, action: &u8, mut emit: impl FnMut(u8, f64)) {
///         if *action == 0 {
///             emit(1, 1.0);
///         } else {
///             emit(1, 0.5);
///             emit(*state, 0.5);
///         }
///     }
/// }
///
/// let objectives = Objectives::new([50.0, 50.0]);
/// let config = Config::default().with_epsilon(1e-9);
/// let solution = mo_value_iteration(&Choice, 0, &|state: &u8| *state == 1, objectives, config)?;
///
/// // Sure: [3, 0]. Coin flip: two attempts in expectation, [2, 2].
/// assert_eq!(solution.policies().len(), 2);
/// # Ok::<(), stochastic_search::Error<u8>>(())
/// ```
pub fn mo_value_iteration<M, G, B, O, const N: usize>(
    mdp: &M,
    initial: M::State,
    goal: &G,
    objectives: Objectives<N>,
    config: Config<B, O>,
) -> MoSolveResult<M, N>
where
    M: MoMdp<N>,
    M::State: Clone + Eq + Hash,
    G: Goal<M::State>,
    B: Budget<Stats>,
    O: Observer,
{
    let mut run = Run::new(mdp, initial, goal, ZeroVector, objectives, config)?;
    let status = finish(run.solve_movi())?;

    Ok(run.into_solution(status))
}

impl<M, G, B, O, const N: usize> Run<'_, M, G, ZeroVector, B, O, N>
where
    M: MoMdp<N>,
    M::State: Clone + Eq + Hash,
    G: Goal<M::State>,
    B: Budget<Stats>,
    O: Observer,
{
    /// Enumerates the reachable states, then sweeps until the residual is below epsilon.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    fn solve_movi(&mut self) -> Result<(), Halt<M::State>> {
        self.enumerate()?;
        let states: Vec<Id> = (0..self.space.len())
            .map(|id| Id::try_from(id).expect("ids fit u32"))
            .collect();
        while self.sweep(&states)? >= self.epsilon {}

        Ok(())
    }

    /// Expands every state reachable from the initial state, in id order.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    fn enumerate(&mut self) -> Result<(), Halt<M::State>> {
        let mut next = 0;
        while next < self.space.len() {
            self.meter.check(&self.space)?;
            let id = Id::try_from(next).expect("ids fit u32");
            self.space
                .expand(&mut self.problem, id, &mut self.scratch)?;
            next += 1;
        }

        Ok(())
    }
}

impl<M, G, H, B, O, const N: usize> Run<'_, M, G, H, B, O, N>
where
    M: MoMdp<N>,
    M::State: Clone + Eq + Hash,
    G: Goal<M::State>,
    H: crate::model::Heuristic<M::State, Vec<[f64; N]>>,
    B: Budget<Stats>,
    O: Observer,
{
    /// Backs up every state of `states` once, last first, and returns the largest
    /// residual. Counts as one sweep.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    pub(crate) fn sweep(&mut self, states: &[Id]) -> Result<f64, Halt<M::State>> {
        self.meter.stats.sweeps += 1;
        let mut largest = 0.0_f64;
        for &state in states.iter().rev() {
            largest = largest.max(self.backup(state)?);
        }

        Ok(largest)
    }
}
