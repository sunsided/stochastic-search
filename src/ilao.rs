//! Improved LAO* (iLAO*, Hansen and Zilberstein 2001): heuristic search over the best partial
//! solution graph with depth-first expansion and postorder backups.
//!
//! ```mermaid
//! graph TD
//!     A[depth-first pass over the greedy graph of s0] --> B[expand unexpanded tips]
//!     B --> C[back up visited states in postorder]
//!     C --> D{no tips and residual below epsilon?}
//!     D -- no --> A
//!     D -- yes --> E{solved check on s0 passes?}
//!     E -- no --> A
//!     E -- yes --> F[return values and frozen policy]
//! ```

use core::hash::Hash;

use alloc::vec::Vec;

use crate::budget::{Budget, Stats};
use crate::config::Config;
use crate::lrtdp::Run;
use crate::meter::{Halt, finish};
use crate::model::{Goal, Heuristic, Mdp};
use crate::observer::Observer;
use crate::solution::SolveResult;
use crate::space::{Id, Kind};

/// Solves an SSP from `initial` with iLAO*.
///
/// LAO* (Hansen and Zilberstein 2001) is AO* for cyclic AND/OR graphs: it grows an explicit
/// graph from `initial`, keeps the best partial solution graph (the states reachable under
/// the greedy policy), and expands its tips until none is left. iLAO*, the improved variant
/// from the same paper, does this in passes: each pass is one depth-first traversal of the
/// greedy graph that expands every unexpanded tip it meets and backs up the visited states in
/// postorder. Passes repeat until one expands nothing and its largest residual is below
/// [`Config::epsilon`] (the paper's convergence test, value iteration on the solution graph);
/// then the solved check of [`lrtdp`](crate::lrtdp()) runs on `initial` and freezes the
/// policy. If that check fails (a backup flipped a greedy action towards states the pass did
/// not visit), passes continue.
///
/// Unlike [`lrtdp`](crate::lrtdp()) it samples nothing, so [`Config::seed`] and
/// [`Config::max_trial_length`] are ignored. Each pass counts in [`Stats::sweeps`](crate::Stats).
///
/// # Guarantee
///
/// A converged run has exactly the guarantee of [`lrtdp`](crate::lrtdp()) (section
/// "Guarantee" there), since the final solved check is the same.
///
/// # Termination
///
/// With the default [`Limits`](crate::Limits) budget the call always returns. Under
/// [`DeadEnds::Infinite`](crate::DeadEnds::Infinite), values in a trap cycle rise with every
/// pass; the dead-end proof of [`lrtdp`](crate::lrtdp()) runs whenever the backup count has
/// doubled since the last analysis, so traps become dead ends and an unavoidable one yields
/// [`Status::Unsolvable`](crate::Status::Unsolvable).
///
/// # Events
///
/// Progress and dead-end proofs as for [`lrtdp`](crate::lrtdp()); suspected divergence is
/// checked after every pass instead of every trial.
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
/// A corridor `0..=3` where each step costs 1 and slips back in place with probability 0.25:
///
/// ```
/// use stochastic_search::{Config, Mdp, heuristic_fn, ilao_star};
///
/// struct Corridor;
///
/// impl Mdp for Corridor {
///     type State = u8;
///     type Action = ();
///
///     fn actions(&self, _state: &u8, mut emit: impl FnMut((), f64)) {
///         emit((), 1.0);
///     }
///
///     fn outcomes(&self, state: &u8, _action: &(), mut emit: impl FnMut(u8, f64)) {
///         emit(state + 1, 0.75);
///         emit(*state, 0.25);
///     }
/// }
///
/// // The remaining distance is an admissible estimate.
/// let heuristic = heuristic_fn(|state: &u8| Some(f64::from(3 - state)));
/// let config = Config::default().with_epsilon(1e-9);
/// let solution = ilao_star(&Corridor, 0, &|state: &u8| *state == 3, heuristic, config)?;
///
/// assert!(solution.is_converged());
/// assert!((solution.initial_value() - 4.0).abs() < 1e-6);
/// # Ok::<(), stochastic_search::Error<u8>>(())
/// ```
pub fn ilao_star<M, G, H, B, O>(
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
    let mut stack = Vec::new();
    let status = finish(run.solve_ilao(&mut stack))?;

    Ok(run.into_solution(status))
}

/// One entry of a pass's depth-first stack: a state, and whether its successors were pushed
/// already (then the next visit is its postorder backup).
type Frame = (Id, bool);

/// What one iLAO* pass found.
#[derive(Debug, Clone, Copy)]
struct Pass {
    /// Whether the pass expanded at least one tip.
    expanded: bool,
    /// The largest residual of the pass's backups.
    residual: f64,
}

impl<M, G, H, B, O> Run<'_, M, G, H, B, O>
where
    M: Mdp,
    M::State: Clone + Eq + Hash,
    G: Goal<M::State>,
    H: Heuristic<M::State, f64>,
    B: Budget<Stats>,
    O: Observer,
{
    /// Runs passes until the initial state is solved.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    fn solve_ilao(&mut self, stack: &mut Vec<Frame>) -> Result<(), Halt<M::State>> {
        let start = self.start;
        while !self.space.node(start).solved {
            self.meter.check(&self.space)?;
            self.meter.stats.sweeps += 1;
            let pass = self.pass(stack)?;
            self.maybe_prove_dead_ends(false);
            self.track_divergence();
            if self.is_settled(pass) {
                self.check_solved(start, false)?;
            }
        }

        Ok(())
    }

    /// Returns `true` if `pass` meets the paper's convergence test: nothing expanded, every
    /// residual below epsilon.
    fn is_settled(&self, pass: Pass) -> bool {
        !pass.expanded && pass.residual < self.epsilon
    }

    /// One depth-first traversal of the greedy graph from the initial state: expands the
    /// unexpanded tips it meets (without descending below them) and backs up every visited
    /// state in postorder.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    fn pass(&mut self, stack: &mut Vec<Frame>) -> Result<Pass, Halt<M::State>> {
        let epoch = self.next_epoch();
        let mut pass = Pass {
            expanded: false,
            residual: 0.0,
        };

        stack.clear();
        self.space.node_mut(self.start).mark = epoch;
        stack.push((self.start, false));
        while let Some(frame) = stack.last_mut() {
            let (state, pushed) = *frame;
            frame.1 = true;
            if pushed || self.is_tip(state, &mut pass) {
                stack.pop();
                pass.residual = pass.residual.max(self.backup(state)?);
            } else {
                self.push_unvisited_children(state, epoch, stack);
            }
        }

        Ok(pass)
    }

    /// Returns `true` for an open state that is not expanded yet, and records in `pass`
    /// that a tip was found. Goals, dead ends and solved states are not tips; the pass
    /// never pushes them.
    fn is_tip(&self, state: Id, pass: &mut Pass) -> bool {
        let node = self.space.node(state);
        let tip = !node.expanded;
        pass.expanded |= tip;

        tip
    }

    /// Pushes the open, unsolved, not yet visited successors of `state`'s greedy action,
    /// marking them visited.
    fn push_unvisited_children(&mut self, state: Id, epoch: u32, stack: &mut Vec<Frame>) {
        let Some(action) = self.space.node(state).policy else {
            return;
        };

        for index in 0..self.space.outcomes(action).len() {
            let target = self.space.outcomes(action)[index].target;
            let node = self.space.node_mut(target);
            if node.kind == Kind::Open && !node.solved && node.mark != epoch {
                node.mark = epoch;
                stack.push((target, false));
            }
        }
    }
}
