//! Classic LAO* (Hansen and Zilberstein 2001): expand one tip of the best partial solution
//! graph at a time, then run value iteration on its ancestors.
//!
//! ```mermaid
//! graph TD
//!     A[collect the best partial solution graph of s0] --> B{unexpanded tip?}
//!     B -- yes --> C[expand one tip]
//!     C --> D[value iteration on the tip and its ancestors along greedy arcs]
//!     D --> A
//!     B -- no --> E[sweep the solution graph]
//!     E --> F{residual below epsilon and solved check on s0 passes?}
//!     F -- no --> A
//!     F -- yes --> G[return values and frozen policy]
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

/// Solves an SSP from `initial` with classic LAO*, one tip at a time.
///
/// LAO* (Hansen and Zilberstein 2001) grows an explicit graph from `initial` and keeps the
/// best partial solution graph: the states reachable from `initial` under the greedy
/// policy. Each step expands one unexpanded tip of that graph (the first in depth-first
/// order) and runs value iteration, until the largest residual is below
/// [`Config::epsilon`], on the set `Z` of the tip and its ancestors along greedy arcs. Once the
/// graph has no tips left, the paper's convergence test runs: sweeps over the solution graph
/// until their residual is below epsilon, then the solved check of [`lrtdp`](crate::lrtdp())
/// on `initial`, which freezes the policy. If a sweep or the check turns the policy towards
/// new tips, expansion resumes.
///
/// This is the paper's original algorithm. Its improved variant,
/// [`ilao_star`](crate::ilao_star()), expands every tip in one pass and is usually much
/// faster: here, every expansion recomputes the solution graph and the ancestor set. It
/// samples nothing, so [`Config::seed`] and [`Config::max_trial_length`] are ignored. Every
/// value iteration sweep (on `Z` or on the solution graph) counts in
/// [`Stats::sweeps`](crate::Stats).
///
/// # Guarantee
///
/// A converged run has exactly the guarantee of [`lrtdp`](crate::lrtdp()) (section
/// "Guarantee" there), since the final solved check is the same.
///
/// # Termination
///
/// With the default [`Limits`](crate::Limits) budget the call always returns. Under
/// [`DeadEnds::Infinite`](crate::DeadEnds::Infinite), value iteration on a set containing a
/// trap cycle raises its values forever; the dead-end proof of [`lrtdp`](crate::lrtdp())
/// runs after every sweep once the backup count has doubled since the last analysis, so
/// traps become dead ends and an unavoidable one yields
/// [`Status::Unsolvable`](crate::Status::Unsolvable).
///
/// # Events
///
/// As for [`ilao_star`](crate::ilao_star()), with suspected divergence checked after every
/// expansion step.
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
/// use stochastic_search::{Config, Mdp, Zero, lao_star};
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
/// let config = Config::default().with_epsilon(1e-9);
/// let solution = lao_star(&Corridor, 0, &|state: &u8| *state == 3, Zero, config)?;
///
/// assert!(solution.is_converged());
/// assert!((solution.initial_value() - 4.0).abs() < 1e-6);
/// # Ok::<(), stochastic_search::Error<u8>>(())
/// ```
pub fn lao_star<M, G, H, B, O>(
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
    let mut buffers = Buffers::default();
    let status = finish(run.solve_lao(&mut buffers))?;

    Ok(run.into_solution(status))
}

/// Reused buffers of one LAO* run.
#[derive(Default)]
struct Buffers {
    /// The best partial solution graph in depth-first preorder.
    graph: Vec<Id>,
    /// The tip and its ancestors along greedy arcs, the tip last.
    ancestors: Vec<Id>,
    /// Greedy arcs inside the solution graph as `(target, source)`, sorted by target.
    arcs: Vec<(Id, Id)>,
    /// The depth-first stack of [`Run::collect_solution_graph`].
    stack: Vec<Id>,
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
    /// Expands tips one at a time until the initial state is solved.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    fn solve_lao(&mut self, buffers: &mut Buffers) -> Result<(), Halt<M::State>> {
        while !self.space.node(self.start).solved {
            self.meter.check(&self.space)?;
            self.collect_solution_graph(buffers);
            let tip = buffers
                .graph
                .iter()
                .copied()
                .find(|&state| !self.space.node(state).expanded);

            if let Some(tip) = tip {
                self.backup(tip)?;
                self.collect_ancestors(tip, buffers);
                self.converge_on(&buffers.ancestors)?;
            } else if self.sweep(&buffers.graph)? < self.epsilon {
                self.check_solved(self.start, false)?;
            }

            self.track_divergence();
        }

        Ok(())
    }

    /// Collects the open, unsolved states reachable from the initial state under the greedy
    /// policy into `buffers.graph`, in depth-first preorder. Does not descend below
    /// unexpanded states (the tips) or states without a policy action.
    fn collect_solution_graph(&mut self, buffers: &mut Buffers) {
        let epoch = self.next_epoch();
        let (graph, stack) = (&mut buffers.graph, &mut buffers.stack);
        graph.clear();
        stack.clear();
        if self.is_unsolved_open(self.start) {
            self.space.node_mut(self.start).mark = epoch;
            stack.push(self.start);
        }

        while let Some(state) = stack.pop() {
            graph.push(state);
            let Some(action) = self.space.node(state).policy else {
                continue;
            };

            for index in 0..self.space.outcomes(action).len() {
                let target = self.space.outcomes(action)[index].target;
                if self.is_unsolved_open(target) && self.space.node(target).mark != epoch {
                    self.space.node_mut(target).mark = epoch;
                    stack.push(target);
                }
            }
        }
    }

    /// Returns `true` for an open state that is not labeled solved.
    pub(crate) fn is_unsolved_open(&self, state: Id) -> bool {
        let node = self.space.node(state);

        node.kind == Kind::Open && !node.solved
    }

    /// Collects `tip` and every state of the solution graph that reaches it along greedy
    /// arcs into `buffers.ancestors`, in reverse breadth-first order from `tip`, so that
    /// [`Run::sweep`] backs up `tip` first and its farthest ancestors last.
    fn collect_ancestors(&mut self, tip: Id, buffers: &mut Buffers) {
        self.collect_greedy_arcs(buffers);

        let epoch = self.next_epoch();
        let (arcs, ancestors) = (&buffers.arcs, &mut buffers.ancestors);
        ancestors.clear();
        ancestors.push(tip);
        self.space.node_mut(tip).mark = epoch;
        let mut next = 0;
        while next < ancestors.len() {
            let target = ancestors[next];
            next += 1;
            let from = arcs.partition_point(|&(t, _)| t < target);
            for &(_, source) in arcs[from..].iter().take_while(|&&(t, _)| t == target) {
                if self.space.node(source).mark != epoch {
                    self.space.node_mut(source).mark = epoch;
                    ancestors.push(source);
                }
            }
        }

        ancestors.reverse();
    }

    /// Collects the greedy arcs between states of the solution graph, sorted by target.
    fn collect_greedy_arcs(&self, buffers: &mut Buffers) {
        let arcs = &mut buffers.arcs;
        arcs.clear();
        for &source in &buffers.graph {
            let Some(action) = self.space.node(source).policy else {
                continue;
            };

            arcs.extend(
                self.space
                    .outcomes(action)
                    .iter()
                    .map(|outcome| (outcome.target, source)),
            );
        }

        arcs.sort_unstable();
    }

    /// Runs value iteration on `states` until a sweep's largest residual is below epsilon.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    fn converge_on(&mut self, states: &[Id]) -> Result<(), Halt<M::State>> {
        while self.sweep(states)? >= self.epsilon {}

        Ok(())
    }

    /// Backs up every state of `states` once, last first, and returns the largest residual.
    /// Counts as one sweep and gives the dead-end proof a chance to run afterwards.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    fn sweep(&mut self, states: &[Id]) -> Result<f64, Halt<M::State>> {
        self.meter.stats.sweeps += 1;
        let mut largest = 0.0_f64;
        for &state in states.iter().rev() {
            largest = largest.max(self.backup(state)?);
        }

        self.maybe_prove_dead_ends(false);

        Ok(largest)
    }
}
