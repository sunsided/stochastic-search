//! MOLAO* and iMOLAO* (Algorithms 3 and 4 of Chen, Trevizan and Thiébaux 2023): LAO* and
//! improved LAO* with the multi-objective backup, over the solution graph of all greedy
//! actions.
//!
//! ```mermaid
//! graph TD
//!     A[collect the solution graph of s0 under all greedy actions] --> B{unexpanded tip?}
//!     B -- "yes, MOLAO*" --> C[expand one tip, MOVI on it and its ancestors]
//!     B -- "yes, iMOLAO*" --> D[expand every tip, back up the graph once in postorder]
//!     C --> A
//!     D --> A
//!     B -- no --> E[back up the graph once]
//!     E --> F{residual below epsilon and graph unchanged?}
//!     F -- no --> A
//!     F -- yes --> G[extract one policy per vector of V of s0]
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

/// Solves a multi-objective SSP from `initial` with MOLAO*, one tip at a time.
///
/// Keeps the solution graph: the states reachable from `initial` under every greedy action
/// (every action whose Q-value contributes a vector, or the bound `b`, to its state's
/// value). Each step expands one unexpanded tip of the graph (the first in depth-first
/// order) and runs multi-objective value iteration, until the largest residual is below
/// [`Config::epsilon`], on the tip and its ancestors in the graph. When the graph has no
/// tips left, one more sweep over the whole graph must stay below epsilon and leave the
/// graph unchanged, else expansion resumes (the paper's Algorithm 3 stops without that
/// sweep).
///
/// Every sweep counts in [`Stats::sweeps`](crate::Stats). [`Config::seed`] and
/// [`Config::max_trial_length`] are ignored. Usually slower than
/// [`imolao_star`](crate::imolao_star()), which backs up the graph once per pass.
///
/// # Guarantee
///
/// With an admissible heuristic (in the scalarised sense of the [`mo`](crate::mo) module docs), a converged
/// run's policies have values epsilon-consistent with the convex coverage set of the
/// proper policies, up to the margin of `objectives`. The solution graph is closed: every
/// state a returned policy reaches is expanded or a goal.
///
/// # Termination
///
/// Values are capped by the bound `b`, so trap cycles climb to `b` in finitely many
/// backups and leave the coverage set. A cycle of zero cost in every objective violates
/// the paper's Assumption 1; the default [`Limits`](crate::Limits) budget still stops the
/// run.
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
/// ```
/// use stochastic_search::{Config, MoMdp, Objectives, Zero, molao_star};
///
/// /// A corridor `0..=2`: `walk` costs `[2, 0]`, `drive` costs `[1, 1]`; both advance.
/// struct Corridor;
///
/// impl MoMdp<2> for Corridor {
///     type State = u8;
///     type Action = &'static str;
///
///     fn actions(&self, _state: &u8, mut emit: impl FnMut(&'static str, [f64; 2])) {
///         emit("walk", [2.0, 0.0]);
///         emit("drive", [1.0, 1.0]);
///     }
///
///     fn outcomes(&self, state: &u8, _action: &&'static str, mut emit: impl FnMut(u8, f64)) {
///         emit(state + 1, 1.0);
///     }
/// }
///
/// let goal = |state: &u8| *state == 2;
/// let solution = molao_star(&Corridor, 0, &goal, Zero, Objectives::new([10.0, 10.0]), Config::default())?;
///
/// // Walk twice [4, 0] or drive twice [2, 2]; mixing lies on the hull edge between them.
/// assert_eq!(solution.policies().len(), 2);
/// # Ok::<(), stochastic_search::Error<u8>>(())
/// ```
pub fn molao_star<M, G, H, B, O, const N: usize>(
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
    let status = finish(run.solve_lao(false))?;

    Ok(run.into_solution(status))
}

/// Solves a multi-objective SSP from `initial` with iMOLAO*.
///
/// Each pass collects the solution graph of all greedy actions, expands every unexpanded
/// tip in it and backs up every state of the graph once, descendants before ancestors
/// (reverse depth-first preorder, which is a postorder on the depth-first tree). Stops
/// when a pass finds no tip, its largest residual is below [`Config::epsilon`], and the
/// pass leaves the graph unchanged (a near tie can switch a greedy action). Every
/// pass counts in [`Stats::sweeps`](crate::Stats).
///
/// Guarantee, termination and errors as for [`molao_star`](crate::molao_star()).
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
/// ```
/// use stochastic_search::{Config, MoMdp, Objectives, Zero, imolao_star};
///
/// /// Retry with `[1, 0]` per attempt (success 0.5) or pay `[0, 3]` once.
/// struct Retry;
///
/// impl MoMdp<2> for Retry {
///     type State = bool;
///     type Action = bool;
///
///     fn actions(&self, _done: &bool, mut emit: impl FnMut(bool, [f64; 2])) {
///         emit(false, [1.0, 0.0]);
///         emit(true, [0.0, 3.0]);
///     }
///
///     fn outcomes(&self, _done: &bool, pay: &bool, mut emit: impl FnMut(bool, f64)) {
///         if *pay {
///             emit(true, 1.0);
///         } else {
///             emit(true, 0.5);
///             emit(false, 0.5);
///         }
///     }
/// }
///
/// let config = Config::default().with_epsilon(1e-8);
/// let solution = imolao_star(&Retry, false, &|done: &bool| *done, Zero, Objectives::new([50.0, 50.0]), config)?;
///
/// let mut front: Vec<[f64; 2]> = solution.policies().map(|policy| *policy.value()).collect();
/// front.sort_by(|a, b| a[0].total_cmp(&b[0]));
/// assert!((front[0][1] - 3.0).abs() < 1e-6 && (front[1][0] - 2.0).abs() < 1e-6);
/// # Ok::<(), stochastic_search::Error<bool>>(())
/// ```
pub fn imolao_star<M, G, H, B, O, const N: usize>(
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
    let status = finish(run.solve_lao(true))?;

    Ok(run.into_solution(status))
}

/// Reused buffers of one MOLAO* or iMOLAO* run.
#[derive(Default)]
struct Buffers {
    /// The solution graph in depth-first preorder.
    graph: Vec<Id>,
    /// The depth-first stack.
    stack: Vec<Id>,
    /// The tip and its ancestors in the solution graph.
    ancestors: Vec<Id>,
    /// Greedy arcs inside the solution graph as `(target, source)`, sorted by target.
    arcs: Vec<(Id, Id)>,
    /// The greedy actions of one state.
    actions: Vec<u32>,
    /// The solution graph collected again after a converging sweep.
    check: Vec<Id>,
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
    /// The main loop of both variants: `improved` selects iMOLAO*'s passes over MOLAO*'s
    /// one-tip steps.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    fn solve_lao(&mut self, improved: bool) -> Result<(), Halt<M::State>> {
        let mut buffers = Buffers::default();
        loop {
            self.meter.check(&self.space)?;
            self.collect_solution_graph(&mut buffers.graph, &mut buffers.stack);
            let tips = buffers.graph.iter().any(|&state| self.is_tip(state));
            if improved || !tips {
                let residual = self.sweep(&buffers.graph)?;
                if !tips && residual < self.epsilon && self.is_graph_unchanged(&mut buffers) {
                    return Ok(());
                }
            } else {
                self.expand_one_tip(&mut buffers)?;
            }
        }
    }

    /// Returns `true` if the solution graph collected now equals `buffers.graph`, the one
    /// the last sweep backed up. A sweep below epsilon can still switch a greedy action
    /// (a near tie) and pull new states, even tips, into the graph.
    fn is_graph_unchanged(&mut self, buffers: &mut Buffers) -> bool {
        let mut graph = core::mem::take(&mut buffers.check);
        self.collect_solution_graph(&mut graph, &mut buffers.stack);
        let unchanged = graph == buffers.graph;
        buffers.check = graph;

        unchanged
    }

    /// Expands the first tip of the solution graph and runs value iteration on it and its
    /// ancestors (lines 3 to 9 of Algorithm 3).
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    fn expand_one_tip(&mut self, buffers: &mut Buffers) -> Result<(), Halt<M::State>> {
        let Some(tip) = buffers.graph.iter().copied().find(|&s| self.is_tip(s)) else {
            return Ok(());
        };

        self.backup(tip)?;
        self.collect_ancestors(tip, buffers);
        while self.sweep(&buffers.ancestors)? >= self.epsilon {}

        Ok(())
    }

    /// Collects `tip` and every state of the solution graph that reaches it along greedy
    /// arcs into `buffers.ancestors`, the tip last, so that a sweep (which runs last
    /// first) backs up the tip first and its farthest ancestors last.
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

    /// Collects the greedy arcs leaving the states of the solution graph, sorted by target.
    fn collect_greedy_arcs(&self, buffers: &mut Buffers) {
        buffers.arcs.clear();
        for &source in &buffers.graph {
            self.greedy_actions(source, &mut buffers.actions);
            for &action in &buffers.actions {
                buffers.arcs.extend(
                    self.space
                        .outcomes(action)
                        .iter()
                        .map(|outcome| (outcome.target, source)),
                );
            }
        }

        buffers.arcs.sort_unstable();
        buffers.arcs.dedup();
    }
}
