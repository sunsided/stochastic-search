//! The state shared by the multi-objective algorithms: problem, space, budget and the
//! multi-objective Bellman backup (equations 7 to 9 of the paper).

use core::hash::Hash;
use core::mem;

use alloc::vec::Vec;

use crate::budget::{Budget, Stats};
use crate::config::{Config, DeadEnds};
use crate::error::Error;
use crate::meter::{Halt, Meter};
use crate::mo::coverage::{NO_ACTION, Point, Scratch, ValueSet, add_scaled, hausdorff, prune};
use crate::mo::solution::MoSolution;
use crate::mo::space::{Problem, Space};
use crate::mo::{MoMdp, Objectives};
use crate::model::{Goal, Heuristic};
use crate::observer::{Event, Observer};
use crate::rng::SplitMix64;
use crate::solution::Status;
use crate::space::{Id, Kind};

/// One multi-objective run over a lazily built space.
pub(crate) struct Run<'a, M: MoMdp<N>, G, H, B, O, const N: usize> {
    /// The problem being solved.
    pub(crate) problem: Problem<'a, M, G, H, N>,
    /// The states touched so far, with their value sets.
    pub(crate) space: Space<M::State, M::Action, N>,
    /// Reused buffers of the coverage set pruning.
    pub(crate) scratch: Scratch<N>,
    /// The residual threshold.
    pub(crate) epsilon: f64,
    /// The step limit of one MOLRTDP trial.
    pub(crate) max_trial_length: usize,
    /// Budget, counters and observer.
    pub(crate) meter: Meter<B, O>,
    /// The initial state.
    pub(crate) start: Id,
    /// The residual of the most recent backup, for progress events.
    last_residual: f64,
    /// The trial sampler.
    pub(crate) rng: SplitMix64,
    /// The epoch of the current graph search, compared against the node marks.
    epoch: u32,
    /// Reused buffer: the partial sums of one Q-value.
    acc: Vec<Point<N>>,
    /// Reused buffer: the next partial sums.
    next: Vec<Point<N>>,
}

impl<'a, M, G, H, B, O, const N: usize> Run<'a, M, G, H, B, O, N>
where
    M: MoMdp<N>,
    M::State: Clone + Eq + Hash,
    G: Goal<M::State>,
    H: Heuristic<M::State, Vec<[f64; N]>>,
    B: Budget<Stats>,
    O: Observer,
{
    /// Validates the settings and interns the initial state.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfig`] for invalid settings (including a dead-end penalty,
    /// which the multi-objective algorithms replace by the bound `b`), and
    /// [`Error::InvalidHeuristic`] for an invalid estimate of the initial state.
    pub(crate) fn new(
        mdp: &'a M,
        initial: M::State,
        goal: &'a G,
        heuristic: H,
        objectives: Objectives<N>,
        config: Config<B, O>,
    ) -> Result<Self, Error<M::State>> {
        config.validate()?;
        if config.dead_ends != DeadEnds::Infinite {
            return Err(Error::InvalidConfig {
                reason: "multi-objective runs take the bound b instead of a dead-end penalty",
            });
        }

        let margin = objectives.validate(config.epsilon)?;
        let mut problem = Problem {
            mdp,
            goal,
            heuristic,
            bound: objectives.bound,
            margin,
        };
        let mut space = Space::new();
        let mut scratch = Scratch::default();
        let start = space.intern(&mut problem, initial, &mut scratch)?;

        Ok(Self {
            problem,
            space,
            scratch,
            epsilon: config.epsilon,
            max_trial_length: config.max_trial_length,
            rng: SplitMix64::new(config.seed),
            meter: Meter::new(config),
            start,
            last_residual: f64::INFINITY,
            epoch: 0,
            acc: Vec::new(),
            next: Vec::new(),
        })
    }

    /// Finishes the run: extracts one policy per vector of the initial value.
    pub(crate) fn into_solution(mut self, status: Status) -> MoSolution<M::State, M::Action, N> {
        self.meter.sync(&self.space);

        MoSolution::new(
            self.space,
            self.start,
            status,
            self.meter.stats,
            self.epsilon,
        )
    }

    /// Expands `state` if needed and backs it up; returns the Hausdorff distance between
    /// its old and new value. Goals and dead ends cost nothing and return 0, except for a
    /// state that this expansion turned into a dead end, whose estimate was replaced by
    /// `b`.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    pub(crate) fn backup(&mut self, state: Id) -> Result<f64, Halt<M::State>> {
        if self.space.node(state).kind != Kind::Open {
            return Ok(0.0);
        }

        if !self.space.node(state).expanded {
            let estimate = self.space.node(state).value.clone();
            self.space
                .expand(&mut self.problem, state, &mut self.scratch)?;
            let node = self.space.node(state);
            if node.kind != Kind::Open {
                return Ok(hausdorff(&estimate, &node.value, &self.problem.bound));
            }
        }

        self.meter.charge_backup(&self.space)?;
        let value = self.bellman(state);
        let node = self.space.node_mut(state);
        let residual = hausdorff(&node.value, &value, &self.problem.bound);
        node.value = value;
        if state == self.start {
            self.last_residual = residual;
        }

        self.report_progress();

        Ok(residual)
    }

    /// The new value of an expanded state: the coverage set of the union of its Q-values,
    /// with `b` where an action's Q-value contains it.
    fn bellman(&mut self, state: Id) -> ValueSet<N> {
        let mut value = ValueSet::default();
        for action in self.space.action_range(state) {
            if self.q_value(action) {
                value.bound.push(action);
            }

            value.points.append(&mut self.acc);
        }

        prune(&mut value.points, self.problem.margin, &mut self.scratch);

        value
    }

    /// Computes the Q-value of `action` (equation 8) into `self.acc`: the bounded set sum
    /// of its cost vector and its successors' values, pruned after every successor.
    /// Returns `true` if the Q-value contains `b`.
    fn q_value(&mut self, action: u32) -> bool {
        let entry = self.space.action(action);
        self.acc.clear();
        self.acc.push(Point {
            vector: entry.cost,
            action,
        });

        let mut bound = false;
        for outcome in self.space.outcomes(action) {
            let set = &self.space.node(outcome.target).value;
            bound |= add_scaled(
                &self.acc,
                set,
                outcome.probability,
                &self.problem.bound,
                &mut self.next,
            );
            prune(&mut self.next, self.problem.margin, &mut self.scratch);
            mem::swap(&mut self.acc, &mut self.next);
            if self.acc.is_empty() {
                break;
            }
        }

        bound
    }

    /// Collects the greedy actions of `state` (the paper's `getActions`): every action
    /// whose Q-value contributes a vector or `b` to the state's value. Sorted, no
    /// duplicates.
    pub(crate) fn greedy_actions(&self, state: Id, out: &mut Vec<u32>) {
        let value = &self.space.node(state).value;
        out.clear();
        out.extend(value.points.iter().map(|point| point.action));
        out.extend(value.bound.iter().copied());
        out.retain(|&action| action != NO_ACTION);
        out.sort_unstable();
        out.dedup();
    }

    /// Starts a new visit epoch. Clears all marks when the counter wraps.
    pub(crate) fn next_epoch(&mut self) -> u32 {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            for node in self.space.nodes_mut() {
                node.mark = 0;
            }

            self.epoch = 1;
        }

        self.epoch
    }

    /// Emits a progress event if one is due.
    fn report_progress(&mut self) {
        if !self.meter.progress_due() {
            return;
        }

        self.meter.sync(&self.space);
        self.meter.schedule_progress();
        let event = Event::FrontProgress {
            backups: self.meter.stats.backups,
            states: self.meter.stats.states,
            front: self.space.node(self.start).value.points.len(),
            residual: self.last_residual,
        };
        self.meter.emit(event);
    }

    /// Collects the states reachable from the initial state under all greedy actions (the
    /// paper's `solutionGraph`) into `graph`, in depth-first preorder. Does not descend
    /// below unexpanded states (the tips), goals and dead ends.
    pub(crate) fn collect_solution_graph(&mut self, graph: &mut Vec<Id>, stack: &mut Vec<Id>) {
        let epoch = self.next_epoch();
        let mut actions = Vec::new();
        graph.clear();
        stack.clear();
        self.space.node_mut(self.start).mark = epoch;
        stack.push(self.start);
        while let Some(state) = stack.pop() {
            graph.push(state);
            let node = self.space.node(state);
            if node.kind != Kind::Open || !node.expanded {
                continue;
            }

            self.greedy_actions(state, &mut actions);
            for &action in &actions {
                for index in 0..self.space.outcomes(action).len() {
                    let target = self.space.outcomes(action)[index].target;
                    let node = self.space.node_mut(target);
                    if node.mark != epoch {
                        node.mark = epoch;
                        stack.push(target);
                    }
                }
            }
        }
    }

    /// Returns `true` for an open state that is not expanded yet: a tip.
    pub(crate) fn is_tip(&self, state: Id) -> bool {
        let node = self.space.node(state);

        node.kind == Kind::Open && !node.expanded
    }
}
