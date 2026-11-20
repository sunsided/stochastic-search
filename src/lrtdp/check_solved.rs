//! The `checkSolved` procedure of LRTDP: label a state solved once its greedy envelope is
//! epsilon-consistent.

use core::hash::Hash;
use core::mem;

use alloc::vec::Vec;

use crate::budget::Budget;
use crate::lrtdp::Run;
use crate::meter::Halt;
use crate::model::{Goal, Heuristic, Mdp};
use crate::observer::Observer;
use crate::space::Id;
use crate::space::backup::residual;

impl<M, G, H, B, O> Run<'_, M, G, H, B, O>
where
    M: Mdp,
    M::State: Clone + Eq + Hash,
    G: Goal<M::State>,
    H: Heuristic<M::State, f64>,
    B: Budget,
    O: Observer,
{
    /// Labels `start` and its greedy envelope solved if every unsolved state in it has a
    /// residual of at most epsilon; otherwise backs up the visited states in reverse order.
    ///
    /// Returns whether the states were labeled.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    pub(super) fn check_solved(&mut self, start: Id) -> Result<bool, Halt<M::State>> {
        let mut open = mem::take(&mut self.open);
        let mut closed = mem::take(&mut self.closed);
        open.clear();
        closed.clear();

        let result = self.search_envelope(start, &mut open, &mut closed);
        let result = match result {
            Ok(true) => {
                self.label_solved(&closed);
                Ok(true)
            }
            Ok(false) => self.back_up_reversed(&closed).map(|()| false),
            Err(halt) => Err(halt),
        };

        self.open = open;
        self.closed = closed;

        result
    }

    /// Depth-first search over the greedy graph from `start`, stopping at solved states and
    /// at states whose residual exceeds epsilon. Freezes the greedy action of every visited
    /// state and collects them in `closed`.
    ///
    /// Returns `true` if no visited state exceeded epsilon.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    fn search_envelope(
        &mut self,
        start: Id,
        open: &mut Vec<Id>,
        closed: &mut Vec<Id>,
    ) -> Result<bool, Halt<M::State>> {
        let epoch = self.next_epoch();
        if !self.space.node(start).solved {
            self.space.node_mut(start).mark = epoch;
            open.push(start);
        }

        let mut consistent = true;
        while let Some(state) = open.pop() {
            closed.push(state);
            if self.freeze_greedy(state)? > self.epsilon {
                consistent = false;
                continue;
            }

            self.push_unvisited_successors(state, epoch, open);
        }

        Ok(consistent)
    }

    /// Expands `state` if needed, stores its greedy action without changing its value, and
    /// returns its residual.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    fn freeze_greedy(&mut self, state: Id) -> Result<f64, Halt<M::State>> {
        self.space.expand(&mut self.problem, state)?;
        self.meter.charge_backup(&self.space)?;

        let backup = self.space.bellman(state);
        let node = self.space.node_mut(state);
        node.policy = backup.policy;
        let residual = residual(node.value, backup.value);
        self.report_progress();

        Ok(residual)
    }

    /// Pushes the successors of `state`'s greedy action that are neither solved nor visited
    /// in this epoch, marking them visited.
    fn push_unvisited_successors(&mut self, state: Id, epoch: u32, open: &mut Vec<Id>) {
        let Some(action) = self.space.node(state).policy else {
            return;
        };

        for index in 0..self.space.outcomes(action).len() {
            let target = self.space.outcomes(action)[index].target;
            let node = self.space.node_mut(target);
            if !node.solved && node.mark != epoch {
                node.mark = epoch;
                open.push(target);
            }
        }
    }

    /// Labels every state in `closed` solved, freezing its greedy action.
    fn label_solved(&mut self, closed: &[Id]) {
        for &state in closed {
            self.space.node_mut(state).solved = true;
        }
    }

    /// Backs up the states in `closed` in reverse visiting order.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    fn back_up_reversed(&mut self, closed: &[Id]) -> Result<(), Halt<M::State>> {
        for &state in closed.iter().rev() {
            self.backup(state)?;
        }

        Ok(())
    }

    /// Starts a new visit epoch. Clears all marks when the counter wraps, so a stale mark
    /// never matches.
    fn next_epoch(&mut self) -> u32 {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            for node in self.space.nodes_mut() {
                node.mark = 0;
            }

            self.epoch = 1;
        }

        self.epoch
    }
}
