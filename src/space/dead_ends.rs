//! Graph analysis that proves states to be dead ends: states without any policy that reaches
//! a goal with probability 1.

use alloc::vec;
use alloc::vec::Vec;

use crate::space::{Id, Kind, Space};

impl<S, A> Space<S, A> {
    /// Turns every expanded, unsolved state that provably has no proper policy into a dead
    /// end valued at infinity, labeled solved and without a policy action. Returns how many
    /// changed. Solved states are left alone, so a frozen policy is never rewritten; marking
    /// is optional for soundness, only needed for termination.
    ///
    /// Sound on a partially expanded space: unexpanded states count as able to reach a goal,
    /// so only states that cannot reach a goal whatever the unexpanded part looks like are
    /// marked. On a fully expanded space the result is exact. Only meaningful under infinite
    /// dead-end cost; callers with a finite penalty must not call it.
    pub(crate) fn mark_dead_ends(&mut self) -> usize {
        let proper = self.optimistically_proper();
        let mut marked = 0;
        for (index, &has_proper_policy) in proper.iter().enumerate() {
            let node = self.node_mut(id(index));
            if !has_proper_policy && node.kind == Kind::Open && node.expanded && !node.solved {
                node.kind = Kind::DeadEnd;
                node.value = f64::INFINITY;
                node.solved = true;
                node.policy = None;
                marked += 1;
            }
        }

        marked
    }

    /// Marks every state that has a policy reaching a goal or an unexpanded state with
    /// probability 1.
    ///
    /// The standard greatest fixed point: start with all states as candidates; repeatedly
    /// keep only the candidates that reach a target through *safe* actions, those whose every
    /// outcome stays among the candidates. Graph-only, so it is exact regardless of the
    /// probability values. Each round is linear in the stored outcomes and removes at least
    /// one state; in practice there are few rounds.
    fn optimistically_proper(&self) -> Vec<bool> {
        let predecessors = Predecessors::build(self);
        let mut candidates = vec![true; self.len()];
        loop {
            let reaching = self.reach_targets_within(&predecessors, &candidates);
            if reaching == candidates {
                return reaching;
            }

            candidates = reaching;
        }
    }

    /// Marks the candidates that reach a target (a goal or an unexpanded open state) with
    /// positive probability using only safe actions.
    fn reach_targets_within(&self, predecessors: &Predecessors, candidates: &[bool]) -> Vec<bool> {
        let safe = self.safe_actions(candidates);
        let mut reaching: Vec<bool> = (0..self.len())
            .map(|index| candidates[index] && self.is_target(id(index)))
            .collect();
        let mut stack: Vec<Id> = (0..self.len())
            .filter(|&index| reaching[index])
            .map(id)
            .collect();

        while let Some(target) = stack.pop() {
            for &(state, action) in predecessors.of(target) {
                let index = state as usize;
                if candidates[index] && !reaching[index] && safe[action as usize] {
                    reaching[index] = true;
                    stack.push(state);
                }
            }
        }

        reaching
    }

    /// Marks, per entry of the action arena, whether every outcome is a candidate.
    fn safe_actions(&self, candidates: &[bool]) -> Vec<bool> {
        (0..self.action_count())
            .map(|index| {
                let index = u32::try_from(index).expect("action indices fit u32");

                self.outcomes(index)
                    .iter()
                    .all(|outcome| candidates[outcome.target as usize])
            })
            .collect()
    }

    /// Returns `true` for states the analysis assumes to reach a goal: goals, and open states
    /// that are not expanded yet.
    fn is_target(&self, state: Id) -> bool {
        let node = self.node(state);

        node.kind == Kind::Goal || (node.kind == Kind::Open && !node.expanded)
    }
}

/// The reverse edges of the stored graph in compressed sparse row form.
struct Predecessors {
    /// `entries[offsets[t]..offsets[t + 1]]` are the predecessors of state `t`.
    offsets: Vec<usize>,
    /// `(state, action index)` pairs whose action has the indexed state as an outcome.
    entries: Vec<(Id, u32)>,
}

impl Predecessors {
    /// Collects the reverse edges of every expanded state's actions.
    fn build<S, A>(space: &Space<S, A>) -> Self {
        let mut offsets = vec![0; space.len() + 1];
        space.for_each_edge(|_, _, target| offsets[target as usize + 1] += 1);
        for index in 1..offsets.len() {
            offsets[index] += offsets[index - 1];
        }

        let mut fill = offsets.clone();
        let mut entries = vec![(0, 0); offsets[space.len()]];
        space.for_each_edge(|state, action, target| {
            let slot = &mut fill[target as usize];
            entries[*slot] = (state, action);
            *slot += 1;
        });

        Self { offsets, entries }
    }

    /// The predecessors of `target`.
    fn of(&self, target: Id) -> &[(Id, u32)] {
        let index = target as usize;

        &self.entries[self.offsets[index]..self.offsets[index + 1]]
    }
}

impl<S, A> Space<S, A> {
    /// Calls `visit(state, action index, target)` for every outcome of every expanded state.
    fn for_each_edge(&self, mut visit: impl FnMut(Id, u32, Id)) {
        for index in 0..self.len() {
            let state = id(index);
            for (action, _) in self.actions_of(state) {
                for outcome in self.outcomes(action) {
                    visit(state, action, outcome.target);
                }
            }
        }
    }
}

/// Converts a state index to an id; indices come from a space capped at `u32::MAX` states.
fn id(index: usize) -> Id {
    Id::try_from(index).expect("interned indices fit an id")
}
