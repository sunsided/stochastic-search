//! An explicit MDP given as tables, shared by all integration tests.

use stochastic_search::Mdp;

/// One action of a [`Table`] state.
#[derive(Debug, Clone)]
pub struct Action {
    /// The action cost.
    pub cost: f64,
    /// The successors and their probabilities.
    pub outcomes: Vec<(usize, f64)>,
}

/// An MDP over states `0..actions.len()`; actions are indices into a state's action list.
#[derive(Debug, Clone)]
pub struct Table {
    /// The actions of each state. Goal states may have any list; it is never queried.
    pub actions: Vec<Vec<Action>>,
    /// The goal state.
    pub goal: usize,
}

impl Table {
    /// The number of states.
    pub fn len(&self) -> usize {
        self.actions.len()
    }

    /// Returns `true` if `state` is the goal.
    pub fn is_goal(&self, state: usize) -> bool {
        state == self.goal
    }

    /// The smallest cost of any action of a non-goal state.
    pub fn min_cost(&self) -> f64 {
        self.actions
            .iter()
            .flatten()
            .map(|action| action.cost)
            .fold(f64::INFINITY, f64::min)
    }
}

impl Mdp for Table {
    type State = usize;
    type Action = usize;

    fn actions(&self, state: &usize, mut emit: impl FnMut(usize, f64)) {
        for (index, action) in self.actions[*state].iter().enumerate() {
            emit(index, action.cost);
        }
    }

    fn outcomes(&self, state: &usize, action: &usize, mut emit: impl FnMut(usize, f64)) {
        for &(target, probability) in &self.actions[*state][*action].outcomes {
            emit(target, probability);
        }
    }
}

/// Shorthand for an [`Action`].
pub fn action(cost: f64, outcomes: &[(usize, f64)]) -> Action {
    Action {
        cost,
        outcomes: outcomes.to_vec(),
    }
}
