//! Graph analysis: which states have a policy that reaches a goal with probability 1.

use alloc::vec;
use alloc::vec::Vec;

use crate::space::{Id, Kind, Space};

/// Marks every state of a fully expanded space that has a proper policy.
///
/// The standard greatest fixed point: start with all states as candidates; repeatedly keep
/// only the candidates that can reach a goal using *safe* actions, those whose every
/// outcome stays among the candidates. Graph-only, so it is exact regardless of the
/// probability values.
///
/// shortcut: quadratic in the number of states per round, fine for the oracle's small
/// problems; switch to predecessor lists if value iteration is ever used at scale.
pub(super) fn reaches_goal_almost_surely<S, A>(space: &Space<S, A>) -> Vec<bool> {
    let mut candidates = vec![true; space.len()];
    loop {
        let reaching = reach_goal_within(space, &candidates);
        if reaching == candidates {
            return reaching;
        }

        candidates = reaching;
    }
}

/// Marks the candidates that reach a goal through safe actions (every outcome a candidate)
/// with positive probability.
fn reach_goal_within<S, A>(space: &Space<S, A>, candidates: &[bool]) -> Vec<bool> {
    let mut reaching: Vec<bool> = (0..space.len())
        .map(|index| candidates[index] && space.node(id(index)).kind == Kind::Goal)
        .collect();

    let mut changed = true;
    while changed {
        changed = false;
        for index in 0..space.len() {
            if reaching[index] || !candidates[index] {
                continue;
            }

            if has_safe_action_towards(space, id(index), candidates, &reaching) {
                reaching[index] = true;
                changed = true;
            }
        }
    }

    reaching
}

/// Returns `true` if `state` has an action whose outcomes are all candidates and at least
/// one of which already reaches a goal.
fn has_safe_action_towards<S, A>(
    space: &Space<S, A>,
    state: Id,
    candidates: &[bool],
    reaching: &[bool],
) -> bool {
    space.actions_of(state).any(|(index, _)| {
        let outcomes = space.outcomes(index);
        let safe = outcomes.iter().all(|o| candidates[o.target as usize]);

        safe && outcomes.iter().any(|o| reaching[o.target as usize])
    })
}

/// Converts a state index to an id; indices come from a space capped at `u32::MAX` states.
fn id(index: usize) -> Id {
    Id::try_from(index).expect("interned indices fit an id")
}
