//! Bellman backups, residuals and successor sampling over the cached expansions.

use crate::space::expand::abs;
use crate::space::{ActionEntry, Id, Kind, Space};

/// The result of evaluating the Bellman equation at one state.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Backup {
    /// The backed-up value: the cheapest Q-value, capped at the dead-end value.
    pub(crate) value: f64,
    /// The greedy action as an index into the action arena; `None` if no action beats the
    /// cap (a dead end or a give-up state) or the state is a goal.
    pub(crate) policy: Option<u32>,
}

impl<S, A> Space<S, A> {
    /// The Q-value of an action under `value`: its cost plus the expected value of its
    /// successors.
    fn q_value(&self, index: u32, entry: &ActionEntry<A>, value: impl Fn(Id) -> f64) -> f64 {
        let expected: f64 = self
            .outcomes(index)
            .iter()
            .map(|outcome| outcome.probability * value(outcome.target))
            .sum();

        entry.cost + expected
    }

    /// Evaluates the Bellman equation at `id` without changing anything.
    ///
    /// Goals and proven dead ends keep their value. An unexpanded open state is treated as
    /// having no actions, so callers expand first. Ties go to the action emitted first.
    pub(crate) fn bellman(&self, id: Id) -> Backup {
        self.bellman_under(id, |state| self.node(state).value)
    }

    /// Evaluates the Bellman equation at `id` under another value function, `value`, with
    /// the rules of [`Space::bellman`]. The bounded algorithms use it for their upper bound.
    pub(crate) fn bellman_under(&self, id: Id, value: impl Fn(Id) -> f64) -> Backup {
        if self.node(id).kind != Kind::Open {
            return Backup {
                value: value(id),
                policy: None,
            };
        }

        let mut best = f64::INFINITY;
        let mut policy = None;
        for (index, entry) in self.actions_of(id) {
            let q = self.q_value(index, entry, &value);
            if q < best {
                best = q;
                policy = Some(index);
            }
        }

        if best <= self.cap() {
            Backup {
                value: best,
                policy,
            }
        } else {
            Backup {
                value: self.cap(),
                policy: None,
            }
        }
    }

    /// Backs up `id`: stores the Bellman value and greedy action, and returns the residual.
    pub(crate) fn update(&mut self, id: Id) -> f64 {
        let backup = self.bellman(id);
        let node = self.node_mut(id);
        let residual = residual(node.value, backup.value);
        node.value = backup.value;
        node.policy = backup.policy;

        residual
    }

    /// Picks the successor of the action at `index` whose cumulative probability bucket
    /// contains `u`, a uniform sample from `[0, 1)`.
    ///
    /// Falls back to the last outcome if rounding leaves `u` past the final bucket.
    pub(crate) fn sample(&self, index: u32, u: f64) -> Id {
        let outcomes = self.outcomes(index);
        let mut cumulative = 0.0;
        for outcome in outcomes {
            cumulative += outcome.probability;
            if u < cumulative {
                return outcome.target;
            }
        }

        outcomes
            .last()
            .expect("cached actions always have at least one outcome")
            .target
    }
}

/// The change between two values, with `inf - inf` defined as `0` instead of NaN.
///
/// Values are never NaN or negative infinity, so this is never NaN.
pub(crate) fn residual(old: f64, new: f64) -> f64 {
    if old.is_infinite() && new.is_infinite() {
        0.0
    } else {
        abs(new - old)
    }
}
