//! The result of a run: a value function and a greedy policy ([`Solution`]).

use core::fmt;
use core::hash::Hash;

use crate::budget::Stats;
use crate::error::Error;
use crate::model::Mdp;
use crate::space::{Id, Space};

/// The result of solving the MDP `M`: a [`Solution`] or the [`Error`] that stopped the run.
pub type SolveResult<M> =
    Result<Solution<<M as Mdp>::State, <M as Mdp>::Action>, Error<<M as Mdp>::State>>;

/// How a run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Status {
    /// The convergence criterion holds: for LRTDP, the initial state is labeled solved; for
    /// value iteration, the last sweep's largest residual was below epsilon.
    Converged,
    /// The [`Budget`](crate::Budget) stopped the run first. Values and policy are the ones
    /// reached so far and carry no guarantee.
    BudgetExhausted,
    /// Proven: no policy reaches a goal with probability 1 from the initial state, so under
    /// [`DeadEnds::Infinite`](crate::DeadEnds::Infinite) its value is infinite and it has no
    /// policy action. Never returned under a finite penalty, where giving up is always
    /// possible.
    Unsolvable,
}

/// A value function and a greedy policy over the states a run touched.
///
/// See `docs/design.md`, section "Convergence criterion", for what a converged solution
/// guarantees.
pub struct Solution<S, A> {
    /// The states, values and policy actions.
    space: Space<S, A>,
    /// The id of the initial state.
    initial: Id,
    /// How the run ended.
    status: Status,
    /// Counters of the run.
    stats: Stats,
}

impl<S, A> Solution<S, A> {
    /// Wraps the state space of a finished run.
    ///
    /// A run whose initial value is infinite becomes [`Status::Unsolvable`], even if the
    /// budget ran out: infinity only enters values through proven dead ends (states without
    /// actions, heuristic dead ends, the dead-end proof) and spreads only through backups,
    /// where it means every action risks a proven dead end.
    pub(crate) fn new(space: Space<S, A>, initial: Id, status: Status, counters: Stats) -> Self {
        let status = if space.node(initial).value.is_infinite() {
            Status::Unsolvable
        } else {
            status
        };

        Self {
            space,
            initial,
            status,
            stats: counters,
        }
    }

    /// How the run ended.
    #[must_use]
    pub fn status(&self) -> Status {
        self.status
    }

    /// Returns `true` if the run met its convergence criterion with a finite initial value.
    #[must_use]
    pub fn is_converged(&self) -> bool {
        self.status == Status::Converged
    }

    /// Counters of the run.
    #[must_use]
    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    /// The initial state.
    #[must_use]
    pub fn initial_state(&self) -> &S {
        self.space.state(self.initial)
    }

    /// The value of the initial state: the expected cost to reach a goal.
    ///
    /// Infinite exactly when the status is [`Status::Unsolvable`] (which relies on the
    /// heuristic returning `None` only for real dead ends).
    #[must_use]
    pub fn initial_value(&self) -> f64 {
        self.space.node(self.initial).value
    }

    /// The policy's action in the initial state; see [`Solution::action`].
    #[must_use]
    pub fn initial_action(&self) -> Option<&A> {
        self.policy_of(self.initial)
    }

    /// The policy action stored for `id`.
    fn policy_of(&self, id: Id) -> Option<&A> {
        let index = self.space.node(id).policy?;

        Some(&self.space.action(index).action)
    }
}

impl<S: Hash + Eq, A> Solution<S, A> {
    /// The value of `state`, or `None` if the run never touched it.
    ///
    /// For LRTDP, values outside the policy's reachable states are only bounds (lower bounds
    /// with an admissible heuristic), not converged values.
    #[must_use]
    pub fn value(&self, state: &S) -> Option<f64> {
        self.space.find(state).map(|id| self.space.node(id).value)
    }

    /// The policy's action in `state`.
    ///
    /// `None` at goals, dead ends, states where giving up is cheaper (under
    /// [`DeadEnds::Penalty`](crate::DeadEnds::Penalty)) and states the run never backed up.
    /// Every state reachable from the initial state under the policy of a converged run has
    /// an action unless it is one of the first three.
    #[must_use]
    pub fn action(&self, state: &S) -> Option<&A> {
        self.space.find(state).and_then(|id| self.policy_of(id))
    }
}

impl<S: fmt::Debug, A> fmt::Debug for Solution<S, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Solution")
            .field("initial_state", self.initial_state())
            .field("initial_value", &self.initial_value())
            .field("status", &self.status)
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}
