//! Capability traits that describe a stochastic shortest path problem: [`Mdp`], [`Goal`] and
//! [`Heuristic`].
//!
//! As in `heuristic-search`, there is no single "problem" trait. The MDP describes actions,
//! costs and outcome distributions; the goal test and the heuristic are separate values, so
//! one MDP can be solved for many goals with many heuristics. [`Goal`] and [`Heuristic`] are
//! implemented for plain closures.

/// A finite Markov decision process with action costs.
///
/// Together with a [`Goal`] it defines a stochastic shortest path (SSP) problem: reach a goal
/// state with minimal expected total cost. Both methods report through a callback, so
/// implementations need no allocation.
///
/// Algorithms query a state's actions and outcomes once and cache them, so the methods may be
/// moderately expensive. They are never called for goal states.
///
/// # Example
///
/// A coin flip that reaches the goal (state `1`) with probability `0.5` per attempt:
///
/// ```
/// use stochastic_search::Mdp;
///
/// struct Retry;
///
/// impl Mdp for Retry {
///     type State = u8;
///     type Action = ();
///
///     fn actions(&self, _state: &u8, mut emit: impl FnMut((), f64)) {
///         emit((), 1.0);
///     }
///
///     fn outcomes(&self, state: &u8, _action: &(), mut emit: impl FnMut(u8, f64)) {
///         emit(1, 0.5);
///         emit(*state, 0.5);
///     }
/// }
/// ```
pub trait Mdp {
    /// A state.
    ///
    /// Algorithms clone states into their state table and hash them to detect duplicates, so
    /// they ask for `Clone + Eq + Hash`.
    type State;
    /// An action label, returned in policies.
    type Action;

    /// Calls `emit` once for every action applicable in `state`, with its cost.
    ///
    /// Costs must be finite and non-negative. The SSP guarantees assume that every action of
    /// a non-goal state costs more than zero (see `docs/design.md`). A state without
    /// applicable actions is a dead end.
    fn actions(&self, state: &Self::State, emit: impl FnMut(Self::Action, f64));

    /// Calls `emit` with every successor of applying `action` in `state` and its probability.
    ///
    /// Each probability must lie in `(0, 1]`, and the probabilities must sum to 1 within
    /// [`PROBABILITY_TOLERANCE`](crate::PROBABILITY_TOLERANCE). A successor emitted twice
    /// counts with its summed probability.
    fn outcomes(
        &self,
        state: &Self::State,
        action: &Self::Action,
        emit: impl FnMut(Self::State, f64),
    );
}

/// A goal test over states of type `S`.
///
/// Implemented for every `Fn(&S) -> bool`. Goal states are absorbing and cost-free.
pub trait Goal<S> {
    /// Returns `true` if `state` is a goal state.
    fn is_goal(&self, state: &S) -> bool;
}

impl<S, F: Fn(&S) -> bool> Goal<S> for F {
    fn is_goal(&self, state: &S) -> bool {
        self(state)
    }
}

/// An estimate of the expected cost from a state to a goal.
///
/// Implemented for every `Fn(&S) -> Option<C>`. The algorithms of this crate use `C = f64`,
/// which closures infer on their own. The shape matches `heuristic-search`'s trait of the
/// same name.
///
/// LRTDP returns optimal values only if the heuristic is admissible (never overestimates the
/// optimal expected cost). Inadmissible heuristics never cause a panic; they only void the
/// optimality bound.
pub trait Heuristic<S, C> {
    /// Estimates the expected remaining cost from `state` to a goal.
    ///
    /// `None` (or an infinite estimate) means no proper policy exists from `state` (a proven
    /// dead end). Algorithms never expand such states, so a heuristic must only return `None`
    /// when that is certain. NaN and negative infinity are rejected with
    /// [`Error::InvalidHeuristic`](crate::Error::InvalidHeuristic).
    fn estimate(&self, state: &S) -> Option<C>;
}

impl<S, C, F: Fn(&S) -> Option<C>> Heuristic<S, C> for F {
    fn estimate(&self, state: &S) -> Option<C> {
        self(state)
    }
}

/// The blind heuristic: estimates `0` everywhere. Admissible for every SSP.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Zero;

impl<S> Heuristic<S, f64> for Zero {
    fn estimate(&self, _state: &S) -> Option<f64> {
        Some(0.0)
    }
}
