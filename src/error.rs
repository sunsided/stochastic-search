//! Errors for invalid input ([`Error`]).

use core::fmt;

/// The tolerance on the sum of an action's outcome probabilities.
///
/// The probabilities emitted by [`Mdp::outcomes`](crate::Mdp::outcomes) must sum to 1 within
/// this absolute tolerance. Wide enough for decimal probabilities such as `0.1` summed in
/// binary floating point, narrow enough to catch a missing outcome.
pub const PROBABILITY_TOLERANCE: f64 = 1e-9;

/// Invalid input found while solving. Each variant names the state where it was found.
///
/// Implements [`core::error::Error`] when the state is [`fmt::Debug`].
#[derive(Debug, Clone, PartialEq)]
pub enum Error<S> {
    /// An action cost is negative, NaN or infinite.
    InvalidCost {
        /// The state whose action has the cost.
        state: S,
        /// The offending cost.
        cost: f64,
    },
    /// An outcome probability is NaN or outside `(0, 1]`.
    InvalidProbability {
        /// The state whose action has the outcome.
        state: S,
        /// The offending probability.
        probability: f64,
    },
    /// The outcome probabilities of an action do not sum to 1 within
    /// [`PROBABILITY_TOLERANCE`]. An action without outcomes sums to 0.
    ProbabilitySum {
        /// The state whose action has the outcomes.
        state: S,
        /// The sum of the outcome probabilities.
        sum: f64,
    },
    /// The heuristic estimated NaN or negative infinity.
    InvalidHeuristic {
        /// The state that was estimated.
        state: S,
    },
    /// A [`Config`](crate::Config) setting is out of range.
    InvalidConfig {
        /// Which setting, and the valid range.
        reason: &'static str,
    },
}

impl<S: fmt::Debug> fmt::Display for Error<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCost { state, cost } => {
                write!(f, "invalid action cost {cost} in state {state:?}")
            }
            Self::InvalidProbability { state, probability } => {
                write!(
                    f,
                    "invalid outcome probability {probability} in state {state:?}"
                )
            }
            Self::ProbabilitySum { state, sum } => {
                write!(
                    f,
                    "outcome probabilities sum to {sum}, not 1, in state {state:?}"
                )
            }
            Self::InvalidHeuristic { state } => {
                write!(
                    f,
                    "the heuristic estimated NaN or negative infinity for state {state:?}"
                )
            }
            Self::InvalidConfig { reason } => write!(f, "invalid configuration: {reason}"),
        }
    }
}

impl<S: fmt::Debug> core::error::Error for Error<S> {}
