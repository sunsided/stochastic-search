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
/// Implements [`core::error::Error`] when the state is [`fmt::Debug`]. The enum is
/// `#[non_exhaustive]`: new kinds of invalid input can be added without a breaking change,
/// so a `match` outside this crate needs a wildcard arm.
#[non_exhaustive]
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
    /// The heuristic estimated NaN or negative infinity, a POMDP upper bound (see
    /// [`despot`](crate::despot())) is NaN, or a [`brtdp`](crate::brtdp()) upper estimate is
    /// NaN, negative infinity or below the lower estimate of the same state.
    InvalidHeuristic {
        /// The state that was estimated.
        state: S,
    },
    /// Under [`DeadEnds::Infinite`](crate::DeadEnds::Infinite), the upper bound of
    /// [`brtdp`](crate::brtdp()) estimated no finite value (`None` or infinity) for a state
    /// the lower bound does not prove to be a dead end.
    ///
    /// The bounded algorithms need a finite upper bound wherever the optimal value is finite: an infinite
    /// one on a cycle never becomes finite through backups. Give a finite upper estimate,
    /// return `None` from the lower bound too where the state is a dead end, or use
    /// [`DeadEnds::Penalty`](crate::DeadEnds::Penalty), whose penalty bounds every value.
    UnboundedUpper {
        /// The state that was estimated.
        state: S,
    },
    /// A [`Pomdp::step`](crate::Pomdp::step) returned a NaN or infinite reward.
    InvalidReward {
        /// The state the step started from.
        state: S,
        /// The offending reward.
        reward: f64,
    },
    /// A [`Pomdp`](crate::Pomdp) emitted no actions.
    NoActions,
    /// A POMDP planner was given a belief without particles.
    EmptyBelief,
    /// A configuration setting is out of range.
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
                    "the heuristic or bound estimated an invalid value for state {state:?}"
                )
            }
            Self::UnboundedUpper { state } => {
                write!(
                    f,
                    "no finite upper bound for state {state:?}, which is not a known dead end"
                )
            }
            Self::InvalidReward { state, reward } => {
                write!(f, "invalid reward {reward} from state {state:?}")
            }
            Self::NoActions => f.write_str("the POMDP has no actions"),
            Self::EmptyBelief => f.write_str("the belief has no particles"),
            Self::InvalidConfig { reason } => write!(f, "invalid configuration: {reason}"),
        }
    }
}

impl<S: fmt::Debug> core::error::Error for Error<S> {}
