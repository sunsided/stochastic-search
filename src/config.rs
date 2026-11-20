//! Run configuration ([`Config`]) and dead-end semantics ([`DeadEnds`]).

use crate::budget::{Budget, Limits};
use crate::error::Error;

/// How states without a proper policy are valued.
///
/// See `docs/design.md`, section "Dead ends".
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum DeadEnds {
    /// Dead ends cost infinity (the "fragile" SSP semantics).
    ///
    /// Exact when a proper policy exists from the initial state. If none exists, the run
    /// ends with [`Status::Unsolvable`](crate::Status::Unsolvable) (or with an exhausted
    /// budget, if LRTDP cannot prove it in time); see [`lrtdp`](crate::lrtdp())'s
    /// termination notes.
    #[default]
    Infinite,
    /// Every non-goal state may give up at this finite, non-negative cost.
    ///
    /// Values are capped at the penalty, every policy is proper, and the algorithms converge
    /// on any input. A state valued at the penalty has no action in the policy.
    Penalty(f64),
}

impl DeadEnds {
    /// The value of a dead end, which is also the cap on every value.
    pub(crate) fn cap(self) -> f64 {
        match self {
            Self::Infinite => f64::INFINITY,
            Self::Penalty(penalty) => penalty,
        }
    }
}

/// Settings shared by all algorithms.
///
/// Build one with [`Config::default`] and struct update syntax, and swap the budget with
/// [`Config::with_budget`]:
///
/// ```
/// use stochastic_search::{Config, DeadEnds, Stats};
///
/// let config = Config {
///     epsilon: 1e-4,
///     dead_ends: DeadEnds::Penalty(100.0),
///     ..Config::default()
/// }
/// .with_budget(|stats: &Stats| stats.backups >= 1_000_000);
/// # let _ = config;
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config<B = Limits> {
    /// Convergence threshold on Bellman residuals. Must be finite and greater than zero.
    ///
    /// Default `1e-6`.
    pub epsilon: f64,
    /// Dead-end semantics. Default [`DeadEnds::Infinite`].
    pub dead_ends: DeadEnds,
    /// Seed of the successor sampler used by LRTDP trials. A seed reproduces a run exactly.
    ///
    /// Default `0`.
    pub seed: u64,
    /// LRTDP cuts a trial off after this many steps and starts the next one. At least `1`.
    ///
    /// Guards against trials that circle a dead-end trap forever. Default `10_000`.
    pub max_trial_length: usize,
    /// Stops a run early. Default [`Limits::DEFAULT`], which is finite, so a run with the
    /// default configuration always returns.
    pub budget: B,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            epsilon: 1e-6,
            dead_ends: DeadEnds::Infinite,
            seed: 0,
            max_trial_length: 10_000,
            budget: Limits::DEFAULT,
        }
    }
}

impl<B> Config<B> {
    /// Returns the same configuration with another budget.
    #[must_use]
    pub fn with_budget<B2: Budget>(self, budget: B2) -> Config<B2> {
        Config {
            epsilon: self.epsilon,
            dead_ends: self.dead_ends,
            seed: self.seed,
            max_trial_length: self.max_trial_length,
            budget,
        }
    }

    /// Checks the numeric settings.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfig`] for an epsilon that is not finite and positive, a
    /// penalty that is not finite and non-negative, or a zero trial length.
    pub(crate) fn validate<S>(&self) -> Result<(), Error<S>> {
        if !(self.epsilon.is_finite() && self.epsilon > 0.0) {
            return Err(Error::InvalidConfig {
                reason: "epsilon must be finite and greater than zero",
            });
        }

        if matches!(self.dead_ends, DeadEnds::Penalty(penalty) if !(penalty.is_finite() && penalty >= 0.0))
        {
            return Err(Error::InvalidConfig {
                reason: "the dead-end penalty must be finite and non-negative",
            });
        }

        if self.max_trial_length == 0 {
            return Err(Error::InvalidConfig {
                reason: "max_trial_length must be at least 1",
            });
        }

        Ok(())
    }
}
