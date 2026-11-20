//! Run configuration ([`Config`]) and dead-end semantics ([`DeadEnds`]).

use crate::budget::{Budget, Limits, Stats};
use crate::error::Error;
use crate::observer::Observer;

/// How states without a proper policy are valued.
///
/// See `docs/design.md`, section "Dead ends".
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum DeadEnds {
    /// Dead ends cost infinity (the "fragile" SSP semantics).
    ///
    /// Exact when a proper policy exists from the initial state. If none exists, the run
    /// ends with [`Status::Unsolvable`](crate::Status::Unsolvable) (or with an exhausted
    /// budget, if a heuristic search cannot prove it in time); see [`lrtdp`](crate::lrtdp())'s
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
/// Build one with [`Config::default`] and the `with_*` methods, which swap one setting
/// each; [`Config::with_budget`] and [`Config::with_observer`] also change the type. The
/// struct is `#[non_exhaustive]`, so new settings can be added without a breaking change;
/// outside this crate, the fields can be read and assigned but not listed in a struct
/// expression.
///
/// ```
/// use stochastic_search::{Config, DeadEnds, Stats};
///
/// let config = Config::default()
///     .with_epsilon(1e-4)
///     .with_dead_ends(DeadEnds::Penalty(100.0))
///     .with_budget(|stats: &Stats| stats.backups >= 1_000_000);
/// # let _ = config;
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config<B = Limits, O = ()> {
    /// Convergence threshold on Bellman residuals. Must be finite and greater than zero.
    ///
    /// Default `1e-6`.
    pub epsilon: f64,
    /// Dead-end semantics. Default [`DeadEnds::Infinite`].
    pub dead_ends: DeadEnds,
    /// Seed of the successor sampler used by LRTDP, RTDP, BRTDP and VPI-RTDP trials (and FRTDP's
    /// fallback trials). A seed reproduces a run
    /// exactly.
    ///
    /// Default `0`.
    pub seed: u64,
    /// LRTDP, RTDP and the bounded algorithms cut a trial off after this many steps and start the next one.
    /// At least `1`.
    ///
    /// Guards against trials that circle a dead-end trap forever. Default `10_000`.
    pub max_trial_length: usize,
    /// Stops a run early. Default [`Limits::DEFAULT`] (`10^8` backups, `10^6` stored states,
    /// about 250 MB for typical states), which is finite, so a run with the default
    /// configuration always returns. Raise a cap with
    /// `Config::default().with_budget(Limits { max_states: Some(n), ..Limits::DEFAULT })`;
    /// [`Limits`] documents the memory per stored state.
    pub budget: B,
    /// Receives [`Event`](crate::Event)s during the run. Default `()`, which ignores them
    /// at no cost.
    pub observer: O,
    /// Emit [`Event::Progress`](crate::Event::Progress) every this many backups. `None`
    /// never emits it. Default `Some(1_000_000)`; ignored by the default observer.
    pub progress_interval: Option<u64>,
    /// The number of consecutive LRTDP or RTDP trials (iLAO* passes, LAO* expansion steps) over which
    /// [`Event::SuspectedDivergence`](crate::Event::SuspectedDivergence) looks for steady
    /// growth of the initial value. At least `1`. Default `16`; ignored by the default
    /// observer.
    pub divergence_window: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            epsilon: 1e-6,
            dead_ends: DeadEnds::Infinite,
            seed: 0,
            max_trial_length: 10_000,
            budget: Limits::DEFAULT,
            observer: (),
            progress_interval: Some(1_000_000),
            divergence_window: 16,
        }
    }
}

impl<B, O> Config<B, O> {
    /// Returns the same configuration with another [`Config::epsilon`].
    #[must_use]
    pub fn with_epsilon(self, epsilon: f64) -> Self {
        Self { epsilon, ..self }
    }

    /// Returns the same configuration with other [`Config::dead_ends`] semantics.
    #[must_use]
    pub fn with_dead_ends(self, dead_ends: DeadEnds) -> Self {
        Self { dead_ends, ..self }
    }

    /// Returns the same configuration with another [`Config::seed`].
    #[must_use]
    pub fn with_seed(self, seed: u64) -> Self {
        Self { seed, ..self }
    }

    /// Returns the same configuration with another [`Config::max_trial_length`].
    #[must_use]
    pub fn with_max_trial_length(self, max_trial_length: usize) -> Self {
        Self {
            max_trial_length,
            ..self
        }
    }

    /// Returns the same configuration with another [`Config::progress_interval`].
    #[must_use]
    pub fn with_progress_interval(self, progress_interval: Option<u64>) -> Self {
        Self {
            progress_interval,
            ..self
        }
    }

    /// Returns the same configuration with another [`Config::divergence_window`].
    #[must_use]
    pub fn with_divergence_window(self, divergence_window: usize) -> Self {
        Self {
            divergence_window,
            ..self
        }
    }

    /// Returns the same configuration with another budget.
    #[must_use]
    pub fn with_budget<B2: Budget<Stats>>(self, budget: B2) -> Config<B2, O> {
        Config {
            epsilon: self.epsilon,
            dead_ends: self.dead_ends,
            seed: self.seed,
            max_trial_length: self.max_trial_length,
            budget,
            observer: self.observer,
            progress_interval: self.progress_interval,
            divergence_window: self.divergence_window,
        }
    }

    /// Returns the same configuration with another observer.
    #[must_use]
    pub fn with_observer<O2: Observer>(self, observer: O2) -> Config<B, O2> {
        Config {
            epsilon: self.epsilon,
            dead_ends: self.dead_ends,
            seed: self.seed,
            max_trial_length: self.max_trial_length,
            budget: self.budget,
            observer,
            progress_interval: self.progress_interval,
            divergence_window: self.divergence_window,
        }
    }

    /// Checks the numeric settings.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfig`] for an epsilon that is not finite and positive, a
    /// penalty that is not finite and non-negative, a zero trial length, a zero progress
    /// interval or a zero divergence window.
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

        if self.progress_interval == Some(0) || self.divergence_window == 0 {
            return Err(Error::InvalidConfig {
                reason: "progress_interval and divergence_window must be at least 1",
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
