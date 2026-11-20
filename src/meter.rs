//! Budget accounting shared by the algorithms ([`Meter`], [`Halt`]).

use crate::budget::{Budget, Stats};
use crate::config::Config;
use crate::error::Error;
use crate::observer::{Event, Observer};
use crate::solution::Status;
use crate::space::Space;

/// Why a run stopped before converging.
pub(crate) enum Halt<S> {
    /// The budget is exhausted; the run returns what it has.
    Budget,
    /// The input is invalid; the run returns the error.
    Invalid(Error<S>),
}

impl<S> From<Error<S>> for Halt<S> {
    fn from(error: Error<S>) -> Self {
        Self::Invalid(error)
    }
}

/// Turns the outcome of a run's main loop into a status or an error.
///
/// # Errors
///
/// Returns the error of a [`Halt::Invalid`].
pub(crate) fn finish<S>(outcome: Result<(), Halt<S>>) -> Result<Status, Error<S>> {
    match outcome {
        Ok(()) => Ok(Status::Converged),
        Err(Halt::Budget) => Ok(Status::BudgetExhausted),
        Err(Halt::Invalid(error)) => Err(error),
    }
}

/// Counts a run's work, asks the budget before each unit of it, and reports events.
pub(crate) struct Meter<B, O> {
    /// The caller's budget.
    budget: B,
    /// The caller's observer.
    observer: O,
    /// Backups between progress events; `None` for none.
    progress_interval: Option<u64>,
    /// The backup count at which the next progress event is due.
    next_progress: u64,
    /// The counters so far.
    pub(crate) stats: Stats,
}

impl<B: Budget<Stats>, O: Observer> Meter<B, O> {
    /// Starts counting from zero with the budget, observer and progress interval of
    /// `config`.
    pub(crate) fn new(config: Config<B, O>) -> Self {
        let progress_interval = config.progress_interval;

        Self {
            budget: config.budget,
            observer: config.observer,
            progress_interval,
            next_progress: progress_interval.unwrap_or(u64::MAX),
            stats: Stats::default(),
        }
    }

    /// Sends `event` to the observer. Compiles to nothing for a disabled observer.
    pub(crate) fn emit(&mut self, event: Event) {
        if O::ENABLED {
            self.observer.observe(event);
        }
    }

    /// Returns `true` if a progress event is due. Always `false` for a disabled observer.
    pub(crate) fn progress_due(&self) -> bool {
        O::ENABLED && self.stats.backups >= self.next_progress
    }

    /// Emits a progress event with the current counters and schedules the next one.
    pub(crate) fn emit_progress<S, A>(
        &mut self,
        space: &Space<S, A>,
        initial_value: f64,
        residual: f64,
    ) {
        self.sync(space);
        self.next_progress = self.progress_interval.map_or(u64::MAX, |interval| {
            self.stats.backups.saturating_add(interval)
        });
        self.emit(Event::Progress {
            backups: self.stats.backups,
            states: self.stats.states,
            initial_value,
            residual,
        });
    }

    /// Copies the state counts of `space` into the stats.
    pub(crate) fn sync<S, A>(&mut self, space: &Space<S, A>) {
        self.stats.states = space.len() as u64;
        self.stats.expanded = space.expanded();
    }

    /// Asks the budget whether the run may continue.
    ///
    /// # Errors
    ///
    /// Returns [`Halt::Budget`] if the budget is exhausted.
    pub(crate) fn check<S, A>(&mut self, space: &Space<S, A>) -> Result<(), Halt<S>> {
        self.sync(space);
        if self.budget.exhausted(&self.stats) {
            return Err(Halt::Budget);
        }

        Ok(())
    }

    /// Asks the budget for one more Bellman backup and counts it.
    ///
    /// # Errors
    ///
    /// Returns [`Halt::Budget`] if the budget is exhausted; the backup is then not counted.
    pub(crate) fn charge_backup<S, A>(&mut self, space: &Space<S, A>) -> Result<(), Halt<S>> {
        self.check(space)?;
        self.stats.backups += 1;

        Ok(())
    }
}
