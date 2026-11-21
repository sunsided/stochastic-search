//! Budget accounting shared by the algorithms ([`Meter`], [`Halt`]).

use crate::budget::{Budget, Stats};
use crate::config::Config;
use crate::error::Error;
use crate::observer::{Event, Observer};
use crate::solution::Status;
use crate::space::Space;

/// The state counts a [`Meter`] copies into [`Stats`]: implemented by the scalar and the
/// multi-objective state spaces.
pub(crate) trait Counts {
    /// The state type, which names the [`Halt`] of a budget check.
    type State;

    /// The number of interned states.
    fn states(&self) -> u64;

    /// The number of expanded states.
    fn expanded(&self) -> u64;
}

impl<S, A> Counts for Space<S, A> {
    type State = S;

    fn states(&self) -> u64 {
        self.len() as u64
    }

    fn expanded(&self) -> u64 {
        Space::expanded(self)
    }
}

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
        self.schedule_progress();
        self.emit(Event::Progress {
            backups: self.stats.backups,
            states: self.stats.states,
            initial_value,
            residual,
        });
    }

    /// Schedules the next progress event one interval after the current backup count.
    pub(crate) fn schedule_progress(&mut self) {
        self.next_progress = self.progress_interval.map_or(u64::MAX, |interval| {
            self.stats.backups.saturating_add(interval)
        });
    }

    /// Copies the state counts of `space` into the stats.
    pub(crate) fn sync<C: Counts>(&mut self, space: &C) {
        self.stats.states = space.states();
        self.stats.expanded = space.expanded();
    }

    /// Asks the budget whether the run may continue.
    ///
    /// # Errors
    ///
    /// Returns [`Halt::Budget`] if the budget is exhausted.
    pub(crate) fn check<C: Counts>(&mut self, space: &C) -> Result<(), Halt<C::State>> {
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
    pub(crate) fn charge_backup<C: Counts>(&mut self, space: &C) -> Result<(), Halt<C::State>> {
        self.check(space)?;
        self.stats.backups += 1;

        Ok(())
    }
}
