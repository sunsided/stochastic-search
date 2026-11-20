//! Budget accounting shared by the algorithms ([`Meter`], [`Halt`]).

use crate::budget::{Budget, Stats};
use crate::error::Error;
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

/// Counts a run's work and asks the budget before each unit of it.
pub(crate) struct Meter<B> {
    /// The caller's budget.
    budget: B,
    /// The counters so far.
    pub(crate) stats: Stats,
}

impl<B: Budget> Meter<B> {
    /// Starts counting from zero.
    pub(crate) fn new(budget: B) -> Self {
        Self {
            budget,
            stats: Stats::default(),
        }
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
