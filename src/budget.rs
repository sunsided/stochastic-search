//! Run counters ([`Stats`]) and early stopping ([`Budget`], [`Limits`]).

/// Counters collected during one run of an algorithm.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stats {
    /// Bellman backups that wrote a value. LRTDP's solved checks also evaluate the Bellman
    /// equation; those evaluations count here too, since they cost the same.
    pub backups: u64,
    /// LRTDP trials started. Always `0` for value iteration.
    pub trials: u64,
    /// Value iteration sweeps over the state space. Always `0` for LRTDP.
    pub sweeps: u64,
    /// States whose actions and outcomes were queried from the MDP.
    pub expanded: u64,
    /// Distinct states stored, expanded or not, including goals and dead ends.
    pub states: u64,
}

/// Decides when a run must stop before it converges.
///
/// Asked before every Bellman backup and every trial, with the counters so far, so a run
/// stopped by [`Limits::max_backups`] `= n` has performed exactly `n` backups. Implemented for
/// [`Limits`] and for every `FnMut(&Stats) -> bool`, which lets callers with a clock (this
/// crate is `no_std`) add a time limit.
pub trait Budget {
    /// Returns `true` if the run must stop now, given the counters so far.
    fn exhausted(&mut self, stats: &Stats) -> bool;
}

impl<F: FnMut(&Stats) -> bool> Budget for F {
    fn exhausted(&mut self, stats: &Stats) -> bool {
        self(stats)
    }
}

/// Simple counter limits. The default is unlimited.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Limits {
    /// Stop once this many Bellman backups have been performed. `None` is unbounded.
    pub max_backups: Option<u64>,
    /// Stop once this many LRTDP trials have been started. `None` is unbounded.
    pub max_trials: Option<u64>,
}

impl Limits {
    /// No limits: the run continues until it converges.
    pub const NONE: Self = Self {
        max_backups: None,
        max_trials: None,
    };

    /// Stops after `max` Bellman backups.
    #[must_use]
    pub const fn backups(max: u64) -> Self {
        Self {
            max_backups: Some(max),
            max_trials: None,
        }
    }
}

impl Budget for Limits {
    fn exhausted(&mut self, stats: &Stats) -> bool {
        self.max_backups.is_some_and(|max| stats.backups >= max)
            || self.max_trials.is_some_and(|max| stats.trials >= max)
    }
}
