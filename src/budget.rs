//! Run counters ([`Stats`]) and early stopping ([`Budget`], [`Limits`]).

/// Counters collected during one run of an algorithm.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stats {
    /// Bellman backups that wrote a value. The solved checks of LRTDP, RTDP and iLAO* also
    /// evaluate the Bellman equation; those evaluations count here too, since they cost the
    /// same.
    pub backups: u64,
    /// LRTDP and RTDP trials started. Always `0` for value iteration and iLAO*.
    pub trials: u64,
    /// Value iteration sweeps over the state space, or iLAO* passes over the greedy graph.
    /// Always `0` for LRTDP and RTDP.
    pub sweeps: u64,
    /// States whose actions and outcomes were queried from the MDP.
    pub expanded: u64,
    /// Distinct states stored, expanded or not, including goals and dead ends.
    pub states: u64,
}

/// Decides when a run must stop before it converges.
///
/// Asked before every Bellman backup, every LRTDP trial and every state expansion of value
/// iteration's enumeration, with the counters so far, so a run
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

/// Simple counter limits.
///
/// The default, [`Limits::DEFAULT`], is generous but finite, so a run with the default
/// [`Config`](crate::Config) always returns. [`Limits::unlimited`] removes every limit.
///
/// Raise a cap with struct update syntax:
///
/// ```
/// use stochastic_search::{Config, Limits};
///
/// let config = Config::default().with_budget(Limits {
///     max_states: Some(20_000_000),
///     max_backups: Some(2_000_000_000),
///     ..Limits::DEFAULT
/// });
/// # let _ = config;
/// ```
///
/// # Memory per stored state
///
/// Roughly `38 + size_of::<S>()` bytes per state (node, state copy, hash table slot), plus
/// `16 + size_of::<A>()` bytes per action and `16` bytes per outcome once the state is
/// expanded, plus whatever heap memory the state itself owns. A state with three actions of
/// three outcomes each and small `S` and `A` takes about 250 bytes, so the default cap of
/// `10^6` states is about 250 MB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Stop once this many Bellman backups have been performed. `None` is unbounded.
    pub max_backups: Option<u64>,
    /// Stop once this many LRTDP or RTDP trials have been started. `None` is unbounded.
    pub max_trials: Option<u64>,
    /// Stop once this many states are stored. `None` is unbounded.
    ///
    /// See [Memory per stored state](Limits#memory-per-stored-state) to size it.
    ///
    /// Bounds memory, and the state enumeration of value iteration on huge or infinite
    /// state spaces. Checked before every expansion and backup, so the final count may
    /// exceed the limit by the successors of one state.
    pub max_states: Option<u64>,
}

impl Limits {
    /// The default limits: `10^8` backups and `10^6` stored states, no trial limit.
    ///
    /// The state cap keeps memory around 250 MB for typical states (see
    /// [Memory per stored state](Limits#memory-per-stored-state)); the backup cap allows
    /// about 100 backups per stored state, which LRTDP and value iteration rarely exceed,
    /// and takes seconds to a minute. A non-converging run (see [`lrtdp`](crate::lrtdp())'s
    /// termination notes) stops instead of hanging. Raise either cap as shown on
    /// [`Limits`].
    pub const DEFAULT: Self = Self {
        max_backups: Some(100_000_000),
        max_trials: None,
        max_states: Some(1_000_000),
    };

    /// No limits: the run continues until it converges, which may be never.
    #[must_use]
    pub const fn unlimited() -> Self {
        Self {
            max_backups: None,
            max_trials: None,
            max_states: None,
        }
    }

    /// Stops after `max` Bellman backups, with no other limit.
    #[must_use]
    pub const fn backups(max: u64) -> Self {
        Self {
            max_backups: Some(max),
            ..Self::unlimited()
        }
    }
}

impl Default for Limits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl Budget for Limits {
    fn exhausted(&mut self, stats: &Stats) -> bool {
        self.max_backups.is_some_and(|max| stats.backups >= max)
            || self.max_trials.is_some_and(|max| stats.trials >= max)
            || self.max_states.is_some_and(|max| stats.states >= max)
    }
}
