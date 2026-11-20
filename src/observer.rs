//! Runtime reporting: [`Observer`] receives [`Event`]s while an algorithm runs.
//!
//! Events are informational only. They never change a run's values, policy or
//! [`Status`](crate::Status); in particular [`Event::SuspectedDivergence`] is a heuristic
//! signal, not a verdict.

/// Something worth reporting while an algorithm runs.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Event {
    /// Periodic progress, every [`Config::progress_interval`](crate::Config) backups.
    Progress {
        /// Bellman backups so far.
        backups: u64,
        /// States stored so far.
        states: u64,
        /// The current value of the initial state.
        initial_value: f64,
        /// LRTDP, RTDP and iLAO*: the change of the initial state's value at its most recent
        /// backup. Value iteration: the largest residual so far in the sweep in progress.
        residual: f64,
    },
    /// A dead-end analysis finished; `count` states were newly proven to have no proper
    /// policy and became dead ends. Emitted after every analysis, also when `count` is 0.
    DeadEndsProven {
        /// States newly turned into dead ends by this analysis.
        count: u64,
    },
    /// The initial state's value keeps rising without its growth slowing down, which is
    /// what a value climbing a trap cycle looks like.
    ///
    /// Heuristic and not authoritative (see
    /// [`Config::divergence_window`](crate::Config)): LRTDP and RTDP check the initial
    /// state's value after every trial, iLAO* after every pass. If it rose in each of the
    /// last `divergence_window` checks and the rise in the last is at least half the rise in
    /// the first of that window,
    /// the event fires and the window starts over. A solvable problem whose values climb
    /// slowly towards a large optimum, at a steady rate, triggers it too.
    SuspectedDivergence {
        /// The current value of the initial state.
        value: f64,
        /// How much the value rose over the window.
        growth: f64,
        /// Bellman backups so far.
        backups: u64,
    },
}

/// Receives [`Event`]s while an algorithm runs.
///
/// The method has a no-op default. Implemented for `()` (the default, which ignores
/// everything), for `&mut O` of any observer `O`, and for closures through
/// [`observer_fn`]. When [`Observer::ENABLED`] is `false`, as for `()`, the algorithms skip
/// all bookkeeping for events, so an unused observer costs nothing.
pub trait Observer {
    /// Whether the algorithms should produce events at all. Leave it `true` for real
    /// observers; it is `false` only for `()`.
    const ENABLED: bool = true;

    /// Receives one event.
    fn observe(&mut self, event: Event) {
        let _ = event;
    }
}

impl Observer for () {
    const ENABLED: bool = false;
}

impl<O: Observer + ?Sized> Observer for &mut O {
    const ENABLED: bool = O::ENABLED;

    fn observe(&mut self, event: Event) {
        (**self).observe(event);
    }
}

/// Turns a closure `FnMut(Event)` into an [`Observer`].
///
/// # Example
///
/// ```
/// use stochastic_search::{Config, Event, observer_fn};
///
/// let mut proven = 0;
/// let config = Config::default().with_observer(observer_fn(|event| {
///     if let Event::DeadEndsProven { count } = event {
///         proven += count;
///     }
/// }));
/// # let _ = config;
/// ```
pub fn observer_fn<F: FnMut(Event)>(f: F) -> ObserverFn<F> {
    ObserverFn(f)
}

/// A closure used as an [`Observer`]; created by [`observer_fn`].
#[derive(Debug, Clone, Copy)]
pub struct ObserverFn<F>(F);

impl<F: FnMut(Event)> Observer for ObserverFn<F> {
    fn observe(&mut self, event: Event) {
        (self.0)(event);
    }
}
