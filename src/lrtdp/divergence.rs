//! The heuristic behind [`Event::SuspectedDivergence`](crate::Event::SuspectedDivergence).

/// Watches the initial state's value across trials for steady, unslowing growth.
///
/// Rule: the value must rise in each of `window` consecutive checks, and the rise in the last
/// check must be at least half the rise in the first. Then the watcher reports and starts a
/// new window. Any check without a finite rise resets it. Converging runs usually rise by
/// shrinking amounts and stay silent; values climbing a trap cycle rise by a steady amount.
#[derive(Debug, Clone)]
pub(super) struct Divergence {
    /// The number of consecutive rising checks that make a report.
    window: usize,
    /// Consecutive rising checks so far in this window.
    streak: usize,
    /// The value before the first rise of this window.
    window_start: f64,
    /// The rise in the first check of this window.
    first_rise: f64,
    /// The value at the previous check.
    last_value: f64,
}

impl Divergence {
    /// Starts watching from `initial_value` with windows of `window` checks (at least 1).
    pub(super) fn new(window: usize, initial_value: f64) -> Self {
        Self {
            window,
            streak: 0,
            window_start: initial_value,
            first_rise: 0.0,
            last_value: initial_value,
        }
    }

    /// Records the value at one check. Returns the growth over the window if it completes a
    /// window of steady growth.
    pub(super) fn check(&mut self, value: f64) -> Option<f64> {
        let rise = value - self.last_value;
        let previous = self.last_value;
        self.last_value = value;
        if !(rise > 0.0 && rise.is_finite()) {
            self.streak = 0;
            return None;
        }

        if self.streak == 0 {
            self.window_start = previous;
            self.first_rise = rise;
        }

        self.streak += 1;
        if self.streak < self.window {
            return None;
        }

        self.streak = 0;

        (rise >= self.first_rise / 2.0).then_some(value - self.window_start)
    }
}

#[cfg(test)]
mod tests {
    use super::Divergence;

    /// Steady growth is reported once per window, with the growth over the window.
    #[test]
    fn steady_growth_is_reported() {
        let mut watch = Divergence::new(3, 0.0);

        assert_eq!(watch.check(1.0), None);
        assert_eq!(watch.check(2.0), None);
        assert_eq!(watch.check(3.0), Some(3.0));
        assert_eq!(watch.check(4.0), None);
    }

    /// Geometrically shrinking growth, as in a converging run, is not reported.
    #[test]
    fn shrinking_growth_is_silent() {
        let mut watch = Divergence::new(3, 0.0);

        assert_eq!(watch.check(1.0), None);
        assert_eq!(watch.check(1.3), None);
        assert_eq!(watch.check(1.35), None);
    }

    /// A check without a rise starts the window over.
    #[test]
    fn a_pause_resets_the_window() {
        let mut watch = Divergence::new(2, 0.0);

        assert_eq!(watch.check(1.0), None);
        assert_eq!(watch.check(1.0), None);
        assert_eq!(watch.check(2.0), None);
        assert_eq!(watch.check(3.0), Some(2.0));
    }
}
