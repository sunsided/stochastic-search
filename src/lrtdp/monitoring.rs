//! Work between LRTDP trials: the dead-end proof schedule and runtime events.

use core::hash::Hash;

use crate::budget::Budget;
use crate::lrtdp::Run;
use crate::model::{Goal, Heuristic, Mdp};
use crate::observer::{Event, Observer};

impl<M, G, H, B, O> Run<'_, M, G, H, B, O>
where
    M: Mdp,
    M::State: Clone + Eq + Hash,
    G: Goal<M::State>,
    H: Heuristic<M::State, f64>,
    B: Budget,
    O: Observer,
{
    /// Under infinite dead-end cost, proves by graph analysis which stored states cannot
    /// reach a goal with probability 1 and turns them into dead ends, when it is due.
    ///
    /// It is due after a cut-off trial (`cut_off`), the symptom of a trap cycle, and
    /// whenever the backup count has doubled since the last analysis, which catches traps
    /// that trials rarely enter but solved checks keep backing up. Either way it only runs
    /// once at least as many backups as stored states plus outcomes happened since the last
    /// one: each analysis round is linear in the stored graph, so its cost stays proportional
    /// to the search effort.
    pub(crate) fn maybe_prove_dead_ends(&mut self, cut_off: bool) {
        let work = (self.space.len() + self.space.outcome_count()) as u64;
        let backups = self.meter.stats.backups;
        let doubled = backups >= self.analyzed_at.saturating_mul(2);
        if self.problem.cap.is_finite()
            || !(cut_off || doubled)
            || backups - self.analyzed_at < work
        {
            return;
        }

        self.analyzed_at = backups;
        let count = self.space.mark_dead_ends() as u64;
        self.meter.emit(Event::DeadEndsProven { count });
    }

    /// Feeds the initial value to the divergence watcher and reports steady growth.
    /// Does nothing for a disabled observer.
    pub(crate) fn track_divergence(&mut self) {
        if !O::ENABLED {
            return;
        }

        let value = self.space.node(self.start).value;
        if let Some(growth) = self.divergence.check(value) {
            let backups = self.meter.stats.backups;
            self.meter.emit(Event::SuspectedDivergence {
                value,
                growth,
                backups,
            });
        }
    }

    /// Emits a progress event if one is due. Does nothing for a disabled observer.
    pub(crate) fn report_progress(&mut self) {
        if self.meter.progress_due() {
            let value = self.space.node(self.start).value;
            self.meter
                .emit_progress(&self.space, value, self.start_residual);
        }
    }
}
