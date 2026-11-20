//! Runtime events: dead-end proofs on rarely entered traps, suspected divergence, progress,
//! and the cost of the default observer.

use stochastic_search::{
    Config, Event, Limits, Mdp, Observer, Solution, Status, Zero, lrtdp, observer_fn,
};

use crate::table::{Table, action};

/// A trap (a cost-1 self-loop) entered with probability `1e-9` from the initial state.
/// With `escape`, the initial state also has a sure action of cost 2.
fn rare_trap(escape: bool) -> Table {
    let mut first = vec![action(1.0, &[(1, 1.0 - 1e-9), (2, 1e-9)])];
    if escape {
        first.push(action(2.0, &[(1, 1.0)]));
    }

    Table {
        actions: vec![first, vec![], vec![action(1.0, &[(2, 1.0)])]],
        goal: 1,
    }
}

/// Runs LRTDP from state 0 with `config` and collects every event.
fn run_collecting(table: &Table, config: Config) -> (Solution<usize, usize>, Vec<Event>) {
    let mut events = Vec::new();
    let solution = lrtdp(
        table,
        0,
        &|s: &usize| table.is_goal(*s),
        Zero,
        config.with_observer(observer_fn(|event| events.push(event))),
    )
    .expect("valid input");

    (solution, events)
}

/// Trials practically never enter the trap, but solved checks keep backing it up; the
/// doubling trigger proves it dead within the default budget.
#[test]
fn rarely_entered_unavoidable_trap_is_proven() {
    let (solution, events) = run_collecting(&rare_trap(false), Config::default());

    assert_eq!(solution.status(), Status::Unsolvable);
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::DeadEndsProven { count } if *count > 0))
    );
}

/// With an escape, the rarely entered trap is proven dead and the escape wins.
#[test]
fn rarely_entered_avoidable_trap_is_avoided() {
    let table = rare_trap(true);
    let solution = lrtdp(
        &table,
        0,
        &|s: &usize| table.is_goal(*s),
        Zero,
        Config::default(),
    )
    .expect("valid input");

    assert_eq!(solution.status(), Status::Converged);
    assert!((solution.initial_value() - 2.0).abs() < 1e-9);
    assert_eq!(solution.initial_action(), Some(&1));
}

/// A ring of states without a goal. Short trials never expand the whole ring, so the dead
/// ends cannot be proven, and the initial value climbs steadily.
struct Ring;

impl Mdp for Ring {
    type State = u32;
    type Action = ();

    fn actions(&self, _state: &u32, mut emit: impl FnMut((), f64)) {
        emit((), 1.0);
    }

    fn outcomes(&self, state: &u32, _action: &(), mut emit: impl FnMut(u32, f64)) {
        emit((state + 1) % 1000, 1.0);
    }
}

/// An unsolvable cycle reports suspected divergence before the run ends.
#[test]
fn unsolvable_cycle_reports_suspected_divergence() {
    let mut first_suspicion = None;
    let config = Config::default()
        .with_max_trial_length(20)
        .with_budget(Limits::backups(100_000))
        .with_observer(observer_fn(|event| {
            if let Event::SuspectedDivergence {
                backups,
                value,
                growth,
            } = event
            {
                assert!(value > 0.0 && growth > 0.0);
                first_suspicion.get_or_insert(backups);
            }
        }));
    let solution = lrtdp(&Ring, 0, &|_: &u32| false, Zero, config).expect("valid input");

    assert!(matches!(
        solution.status(),
        Status::Unsolvable | Status::BudgetExhausted
    ));
    let suspicion = first_suspicion.expect("divergence was suspected");
    assert!(suspicion < solution.stats().backups);
}

/// A solvable retry loop with success probability `0.001` climbs slowly to its optimum of
/// 1000: it may look like divergence, but the run still converges, and the observer does
/// not change the result.
#[test]
fn slow_solvable_problem_still_converges() {
    let table = Table {
        actions: vec![vec![action(1.0, &[(1, 0.001), (0, 0.999)])], vec![]],
        goal: 1,
    };
    let config = Config::default()
        .with_max_trial_length(1)
        .with_progress_interval(Some(1000));

    let (observed, events) = run_collecting(&table, config);
    let quiet =
        lrtdp(&table, 0, &|s: &usize| table.is_goal(*s), Zero, config).expect("valid input");

    assert_eq!(observed.status(), Status::Converged);
    assert!((observed.initial_value() - 1000.0).abs() < 1e-2);
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::SuspectedDivergence { .. }))
    );
    assert!(events.iter().any(|e| matches!(e, Event::Progress { .. })));
    assert_eq!(observed.stats(), quiet.stats());
    assert_eq!(
        observed.initial_value().to_bits(),
        quiet.initial_value().to_bits()
    );
}

/// Progress events arrive at the configured rate.
#[test]
fn progress_follows_the_interval() {
    let table = rare_trap(true);
    let (solution, events) =
        run_collecting(&table, Config::default().with_progress_interval(Some(1)));
    let progress = events
        .iter()
        .filter(|e| matches!(e, Event::Progress { .. }))
        .count();

    assert_eq!(
        u64::try_from(progress).expect("small"),
        solution.stats().backups
    );
}

/// The default observer is a zero-sized no-op whose bookkeeping is compiled out.
#[test]
fn default_observer_costs_nothing() {
    const { assert!(!<() as Observer>::ENABLED) };
    assert_eq!(size_of_val(&Config::default().observer), 0);
}
