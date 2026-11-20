//! Bad heuristics, invalid input and budgets: errors or degraded answers, never panics or
//! hangs.

use proptest::prelude::*;
use stochastic_search::{
    Config, DeadEnds, Error, Heuristic, Limits, Status, Zero, heuristic_fn, ilao_star, lao_star,
    lrtdp, rtdp, value_iteration,
};

use crate::properties::assert_consistent;
use crate::random::{Shape, ssp};
use crate::table::{Table, action};

/// A configuration whose budget bounds every run.
fn bounded() -> Config {
    Config::default()
        .with_epsilon(1e-4)
        .with_max_trial_length(200)
        .with_budget(Limits::backups(200_000))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Inadmissible and otherwise wrong (but not NaN) heuristics never panic in LRTDP, RTDP
    /// iLAO* or LAO*: each run returns a solution.
    #[test]
    fn bad_heuristics_do_not_crash(states in 2_usize..10, seed in any::<u64>(), kind in 0_u8..5) {
        let table = ssp(states, seed, Shape::Traps);
        let mut heuristic = heuristic_fn(|state: &usize| -> Option<f64> {
            let spread = f64::from(u32::try_from((*state * 7919 + 13) % 101).expect("small"));
            match kind {
                0 => Some(1000.0 + spread),           // wild overestimate
                1 => Some(-100.0 - spread),           // negative
                2 => (spread < 50.0).then_some(spread), // false dead ends
                3 => Some(if spread < 30.0 { f64::INFINITY } else { spread }),
                _ => Some(if spread < 50.0 { 0.0 } else { 1e9 }),
            }
        });

        for (dead_ends, algorithm) in [DeadEnds::Infinite, DeadEnds::Penalty(20.0)]
            .into_iter()
            .flat_map(|dead_ends| [0, 1, 2, 3].map(|algorithm| (dead_ends, algorithm)))
        {
            let config = bounded().with_dead_ends(dead_ends);
            let goal = |s: &usize| table.is_goal(*s);
            let solution = match algorithm {
                0 => lrtdp(&table, 0, &goal, &mut heuristic, config),
                1 => rtdp(&table, 0, &goal, &mut heuristic, config),
                2 => ilao_star(&table, 0, &goal, &mut heuristic, config),
                _ => lao_star(&table, 0, &goal, &mut heuristic, config),
            };
            prop_assert!(solution.is_ok());

            // The frozen policy keeps its residual bound even for bad heuristics.
            let solution = solution.expect("checked above");
            if solution.is_converged() {
                assert_consistent(&table, &solution, false)?;
            }
        }
    }
}

/// A NaN estimate is reported, not used.
#[test]
fn nan_heuristic_is_an_error() {
    let table = ssp(4, 1, Shape::Proper);
    let result = lrtdp(
        &table,
        0,
        &|s: &usize| table.is_goal(*s),
        heuristic_fn(|_: &usize| Some(f64::NAN)),
        bounded(),
    );

    assert!(matches!(result, Err(Error::InvalidHeuristic { state: 0 })));
}

/// Probabilities that do not sum to 1 are reported with the state.
#[test]
fn probability_sum_is_validated() {
    let table = Table {
        actions: vec![vec![action(1.0, &[(1, 0.5), (0, 0.4)])], vec![]],
        goal: 1,
    };
    let goal = |s: &usize| table.is_goal(*s);

    assert!(matches!(
        lrtdp(&table, 0, &goal, Zero, bounded()),
        Err(Error::ProbabilitySum { state: 0, .. })
    ));
    assert!(matches!(
        value_iteration(&table, 0, &goal, bounded()),
        Err(Error::ProbabilitySum { state: 0, .. })
    ));
}

/// Out-of-range probabilities and costs are reported.
#[test]
fn probabilities_and_costs_are_validated() {
    let bad_probability = Table {
        actions: vec![vec![action(1.0, &[(1, 1.5), (0, -0.5)])], vec![]],
        goal: 1,
    };
    let bad_cost = Table {
        actions: vec![vec![action(-1.0, &[(1, 1.0)])], vec![]],
        goal: 1,
    };

    assert!(matches!(
        lrtdp(&bad_probability, 0, &|s: &usize| *s == 1, Zero, bounded()),
        Err(Error::InvalidProbability { state: 0, .. })
    ));
    assert!(matches!(
        lrtdp(&bad_cost, 0, &|s: &usize| *s == 1, Zero, bounded()),
        Err(Error::InvalidCost { state: 0, .. })
    ));
}

/// Invalid settings are rejected before any work.
#[test]
fn config_is_validated() {
    let table = ssp(3, 1, Shape::Proper);
    let goal = |s: &usize| table.is_goal(*s);
    let configs = [
        bounded().with_epsilon(0.0),
        bounded().with_epsilon(f64::NAN),
        bounded().with_dead_ends(DeadEnds::Penalty(f64::INFINITY)),
        bounded().with_max_trial_length(0),
    ];

    for config in configs {
        assert!(matches!(
            lrtdp(&table, 0, &goal, Zero, config),
            Err(Error::InvalidConfig { .. })
        ));
    }
}

/// A backup budget stops LRTDP after exactly that many backups.
#[test]
fn budget_stops_the_run() {
    let table = ssp(8, 3, Shape::Proper);
    let config = Config::default().with_budget(Limits::backups(3));
    let solution =
        lrtdp(&table, 0, &|s: &usize| table.is_goal(*s), Zero, config).expect("valid input");

    assert_eq!(solution.status(), Status::BudgetExhausted);
    assert_eq!(solution.stats().backups, 3);
}

/// A closure budget works too, with no type annotations beyond the closure argument.
#[test]
fn closure_budget() {
    let table = ssp(8, 3, Shape::Proper);
    let config =
        Config::default().with_budget(|stats: &stochastic_search::Stats| stats.trials >= 1);
    let solution =
        value_iteration(&table, 0, &|s: &usize| table.is_goal(*s), config).expect("valid input");

    assert!(solution.is_converged(), "value iteration runs no trials");
}

/// An MDP with infinitely many states: a counter that only ever increases.
struct Unbounded;

impl stochastic_search::Mdp for Unbounded {
    type State = u64;
    type Action = ();

    fn actions(&self, _state: &u64, mut emit: impl FnMut((), f64)) {
        emit((), 1.0);
    }

    fn outcomes(&self, state: &u64, _action: &(), mut emit: impl FnMut(u64, f64)) {
        emit(state + 1, 0.5);
        emit(state + 2, 0.5);
    }
}

/// Value iteration's state enumeration is budgeted: an infinite state space stops at the
/// state limit instead of exhausting memory.
#[test]
fn value_iteration_exploration_is_budgeted() {
    let config = Config::default().with_budget(Limits {
        max_states: Some(1000),
        ..Limits::unlimited()
    });
    let solution = value_iteration(&Unbounded, 0, &|_: &u64| false, config).expect("valid input");

    assert_eq!(solution.status(), Status::BudgetExhausted);
    assert!(solution.stats().states <= 1001);
}

/// LRTDP on an infinite state space without a goal stops at the default limits (with the
/// state limit lowered from its default to keep the test small).
#[test]
fn default_limits_stop_unbounded_runs() {
    let config = Config::default().with_epsilon(1e-4).with_budget(Limits {
        max_states: Some(10_000),
        ..Limits::DEFAULT
    });
    let solution = lrtdp(&Unbounded, 0, &|_: &u64| false, Zero, config).expect("valid input");

    assert_eq!(solution.status(), Status::BudgetExhausted);
}

/// A heuristic that counts its calls.
struct Counting {
    /// The number of estimates so far.
    calls: u64,
}

impl Heuristic<usize, f64> for Counting {
    fn estimate(&mut self, _state: &usize) -> Option<f64> {
        self.calls += 1;
        Some(0.0)
    }
}

/// A stateful heuristic passed as `&mut h` stays usable after the run.
#[test]
fn heuristic_by_mutable_reference() {
    let table = ssp(6, 9, Shape::Proper);
    let mut heuristic = Counting { calls: 0 };
    let goal = |s: &usize| table.is_goal(*s);
    let solution = lrtdp(&table, 0, &goal, &mut heuristic, bounded()).expect("valid input");

    assert!(solution.is_converged());
    // Every stored state but the goal is estimated exactly once.
    let goals = u64::from(solution.value(&table.goal).is_some());
    assert_eq!(heuristic.calls + goals, solution.stats().states);
}

/// The default budget is finite, so a default run always returns.
#[test]
fn default_budget_is_finite() {
    let limits = Config::default().budget;

    assert!(limits.max_backups.is_some());
    assert!(limits.max_states.is_some());
    assert_eq!(Limits::default(), Limits::DEFAULT);
}
