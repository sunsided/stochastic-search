//! Bad heuristics, invalid input and budgets: errors or degraded answers, never panics or
//! hangs.

use proptest::prelude::*;
use stochastic_search::{Config, DeadEnds, Error, Limits, Status, Zero, lrtdp, value_iteration};

use crate::random::{Shape, ssp};
use crate::table::{Table, action};

/// A configuration whose budget bounds every run.
fn bounded() -> Config {
    Config {
        epsilon: 1e-4,
        max_trial_length: 200,
        budget: Limits::backups(200_000),
        ..Config::default()
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Inadmissible and otherwise wrong (but not NaN) heuristics never panic: each run
    /// returns a solution.
    #[test]
    fn bad_heuristics_do_not_crash(states in 2_usize..10, seed in any::<u64>(), kind in 0_u8..5) {
        let table = ssp(states, seed, Shape::Traps);
        let heuristic = |state: &usize| -> Option<f64> {
            let spread = f64::from(u32::try_from((*state * 7919 + 13) % 101).expect("small"));
            match kind {
                0 => Some(1000.0 + spread),           // wild overestimate
                1 => Some(-100.0 - spread),           // negative
                2 => (spread < 50.0).then_some(spread), // false dead ends
                3 => Some(if spread < 30.0 { f64::INFINITY } else { spread }),
                _ => Some(if spread < 50.0 { 0.0 } else { 1e9 }),
            }
        };

        for dead_ends in [DeadEnds::Infinite, DeadEnds::Penalty(20.0)] {
            let config = Config { dead_ends, ..bounded() };
            let solution = lrtdp(&table, 0, &|s: &usize| table.is_goal(*s), &heuristic, config);
            prop_assert!(solution.is_ok());
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
        &|_: &usize| Some(f64::NAN),
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
        lrtdp(&table, 0, &goal, &Zero, bounded()),
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
        lrtdp(&bad_probability, 0, &|s: &usize| *s == 1, &Zero, bounded()),
        Err(Error::InvalidProbability { state: 0, .. })
    ));
    assert!(matches!(
        lrtdp(&bad_cost, 0, &|s: &usize| *s == 1, &Zero, bounded()),
        Err(Error::InvalidCost { state: 0, .. })
    ));
}

/// Invalid settings are rejected before any work.
#[test]
fn config_is_validated() {
    let table = ssp(3, 1, Shape::Proper);
    let goal = |s: &usize| table.is_goal(*s);
    let configs = [
        Config {
            epsilon: 0.0,
            ..bounded()
        },
        Config {
            epsilon: f64::NAN,
            ..bounded()
        },
        Config {
            dead_ends: DeadEnds::Penalty(f64::INFINITY),
            ..bounded()
        },
        Config {
            max_trial_length: 0,
            ..bounded()
        },
    ];

    for config in configs {
        assert!(matches!(
            lrtdp(&table, 0, &goal, &Zero, config),
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
        lrtdp(&table, 0, &|s: &usize| table.is_goal(*s), &Zero, config).expect("valid input");

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
