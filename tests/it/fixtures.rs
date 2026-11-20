//! Hand-made SSPs with closed-form answers, solved by every SSP algorithm.

use stochastic_search::{
    Config, DeadEnds, Limits, Solution, Status, Zero, cg_ilao_star, ilao_star, lao_star, lrtdp,
    rtdp, value_iteration,
};

use crate::table::{Table, action};

/// Tolerance for closed-form comparisons; LRTDP runs with a tighter epsilon.
const TOLERANCE: f64 = 1e-6;

/// Solves `table` from state 0 with value iteration, LRTDP, RTDP, iLAO*, LAO* and CG-iLAO*
/// under `dead_ends`.
fn solve_both(table: &Table, dead_ends: DeadEnds) -> [Solution<usize, usize>; 6] {
    let config = Config::default()
        .with_epsilon(1e-10)
        .with_dead_ends(dead_ends)
        .with_max_trial_length(100);
    let goal = |s: &usize| table.is_goal(*s);

    [
        value_iteration(table, 0, &goal, config).expect("valid input"),
        lrtdp(table, 0, &goal, Zero, config).expect("valid input"),
        rtdp(table, 0, &goal, Zero, config).expect("valid input"),
        ilao_star(table, 0, &goal, Zero, config).expect("valid input"),
        lao_star(table, 0, &goal, Zero, config).expect("valid input"),
        cg_ilao_star(table, 0, &goal, Zero, config).expect("valid input"),
    ]
}

/// Retrying a coin flip of success probability `p` at cost `c` costs `c / p`.
#[test]
fn retry_loop_costs_cost_over_probability() {
    let table = Table {
        actions: vec![vec![action(2.0, &[(1, 0.25), (0, 0.75)])], vec![]],
        goal: 1,
    };

    for solution in solve_both(&table, DeadEnds::Infinite) {
        assert!(solution.is_converged());
        assert!((solution.initial_value() - 8.0).abs() < TOLERANCE);
        assert_eq!(solution.initial_action(), Some(&0));
    }
}

/// A risky action that may hit a dead end loses under infinite cost and wins under a
/// small penalty.
#[test]
fn risky_action_depends_on_dead_end_semantics() {
    let table = Table {
        actions: vec![
            vec![
                action(10.0, &[(1, 1.0)]),
                action(1.0, &[(1, 0.5), (2, 0.5)]),
            ],
            vec![],
            vec![],
        ],
        goal: 1,
    };

    for solution in solve_both(&table, DeadEnds::Infinite) {
        assert!((solution.initial_value() - 10.0).abs() < TOLERANCE);
        assert_eq!(solution.initial_action(), Some(&0));
        assert_eq!(solution.value(&2), Some(f64::INFINITY));
    }

    for solution in solve_both(&table, DeadEnds::Penalty(4.0)) {
        assert!((solution.initial_value() - 3.0).abs() < TOLERANCE);
        assert_eq!(solution.initial_action(), Some(&1));
        assert_eq!(solution.action(&2), None);
    }
}

/// Without a proper policy, the value is infinite and there is no action; with a penalty,
/// the agent gives up at once.
#[test]
fn unavoidable_dead_end() {
    let table = Table {
        actions: vec![vec![action(1.0, &[(1, 0.5), (2, 0.5)])], vec![], vec![]],
        goal: 1,
    };

    for solution in solve_both(&table, DeadEnds::Infinite) {
        assert_eq!(solution.status(), Status::Unsolvable);
        assert_eq!(solution.initial_value(), f64::INFINITY);
        assert_eq!(solution.initial_action(), None);
    }

    // Trying costs 1 + 0.5 * 1.5 = 1.75 against 1.5 for giving up.
    for solution in solve_both(&table, DeadEnds::Penalty(1.5)) {
        assert!(solution.is_converged());
        assert!((solution.initial_value() - 1.5).abs() < TOLERANCE);
        assert_eq!(solution.initial_action(), None);
    }
}

/// An unavoidable trap cycle under infinite dead-end cost: every algorithm proves it by graph
/// analysis with the default configuration (RTDP and LRTDP after their trials get cut off,
/// iLAO* once its backups double).
#[test]
fn unavoidable_trap_cycle_is_unsolvable() {
    let table = Table {
        actions: vec![
            vec![action(1.0, &[(1, 0.5), (2, 0.5)])],
            vec![],
            vec![action(1.0, &[(2, 1.0)])],
        ],
        goal: 1,
    };
    let goal = |s: &usize| table.is_goal(*s);

    let exact = value_iteration(&table, 0, &goal, Config::default()).expect("valid input");
    assert_eq!(exact.status(), Status::Unsolvable);
    assert_eq!(exact.initial_value(), f64::INFINITY);

    for proven in [
        lrtdp(&table, 0, &goal, Zero, Config::default()),
        rtdp(&table, 0, &goal, Zero, Config::default()),
        ilao_star(&table, 0, &goal, Zero, Config::default()),
        lao_star(&table, 0, &goal, Zero, Config::default()),
        cg_ilao_star(&table, 0, &goal, Zero, Config::default()),
    ] {
        let proven = proven.expect("valid input");
        assert_eq!(proven.status(), Status::Unsolvable);
        assert_eq!(proven.initial_value(), f64::INFINITY);
        assert_eq!(proven.initial_action(), None);
    }
}

/// A short trial length limit proves the trap long before a generous budget runs out.
#[test]
fn trap_proof_does_not_need_the_budget() {
    let table = Table {
        actions: vec![
            vec![action(1.0, &[(1, 0.5), (2, 0.5)])],
            vec![],
            vec![action(1.0, &[(2, 1.0)])],
        ],
        goal: 1,
    };
    let config = Config::default()
        .with_max_trial_length(3)
        .with_budget(Limits::backups(1_000_000));
    let solution =
        lrtdp(&table, 0, &|s: &usize| table.is_goal(*s), Zero, config).expect("valid input");

    assert_eq!(solution.status(), Status::Unsolvable);
}

/// An avoidable trap cycle (the goal is unreachable from it) does not hang LRTDP under
/// infinite dead-end cost.
#[test]
fn avoidable_trap_cycle() {
    let table = Table {
        actions: vec![
            vec![action(1.0, &[(1, 0.5), (2, 0.5)]), action(5.0, &[(1, 1.0)])],
            vec![],
            vec![action(1.0, &[(2, 1.0)])],
        ],
        goal: 1,
    };

    for solution in solve_both(&table, DeadEnds::Infinite) {
        assert!(solution.is_converged());
        assert!((solution.initial_value() - 5.0).abs() < TOLERANCE);
        assert_eq!(solution.initial_action(), Some(&1));
    }
}

/// An initial goal state costs nothing and needs no work.
#[test]
fn initial_goal() {
    let table = Table {
        actions: vec![vec![]],
        goal: 0,
    };

    for solution in solve_both(&table, DeadEnds::Infinite) {
        assert!(solution.is_converged());
        assert_eq!(solution.initial_value(), 0.0);
        assert_eq!(solution.initial_action(), None);
        assert_eq!(solution.stats().backups, 0);
    }
}
