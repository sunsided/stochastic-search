//! BRTDP: both bounds bracket the value iteration answer on random SSPs, converged or cut
//! off by a budget, under both dead-end semantics.

use proptest::prelude::*;
use stochastic_search::{
    Config, DeadEnds, Error, Limits, Solution, Zero, brtdp, heuristic_fn, value_iteration,
};

use crate::oracle::{determinization, first_action_values};
use crate::random::{Shape, ssp};
use crate::table::{Table, action};

/// The BRTDP gap threshold used by these tests.
const EPSILON: f64 = 1e-4;

/// Slack for floating-point rounding and for value iteration stopping just below `V*`.
const ROUNDING: f64 = 1e-6;

/// Solves `table` from state 0 with BRTDP and the given bound values (infinity is `None`).
fn solve(
    table: &Table,
    lower: &[f64],
    upper: &[f64],
    dead_ends: DeadEnds,
    seed: u64,
    budget: Limits,
) -> Result<Solution<usize, usize>, Error<usize>> {
    let config = Config {
        epsilon: EPSILON,
        dead_ends,
        seed,
        max_trial_length: 200,
        ..Config::default()
    }
    .with_budget(budget);
    let bound = |values: &[f64], state: usize| Some(values[state]).filter(|v| v.is_finite());
    let lower = heuristic_fn(|state: &usize| bound(lower, *state));
    let upper = heuristic_fn(|state: &usize| bound(upper, *state));

    brtdp(
        table,
        0,
        &|s: &usize| table.is_goal(*s),
        lower,
        upper,
        config,
    )
}

/// Optimal values of every state by value iteration from that state, to near machine
/// precision; infinity where no proper policy exists.
fn optimal(table: &Table, dead_ends: DeadEnds) -> Vec<f64> {
    let config = Config {
        epsilon: 1e-11,
        dead_ends,
        ..Config::default()
    };

    (0..table.len())
        .map(|state| {
            value_iteration(table, state, &|s: &usize| table.is_goal(*s), config)
                .expect("valid input")
                .initial_value()
        })
        .collect()
}

/// Asserts `lower <= V* <= upper` at every state the solution stored.
fn assert_brackets(
    solution: &Solution<usize, usize>,
    optimal: &[f64],
) -> Result<(), TestCaseError> {
    for (state, &v) in optimal.iter().enumerate() {
        let Some(lower) = solution.value(&state) else {
            continue;
        };

        let upper = solution
            .upper_bound(&state)
            .expect("stored states have both bounds");
        prop_assert!(
            lower <= v + ROUNDING,
            "lower {lower} above V* {v} at {state}"
        );
        prop_assert!(
            v <= upper + ROUNDING,
            "upper {upper} below V* {v} at {state}"
        );
    }

    Ok(())
}

/// Asserts that the run converged with a gap of at most epsilon at the initial state.
fn assert_converged(solution: &Solution<usize, usize>) -> Result<(), TestCaseError> {
    let upper = solution
        .initial_upper_bound()
        .expect("BRTDP keeps upper bounds");
    prop_assert!(solution.is_converged(), "{:?}", solution.status());
    prop_assert!(upper - solution.initial_value() <= EPSILON);

    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Under infinite dead-end cost, with the blind or the determinization lower bound and
    /// the always-action-0 policy value as upper bound, BRTDP converges and brackets `V*`
    /// everywhere, also with avoidable traps (which the determinization proves).
    #[test]
    fn brtdp_brackets_value_iteration(states in 2_usize..10, seed in any::<u64>()) {
        for (shape, blind) in [(Shape::Proper, true), (Shape::Proper, false), (Shape::Traps, false)] {
            let table = ssp(states, seed, shape);
            let lower = if blind { vec![0.0; table.len()] } else { determinization(&table) };
            let optimal = optimal(&table, DeadEnds::Infinite);
            let budget = Limits { max_states: Some(1_000), ..Limits::backups(1_000_000) };
            let solution = solve(&table, &lower, &first_action_values(&table), DeadEnds::Infinite, seed, budget)
                .expect("valid input");

            assert_converged(&solution)?;
            assert_brackets(&solution, &optimal)?;
        }
    }

    /// Under a finite penalty, the blind lower bound and no upper estimate (the penalty)
    /// suffice; BRTDP converges and brackets `V*`.
    #[test]
    fn brtdp_brackets_value_iteration_with_penalty(states in 2_usize..10, seed in any::<u64>(), penalty in 1_u32..60) {
        let table = ssp(states, seed, Shape::Traps);
        let dead_ends = DeadEnds::Penalty(f64::from(penalty));
        let none = vec![f64::INFINITY; table.len()];
        let solution = solve(&table, &vec![0.0; table.len()], &none, dead_ends, seed, Limits::backups(1_000_000))
            .expect("valid input");

        assert_converged(&solution)?;
        assert_brackets(&solution, &optimal(&table, dead_ends))?;
        prop_assert!(solution.initial_upper_bound() <= Some(f64::from(penalty)));
    }

    /// The bounds bracket `V*` at any time, also when a small budget stops the run.
    #[test]
    fn brtdp_brackets_when_cut_off(states in 2_usize..10, seed in any::<u64>(), backups in 1_u64..200) {
        let table = ssp(states, seed, Shape::Traps);
        let lower = determinization(&table);
        let solution = solve(&table, &lower, &first_action_values(&table), DeadEnds::Infinite, seed, Limits::backups(backups))
            .expect("valid input");

        assert_brackets(&solution, &optimal(&table, DeadEnds::Infinite))?;
    }

    /// The same seed gives bit-identical bounds, policies and statistics.
    #[test]
    fn brtdp_is_deterministic(states in 2_usize..10, seed in any::<u64>()) {
        let table = ssp(states, seed, Shape::Traps);
        let (lower, upper) = (vec![0.0; table.len()], vec![f64::INFINITY; table.len()]);
        let run = || solve(&table, &lower, &upper, DeadEnds::Penalty(30.0), seed, Limits::DEFAULT)
            .expect("valid input");
        let (first, second) = (run(), run());

        prop_assert_eq!(first.stats(), second.stats());
        for state in 0..table.len() {
            prop_assert_eq!(first.value(&state).map(f64::to_bits), second.value(&state).map(f64::to_bits));
            prop_assert_eq!(first.upper_bound(&state).map(f64::to_bits), second.upper_bound(&state).map(f64::to_bits));
            prop_assert_eq!(first.action(&state), second.action(&state));
        }
    }
}

/// A retry loop next to a sure route and a trap: state 0 can pay 5 to finish, or flip a coin
/// for 1 that finishes or lands in the trap 2 (no actions). State 1 is the goal.
fn retry_with_trap() -> Table {
    Table {
        actions: vec![
            vec![action(5.0, &[(1, 1.0)]), action(1.0, &[(1, 0.5), (2, 0.5)])],
            Vec::new(),
            Vec::new(),
        ],
        goal: 1,
    }
}

/// Under infinite dead-end cost, a dead end the lower bound misses has no finite upper
/// bound, and BRTDP reports it instead of running with an infinite gap.
#[test]
fn brtdp_rejects_unbounded_upper_bounds() {
    let table = retry_with_trap();
    let result = brtdp(
        &table,
        0,
        &|s: &usize| *s == 1,
        Zero,
        heuristic_fn(|s: &usize| (*s == 0).then_some(5.0)),
        Config::default(),
    );

    assert!(matches!(result, Err(Error::UnboundedUpper { state: 2 })));
}

/// With the trap proven by the lower bound, the same problem solves: paying 5 is optimal.
#[test]
fn brtdp_skips_proven_dead_ends() {
    let table = retry_with_trap();
    let solution = brtdp(
        &table,
        0,
        &|s: &usize| *s == 1,
        heuristic_fn(|s: &usize| (*s != 2).then_some(0.0)),
        heuristic_fn(|s: &usize| (*s == 0).then_some(5.0)),
        Config::default(),
    )
    .expect("valid input");

    assert!(solution.is_converged());
    assert_eq!(solution.initial_upper_bound(), Some(5.0));
    assert!((solution.initial_value() - 5.0).abs() <= 1e-6);
    assert_eq!(solution.initial_action(), Some(&0));
    assert_eq!(solution.upper_bound(&2), Some(f64::INFINITY));
}

/// Invalid bounds and settings are reported.
#[test]
fn brtdp_validates_its_input() {
    let table = retry_with_trap();
    let goal = |s: &usize| *s == 1;
    let lower = || heuristic_fn(|s: &usize| (*s != 2).then_some(1.0));
    let upper = |u: f64| heuristic_fn(move |_: &usize| Some(u));

    let crossed = brtdp(&table, 0, &goal, lower(), upper(0.5), Config::default());
    assert!(matches!(crossed, Err(Error::InvalidHeuristic { state: 0 })));

    let nan = brtdp(
        &table,
        0,
        &goal,
        lower(),
        upper(f64::NAN),
        Config::default(),
    );
    assert!(matches!(nan, Err(Error::InvalidHeuristic { state: 0 })));

    for tau in [1.0, 0.5, f64::INFINITY, f64::NAN] {
        let config = Config {
            tau,
            ..Config::default()
        };
        let result = brtdp(&table, 0, &goal, lower(), upper(5.0), config);
        assert!(
            matches!(result, Err(Error::InvalidConfig { .. })),
            "tau {tau}"
        );
    }
}

/// A goal as the initial state converges at once with both bounds at 0.
#[test]
fn brtdp_at_a_goal() {
    let table = retry_with_trap();
    let solution = brtdp(
        &table,
        1,
        &|s: &usize| *s == 1,
        Zero,
        Zero,
        Config::default(),
    )
    .expect("valid input");

    assert!(solution.is_converged());
    assert_eq!(solution.initial_upper_bound(), Some(0.0));
    assert_eq!(solution.stats().trials, 0);
}
