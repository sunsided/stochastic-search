//! Property tests on random SSPs: LRTDP against value iteration and the raw-table oracle.

use proptest::prelude::*;
use stochastic_search::{Config, DeadEnds, Solution, lrtdp, value_iteration};

use crate::oracle::{determinization, optimal_values, policy_envelope, policy_is_proper, q_value};
use crate::random::{Shape, ssp};
use crate::table::Table;

/// The LRTDP convergence threshold used by the property tests.
const EPSILON: f64 = 1e-4;

/// The value iteration threshold, small enough to treat its result as `V*`.
const ORACLE_EPSILON: f64 = 1e-11;

/// Slack for floating-point rounding in comparisons.
const ROUNDING: f64 = 1e-9;

/// Solves `table` from state 0 with LRTDP and the given heuristic values.
fn solve_lrtdp(table: &Table, h: &[f64], dead_ends: DeadEnds, seed: u64) -> Solution<usize, usize> {
    let config = Config {
        epsilon: EPSILON,
        dead_ends,
        seed,
        max_trial_length: 200,
        ..Config::default()
    };
    let heuristic = |state: &usize| Some(h[*state]);

    lrtdp(table, 0, &|s: &usize| table.is_goal(*s), &heuristic, config).expect("valid input")
}

/// Solves `table` from state 0 with value iteration to near machine precision.
fn solve_vi(table: &Table, dead_ends: DeadEnds) -> Solution<usize, usize> {
    let config = Config {
        epsilon: ORACLE_EPSILON,
        dead_ends,
        ..Config::default()
    };

    value_iteration(table, 0, &|s: &usize| table.is_goal(*s), config).expect("valid input")
}

/// Asserts the LRTDP bound from the design: `V <= V*` and
/// `V* - V <= epsilon * V / (c_min - epsilon)`.
fn assert_within_bound(lrtdp_value: f64, optimal: f64, c_min: f64) -> Result<(), TestCaseError> {
    let gap_bound = EPSILON * lrtdp_value / (c_min - EPSILON);
    prop_assert!(
        lrtdp_value <= optimal + ROUNDING,
        "LRTDP {lrtdp_value} above V* {optimal}"
    );
    prop_assert!(
        optimal - lrtdp_value <= gap_bound + ROUNDING,
        "gap {} exceeds bound {gap_bound}",
        optimal - lrtdp_value
    );

    Ok(())
}

/// Asserts epsilon-consistency on the policy envelope: the policy residual always, and the
/// full Bellman residual when `full` (consistent heuristics).
fn assert_consistent(
    table: &Table,
    solution: &Solution<usize, usize>,
    full: bool,
) -> Result<(), TestCaseError> {
    for state in policy_envelope(table, solution, 0) {
        let Some(&action) = solution.action(&state) else {
            continue;
        };

        let value = solution.value(&state).expect("envelope states have values");
        let policy_residual = (value - q_value(table, solution, state, action)).abs();
        prop_assert!(
            policy_residual <= EPSILON + ROUNDING,
            "policy residual {policy_residual} at {state}"
        );

        if full {
            let best = (0..table.actions[state].len())
                .map(|a| q_value(table, solution, state, a))
                .fold(f64::INFINITY, f64::min);
            prop_assert!(
                (value - best).abs() <= EPSILON + ROUNDING,
                "residual at {state}"
            );
        }
    }

    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Value iteration agrees with the raw-table oracle on SSPs with proper policies everywhere.
    #[test]
    fn value_iteration_matches_oracle(states in 2_usize..10, seed in any::<u64>()) {
        let table = ssp(states, seed, Shape::Proper);
        let expected = optimal_values(&table);
        let solution = solve_vi(&table, DeadEnds::Infinite);

        prop_assert!(solution.is_converged());
        prop_assert!((solution.initial_value() - expected[0]).abs() < 1e-8);
    }

    /// LRTDP with the blind and the determinization heuristic stays within the design bound
    /// of `V*`, is epsilon-consistent on its envelope, and returns a proper policy.
    #[test]
    fn lrtdp_matches_value_iteration(states in 2_usize..10, seed in any::<u64>()) {
        let table = ssp(states, seed, Shape::Proper);
        let optimal = solve_vi(&table, DeadEnds::Infinite).initial_value();

        for h in [vec![0.0; table.len()], determinization(&table)] {
            let solution = solve_lrtdp(&table, &h, DeadEnds::Infinite, seed);
            prop_assert!(solution.is_converged());
            assert_within_bound(solution.initial_value(), optimal, table.min_cost())?;
            assert_consistent(&table, &solution, true)?;
            prop_assert!(policy_is_proper(&table, &solution, 0));
        }
    }

    /// With avoidable traps under infinite dead-end cost, LRTDP still converges to the
    /// value iteration answer (which finds the traps by graph analysis) and avoids the traps.
    #[test]
    fn lrtdp_avoids_traps(states in 2_usize..10, seed in any::<u64>()) {
        let table = ssp(states, seed, Shape::Traps);
        let optimal = solve_vi(&table, DeadEnds::Infinite).initial_value();
        prop_assert!(optimal.is_finite());

        for h in [vec![0.0; table.len()], determinization(&table)] {
            let solution = solve_lrtdp(&table, &h, DeadEnds::Infinite, seed);
            prop_assert!(solution.is_converged());
            assert_within_bound(solution.initial_value(), optimal, table.min_cost())?;
            assert_consistent(&table, &solution, false)?;
            prop_assert!(policy_is_proper(&table, &solution, 0));
        }
    }

    /// Under a finite penalty, LRTDP agrees with value iteration; give-up states are allowed
    /// in the envelope and count as terminal.
    #[test]
    fn lrtdp_matches_value_iteration_with_penalty(states in 2_usize..10, seed in any::<u64>(), penalty in 1_u32..60) {
        let table = ssp(states, seed, Shape::Traps);
        let dead_ends = DeadEnds::Penalty(f64::from(penalty));
        let optimal = solve_vi(&table, dead_ends).initial_value();
        let solution = solve_lrtdp(&table, &vec![0.0; table.len()], dead_ends, seed);

        prop_assert!(solution.is_converged());
        prop_assert!(solution.initial_value() <= f64::from(penalty));
        assert_within_bound(solution.initial_value(), optimal, table.min_cost().min(f64::from(penalty)))?;
        assert_consistent(&table, &solution, true)?;
    }

    /// The same seed gives bit-identical values, policies and statistics.
    #[test]
    fn lrtdp_is_deterministic(states in 2_usize..10, seed in any::<u64>()) {
        let table = ssp(states, seed, Shape::Traps);
        let h = vec![0.0; table.len()];
        let first = solve_lrtdp(&table, &h, DeadEnds::Infinite, seed);
        let second = solve_lrtdp(&table, &h, DeadEnds::Infinite, seed);

        prop_assert_eq!(first.stats(), second.stats());
        for state in 0..table.len() {
            prop_assert_eq!(first.value(&state).map(f64::to_bits), second.value(&state).map(f64::to_bits));
            prop_assert_eq!(first.action(&state), second.action(&state));
        }
    }
}
