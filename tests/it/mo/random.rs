//! Random small MOSSPs against the brute-force oracle and value iteration, and the
//! one-objective reduction to the scalar algorithms.

use proptest::prelude::*;
use stochastic_search::{
    Abstraction, Config, Determinized, IdealPoint, Limits, Objectives, Zero, ZeroVector,
    heuristic_fn, lao_star, lrtdp, value_iteration,
};

use super::oracle::{
    MoAction, MoTable, Scalar, assert_policies_evaluate, hull, proper_values, support,
};
use super::{Algorithm, front};
use crate::random::{Shape, ssp};

/// The bound `b` of every objective: twice the largest proper policy cost the tests
/// accept. A small bound matters: until a trap's value climbs to `b`, its truncated
/// values add vertices to every value set that can reach it.
pub const BOUND: f64 = 40.0;

/// The convergence threshold of the two-objective tests.
const EPSILON: f64 = 1e-6;

/// Slack when comparing values with the oracle, two objectives.
const TOLERANCE: f64 = 1e-4;

/// The convergence threshold of the three-objective tests: transient value sets grow
/// quickly as the margin (`10 * epsilon`) shrinks.
const EPSILON_3D: f64 = 1e-4;

/// Slack when comparing values with the oracle, three objectives.
const TOLERANCE_3D: f64 = 1e-2;

/// A random MOSSP over `states` regular states plus two traps (see
/// [`ssp`](crate::random::ssp)), with cost components in `0..=3` and at least one positive
/// component per action, so that every improper policy has infinite cost in some
/// objective.
pub fn mossp<const N: usize>(states: usize, seed: u64) -> MoTable<N> {
    let shape = ssp(states, seed, Shape::Traps);
    let mut rng = seed | 1;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;

        f64::from(u8::try_from(rng % 4).expect("below 4"))
    };
    let actions = shape
        .actions
        .iter()
        .map(|list| {
            list.iter()
                .map(|action| {
                    let mut cost: [f64; N] = std::array::from_fn(|_| next());
                    if cost.iter().all(|&c| c == 0.0) {
                        cost[0] = 1.0;
                    }

                    MoAction {
                        cost,
                        outcomes: action.outcomes.clone(),
                    }
                })
                .collect()
        })
        .collect();

    MoTable {
        actions,
        goal: shape.goal,
    }
}

/// Returns `true` if the paper's Assumption 1 holds for [`BOUND`] with room to spare:
/// every proper policy from every state costs less than half of it in every objective.
fn bound_holds<const N: usize>(table: &MoTable<N>) -> bool {
    (0..table.len()).all(|state| {
        proper_values(table, state)
            .iter()
            .all(|v| v.iter().all(|&c| c < BOUND / 2.0))
    })
}

/// The weights `(t, 1 - t)` for `t` in steps of 1/40.
fn weights_2d() -> Vec<[f64; 2]> {
    (0..=40)
        .map(|i| f64::from(i) / 40.0)
        .map(|t| [t, 1.0 - t])
        .collect()
}

/// The weights on a grid of the 3-simplex with step 1/8.
fn weights_3d() -> Vec<[f64; 3]> {
    let mut weights = Vec::new();
    for i in 0..=8 {
        for j in 0..=8 - i {
            let (a, b) = (f64::from(i) / 8.0, f64::from(j) / 8.0);
            weights.push([a, b, 1.0 - a - b]);
        }
    }

    weights
}

/// Asserts that two vector sets have the same support function on `weights`.
fn assert_same_support<const N: usize>(
    actual: &[[f64; N]],
    expected: &[[f64; N]],
    weights: &[[f64; N]],
    tolerance: f64,
) {
    for weight in weights {
        let (a, e) = (support(actual, weight), support(expected, weight));
        assert!(
            (a - e).abs() <= tolerance * (1.0 + e.abs()),
            "w {weight:?}: {a} vs {e}; {actual:?} vs {expected:?}"
        );
    }
}

/// The ideal-point heuristic of the exact per-objective optimal values.
fn ideal_point<const N: usize>(
    table: &MoTable<N>,
) -> impl stochastic_search::Heuristic<usize, Vec<[f64; N]>> {
    IdealPoint(std::array::from_fn(|objective| {
        let projection = table.projection(objective);
        let goal = projection.goal;
        let values: Vec<Option<f64>> = (0..projection.len())
            .map(|state| {
                let solution = value_iteration(
                    &projection,
                    state,
                    &|s: &usize| *s == goal,
                    Config::default().with_epsilon(1e-10),
                )
                .expect("valid table");
                let value = solution.initial_value();
                // Value iteration converges from below, so it is a lower bound.
                value.is_finite().then_some(value)
            })
            .collect();

        heuristic_fn(move |state: &usize| values[*state])
    }))
}

/// Solves `table` with every algorithm and heuristic, checks each policy by exact
/// evaluation and returns the fronts.
fn solve_all<const N: usize>(
    table: &MoTable<N>,
    epsilon: f64,
    tolerance: f64,
    max_backups: Option<u64>,
) -> Option<Vec<Vec<[f64; N]>>> {
    let goal = |state: &usize| *state == table.goal;
    let objectives = Objectives::new([BOUND; N]);
    let budget = Limits {
        max_backups,
        ..Limits::DEFAULT
    };
    let config = Config::default().with_epsilon(epsilon).with_budget(budget);
    // Simple determinised paths cost at most 6 steps of 3 per objective, below the bound.
    let determinized = Determinized(table);
    let mut fronts = Vec::new();
    for algorithm in Algorithm::ALL {
        let solutions = [
            algorithm.solve(table, 0, &goal, ZeroVector, objectives, config),
            algorithm.solve(table, 0, &goal, ideal_point(table), objectives, config),
            algorithm.solve(
                table,
                0,
                &goal,
                Abstraction::new(
                    &determinized,
                    |s: &usize| *s,
                    &goal,
                    Objectives::new([BOUND; N]),
                )
                .with_config(config),
                objectives,
                config,
            ),
        ];
        for solution in solutions {
            let solution = solution.expect("valid input");
            if solution.status() == stochastic_search::Status::BudgetExhausted
                && max_backups.is_some()
            {
                return None;
            }

            assert!(
                solution.is_converged()
                    || solution.status() == stochastic_search::Status::Unsolvable,
                "{algorithm:?}: {solution:?}"
            );
            assert_policies_evaluate(table, &solution, 0, tolerance);
            fronts.push(front(&solution));
        }
    }

    Some(fronts)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// Two objectives: value iteration returns exactly the hull vertices of all proper
    /// deterministic policies, and every search agrees with it on every weight.
    #[test]
    fn bi_objective_matches_the_oracle(states in 2_usize..7, seed in any::<u64>()) {
        let table = mossp::<2>(states, seed);
        prop_assume!(bound_holds(&table));
        let values = proper_values(&table, 0);
        let vertices = hull(&values, 1e-6);

        let fronts = solve_all(&table, EPSILON, TOLERANCE, None).expect("no budget");
        let movi = &fronts[0];
        prop_assert_eq!(movi.len(), vertices.len(), "{:?} vs {:?}", movi, vertices);
        for front in &fronts {
            if vertices.is_empty() {
                prop_assert!(front.is_empty());
                continue;
            }

            assert_same_support(front, &vertices, &weights_2d(), TOLERANCE);
            prop_assert!(front.len() <= vertices.len());
        }
    }

}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Three objectives: every search agrees with the brute-force oracle on a weight grid.
    #[test]
    fn three_objectives_match_the_oracle(states in 2_usize..4, seed in any::<u64>()) {
        let table = mossp::<3>(states, seed);
        prop_assume!(bound_holds(&table));
        let values = proper_values(&table, 0);

        // A few instances have transient sets in the thousands; they are skipped.
        let fronts = solve_all(&table, EPSILON_3D, TOLERANCE_3D, Some(5_000));
        prop_assume!(fronts.is_some());
        for front in fronts.expect("assumed") {
            if values.is_empty() {
                prop_assert!(front.is_empty());
            } else {
                assert_same_support(&front, &values, &weights_3d(), TOLERANCE_3D);
            }
        }
    }

    /// One objective: every multi-objective algorithm finds the single optimal value of
    /// LAO* and LRTDP.
    #[test]
    fn one_objective_reduces_to_the_scalar_algorithms(states in 2_usize..10, seed in any::<u64>()) {
        let table = ssp(states, seed, Shape::Traps);
        let goal = |state: &usize| *state == table.goal;
        let config = Config::default().with_epsilon(1e-9);
        let lao = lao_star(&table, 0, &goal, Zero, config).expect("valid input");
        let rtdp = lrtdp(&table, 0, &goal, Zero, config).expect("valid input");
        prop_assert!((lao.initial_value() - rtdp.initial_value()).abs() < 1e-5);

        for algorithm in Algorithm::ALL {
            let solution = algorithm
                .solve(&Scalar(&table), 0, &goal, ZeroVector, Objectives::new([1e4]), config)
                .expect("valid input");
            prop_assert!(solution.is_converged());
            let front = front(&solution);
            prop_assert_eq!(front.len(), 1);
            prop_assert!((front[0][0] - lao.initial_value()).abs() < 1e-5, "{:?}: {:?} vs {}", algorithm, front, lao.initial_value());
        }
    }
}
