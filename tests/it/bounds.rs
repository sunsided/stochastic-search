//! BRTDP, FRTDP and VPI-RTDP: both bounds bracket the value iteration answer on random SSPs,
//! converged or cut off by a budget, under both dead-end semantics, and the upper-greedy
//! policy of a converged run costs at most the initial upper bound.

use proptest::prelude::*;
use stochastic_search::{
    BrtdpConfig, Config, DeadEnds, Error, FrtdpConfig, Heuristic, Limits, Solution, VpiRtdpConfig,
    Zero, brtdp, frtdp, heuristic_fn, value_iteration, vpi_rtdp,
};

use crate::oracle::{determinization, first_action_values, policy_envelope};
use crate::random::{Shape, ssp};
use crate::table::{Table, action};

/// The gap threshold used by these tests.
const EPSILON: f64 = 1e-4;

/// Slack for floating-point rounding and for value iteration stopping just below `V*`.
const ROUNDING: f64 = 1e-6;

/// The bounded algorithms under test; all share BRTDP's guarantee.
#[derive(Debug, Clone, Copy)]
enum Algorithm {
    /// [`brtdp`].
    Brtdp,
    /// [`frtdp`].
    Frtdp,
    /// [`vpi_rtdp`].
    Vpi,
}

/// Every [`Algorithm`].
const ALGORITHMS: [Algorithm; 3] = [Algorithm::Brtdp, Algorithm::Frtdp, Algorithm::Vpi];

/// Runs `algorithm` on `table` from `initial` with the default settings around `config`.
fn run<L, U>(
    algorithm: Algorithm,
    table: &Table,
    initial: usize,
    goal: &impl Fn(&usize) -> bool,
    (lower, upper): (L, U),
    config: Config<Limits>,
) -> Result<Solution<usize, usize>, Error<usize>>
where
    L: Heuristic<usize, f64>,
    U: Heuristic<usize, f64>,
{
    match algorithm {
        Algorithm::Brtdp => brtdp(table, initial, goal, lower, upper, config.into()),
        Algorithm::Frtdp => frtdp(table, initial, goal, lower, upper, config.into()),
        Algorithm::Vpi => vpi_rtdp(table, initial, goal, lower, upper, config.into()),
    }
}

/// Solves `table` from state 0 with `algorithm` and the given bound values (infinity is
/// `None`).
fn solve(
    algorithm: Algorithm,
    table: &Table,
    (lower, upper): (&[f64], &[f64]),
    dead_ends: DeadEnds,
    seed: u64,
    budget: Limits,
) -> Result<Solution<usize, usize>, Error<usize>> {
    let config = Config::default()
        .with_epsilon(EPSILON)
        .with_dead_ends(dead_ends)
        .with_seed(seed)
        .with_max_trial_length(200)
        .with_budget(budget);
    let bound = |values: &[f64], state: usize| Some(values[state]).filter(|v| v.is_finite());
    let lower = heuristic_fn(|state: &usize| bound(lower, *state));
    let upper = heuristic_fn(|state: &usize| bound(upper, *state));

    run(
        algorithm,
        table,
        0,
        &|s: &usize| table.is_goal(*s),
        (lower, upper),
        config,
    )
}

/// Optimal values of every state by value iteration from that state, to near machine
/// precision; infinity where no proper policy exists.
fn optimal(table: &Table, dead_ends: DeadEnds) -> Vec<f64> {
    let config = Config::default()
        .with_epsilon(1e-11)
        .with_dead_ends(dead_ends);

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

/// Asserts `U(s) >= Q_U(s, pi(s))` at every stored state with a policy action, which holds
/// for a monotone upper bound such as the penalty.
fn assert_upper_policy_bounded(
    table: &Table,
    solution: &Solution<usize, usize>,
) -> Result<(), TestCaseError> {
    for (state, actions) in table.actions.iter().enumerate() {
        let Some(&action) = solution.action(&state) else {
            continue;
        };

        let entry = &actions[action];
        let upper = |s: usize| solution.upper_bound(&s).expect("successors are stored");
        let q: f64 = entry.cost
            + entry
                .outcomes
                .iter()
                .map(|&(t, p)| p * upper(t))
                .sum::<f64>();
        prop_assert!(
            upper(state) >= q - ROUNDING,
            "U {} below Q_U {q} at {state}",
            upper(state)
        );
    }

    Ok(())
}

/// The expected cost of the solution's policy from every envelope state, by policy
/// evaluation from the raw tables; a non-goal state without an action gives up at
/// `give_up`.
fn policy_cost(table: &Table, solution: &Solution<usize, usize>, give_up: f64) -> Vec<f64> {
    let envelope = policy_envelope(table, solution, 0);
    let mut values = vec![0.0; table.len()];
    for _ in 0..1_000_000 {
        let mut change = 0.0_f64;
        for &state in &envelope {
            let next = match solution.action(&state) {
                _ if table.is_goal(state) => 0.0,
                None => give_up,
                Some(&a) => {
                    let entry = &table.actions[state][a];
                    entry.cost
                        + entry
                            .outcomes
                            .iter()
                            .map(|&(t, p)| p * values[t])
                            .sum::<f64>()
                }
            };
            change = change.max((next - values[state]).abs());
            values[state] = next;
        }

        if change < 1e-12 {
            break;
        }
    }

    values
}

/// Asserts that following the policy of a converged run costs at most the initial upper
/// bound, with the fill pass having given every envelope state an action where one exists.
fn assert_policy_within_upper_bound(
    table: &Table,
    solution: &Solution<usize, usize>,
    dead_ends: DeadEnds,
) -> Result<(), TestCaseError> {
    let give_up = match dead_ends {
        DeadEnds::Penalty(p) => p,
        DeadEnds::Infinite => f64::INFINITY,
    };

    let cost = policy_cost(table, solution, give_up)[0];
    let upper = solution
        .initial_upper_bound()
        .expect("bounded runs keep upper bounds");
    prop_assert!(
        cost <= upper + ROUNDING,
        "policy cost {cost} above upper bound {upper}"
    );

    Ok(())
}

/// Asserts that the run converged with a gap of at most epsilon at the initial state.
fn assert_converged(solution: &Solution<usize, usize>) -> Result<(), TestCaseError> {
    let upper = solution
        .initial_upper_bound()
        .expect("bounded runs keep upper bounds");
    prop_assert!(solution.is_converged(), "{:?}", solution.status());
    prop_assert!(upper - solution.initial_value() <= EPSILON);

    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Under infinite dead-end cost, with the blind or the determinization lower bound and
    /// the always-action-0 policy value as upper bound, every bounded algorithm converges,
    /// brackets `V*` everywhere and returns a policy within its upper bound, also with
    /// avoidable traps (which the determinization proves).
    #[test]
    fn bounded_algorithms_bracket_value_iteration(states in 2_usize..10, seed in any::<u64>()) {
        for (shape, blind, algorithm) in [(Shape::Proper, true), (Shape::Proper, false), (Shape::Traps, false)]
            .into_iter()
            .flat_map(|(shape, blind)| ALGORITHMS.map(|algorithm| (shape, blind, algorithm)))
        {
            let table = ssp(states, seed, shape);
            let lower = if blind { vec![0.0; table.len()] } else { determinization(&table) };
            let optimal = optimal(&table, DeadEnds::Infinite);
            let budget = Limits { max_states: Some(1_000), ..Limits::backups(1_000_000) };
            let bounds = (lower.as_slice(), first_action_values(&table));
            let solution = solve(algorithm, &table, (bounds.0, &bounds.1), DeadEnds::Infinite, seed, budget)
                .expect("valid input");

            assert_converged(&solution)?;
            assert_brackets(&solution, &optimal)?;
            assert_policy_within_upper_bound(&table, &solution, DeadEnds::Infinite)?;
        }
    }

    /// Under a finite penalty, the blind lower bound and no upper estimate (the penalty)
    /// suffice; every bounded algorithm converges, brackets `V*`, keeps its monotone upper
    /// bound above the policy's Q-values and returns a policy within it.
    #[test]
    fn bounded_algorithms_bracket_value_iteration_with_penalty(states in 2_usize..10, seed in any::<u64>(), penalty in 1_u32..60) {
        let table = ssp(states, seed, Shape::Traps);
        let dead_ends = DeadEnds::Penalty(f64::from(penalty));
        let (zero, none) = (vec![0.0; table.len()], vec![f64::INFINITY; table.len()]);
        let optimal = optimal(&table, dead_ends);
        for algorithm in ALGORITHMS {
            let solution = solve(algorithm, &table, (&zero, &none), dead_ends, seed, Limits::backups(1_000_000))
                .expect("valid input");

            assert_converged(&solution)?;
            assert_brackets(&solution, &optimal)?;
            prop_assert!(solution.initial_upper_bound() <= Some(f64::from(penalty)));
            assert_upper_policy_bounded(&table, &solution)?;
            assert_policy_within_upper_bound(&table, &solution, dead_ends)?;
        }
    }

    /// The bounds bracket `V*` at any time, also when a small budget stops the run.
    #[test]
    fn bounded_algorithms_bracket_when_cut_off(states in 2_usize..10, seed in any::<u64>(), backups in 1_u64..200) {
        let table = ssp(states, seed, Shape::Traps);
        let lower = determinization(&table);
        let upper = first_action_values(&table);
        let optimal = optimal(&table, DeadEnds::Infinite);
        for algorithm in ALGORITHMS {
            let solution = solve(algorithm, &table, (&lower, &upper), DeadEnds::Infinite, seed, Limits::backups(backups))
                .expect("valid input");

            assert_brackets(&solution, &optimal)?;
        }
    }

    /// The same seed gives bit-identical bounds, policies and statistics.
    #[test]
    fn bounded_algorithms_are_deterministic(states in 2_usize..10, seed in any::<u64>()) {
        let table = ssp(states, seed, Shape::Traps);
        let (lower, upper) = (vec![0.0; table.len()], vec![f64::INFINITY; table.len()]);
        for algorithm in ALGORITHMS {
            let solve = || solve(algorithm, &table, (&lower, &upper), DeadEnds::Penalty(30.0), seed, Limits::DEFAULT)
                .expect("valid input");
            let (first, second) = (solve(), solve());

            prop_assert_eq!(first.stats(), second.stats());
            for state in 0..table.len() {
                prop_assert_eq!(first.value(&state).map(f64::to_bits), second.value(&state).map(f64::to_bits));
                prop_assert_eq!(first.upper_bound(&state).map(f64::to_bits), second.upper_bound(&state).map(f64::to_bits));
                prop_assert_eq!(first.action(&state), second.action(&state));
            }
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

/// The goal test of [`retry_with_trap`]; takes a reference to fit [`stochastic_search::Goal`].
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "goal tests take states by reference"
)]
fn retry_goal(state: &usize) -> bool {
    *state == 1
}

/// Under infinite dead-end cost, a dead end the lower bound misses has no finite upper
/// bound, and every bounded algorithm reports it instead of running with an infinite gap.
#[test]
fn bounded_algorithms_reject_unbounded_upper_bounds() {
    let table = retry_with_trap();
    for algorithm in ALGORITHMS {
        let upper = heuristic_fn(|s: &usize| (*s == 0).then_some(5.0));
        let result = run(
            algorithm,
            &table,
            0,
            &retry_goal,
            (Zero, upper),
            Config::default(),
        );

        assert!(
            matches!(result, Err(Error::UnboundedUpper { state: 2 })),
            "{algorithm:?}"
        );
    }
}

/// With the trap proven by the lower bound, the same problem solves: paying 5 is optimal.
#[test]
fn bounded_algorithms_skip_proven_dead_ends() {
    let table = retry_with_trap();
    for algorithm in ALGORITHMS {
        let lower = heuristic_fn(|s: &usize| (*s != 2).then_some(0.0));
        let upper = heuristic_fn(|s: &usize| (*s == 0).then_some(5.0));
        let solution = run(
            algorithm,
            &table,
            0,
            &retry_goal,
            (lower, upper),
            Config::default(),
        )
        .expect("valid input");

        assert!(solution.is_converged(), "{algorithm:?}");
        assert_eq!(solution.initial_upper_bound(), Some(5.0));
        assert!((solution.initial_value() - 5.0).abs() <= 1e-6);
        assert_eq!(solution.initial_action(), Some(&0));
        assert_eq!(solution.upper_bound(&2), Some(f64::INFINITY));
    }
}

/// Invalid bounds and settings are reported.
#[test]
fn bounded_algorithms_validate_their_input() {
    let table = retry_with_trap();
    let lower = || heuristic_fn(|s: &usize| (*s != 2).then_some(1.0));
    let upper = |u: f64| heuristic_fn(move |_: &usize| Some(u));
    for algorithm in ALGORITHMS {
        for u in [0.5, f64::NAN] {
            let result = run(
                algorithm,
                &table,
                0,
                &retry_goal,
                (lower(), upper(u)),
                Config::default(),
            );
            assert!(
                matches!(result, Err(Error::InvalidHeuristic { state: 0 })),
                "{algorithm:?} {u}"
            );
        }
    }

    let invalid = |result: Result<Solution<usize, usize>, Error<usize>>| {
        matches!(result, Err(Error::InvalidConfig { .. }))
    };
    for tau in [1.0, 0.5, f64::INFINITY, f64::NAN] {
        let config = BrtdpConfig::default().with_tau(tau);
        let result = brtdp(&table, 0, &retry_goal, lower(), upper(5.0), config);
        assert!(invalid(result), "tau {tau}");
    }

    for (depth, factor) in [
        (0.5, 1.1),
        (10.0, 0.9),
        (f64::NAN, 1.1),
        (10.0, f64::INFINITY),
    ] {
        let config = FrtdpConfig::default().with_depth(depth, factor);
        let result = frtdp(&table, 0, &retry_goal, lower(), upper(5.0), config);
        assert!(invalid(result), "depth {depth} factor {factor}");
    }

    for (alpha, beta) in [(-0.1, 0.95), (1.5, 0.95), (0.001, 0.0), (0.001, 1.5)] {
        let config = VpiRtdpConfig::default().with_alpha(alpha).with_beta(beta);
        let result = vpi_rtdp(&table, 0, &retry_goal, lower(), upper(5.0), config);
        assert!(invalid(result), "alpha {alpha} beta {beta}");
    }
}

/// A goal as the initial state converges at once with both bounds at 0.
#[test]
fn bounded_algorithms_at_a_goal() {
    let table = retry_with_trap();
    for algorithm in ALGORITHMS {
        let solution = run(
            algorithm,
            &table,
            1,
            &retry_goal,
            (Zero, Zero),
            Config::default(),
        )
        .expect("valid input");

        assert!(solution.is_converged());
        assert_eq!(solution.initial_upper_bound(), Some(0.0));
        assert_eq!(solution.stats().trials, 0);
    }
}

/// A state whose bounds are exact from the start is never backed up by a trial (its gap is
/// 0), but the upper-greedy policy reaches it; the fill pass gives it its action.
#[test]
fn bounded_algorithms_fill_policy_gaps() {
    // 0 --(1)--> 2 --(1)--> 1 (goal); the bounds at 2 are exact.
    let table = Table {
        actions: vec![
            vec![action(1.0, &[(2, 1.0)])],
            Vec::new(),
            vec![action(1.0, &[(1, 1.0)])],
        ],
        goal: 1,
    };
    let exact_at_two = |s: &usize| Some(if *s == 2 { 1.0 } else { 0.0 });
    let loose = |s: &usize| Some(if *s == 2 { 1.0 } else { 10.0 });
    for algorithm in ALGORITHMS {
        let solution = run(
            algorithm,
            &table,
            0,
            &retry_goal,
            (heuristic_fn(exact_at_two), heuristic_fn(loose)),
            Config::default(),
        )
        .expect("valid input");

        assert!(solution.is_converged(), "{algorithm:?}");
        assert_eq!(solution.initial_action(), Some(&0));
        assert_eq!(solution.action(&2), Some(&0), "{algorithm:?}");
    }
}
