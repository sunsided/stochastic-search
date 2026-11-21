//! Multi-objective SSPs: the paper's figures, closed-form fixtures, the one-objective
//! reduction to LAO* and LRTDP, and random MOSSPs against a brute-force oracle.

mod oracle;
mod random;

use stochastic_search::{
    Abstraction, Config, DeadEnds, Determinized, Error, Heuristic, MoMdp, MoSolution,
    MoSolveResult, Objectives, Status, ZeroVector, imolao_star, mo_value_iteration, molao_star,
    molrtdp,
};

use oracle::{MoAction, MoTable, assert_policies_evaluate};

/// The convergence threshold of the fixtures.
const EPSILON: f64 = 1e-8;

/// The multi-objective solvers under test.
#[derive(Debug, Clone, Copy)]
pub enum Algorithm {
    /// [`mo_value_iteration`].
    Movi,
    /// [`molao_star`].
    Molao,
    /// [`imolao_star`].
    Imolao,
    /// [`molrtdp`].
    Molrtdp,
}

impl Algorithm {
    /// Every solver.
    pub const ALL: [Self; 4] = [Self::Movi, Self::Molao, Self::Imolao, Self::Molrtdp];

    /// Solves `mdp` from `initial`; value iteration ignores the heuristic.
    pub fn solve<M, G, H, const N: usize>(
        self,
        mdp: &M,
        initial: M::State,
        goal: &G,
        heuristic: H,
        objectives: Objectives<N>,
        config: Config,
    ) -> MoSolveResult<M, N>
    where
        M: MoMdp<N>,
        M::State: Clone + Eq + std::hash::Hash,
        G: stochastic_search::Goal<M::State>,
        H: Heuristic<M::State, Vec<[f64; N]>>,
    {
        match self {
            Self::Movi => mo_value_iteration(mdp, initial, goal, objectives, config),
            Self::Molao => molao_star(mdp, initial, goal, heuristic, objectives, config),
            Self::Imolao => imolao_star(mdp, initial, goal, heuristic, objectives, config),
            Self::Molrtdp => molrtdp(mdp, initial, goal, heuristic, objectives, config),
        }
    }
}

/// Shorthand for an action.
fn action<const N: usize>(cost: [f64; N], outcomes: &[(usize, f64)]) -> MoAction<N> {
    MoAction {
        cost,
        outcomes: outcomes.to_vec(),
    }
}

/// The front of a solution, sorted lexicographically.
pub fn front<const N: usize>(solution: &MoSolution<usize, usize, N>) -> Vec<[f64; N]> {
    let mut front: Vec<[f64; N]> = solution.policies().map(|policy| *policy.value()).collect();
    front.sort_by(|a, b| a.partial_cmp(b).expect("finite"));

    front
}

/// Asserts that two sorted fronts agree within `tolerance`.
fn assert_front<const N: usize>(actual: &[[f64; N]], expected: &[[f64; N]], tolerance: f64) {
    assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
    for (a, e) in actual.iter().zip(expected) {
        for (x, y) in a.iter().zip(e) {
            assert!((x - y).abs() <= tolerance, "{actual:?} vs {expected:?}");
        }
    }
}

/// Solves `table` from state 0 with every algorithm and checks the front and the policies.
fn assert_all_solve<const N: usize>(table: &MoTable<N>, bound: [f64; N], expected: &[[f64; N]]) {
    let goal = |state: &usize| *state == table.goal;
    for algorithm in Algorithm::ALL {
        let config = Config::default().with_epsilon(EPSILON);
        let solution = algorithm
            .solve(table, 0, &goal, ZeroVector, Objectives::new(bound), config)
            .expect("valid input");
        assert!(solution.is_converged(), "{algorithm:?}: {solution:?}");
        assert_front(&front(&solution), expected, 1e-5);
        assert_policies_evaluate(table, &solution, 0, 1e-5);
    }
}

/// Figure 1: two retry actions, each free in one objective.
#[test]
fn figure_1() {
    let table = MoTable {
        actions: vec![
            vec![
                action([1.0, 0.0], &[(1, 0.5), (0, 0.5)]),
                action([0.0, 1.0], &[(2, 0.5), (0, 0.5)]),
            ],
            Vec::new(),
            Vec::new(),
        ],
        goal: 1,
    };
    let mut both = table.clone();
    both.actions[0][1].outcomes = vec![(1, 0.5), (0, 0.5)];

    assert_all_solve(&both, [100.0, 100.0], &[[0.0, 2.0], [2.0, 0.0]]);
}

/// Figure 2: the improper policy costs `[inf, 0]`, which only the bound `b` exposes.
#[test]
fn figure_2() {
    let table = MoTable {
        actions: vec![
            vec![
                action([1.0, 0.0], &[(1, 1.0)]),
                action([0.0, 1.0], &[(2, 1.0)]),
            ],
            vec![action([1.0, 0.0], &[(0, 1.0)])],
            Vec::new(),
        ],
        goal: 2,
    };

    assert_all_solve(&table, [2.0, 2.0], &[[0.0, 1.0]]);
}

/// An action that risks a dead end is improper and never in the front, even though it is
/// free.
#[test]
fn risking_a_dead_end_is_improper() {
    let table = MoTable {
        actions: vec![
            vec![
                action([0.0, 0.0], &[(1, 0.5), (2, 0.5)]),
                action([1.0, 1.0], &[(1, 1.0)]),
            ],
            Vec::new(),
            Vec::new(),
        ],
        goal: 1,
    };

    assert_all_solve(&table, [10.0, 10.0], &[[1.0, 1.0]]);
}

/// Truncated non-stationary policies sit just below the hull edge between the two vertices
/// and only reach it in the limit; the margin prunes them, so exactly two policies come
/// back.
#[test]
fn truncated_mixtures_are_pruned() {
    let table = MoTable {
        actions: vec![
            vec![
                action([1.0, 0.0], &[(1, 0.5), (0, 0.5)]),
                action([0.0, 1.0], &[(1, 0.9), (0, 0.1)]),
            ],
            Vec::new(),
        ],
        goal: 1,
    };

    for epsilon in [1e-4, 1e-6, 1e-9] {
        for algorithm in Algorithm::ALL {
            let config = Config::default().with_epsilon(epsilon);
            let goal = |state: &usize| *state == 1;
            let solution = algorithm
                .solve(
                    &table,
                    0,
                    &goal,
                    ZeroVector,
                    Objectives::new([50.0; 2]),
                    config,
                )
                .expect("valid input");
            assert_front(&front(&solution), &[[0.0, 1.0 / 0.9], [2.0, 0.0]], 1e-3);
            assert_policies_evaluate(&table, &solution, 0, 1e-3);
        }
    }
}

/// No proper policy: every algorithm reports the problem unsolvable, with no policy.
#[test]
fn unsolvable() {
    let table: MoTable<2> = MoTable {
        actions: vec![
            vec![
                action([1.0, 0.0], &[(0, 0.5), (2, 0.5)]),
                action([0.0, 1.0], &[(0, 1.0)]),
            ],
            Vec::new(),
            Vec::new(),
        ],
        goal: 1,
    };
    let goal = |state: &usize| *state == 1;
    for algorithm in Algorithm::ALL {
        let solution = algorithm
            .solve(
                &table,
                0,
                &goal,
                ZeroVector,
                Objectives::new([5.0; 2]),
                Config::default(),
            )
            .expect("valid input");
        assert_eq!(solution.status(), Status::Unsolvable, "{algorithm:?}");
        assert_eq!(solution.policies().len(), 0);
    }
}

/// A goal initial state has the zero vector and one empty policy.
#[test]
fn goal_initial_state() {
    let table: MoTable<2> = MoTable {
        actions: vec![Vec::new()],
        goal: 0,
    };
    for algorithm in Algorithm::ALL {
        let goal = |state: &usize| *state == 0;
        let solution = algorithm
            .solve(
                &table,
                0,
                &goal,
                ZeroVector,
                Objectives::new([5.0; 2]),
                Config::default(),
            )
            .expect("valid input");
        assert!(solution.is_converged());
        assert_eq!(front(&solution), vec![[0.0, 0.0]]);
        assert!(solution.policies().all(|policy| policy.is_empty()));
    }
}

/// Invalid settings and MDP output are reported, not panicked on.
#[test]
fn invalid_input() {
    let table = MoTable {
        actions: vec![vec![action([1.0, -1.0], &[(1, 1.0)])], Vec::new()],
        goal: 1,
    };
    let goal = |state: &usize| *state == 1;
    for algorithm in Algorithm::ALL {
        let error = algorithm
            .solve(
                &table,
                0,
                &goal,
                ZeroVector,
                Objectives::new([5.0; 2]),
                Config::default(),
            )
            .expect_err("negative cost");
        assert!(matches!(error, Error::InvalidCost { cost, .. } if cost < 0.0));

        let bad_bound = Objectives::new([5.0, f64::NAN]);
        let error = algorithm
            .solve(&table, 0, &goal, ZeroVector, bad_bound, Config::default())
            .expect_err("bad bound");
        assert!(matches!(error, Error::InvalidConfig { .. }));

        let penalty = Config::default().with_dead_ends(DeadEnds::Penalty(3.0));
        let error = algorithm
            .solve(
                &table,
                0,
                &goal,
                ZeroVector,
                Objectives::new([5.0; 2]),
                penalty,
            )
            .expect_err("penalty");
        assert!(matches!(error, Error::InvalidConfig { .. }));
    }
}

/// A NaN heuristic vector is reported.
#[test]
fn invalid_heuristic() {
    let table = MoTable {
        actions: vec![vec![action([1.0, 1.0], &[(1, 1.0)])], Vec::new()],
        goal: 1,
    };
    let goal = |state: &usize| *state == 1;
    let nan = stochastic_search::heuristic_fn(|_: &usize| Some(vec![[f64::NAN, 0.0]]));
    let error = molrtdp(
        &table,
        0,
        &goal,
        nan,
        Objectives::new([5.0; 2]),
        Config::default(),
    )
    .expect_err("NaN");
    assert!(matches!(error, Error::InvalidHeuristic { state: 0 }));
}

/// The determinisation heuristic is not admissible in the set-domination sense, but the
/// search still finds the optimum because it is admissible for every weight.
#[test]
fn determinization_heuristic_on_the_coin() {
    let table = MoTable {
        actions: vec![
            vec![action([0.0, 0.0], &[(1, 0.5), (2, 0.5)])],
            vec![action([1.0, 0.0], &[(3, 1.0)])],
            vec![action([0.0, 1.0], &[(3, 1.0)])],
            Vec::new(),
        ],
        goal: 3,
    };
    let goal = |state: &usize| *state == 3;
    let determinized = Determinized(&table);
    let objectives = Objectives::new([10.0; 2]);
    let mut heuristic = Abstraction::new(&determinized, |state: &usize| *state, &goal, objectives);
    let estimate = heuristic.estimate(&0).expect("solvable");
    assert_eq!(estimate.len(), 2);

    for algorithm in [Algorithm::Molao, Algorithm::Imolao, Algorithm::Molrtdp] {
        let heuristic = Abstraction::new(&determinized, |state: &usize| *state, &goal, objectives);
        let solution = algorithm
            .solve(&table, 0, &goal, heuristic, objectives, Config::default())
            .expect("valid input");
        assert_front(&front(&solution), &[[0.5, 0.5]], 1e-6);
    }
}

/// One seed gives identical MOLRTDP runs.
#[test]
fn molrtdp_is_deterministic() {
    let table = random::mossp::<2>(6, 42);
    let goal = |state: &usize| *state == table.goal;
    let run = || {
        molrtdp(
            &table,
            0,
            &goal,
            ZeroVector,
            Objectives::new([random::BOUND; 2]),
            Config::default().with_seed(9),
        )
        .expect("valid input")
    };
    let (a, b) = (run(), run());
    assert_eq!(front(&a), front(&b));
    assert_eq!(a.stats(), b.stats());
}

/// A budget stop never claims unsolvability, even when the initial value has no vector
/// yet.
#[test]
fn budget_stop_is_not_a_verdict() {
    let table = MoTable {
        actions: vec![
            vec![action([1.0, 0.0], &[(1, 1.0)])],
            vec![action([1.0, 1.0], &[(2, 1.0)])],
            Vec::new(),
        ],
        goal: 2,
    };
    let goal = |state: &usize| *state == 2;
    for algorithm in Algorithm::ALL {
        let config = Config::default().with_budget(stochastic_search::Limits::backups(1));
        let solution = algorithm
            .solve(
                &table,
                0,
                &goal,
                ZeroVector,
                Objectives::new([5.0; 2]),
                config,
            )
            .expect("valid input");
        assert_eq!(solution.status(), Status::BudgetExhausted, "{algorithm:?}");
    }
}

/// Every determinised path overshoots the bound while the expected cost stays below it;
/// the abstraction then estimates blind instead of proving a dead end.
#[test]
fn abstraction_capped_by_its_bound_is_not_a_dead_end() {
    let table = MoTable {
        actions: vec![
            vec![action([6.0, 6.0], &[(1, 0.5), (2, 0.5)])],
            vec![action([7.0, 0.0], &[(3, 1.0)])],
            vec![action([0.0, 7.0], &[(3, 1.0)])],
            Vec::new(),
        ],
        goal: 3,
    };
    let goal = |state: &usize| *state == 3;
    let objectives = Objectives::new([10.0; 2]);
    let determinized = Determinized(&table);
    for algorithm in [Algorithm::Molao, Algorithm::Imolao, Algorithm::Molrtdp] {
        let heuristic = Abstraction::new(&determinized, |state: &usize| *state, &goal, objectives);
        let solution = algorithm
            .solve(&table, 0, &goal, heuristic, objectives, Config::default())
            .expect("valid input");
        assert!(solution.is_converged(), "{algorithm:?}: {solution:?}");
        assert_front(&front(&solution), &[[9.5, 9.5]], 1e-6);
    }
}
