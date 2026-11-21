//! Fuzzes the multi-objective planners with small decoded bi-objective tables, valid and
//! invalid. Each run must return a solution or an error, never panic or hang, and every
//! policy value of a solution is free of NaN.

#![no_main]
#![forbid(unsafe_code)]

use libfuzzer_sys::arbitrary::{Result, Unstructured};
use libfuzzer_sys::fuzz_target;
use stochastic_search::{
    Config, Limits, MoSolveResult, Objectives, ZeroVector, imolao_star, mo_value_iteration,
    molao_star, molrtdp,
};
use stochastic_search_fuzz::{Table, number};

/// The number of objectives.
const N: usize = 2;

/// The backup budget of every run. A backup prunes a coverage set of up to ~50 vectors on
/// improper inputs, which takes about 0.35 ms in a release build and ten times that under
/// the sanitizer, so this keeps each run well under libFuzzer's timeout.
const MAX_BACKUPS: u64 = 300;

/// A decoded bi-objective problem.
#[derive(Debug)]
struct Input {
    /// The MDP, solved from state 0.
    table: Table<[f64; N]>,
    /// The bound `b` of the weak improper policy assumption, per objective.
    bound: [f64; N],
    /// The planners' random seed.
    seed: u64,
}

impl Input {
    /// Decodes an input with costs in `[0, 4]` and bounds in `[0, 1000]` per objective, mostly.
    fn decode(u: &mut Unstructured<'_>) -> Result<Self> {
        let table = Table::decode(u, |u| Ok([number(u, 0.0, 4.0)?, number(u, 0.0, 4.0)?]))?;
        let bound = [number(u, 0.0, 1000.0)?, number(u, 0.0, 1000.0)?];

        Ok(Self {
            table,
            bound,
            seed: u.arbitrary()?,
        })
    }

    /// The configuration every planner runs with.
    fn config(&self) -> Config {
        Config::default()
            .with_epsilon(1e-3)
            .with_seed(self.seed)
            .with_max_trial_length(50)
            .with_budget(Limits::backups(MAX_BACKUPS))
    }
}

/// Runs the planner selected by `algorithm` on `input`.
fn solve(input: &Input, algorithm: u8) -> MoSolveResult<Table<[f64; N]>, N> {
    let table = &input.table;
    let goal = |state: &usize| table.is_goal(*state);
    let objectives = Objectives::new(input.bound);
    let config = input.config();

    match algorithm % 4 {
        0 => mo_value_iteration(table, 0, &goal, objectives, config),
        1 => molrtdp(table, 0, &goal, ZeroVector, objectives, config),
        2 => molao_star(table, 0, &goal, ZeroVector, objectives, config),
        _ => imolao_star(table, 0, &goal, ZeroVector, objectives, config),
    }
}

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    let Ok(algorithm) = u.arbitrary::<u8>() else {
        return;
    };
    let Ok(input) = Input::decode(&mut u) else {
        return;
    };

    if let Ok(solution) = solve(&input, algorithm) {
        for policy in solution.policies() {
            let value = policy.value();
            assert!(
                !value.iter().any(|v| v.is_nan()),
                "NaN value from {input:?}"
            );
        }
    }
});
