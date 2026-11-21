//! Fuzzes every single-objective SSP planner with small decoded tables, heuristics, bounds
//! and dead-end models, valid and invalid. Each run must return a solution or an error, never
//! panic or hang, and a solution's initial value is never NaN.

#![no_main]
#![forbid(unsafe_code)]

use libfuzzer_sys::arbitrary::{Result, Unstructured};
use libfuzzer_sys::fuzz_target;
use stochastic_search::{
    Config, DeadEnds, Limits, SolveResult, brtdp, cg_ilao_star, frtdp, heuristic_fn, ilao_star,
    lao_star, lrtdp, rtdp, value_iteration, vpi_rtdp,
};
use stochastic_search_fuzz::{Table, number};

/// The backup budget of every run; small enough to keep a run well under a millisecond.
const MAX_BACKUPS: u64 = 5_000;

/// A decoded planning problem.
#[derive(Debug)]
struct Input {
    /// The MDP, solved from state 0.
    table: Table<f64>,
    /// The heuristic (and lower bound) per state; `None` claims a dead end.
    lower: Vec<Option<f64>>,
    /// The upper bound per state for the bounded planners.
    upper: Vec<Option<f64>>,
    /// How dead ends are valued.
    dead_ends: DeadEnds,
    /// The planners' random seed.
    seed: u64,
}

impl Input {
    /// Decodes an input with costs in `[0, 4]`, estimates in `[0, 8]` and upper bounds in
    /// `[0, 50]`, mostly.
    fn decode(u: &mut Unstructured<'_>) -> Result<Self> {
        let table = Table::decode(u, |u| number(u, 0.0, 4.0))?;
        let mut lower = Vec::with_capacity(table.len());
        let mut upper = Vec::with_capacity(table.len());
        for _ in 0..table.len() {
            lower.push(estimate(u, 8.0)?);
            upper.push(estimate(u, 50.0)?);
        }

        let dead_ends = if u.arbitrary()? {
            DeadEnds::Infinite
        } else {
            DeadEnds::Penalty(number(u, 0.0, 100.0)?)
        };

        Ok(Self {
            table,
            lower,
            upper,
            dead_ends,
            seed: u.arbitrary()?,
        })
    }

    /// The configuration every planner runs with.
    fn config(&self) -> Config {
        Config::default()
            .with_epsilon(1e-3)
            .with_dead_ends(self.dead_ends)
            .with_seed(self.seed)
            .with_max_trial_length(50)
            .with_budget(Limits::backups(MAX_BACKUPS))
    }
}

/// Draws a per-state estimate in `[0, high]`; one time in eight `None`.
fn estimate(u: &mut Unstructured<'_>, high: f64) -> Result<Option<f64>> {
    if u.ratio(1, 8)? {
        return Ok(None);
    }

    number(u, 0.0, high).map(Some)
}

/// Runs the planner selected by `algorithm` on `input`.
fn solve(input: &Input, algorithm: u8) -> SolveResult<Table<f64>> {
    let table = &input.table;
    let goal = |state: &usize| table.is_goal(*state);
    let lower = heuristic_fn(|state: &usize| input.lower[*state]);
    let upper = heuristic_fn(|state: &usize| input.upper[*state]);
    let config = input.config();

    match algorithm % 9 {
        0 => value_iteration(table, 0, &goal, config),
        1 => lrtdp(table, 0, &goal, lower, config),
        2 => rtdp(table, 0, &goal, lower, config),
        3 => lao_star(table, 0, &goal, lower, config),
        4 => ilao_star(table, 0, &goal, lower, config),
        5 => cg_ilao_star(table, 0, &goal, lower, config),
        6 => brtdp(table, 0, &goal, lower, upper, config.into()),
        7 => frtdp(table, 0, &goal, lower, upper, config.into()),
        _ => vpi_rtdp(table, 0, &goal, lower, upper, config.into()),
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
        let value = solution.initial_value();
        assert!(!value.is_nan(), "NaN initial value from {input:?}");
        assert!(
            solution.stats().backups <= MAX_BACKUPS + 1,
            "budget overrun"
        );
    }
});
