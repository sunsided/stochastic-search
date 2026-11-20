//! Benchmarks for the hot paths: LRTDP trials on a corridor SSP, and POMCP simulations and
//! DESPOT explorations on Tiger.

use std::hint::black_box;

use criterion::Criterion;
use rand_core::Rng;
use stochastic_search::{
    Config, DespotConfig, Limits, Mdp, PomcpConfig, Pomdp, RolloutPolicy, Step, Zero, despot,
    lrtdp, pomcp, unit_interval,
};

/// Runs every benchmark in this file with criterion's command-line flags.
fn main() {
    let mut criterion = Criterion::default().configure_from_args();

    bench_lrtdp(&mut criterion);
    bench_pomcp(&mut criterion);
    bench_despot(&mut criterion);
    criterion.final_summary();
}

/// A corridor of 1000 cells; each step costs 1 and slips back in place with probability 0.2.
struct Corridor;

impl Mdp for Corridor {
    type State = u32;
    type Action = ();

    fn actions(&self, _state: &u32, mut emit: impl FnMut((), f64)) {
        emit((), 1.0);
    }

    fn outcomes(&self, state: &u32, _action: &(), mut emit: impl FnMut(u32, f64)) {
        emit(state + 1, 0.8);
        emit(*state, 0.2);
    }
}

/// LRTDP solving the corridor from scratch with the blind heuristic.
fn bench_lrtdp(criterion: &mut Criterion) {
    criterion.bench_function("lrtdp corridor 1000", |bencher| {
        bencher.iter(|| {
            let solution = lrtdp(
                &Corridor,
                0,
                &|state: &u32| *state == 1000,
                Zero,
                Config::default(),
            )
            .expect("valid input");

            black_box(solution.initial_value())
        });
    });
}

/// The Tiger problem: the state is `true` if the tiger is behind the left door; actions are
/// listen (0), open left (1) and open right (2).
struct Tiger;

impl Pomdp for Tiger {
    type State = bool;
    type Action = u8;
    type Observation = Option<bool>;

    fn actions(&self, mut emit: impl FnMut(u8)) {
        emit(0);
        emit(1);
        emit(2);
    }

    fn step<R: Rng + ?Sized>(
        &self,
        left: &bool,
        action: &u8,
        rng: &mut R,
    ) -> Step<bool, Option<bool>> {
        if *action == 0 {
            let correct = unit_interval(rng) < 0.85;
            return Step::new(*left, Some(*left == correct), -1.0);
        }

        let reward = if (*action == 1) == *left {
            -100.0
        } else {
            10.0
        };

        Step::new(unit_interval(rng) < 0.5, None, reward)
    }
}

/// The usual Tiger rollout policy: always listen.
struct Listen;

impl RolloutPolicy<bool> for Listen {
    fn action<R: Rng + ?Sized>(&mut self, _state: &bool, _actions: usize, _rng: &mut R) -> usize {
        0
    }
}

/// 10 000 POMCP simulations from the uniform Tiger belief.
fn bench_pomcp(criterion: &mut Criterion) {
    let config = PomcpConfig {
        exploration: 110.0,
        max_depth: 60,
        ..PomcpConfig::default()
    }
    .with_budget(Limits {
        max_trials: Some(10_000),
        ..PomcpConfig::DEFAULT_LIMITS
    });

    criterion.bench_function("pomcp tiger 10k simulations", |bencher| {
        bencher.iter(|| {
            let plan = pomcp(&Tiger, &[true, false], Listen, config).expect("valid input");

            black_box(plan.value)
        });
    });
}

/// 200 DESPOT explorations with 200 scenarios from the uniform Tiger belief.
fn bench_despot(criterion: &mut Criterion) {
    let config = DespotConfig {
        scenarios: 200,
        max_depth: 40,
        ..DespotConfig::default()
    }
    .with_budget(Limits {
        max_trials: Some(200),
        ..DespotConfig::DEFAULT_LIMITS
    });

    criterion.bench_function("despot tiger 200 explorations", |bencher| {
        bencher.iter(|| {
            let plan = despot(&Tiger, &[true, false], Listen, |_: &bool| 200.0, config)
                .expect("valid input");

            black_box(plan.value)
        });
    });
}
