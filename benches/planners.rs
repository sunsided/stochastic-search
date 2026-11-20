//! Benchmarks for the hot paths: LRTDP trials on a corridor SSP, LRTDP against the bounded
//! RTDP variants on a noisy corridor, LAO* against iLAO* and CG-iLAO*, POMCP simulations
//! and DESPOT explorations on Tiger, tree reuse between decisions, and the particle filter.

use std::hint::black_box;

use criterion::Criterion;
use rand_core::Rng;
use stochastic_search::{
    BeliefFilter, BrtdpConfig, Config, Despot, DespotConfig, FrtdpConfig, Limits, Mdp, Pomcp,
    PomcpConfig, Pomdp, RolloutPolicy, SplitMix64, Step, VpiRtdpConfig, Zero, brtdp, cg_ilao_star,
    despot, frtdp, heuristic_fn, ilao_star, lao_star, lrtdp, pomcp, unit_interval, update_belief,
    vpi_rtdp,
};

/// Runs every benchmark in this file with criterion's command-line flags.
fn main() {
    let mut criterion = Criterion::default().configure_from_args();

    bench_lrtdp(&mut criterion);
    bench_bounded(&mut criterion);
    bench_lao(&mut criterion);
    bench_pomcp(&mut criterion);
    bench_despot(&mut criterion);
    bench_reuse(&mut criterion);
    bench_belief(&mut criterion);
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

/// A noisy corridor of 200 cells: each step costs 1, moves on with probability 0.6, stays
/// with 0.2 and slips back one cell with 0.2 (staying put at cell 0).
struct NoisyCorridor;

impl Mdp for NoisyCorridor {
    type State = u32;
    type Action = ();

    fn actions(&self, _state: &u32, mut emit: impl FnMut((), f64)) {
        emit((), 1.0);
    }

    fn outcomes(&self, state: &u32, _action: &(), mut emit: impl FnMut(u32, f64)) {
        emit(state + 1, 0.6);
        emit(*state, 0.2);
        emit(state.saturating_sub(1), 0.2);
    }
}

/// LRTDP, BRTDP, FRTDP and VPI-RTDP on the noisy corridor with the blind lower bound; the
/// bounded ones get the upper bound `2.5 * distance + 1` (the walk drifts forward by at
/// least 0.4 cells per step).
fn bench_bounded(criterion: &mut Criterion) {
    let goal = |state: &u32| *state == 200;
    let upper = || heuristic_fn(|state: &u32| Some(2.5 * f64::from(200 - state) + 1.0));
    let mut group = criterion.benchmark_group("noisy corridor 200");
    group.bench_function("lrtdp", |bencher| {
        bencher.iter(|| {
            let solution = lrtdp(&NoisyCorridor, 0, &goal, Zero, Config::default());
            black_box(solution.expect("valid input").initial_value())
        });
    });
    group.bench_function("brtdp", |bencher| {
        bencher.iter(|| {
            let config = BrtdpConfig::default();
            let solution = brtdp(&NoisyCorridor, 0, &goal, Zero, upper(), config);
            black_box(solution.expect("valid input").initial_value())
        });
    });
    group.bench_function("frtdp", |bencher| {
        bencher.iter(|| {
            let config = FrtdpConfig::default();
            let solution = frtdp(&NoisyCorridor, 0, &goal, Zero, upper(), config);
            black_box(solution.expect("valid input").initial_value())
        });
    });
    group.bench_function("vpi-rtdp", |bencher| {
        bencher.iter(|| {
            let config = VpiRtdpConfig::default();
            let solution = vpi_rtdp(&NoisyCorridor, 0, &goal, Zero, upper(), config);
            black_box(solution.expect("valid input").initial_value())
        });
    });
    group.finish();
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

/// Classic LAO*, iLAO* and CG-iLAO* on the noisy corridor with the blind heuristic.
fn bench_lao(criterion: &mut Criterion) {
    let goal = |state: &u32| *state == 200;
    let mut group = criterion.benchmark_group("noisy corridor 200, solution graphs");
    group.bench_function("lao*", |bencher| {
        bencher.iter(|| {
            let solution = lao_star(&NoisyCorridor, 0, &goal, Zero, Config::default());
            black_box(solution.expect("valid input").initial_value())
        });
    });
    group.bench_function("ilao*", |bencher| {
        bencher.iter(|| {
            let solution = ilao_star(&NoisyCorridor, 0, &goal, Zero, Config::default());
            black_box(solution.expect("valid input").initial_value())
        });
    });
    group.bench_function("cg-ilao*", |bencher| {
        bencher.iter(|| {
            let solution = cg_ilao_star(&NoisyCorridor, 0, &goal, Zero, Config::default());
            black_box(solution.expect("valid input").initial_value())
        });
    });
    group.finish();
}

/// Two Tiger decisions (listen, hear left) with 5 000 simulations or 100 explorations each:
/// with the tree kept by `advance`, and with a fresh planner for the second decision.
fn bench_reuse(criterion: &mut Criterion) {
    let pomcp_config = PomcpConfig::default()
        .with_exploration(110.0)
        .with_max_depth(60)
        .with_budget(Limits {
            max_trials: Some(5_000),
            ..PomcpConfig::DEFAULT_LIMITS
        });
    let despot_config = DespotConfig::default()
        .with_scenarios(200)
        .with_max_depth(40)
        .with_budget(Limits {
            max_trials: Some(100),
            ..DespotConfig::DEFAULT_LIMITS
        });
    let heard_left = [true, true, true, true, true, true, false];

    let mut group = criterion.benchmark_group("tiger two decisions");
    group.bench_function("pomcp kept tree", |bencher| {
        bencher.iter(|| {
            let mut planner = Pomcp::new(&Tiger, Listen, pomcp_config).expect("valid");
            planner.plan(&[true, false]).expect("valid");
            planner.advance(&0, &Some(true));
            black_box(planner.plan(&heard_left).expect("valid").value)
        });
    });
    group.bench_function("pomcp fresh tree", |bencher| {
        bencher.iter(|| {
            let mut planner = Pomcp::new(&Tiger, Listen, pomcp_config).expect("valid");
            planner.plan(&[true, false]).expect("valid");
            let mut planner = Pomcp::new(&Tiger, Listen, pomcp_config).expect("valid");
            black_box(planner.plan(&heard_left).expect("valid").value)
        });
    });
    group.bench_function("despot kept tree", |bencher| {
        bencher.iter(|| {
            let upper = |_: &bool| 200.0;
            let mut planner =
                Despot::new(&Tiger, &[true, false], Listen, upper, despot_config).expect("valid");
            planner.plan().expect("valid");
            planner.advance(&0, &Some(true)).expect("a seen history");
            black_box(planner.plan().expect("valid").value)
        });
    });
    group.bench_function("despot fresh tree", |bencher| {
        bencher.iter(|| {
            let upper = |_: &bool| 200.0;
            let mut planner =
                Despot::new(&Tiger, &[true, false], Listen, upper, despot_config).expect("valid");
            planner.plan().expect("valid");
            let mut planner =
                Despot::new(&Tiger, &heard_left, Listen, upper, despot_config).expect("valid");
            black_box(planner.plan().expect("valid").value)
        });
    });
    group.finish();
}

/// The particle filter: 10 000 particles after listening and hearing the tiger on the left.
fn bench_belief(criterion: &mut Criterion) {
    let belief: Vec<bool> = (0..10_000).map(|i| i % 2 == 0).collect();
    let filter = BeliefFilter::default()
        .with_particles(10_000)
        .with_reinvigorate(500);
    criterion.bench_function("update_belief tiger 10k particles", |bencher| {
        let mut rng = SplitMix64::new(0);
        bencher.iter(|| {
            let update = update_belief(
                &Tiger,
                &belief,
                &0,
                &Some(true),
                filter,
                &mut rng,
                |s, _| *s,
            )
            .expect("the observation is possible");
            black_box(update.particles.len())
        });
    });
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
    let config = PomcpConfig::default()
        .with_exploration(110.0)
        .with_max_depth(60)
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
    let config = DespotConfig::default()
        .with_scenarios(200)
        .with_max_depth(40)
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
