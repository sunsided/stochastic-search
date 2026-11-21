//! Fuzzes POMCP, DESPOT and the particle filter with small decoded generative POMDPs, valid
//! and invalid. Each run must return a plan, a belief or an error, never panic or hang. A
//! plan names one of the model's actions and has a value that is not NaN; a belief update
//! keeps its counts consistent.

#![no_main]
#![forbid(unsafe_code)]

use libfuzzer_sys::arbitrary::{Result, Unstructured};
use libfuzzer_sys::fuzz_target;
use stochastic_search::{
    BeliefFilter, DespotConfig, Error, Limits, Plan, PomcpConfig, SplitMix64, UniformRandom,
    despot, pomcp, update_belief,
};
use stochastic_search_fuzz::{GenerativeTable, number, state};

/// The most particles in a decoded belief.
const MAX_PARTICLES: usize = 8;

/// The simulation (POMCP) or exploration (DESPOT) budget of every run.
const MAX_TRIALS: u64 = 64;

/// The largest reward or bound magnitude whose discounted sums over a run stay finite.
///
/// Finite rewards near `f64::MAX` overflow when summed to infinite returns, whose averages
/// are NaN. The planners accept such rewards (they are finite), so the NaN check only covers
/// inputs below this magnitude.
const MAX_SAFE_MAGNITUDE: f64 = 1e300;

/// A decoded planning problem.
#[derive(Debug)]
struct Input {
    /// The POMDP.
    pomdp: GenerativeTable,
    /// The particle belief to plan from; may be empty, which is invalid input.
    belief: Vec<usize>,
    /// The discount factor, in `[0, 1]` mostly.
    discount: f64,
    /// DESPOT's upper bound on the return from any state.
    upper: f64,
    /// The action and observation the belief update conditions on.
    update: (usize, u8),
    /// The planners' and the filter's random seed.
    seed: u64,
}

impl Input {
    /// Decodes an input.
    fn decode(u: &mut Unstructured<'_>) -> Result<Self> {
        let pomdp = GenerativeTable::decode(u)?;
        let particles = u.int_in_range(0..=MAX_PARTICLES)?;
        let mut belief = Vec::with_capacity(particles);
        for _ in 0..particles {
            belief.push(state(u, pomdp.states())?);
        }

        let discount = number(u, 0.0, 1.0)?;
        let upper = number(u, -10.0, 200.0)?;
        let update = (
            u.int_in_range(0..=pomdp.actions.max(1) - 1)?,
            u.int_in_range(0..=GenerativeTable::MAX_OBSERVATIONS - 1)?,
        );

        Ok(Self {
            pomdp,
            belief,
            discount,
            upper,
            update,
            seed: u.arbitrary()?,
        })
    }
}

impl Input {
    /// Returns `true` if every reward and the upper bound are small enough that no return
    /// can overflow (see [`MAX_SAFE_MAGNITUDE`]). Non-finite ones count as safe: the planners
    /// must reject them.
    fn has_safe_magnitudes(&self) -> bool {
        let rewards = self.pomdp.steps.iter().flatten().flatten();

        (!self.upper.is_finite() || self.upper.abs() <= MAX_SAFE_MAGNITUDE)
            && rewards
                .into_iter()
                .all(|(t, _)| !t.reward.is_finite() || t.reward.abs() <= MAX_SAFE_MAGNITUDE)
    }
}

/// Plans once with POMCP or DESPOT, as `despot` selects.
fn plan(input: &Input, use_despot: bool) -> core::result::Result<Plan<usize>, Error<usize>> {
    let limits = Limits {
        max_trials: Some(MAX_TRIALS),
        ..Limits::default()
    };

    if use_despot {
        let config = DespotConfig::default()
            .with_discount(input.discount)
            .with_scenarios(8)
            .with_max_depth(8)
            .with_seed(input.seed)
            .with_budget(limits);

        despot(
            &input.pomdp,
            &input.belief,
            UniformRandom,
            |_: &usize| input.upper,
            config,
        )
    } else {
        let config = PomcpConfig::default()
            .with_discount(input.discount)
            .with_max_depth(8)
            .with_seed(input.seed)
            .with_budget(limits);

        pomcp(&input.pomdp, &input.belief, UniformRandom, config)
    }
}

/// Updates the belief after the decoded action and observation and checks the counts.
fn check_update(input: &Input) {
    let (action, observation) = input.update;
    if action >= input.pomdp.actions {
        return;
    }

    let filter = BeliefFilter::default()
        .with_particles(16)
        .with_max_attempts(256)
        .with_reinvigorate(2);
    let mut rng = SplitMix64::new(input.seed);
    let result = update_belief(
        &input.pomdp,
        &input.belief,
        &action,
        &observation,
        filter,
        &mut rng,
        |state, _| *state,
    );

    if let Ok(update) = result {
        assert!(update.accepted >= 1 && update.accepted <= 16);
        assert_eq!(update.particles.len(), 18);
        assert!(update.particles.iter().all(|&s| s < input.pomdp.states()));
    }
}

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    let Ok(use_despot) = u.arbitrary::<bool>() else {
        return;
    };
    let Ok(input) = Input::decode(&mut u) else {
        return;
    };

    if let Ok(plan) = plan(&input, use_despot) {
        assert!(
            plan.action < input.pomdp.actions,
            "unknown action from {input:?}"
        );
        assert!(
            !plan.value.is_nan() || !input.has_safe_magnitudes(),
            "NaN value from {input:?}"
        );
    }

    check_update(&input);
});
