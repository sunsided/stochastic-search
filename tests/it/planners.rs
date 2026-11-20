//! POMCP on Tiger and a one-rock `RockSample`: the known-good action over several seeds.
//!
//! The expected actions are derived by hand in each test's docs, with their margins, so
//! they do not depend on the planner under test. The rollout policies are the usual
//! domain defaults (always listen, head east); uniformly random rollouts drown Tiger's
//! first-step difference in noise.

use stochastic_search::{Error, Limits, PomcpConfig, UniformRandom, pomcp};

use crate::pomdps::{
    AlwaysListen, DISCOUNT, HeadEast, RockAction, Tiger, TigerAction, one_rock, one_rock_belief,
    tiger_belief,
};

/// The seeds every planner test runs with.
const SEEDS: [u64; 4] = [0, 1, 2, 3];

/// A POMCP configuration for returns spanning `range`, with `simulations` simulations.
fn pomcp_config(seed: u64, range: f64, simulations: u64) -> PomcpConfig {
    PomcpConfig {
        discount: DISCOUNT,
        exploration: range,
        max_depth: 60,
        seed,
        ..PomcpConfig::default()
    }
    .with_budget(Limits {
        max_trials: Some(simulations),
        ..PomcpConfig::DEFAULT_LIMITS
    })
}

/// From the uniform belief, opening a door has expected reward
/// `0.5 * 10 - 0.5 * 100 = -45`; listening costs 1 and is the known optimal first action.
#[test]
fn pomcp_listens_to_the_tiger_first() {
    for seed in SEEDS {
        let plan = pomcp(
            &Tiger,
            &tiger_belief(1, 2),
            AlwaysListen,
            pomcp_config(seed, 110.0, 20_000),
        )
        .expect("valid input");

        assert_eq!(plan.action, TigerAction::Listen, "seed {seed}");
        assert_eq!(plan.stats.trials, 20_000);
    }
}

/// After three more "left" than "right" observations, the tiger is on the left with
/// probability `0.85^3 / (0.85^3 + 0.15^3) = 0.9945` (199 of 200 particles here, 0.995).
/// Opening the right door now yields `0.995 * 10 - 0.005 * 100 = 9.45` plus the discounted
/// value of a fresh round; listening once more first costs 1 and delays both, about 1.1
/// worse even if it made the decision certain.
#[test]
fn pomcp_opens_the_far_door_when_confident() {
    for seed in SEEDS {
        let plan = pomcp(
            &Tiger,
            &tiger_belief(199, 200),
            AlwaysListen,
            pomcp_config(seed, 110.0, 20_000),
        )
        .expect("valid input");

        assert_eq!(plan.action, TigerAction::OpenRight, "seed {seed}");
    }
}

/// The agent stands on a rock that is good with probability 0.5, three moves from the exit.
/// With `gamma = 0.95`:
///
/// - exit at once: `10 * gamma^2 = 9.03`;
/// - sample blindly: `0` expected now, then exit: `10 * gamma^3 = 8.57`;
/// - check (perfect at distance 0), sample only if good, exit:
///   `0.5 * (10 * gamma + 10 * gamma^4) + 0.5 * 10 * gamma^3 = 13.1`.
///
/// Checking wins by about 4.
#[test]
fn pomcp_checks_an_uncertain_rock() {
    let model = one_rock();
    for seed in SEEDS {
        let plan = pomcp(
            &model,
            &one_rock_belief(50, 100),
            HeadEast,
            pomcp_config(seed, 20.0, 20_000),
        )
        .expect("valid input");

        assert_eq!(plan.action, RockAction::Check(0), "seed {seed}");
    }
}

/// On a rock known to be bad, sampling costs 10 and checking only wastes a step; heading
/// east (`10 * gamma^2 = 9.03`) beats every other first action (at most
/// `10 * gamma^3 = 8.57`).
#[test]
fn pomcp_leaves_a_bad_rock() {
    let model = one_rock();
    for seed in SEEDS {
        let plan = pomcp(
            &model,
            &one_rock_belief(0, 10),
            HeadEast,
            pomcp_config(seed, 20.0, 20_000),
        )
        .expect("valid input");

        assert_eq!(plan.action, RockAction::East, "seed {seed}");
    }
}

/// The same seed gives the same plan, bit for bit.
#[test]
fn pomcp_is_deterministic() {
    let first = pomcp(
        &Tiger,
        &tiger_belief(1, 2),
        UniformRandom,
        pomcp_config(7, 110.0, 2_000),
    );
    let second = pomcp(
        &Tiger,
        &tiger_belief(1, 2),
        UniformRandom,
        pomcp_config(7, 110.0, 2_000),
    );

    let (first, second) = (first.expect("valid"), second.expect("valid"));
    assert_eq!(first.action, second.action);
    assert_eq!(first.value.to_bits(), second.value.to_bits());
    assert_eq!(first.stats, second.stats);
}

/// Empty beliefs and bad settings are errors, not panics.
#[test]
fn pomcp_validates_its_input() {
    let empty: [bool; 0] = [];
    assert_eq!(
        pomcp(&Tiger, &empty, UniformRandom, PomcpConfig::default()),
        Err(Error::EmptyBelief)
    );

    for config in [
        PomcpConfig {
            discount: 0.0,
            ..PomcpConfig::default()
        },
        PomcpConfig {
            exploration: f64::NAN,
            ..PomcpConfig::default()
        },
        PomcpConfig {
            max_depth: 0,
            ..PomcpConfig::default()
        },
    ] {
        assert!(matches!(
            pomcp(&Tiger, &[true], UniformRandom, config),
            Err(Error::InvalidConfig { .. })
        ));
    }
}
