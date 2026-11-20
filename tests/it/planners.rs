//! POMCP and DESPOT on Tiger and a one-rock `RockSample`: the known-good action over
//! several seeds.
//!
//! The expected actions are derived by hand in each test's docs, with their margins, so
//! they do not depend on the planner under test. The rollout policies are the usual
//! domain defaults (always listen, head east); uniformly random rollouts drown Tiger's
//! first-step difference in noise.

use stochastic_search::{
    DespotConfig, Error, Limits, Plan, PomcpConfig, Status, UniformRandom, despot, pomcp,
};

use crate::pomdps::{
    AlwaysListen, DISCOUNT, HeadEast, RockAction, TIGER_UPPER_BOUND, Tiger, TigerAction, one_rock,
    one_rock_belief, rock_upper_bound, tiger_belief,
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
        assert_eq!(plan.stats.simulations, 20_000);
        assert!(plan.stats.steps >= plan.stats.simulations);
        assert!(plan.stats.depth >= 1 && plan.stats.nodes > 1);
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

/// A DESPOT configuration with `explorations` explorations.
fn despot_config(seed: u64, explorations: u64) -> DespotConfig {
    DespotConfig {
        discount: DISCOUNT,
        scenarios: 200,
        max_depth: 40,
        seed,
        ..DespotConfig::default()
    }
    .with_budget(Limits {
        max_trials: Some(explorations),
        ..DespotConfig::DEFAULT_LIMITS
    })
}

/// Plans for Tiger from a belief with `left` of `total` particles on the left.
fn despot_tiger(left: usize, total: usize, seed: u64) -> Plan<TigerAction> {
    despot(
        &Tiger,
        &tiger_belief(left, total),
        AlwaysListen,
        |_: &bool| TIGER_UPPER_BOUND,
        despot_config(seed, 2_000),
    )
    .expect("valid input")
}

/// As [`pomcp_listens_to_the_tiger_first`], for DESPOT.
#[test]
fn despot_listens_to_the_tiger_first() {
    for seed in SEEDS {
        let plan = despot_tiger(1, 2, seed);
        assert_eq!(plan.action, TigerAction::Listen, "seed {seed}");
        assert!(plan.stats.steps > plan.stats.nodes && plan.stats.depth >= 1);
    }
}

/// As [`pomcp_opens_the_far_door_when_confident`], for DESPOT. The recommendation goes by
/// lower bounds, and opening resets the problem into a subtree that needs explorations of
/// its own before its lower bound beats listening; 500 explorations are not always enough.
#[test]
fn despot_opens_the_far_door_when_confident() {
    for seed in SEEDS {
        let plan = despot_tiger(199, 200, seed);
        assert_eq!(plan.action, TigerAction::OpenRight, "seed {seed}");
    }
}

/// As [`pomcp_checks_an_uncertain_rock`], for DESPOT.
#[test]
fn despot_checks_an_uncertain_rock() {
    let model = one_rock();
    for seed in SEEDS {
        let plan = despot(
            &model,
            &one_rock_belief(50, 100),
            HeadEast,
            rock_upper_bound,
            despot_config(seed, 500),
        )
        .expect("valid input");

        assert_eq!(plan.action, RockAction::Check(0), "seed {seed}");
        assert!(
            (plan.value - 13.1).abs() < 0.6,
            "seed {seed}: {}",
            plan.value
        );
    }
}

/// On a rock known to be good, sampling at once and heading east earns
/// `10 + 10 * gamma^3 = 18.57`; checking first only delays it. The scenarios are
/// deterministic, so DESPOT closes the gap and reports that value exactly.
#[test]
fn despot_samples_a_good_rock() {
    let model = one_rock();
    for seed in SEEDS {
        let plan = despot(
            &model,
            &one_rock_belief(10, 10),
            HeadEast,
            rock_upper_bound,
            despot_config(seed, 500),
        )
        .expect("valid input");

        assert_eq!(plan.action, RockAction::Sample, "seed {seed}");
        assert_eq!(plan.status, Status::Converged, "seed {seed}");
        assert!((plan.value - (10.0 + 10.0 * DISCOUNT.powi(3))).abs() < 1e-9);
    }
}

/// As [`pomcp_leaves_a_bad_rock`], for DESPOT. The default policy already heads east, so
/// the bounds close quickly and the search converges.
#[test]
fn despot_leaves_a_bad_rock() {
    let model = one_rock();
    for seed in SEEDS {
        let plan = despot(
            &model,
            &one_rock_belief(0, 10),
            HeadEast,
            rock_upper_bound,
            despot_config(seed, 500),
        )
        .expect("valid input");

        assert_eq!(plan.action, RockAction::East, "seed {seed}");
        assert_eq!(plan.status, Status::Converged, "seed {seed}");
        assert!((plan.value - 10.0 * DISCOUNT.powi(2)).abs() < 1e-9);
    }
}

/// The same seed gives the same plan, bit for bit.
#[test]
fn despot_is_deterministic() {
    let (first, second) = (despot_tiger(1, 2, 5), despot_tiger(1, 2, 5));

    assert_eq!(first.action, second.action);
    assert_eq!(first.value.to_bits(), second.value.to_bits());
    assert_eq!(first.stats, second.stats);
}

/// A regularization that no gap can pay for leaves a one-node tree and the default
/// policy's action.
#[test]
fn despot_with_huge_regularization_follows_the_default_policy() {
    let config = DespotConfig {
        regularization: 1e9,
        ..despot_config(0, 500)
    };
    let plan = despot(
        &Tiger,
        &tiger_belief(1, 2),
        AlwaysListen,
        |_: &bool| TIGER_UPPER_BOUND,
        config,
    )
    .expect("valid input");

    assert_eq!(plan.action, TigerAction::Listen);
    assert_eq!(plan.status, Status::Converged);
    assert_eq!(plan.stats.simulations, 0);
    assert_eq!(plan.stats.nodes, 1);
}

/// Bad settings, empty beliefs and NaN bounds are errors, not panics.
#[test]
fn despot_validates_its_input() {
    let bound = |_: &bool| TIGER_UPPER_BOUND;
    let empty: [bool; 0] = [];
    assert_eq!(
        despot(
            &Tiger,
            &empty,
            UniformRandom,
            bound,
            DespotConfig::default()
        ),
        Err(Error::EmptyBelief)
    );
    assert!(matches!(
        despot(
            &Tiger,
            &[true],
            UniformRandom,
            |_: &bool| f64::NAN,
            DespotConfig::default()
        ),
        Err(Error::InvalidHeuristic { .. })
    ));

    for config in [
        DespotConfig {
            scenarios: 0,
            ..DespotConfig::default()
        },
        DespotConfig {
            regularization: -1.0,
            ..DespotConfig::default()
        },
        DespotConfig {
            xi: 1.0,
            ..DespotConfig::default()
        },
        DespotConfig {
            target_gap: f64::NAN,
            ..DespotConfig::default()
        },
        DespotConfig {
            discount: 1.5,
            ..DespotConfig::default()
        },
    ] {
        assert!(matches!(
            despot(&Tiger, &[true], UniformRandom, bound, config),
            Err(Error::InvalidConfig { .. })
        ));
    }
}

/// Pins one POMCP and one DESPOT plan to the bits of the one-shot planners as first
/// released, so refactors of the planners (tree reuse, counters, events) keep them.
#[test]
fn planners_keep_their_pinned_values() {
    let plan = pomcp(
        &Tiger,
        &tiger_belief(1, 2),
        UniformRandom,
        pomcp_config(7, 110.0, 2_000),
    )
    .expect("valid input");
    assert_eq!(plan.action, TigerAction::Listen);
    assert_eq!(plan.value.to_bits(), 0xc078_f152_1e03_7afb);

    let plan = despot_tiger(1, 2, 5);
    assert_eq!(plan.action, TigerAction::Listen);
    assert_eq!(plan.value.to_bits(), 0x403a_ce5a_27f6_1342);
}
