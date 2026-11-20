//! POMCP and DESPOT on Tiger and a one-rock `RockSample`: the known-good action over
//! several seeds.
//!
//! The expected actions are derived by hand in each test's docs, with their margins, so
//! they do not depend on the planner under test. The rollout policies are the usual
//! domain defaults (always listen, head east); uniformly random rollouts drown Tiger's
//! first-step difference in noise.

use stochastic_search::{
    Despot, DespotConfig, Error, Event, Limits, Plan, Pomcp, PomcpConfig, Status, UniformRandom,
    despot, observer_fn, pomcp, pomcp_exploration,
};

use crate::pomdps::{
    AlwaysListen, DISCOUNT, HeadEast, RockAction, TIGER_UPPER_BOUND, Tiger, TigerAction, one_rock,
    one_rock_belief, rock_upper_bound, tiger_belief,
};

/// The seeds every planner test runs with.
const SEEDS: [u64; 4] = [0, 1, 2, 3];

/// A POMCP configuration for returns spanning `range`, with `simulations` simulations.
fn pomcp_config(seed: u64, range: f64, simulations: u64) -> PomcpConfig {
    PomcpConfig::default()
        .with_discount(DISCOUNT)
        .with_exploration(range)
        .with_max_depth(60)
        .with_seed(seed)
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
        PomcpConfig::default().with_discount(0.0),
        PomcpConfig::default().with_exploration(f64::NAN),
        PomcpConfig::default().with_max_depth(0),
    ] {
        assert!(matches!(
            pomcp(&Tiger, &[true], UniformRandom, config),
            Err(Error::InvalidConfig { .. })
        ));
    }
}

/// A DESPOT configuration with `explorations` explorations.
fn despot_config(seed: u64, explorations: u64) -> DespotConfig {
    DespotConfig::default()
        .with_discount(DISCOUNT)
        .with_scenarios(200)
        .with_max_depth(40)
        .with_seed(seed)
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
    let config = despot_config(0, 500).with_regularization(1e9);
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
        DespotConfig::default().with_scenarios(0),
        DespotConfig::default().with_regularization(-1.0),
        DespotConfig::default().with_xi(1.0),
        DespotConfig::default().with_target_gap(f64::NAN),
        DespotConfig::default().with_discount(1.5),
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

/// POMCP and DESPOT report progress every `progress_interval` simulations, and an
/// observer changes nothing about the plan.
#[test]
fn tree_planners_report_progress_without_changing_the_plan() {
    let mut events = Vec::new();
    let config = pomcp_config(7, 110.0, 2_000);
    let observed = pomcp(
        &Tiger,
        &tiger_belief(1, 2),
        UniformRandom,
        config
            .with_progress_interval(Some(500))
            .with_observer(observer_fn(|event| events.push(event))),
    )
    .expect("valid input");
    let plain = pomcp(&Tiger, &tiger_belief(1, 2), UniformRandom, config).expect("valid input");

    assert_eq!(observed.value.to_bits(), plain.value.to_bits());
    assert_eq!(observed.stats, plain.stats);
    let simulations: Vec<u64> = events
        .iter()
        .map(|event| match event {
            Event::TreeProgress { stats, gap, .. } => {
                assert!(gap.is_none());
                stats.simulations
            }
            _ => panic!("unexpected event {event:?}"),
        })
        .collect();
    assert_eq!(simulations, [500, 1_000, 1_500, 2_000]);

    let mut gaps = Vec::new();
    let config = despot_config(5, 300).with_progress_interval(Some(100));
    let observed = despot(
        &Tiger,
        &tiger_belief(1, 2),
        AlwaysListen,
        |_: &bool| TIGER_UPPER_BOUND,
        config.with_observer(observer_fn(|event| {
            if let Event::TreeProgress { gap: Some(gap), .. } = event {
                gaps.push(gap);
            }
        })),
    )
    .expect("valid input");
    let plain = despot(
        &Tiger,
        &tiger_belief(1, 2),
        AlwaysListen,
        |_: &bool| TIGER_UPPER_BOUND,
        config,
    )
    .expect("valid input");

    assert_eq!(observed.value.to_bits(), plain.value.to_bits());
    assert_eq!(gaps.len(), 3);
    assert!(gaps.windows(2).all(|pair| pair[1] <= pair[0] + 1e-9));
}

/// A kept planner reuses the subtree after listening and hearing the tiger on the left.
/// With no or one net "left" observation, listening is clearly best (the belief is at most
/// 0.85 sure); after three, opening the right door is (see
/// [`pomcp_opens_the_far_door_when_confident`]). After two, the two are close, so that
/// decision is not checked. An unseen observation starts over with a fresh tree.
#[test]
fn pomcp_reuses_its_tree_between_decisions() {
    for seed in SEEDS {
        let mut planner =
            Pomcp::new(&Tiger, AlwaysListen, pomcp_config(seed, 110.0, 20_000)).expect("valid");
        let mut belief = tiger_belief(1, 2);
        for heard in 0..3 {
            if heard < 2 {
                let plan = planner.plan(&belief).expect("valid input");
                assert_eq!(
                    plan.action,
                    TigerAction::Listen,
                    "seed {seed}, step {heard}"
                );
                assert!(planner.advance(&TigerAction::Listen, &Some(true)));
                assert!(planner.tree_size() > 1, "the subtree is kept");
            } else {
                planner.advance(&TigerAction::Listen, &Some(true));
            }

            // The exact posterior after one more "left": odds times 0.85 / 0.15.
            let left = belief.iter().filter(|&&l| l).count() * 85;
            let right = (belief.len() - belief.iter().filter(|&&l| l).count()) * 15;
            belief = tiger_belief(left, left + right);
        }

        let plan = planner.plan(&belief).expect("valid input");
        assert_eq!(plan.action, TigerAction::OpenRight, "seed {seed}");
    }

    let mut planner = Pomcp::new(&Tiger, AlwaysListen, pomcp_config(0, 110.0, 100)).expect("valid");
    planner.plan(&tiger_belief(1, 2)).expect("valid input");
    assert!(!planner.advance(&TigerAction::OpenLeft, &Some(true)));
    assert_eq!(planner.tree_size(), 1);
}

/// The paper's exploration constant, measured on Tiger, keeps the known first action, and
/// the reward-range bound is the closed form.
#[test]
fn pomcp_exploration_constants() {
    for seed in SEEDS {
        let base = pomcp_config(seed, 0.0, 20_000);
        let c = pomcp_exploration(&Tiger, &tiger_belief(1, 2), AlwaysListen, &base, 1_000)
            .expect("valid input");
        assert!(c > 0.0 && c.is_finite(), "seed {seed}: {c}");

        let plan = pomcp(
            &Tiger,
            &tiger_belief(1, 2),
            AlwaysListen,
            base.with_exploration(c),
        )
        .expect("valid input");
        assert_eq!(plan.action, TigerAction::Listen, "seed {seed}");
    }

    let config = PomcpConfig::default()
        .with_discount(0.5)
        .with_max_depth(3)
        .with_reward_range(-100.0, 10.0);
    assert!((config.exploration - 110.0 * 1.75).abs() < 1e-12);
    assert!(matches!(
        pomcp_exploration(&Tiger, &tiger_belief(1, 2), AlwaysListen, &config, 0),
        Err(Error::InvalidConfig { .. })
    ));
}

/// A kept DESPOT planner moves its root after listening and hearing the tiger on the left:
/// the new root's scenarios are the ones that heard it there (about 85% with the tiger on
/// the left), and it still listens with one net observation.
#[test]
fn despot_reuses_its_tree_between_decisions() {
    for seed in SEEDS {
        let mut planner = Despot::new(
            &Tiger,
            &tiger_belief(1, 2),
            AlwaysListen,
            |_: &bool| TIGER_UPPER_BOUND,
            despot_config(seed, 2_000),
        )
        .expect("valid input");
        assert_eq!(planner.plan().expect("valid").action, TigerAction::Listen);

        planner
            .advance(&TigerAction::Listen, &Some(true))
            .expect("a seen history");
        let belief: Vec<bool> = planner.belief().copied().collect();
        let left = belief.iter().filter(|&&left| left).count();
        assert!(
            belief.len() < 200 && left * 100 > belief.len() * 75,
            "seed {seed}"
        );
        assert_eq!(
            planner.plan().expect("valid").action,
            TigerAction::Listen,
            "seed {seed}"
        );
    }
}
