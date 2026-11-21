//! Property tests for the belief-space side: the particle filter's counts and POMCP's and
//! DESPOT's plans over random beliefs, seeds and budgets on Tiger.

use proptest::prelude::*;
use stochastic_search::{
    BeliefFilter, DespotConfig, Error, Limits, PomcpConfig, SplitMix64, despot, pomcp,
    update_belief,
};

use crate::pomdps::{AlwaysListen, DISCOUNT, TIGER_UPPER_BOUND, Tiger, TigerAction, tiger_belief};

/// The lowest discounted return Tiger allows: -100 on every step.
const TIGER_LOWER_BOUND: f64 = -100.0 / (1.0 - DISCOUNT);

/// Slack for floating-point rounding in the value bounds.
const ROUNDING: f64 = 1e-9;

/// Returns `true` if `value` is a discounted Tiger return: finite and within the bounds that
/// the rewards -100 and +10 allow.
fn is_tiger_return(value: f64) -> bool {
    (TIGER_LOWER_BOUND - ROUNDING..=TIGER_UPPER_BOUND + ROUNDING).contains(&value)
}

proptest! {
    /// Listening never moves the tiger and the transform keeps states, so every particle of
    /// the new belief is a state of the old one. The counts add up: the accepted particles
    /// fill the requested number, reinvigoration covers the shortfall plus the extra
    /// particles, and the rejection phase stays within its attempt budget. Depletion is the
    /// only error, and only when the budget is too small to accept anything.
    #[test]
    fn belief_update_counts_add_up(
        left in 0_usize..=20,
        total in 1_usize..=20,
        particles in 1_usize..200,
        reinvigorate in 0_usize..20,
        max_attempts in 1_usize..2_000,
        heard_left in any::<bool>(),
        seed in any::<u64>(),
    ) {
        let belief = tiger_belief(left.min(total), total);
        let filter = BeliefFilter::default()
            .with_particles(particles)
            .with_max_attempts(max_attempts)
            .with_reinvigorate(reinvigorate);
        let mut rng = SplitMix64::new(seed);

        let result = update_belief(
            &Tiger,
            &belief,
            &TigerAction::Listen,
            &Some(heard_left),
            filter,
            &mut rng,
            |state, _| *state,
        );

        match result {
            Ok(update) => {
                prop_assert!(update.accepted >= 1 && update.accepted <= particles);
                prop_assert!(update.attempts <= max_attempts);
                prop_assert_eq!(update.reinvigorated, particles - update.accepted + reinvigorate);
                prop_assert_eq!(update.particles.len(), particles + reinvigorate);
                prop_assert!(update.particles.iter().all(|state| belief.contains(state)));
            }
            Err(Error::BeliefDepleted { attempts }) => prop_assert!(attempts <= max_attempts),
            Err(other) => prop_assert!(false, "unexpected error {other:?}"),
        }
    }

    /// POMCP runs exactly its simulation budget and returns a plan whose value is a possible
    /// discounted return.
    #[test]
    fn pomcp_plans_stay_within_the_returns(
        left in 0_usize..=10,
        simulations in 1_u64..300,
        seed in any::<u64>(),
    ) {
        let config = PomcpConfig::default()
            .with_discount(DISCOUNT)
            .with_exploration(110.0)
            .with_max_depth(30)
            .with_seed(seed)
            .with_budget(Limits {
                max_trials: Some(simulations),
                ..PomcpConfig::DEFAULT_LIMITS
            });

        let plan = pomcp(&Tiger, &tiger_belief(left, 10), AlwaysListen, config)
            .expect("valid input");

        prop_assert_eq!(plan.stats.simulations, simulations);
        prop_assert!(is_tiger_return(plan.value), "value {}", plan.value);
    }

    /// DESPOT stays within its exploration budget and returns a plan whose value is a
    /// possible discounted return.
    #[test]
    fn despot_plans_stay_within_the_returns(
        left in 0_usize..=10,
        explorations in 1_u64..100,
        scenarios in 1_usize..50,
        seed in any::<u64>(),
    ) {
        let config = DespotConfig::default()
            .with_discount(DISCOUNT)
            .with_scenarios(scenarios)
            .with_max_depth(20)
            .with_seed(seed)
            .with_budget(Limits {
                max_trials: Some(explorations),
                ..DespotConfig::DEFAULT_LIMITS
            });

        let plan = despot(
            &Tiger,
            &tiger_belief(left, 10),
            AlwaysListen,
            |_: &bool| TIGER_UPPER_BOUND,
            config,
        )
        .expect("valid input");

        prop_assert!(plan.stats.simulations <= explorations);
        prop_assert!(is_tiger_return(plan.value), "value {}", plan.value);
    }
}
