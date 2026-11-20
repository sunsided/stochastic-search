//! The particle filter against Tiger's exact posterior, its reinvigoration and its errors.

use stochastic_search::{BeliefFilter, Error, SplitMix64, update_belief};

use crate::pomdps::{Tiger, TigerAction, tiger_belief};

/// The fraction of particles that put the tiger on the left.
fn left_fraction(particles: &[bool]) -> f64 {
    let left = particles.iter().filter(|&&left| left).count();

    f64::from(u32::try_from(left).expect("small"))
        / f64::from(u32::try_from(particles.len()).expect("small"))
}

/// After hearing the tiger on the left once, the exact posterior is 0.85; twice, it is
/// `0.85^2 / (0.85^2 + 0.15^2) = 0.9698`.
#[test]
fn filter_matches_the_exact_tiger_posterior() {
    let mut rng = SplitMix64::new(3);
    let filter = BeliefFilter::default()
        .with_particles(40_000)
        .with_max_attempts(1_000_000);
    let listen = |belief: &[bool], rng: &mut SplitMix64| {
        update_belief(
            &Tiger,
            belief,
            &TigerAction::Listen,
            &Some(true),
            filter,
            rng,
            |s, _| *s,
        )
        .expect("the observation is possible")
    };

    let once = listen(&tiger_belief(1, 2), &mut rng);
    assert_eq!(once.accepted, 40_000);
    assert_eq!(once.reinvigorated, 0);
    assert!((left_fraction(&once.particles) - 0.85).abs() < 0.01);

    let twice = listen(&once.particles, &mut rng);
    let exact = 0.85_f64.powi(2) / (0.85_f64.powi(2) + 0.15_f64.powi(2));
    assert!((left_fraction(&twice.particles) - exact).abs() < 0.005);
}

/// A short attempt budget leaves a shortfall, which reinvigoration fills, plus the extra
/// particles asked for; the transform sees accepted particles only.
#[test]
fn reinvigoration_fills_the_shortfall() {
    let mut rng = SplitMix64::new(5);
    let filter = BeliefFilter::default()
        .with_particles(100)
        .with_max_attempts(50)
        .with_reinvigorate(10);
    let mut transformed = 0;
    let update = update_belief(
        &Tiger,
        &tiger_belief(1, 2),
        &TigerAction::Listen,
        &Some(false),
        filter,
        &mut rng,
        |s, _| {
            transformed += 1;
            *s
        },
    )
    .expect("some particle is accepted");

    assert_eq!(update.attempts, 50);
    assert!(update.accepted < 50);
    assert_eq!(update.reinvigorated, 100 - update.accepted + 10);
    assert_eq!(transformed, update.reinvigorated);
    assert_eq!(update.particles.len(), 110);
}

/// An impossible observation depletes the belief, which is reported; empty beliefs and
/// zero sizes are errors too.
#[test]
fn filter_reports_depletion_and_bad_input() {
    let mut rng = SplitMix64::new(1);
    let filter = BeliefFilter::default().with_max_attempts(200);
    let keep = |s: &bool, _: &mut SplitMix64| *s;

    // Opening a door always observes `None`.
    let depleted = update_belief(
        &Tiger,
        &tiger_belief(1, 2),
        &TigerAction::OpenLeft,
        &Some(true),
        filter,
        &mut rng,
        keep,
    );
    assert_eq!(depleted, Err(Error::BeliefDepleted { attempts: 200 }));

    let empty = update_belief(
        &Tiger,
        &[],
        &TigerAction::Listen,
        &Some(true),
        filter,
        &mut rng,
        keep,
    );
    assert_eq!(empty, Err(Error::EmptyBelief));

    let zero = filter.with_particles(0);
    let invalid = update_belief(
        &Tiger,
        &tiger_belief(1, 2),
        &TigerAction::Listen,
        &Some(true),
        zero,
        &mut rng,
        keep,
    );
    assert!(matches!(invalid, Err(Error::InvalidConfig { .. })));
}

/// A POMDP whose only step ends the episode with the observation `()`.
struct OneShot;

impl stochastic_search::Pomdp for OneShot {
    type State = u8;
    type Action = ();
    type Observation = ();

    fn actions(&self, mut emit: impl FnMut(())) {
        emit(());
    }

    fn step<R: rand_core::Rng + ?Sized>(
        &self,
        state: &u8,
        _action: &(),
        _rng: &mut R,
    ) -> stochastic_search::Step<u8, ()> {
        stochastic_search::Step::new(*state, (), 0.0).terminal()
    }
}

/// Terminal successors are rejected even when the observation matches: an episode that
/// ended has no next decision.
#[test]
fn filter_rejects_terminal_successors() {
    let mut rng = SplitMix64::new(2);
    let filter = BeliefFilter::default().with_max_attempts(10);
    let update = update_belief(&OneShot, &[1, 2], &(), &(), filter, &mut rng, |s, _| *s);

    assert_eq!(update, Err(Error::BeliefDepleted { attempts: 10 }));
}
