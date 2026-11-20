//! Belief tracking between decisions: POMCP's particle filter with reinvigoration
//! ([`update_belief`]).

use alloc::vec::Vec;

use rand_core::Rng;

use crate::error::Error;
use crate::pomdp::{Pomdp, below, check_reward};

/// Settings of [`update_belief`].
///
/// The struct is `#[non_exhaustive]`: outside this crate, build it from `default()` with
/// the `with_*` methods; fields stay readable and assignable.
///
/// ```
/// use stochastic_search::BeliefFilter;
///
/// let filter = BeliefFilter::default().with_particles(4_000).with_reinvigorate(250);
/// # let _ = filter;
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BeliefFilter {
    /// The number `K` of particles the rejection phase aims for. At least `1`. Default
    /// `1_000`.
    pub particles: usize,
    /// The most simulated steps the rejection phase may take before it gives up on
    /// reaching `particles`. At least `1`. Default `100_000`.
    pub max_attempts: usize,
    /// Particles added by reinvigoration on top of the accepted ones (the paper adds `n / 16`
    /// for `n` simulations per decision). Default `0`.
    pub reinvigorate: usize,
}

impl BeliefFilter {
    /// Returns the same filter with another [`BeliefFilter::particles`].
    #[must_use]
    pub fn with_particles(self, particles: usize) -> Self {
        Self { particles, ..self }
    }

    /// Returns the same filter with another [`BeliefFilter::max_attempts`].
    #[must_use]
    pub fn with_max_attempts(self, max_attempts: usize) -> Self {
        Self {
            max_attempts,
            ..self
        }
    }

    /// Returns the same filter with another [`BeliefFilter::reinvigorate`].
    #[must_use]
    pub fn with_reinvigorate(self, reinvigorate: usize) -> Self {
        Self {
            reinvigorate,
            ..self
        }
    }
}

impl Default for BeliefFilter {
    fn default() -> Self {
        Self {
            particles: 1_000,
            max_attempts: 100_000,
            reinvigorate: 0,
        }
    }
}

/// The posterior particles of [`update_belief`] and how they came about.
#[derive(Debug, Clone, PartialEq)]
pub struct BeliefUpdate<S> {
    /// The new belief: the accepted successors, then the reinvigorated particles. Never
    /// empty.
    pub particles: Vec<S>,
    /// How many particles the rejection phase accepted, at most
    /// [`BeliefFilter::particles`].
    pub accepted: usize,
    /// How many steps the rejection phase simulated.
    pub attempts: usize,
    /// How many particles came from the reinvigoration transform: the shortfall of the
    /// rejection phase plus [`BeliefFilter::reinvigorate`].
    pub reinvigorated: usize,
}

/// Updates the particle belief `belief` after taking `action` and receiving `observation`,
/// with the Monte-Carlo particle filter of POMCP (Silver and Veness 2010, Section 3.2).
///
/// The rejection phase draws a particle uniformly from `belief`, simulates `action` with
/// [`Pomdp::step`] and keeps the successor if the step is not terminal and its observation
/// equals `observation`, until [`BeliefFilter::particles`] are kept or
/// [`BeliefFilter::max_attempts`] steps are spent. Terminal successors are rejected: an
/// episode that ended has no next decision.
///
/// Reinvigoration then fights particle deprivation: it applies `transform` to accepted
/// particles drawn uniformly, once for every particle the rejection phase fell short of
/// and [`BeliefFilter::reinvigorate`] more times. The paper leaves `transform` to the
/// domain ("a domain specific local transformation to existing particles", such as adding
/// artificial noise); states are opaque here, so it is the caller's. It should return a state
/// that is plausible given the observations so far; that is not checked. `|s, _| s.clone()`
/// just duplicates particles (plain resampling).
///
/// All randomness comes from `rng`, so a seeded generator reproduces an update.
///
/// # Errors
///
/// Returns [`Error::EmptyBelief`] for an empty `belief`, [`Error::BeliefDepleted`] if no
/// particle was accepted (the observation is impossible under the belief, or too rare for
/// the attempt budget), [`Error::InvalidReward`] for a non-finite reward and
/// [`Error::InvalidConfig`] for zero `particles` or `max_attempts`.
///
/// # Example
///
/// The hidden coin from [`Pomdp`]: after peeking and seeing heads, about 90% of the
/// particles are heads.
///
/// ```
/// # use rand_core::Rng;
/// # use stochastic_search::{Pomdp, Step, unit_interval};
/// # struct Coin;
/// # impl Pomdp for Coin {
/// #     type State = bool;
/// #     type Action = &'static str;
/// #     type Observation = Option<bool>;
/// #     fn actions(&self, mut emit: impl FnMut(&'static str)) { emit("peek"); emit("heads"); emit("tails"); }
/// #     fn step<R: Rng + ?Sized>(&self, heads: &bool, action: &&'static str, rng: &mut R)
/// #         -> Step<bool, Option<bool>> {
/// #         if *action == "peek" {
/// #             let truthful = unit_interval(rng) < 0.9;
/// #             Step::new(*heads, Some(*heads == truthful), -1.0)
/// #         } else {
/// #             let right = *heads == (*action == "heads");
/// #             Step::new(*heads, None, if right { 10.0 } else { -10.0 }).terminal()
/// #         }
/// #     }
/// # }
/// use stochastic_search::{BeliefFilter, SplitMix64, update_belief};
///
/// let mut rng = SplitMix64::new(7);
/// let filter = BeliefFilter::default().with_particles(10_000);
/// let update = update_belief(&Coin, &[true, false], &"peek", &Some(true), filter, &mut rng, |s, _| *s)?;
///
/// let heads = update.particles.iter().filter(|&&h| h).count();
/// assert!((heads as f64 / 10_000.0 - 0.9).abs() < 0.02);
/// # Ok::<(), stochastic_search::Error<bool>>(())
/// ```
pub fn update_belief<P, R, F>(
    pomdp: &P,
    belief: &[P::State],
    action: &P::Action,
    observation: &P::Observation,
    filter: BeliefFilter,
    rng: &mut R,
    mut transform: F,
) -> Result<BeliefUpdate<P::State>, Error<P::State>>
where
    P: Pomdp,
    P::State: Clone,
    P::Observation: PartialEq,
    R: Rng + ?Sized,
    F: FnMut(&P::State, &mut R) -> P::State,
{
    if filter.particles == 0 || filter.max_attempts == 0 {
        return Err(Error::InvalidConfig {
            reason: "the belief filter needs at least one particle and one attempt",
        });
    }

    if belief.is_empty() {
        return Err(Error::EmptyBelief);
    }

    let (mut particles, attempts) = reject(pomdp, belief, action, observation, filter, rng)?;
    let accepted = particles.len();
    if accepted == 0 {
        return Err(Error::BeliefDepleted { attempts });
    }

    let reinvigorated = filter.particles - accepted + filter.reinvigorate;
    for _ in 0..reinvigorated {
        let source = &particles[below(rng, accepted)];
        let particle = transform(source, rng);
        particles.push(particle);
    }

    Ok(BeliefUpdate {
        particles,
        accepted,
        attempts,
        reinvigorated,
    })
}

/// The accepted particles of a rejection phase and the number of steps it simulated.
type Accepted<S> = (Vec<S>, usize);

/// The rejection phase of [`update_belief`]: the accepted successors and the number of
/// steps simulated.
///
/// # Errors
///
/// Returns [`Error::InvalidReward`] for a non-finite reward.
fn reject<P, R>(
    pomdp: &P,
    belief: &[P::State],
    action: &P::Action,
    observation: &P::Observation,
    filter: BeliefFilter,
    rng: &mut R,
) -> Result<Accepted<P::State>, Error<P::State>>
where
    P: Pomdp,
    P::State: Clone,
    P::Observation: PartialEq,
    R: Rng + ?Sized,
{
    let mut particles = Vec::with_capacity(filter.particles + filter.reinvigorate);
    let mut attempts = 0;
    while particles.len() < filter.particles && attempts < filter.max_attempts {
        attempts += 1;
        let state = &belief[below(rng, belief.len())];
        let step = pomdp.step(state, action, rng);
        check_reward(state, step.reward)?;
        if !step.terminal && step.observation == *observation {
            particles.push(step.state);
        }
    }

    Ok((particles, attempts))
}
