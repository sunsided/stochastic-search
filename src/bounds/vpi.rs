//! VPI-RTDP (Sanner, Goetschalckx, Driessens and Shani 2009): successor choice by the
//! myopic value of perfect information about a successor's value.

use core::hash::Hash;

use alloc::vec::Vec;

use crate::bounds::{Explorer, Run, gap, solve};
use crate::budget::{Budget, Limits, Stats};
use crate::config::Config;
use crate::error::Error;
use crate::model::{Goal, Heuristic, Mdp};
use crate::observer::Observer;
use crate::solution::SolveResult;
use crate::space::Id;

/// Settings of [`vpi_rtdp`]: the shared [`Config`] and VPI-RTDP's two thresholds.
///
/// Build one with [`VpiRtdpConfig::default`], or with [`VpiRtdpConfig::new`] (or `into`)
/// from a [`Config`], and adjust it with [`VpiRtdpConfig::with_alpha`] and
/// [`VpiRtdpConfig::with_beta`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VpiRtdpConfig<B = Limits, O = ()> {
    /// Epsilon (the gap that ends the run), dead ends, seed, trial length, budget and
    /// observer.
    pub base: Config<B, O>,
    /// The probability `alpha` of continuing a trial by gap-weighted sampling where every
    /// successor has zero value of information. In `[0, 1]`; above `0`, every state BRTDP
    /// could update keeps a positive probability of an update, which is what the paper's
    /// convergence argument needs. Default `0.001`, as in the paper.
    pub alpha: f64,
    /// The fraction `beta` of the initial state's initial gap above which a successor's
    /// gap counts as "close to maximal uncertainty": if any successor's gap exceeds it,
    /// the trial samples by gap as BRTDP does, since the value of information is
    /// uninformative there. In `(0, 1]`. Default `0.95`, the paper's "95% of the maximum
    /// possible bound", read here as relative to the initial gap of `initial`.
    pub beta: f64,
    /// BRTDP's trial-end ratio, used where every successor has zero value of information:
    /// the trial still continues by gap while the expected successor gap is at least the
    /// initial state's gap divided by `tau`. Finite and greater than `1`. Default `10`, as
    /// for [`BrtdpConfig`](crate::BrtdpConfig). Not in the paper; see [`vpi_rtdp`].
    pub tau: f64,
}

impl Default for VpiRtdpConfig {
    fn default() -> Self {
        Self::new(Config::default())
    }
}

impl<B, O> VpiRtdpConfig<B, O> {
    /// Wraps `base` with the paper's thresholds, `alpha = 0.001` and `beta = 0.95`, and
    /// `tau = 10`.
    #[must_use]
    pub fn new(base: Config<B, O>) -> Self {
        Self {
            base,
            alpha: 0.001,
            beta: 0.95,
            tau: crate::BrtdpConfig::DEFAULT_TAU,
        }
    }

    /// Returns the same settings with another `tau`.
    #[must_use]
    pub fn with_tau(self, tau: f64) -> Self {
        Self { tau, ..self }
    }

    /// Returns the same settings with another `alpha`.
    #[must_use]
    pub fn with_alpha(self, alpha: f64) -> Self {
        Self { alpha, ..self }
    }

    /// Returns the same settings with another `beta`.
    #[must_use]
    pub fn with_beta(self, beta: f64) -> Self {
        Self { beta, ..self }
    }

    /// Checks the thresholds.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfig`] for an `alpha` outside `[0, 1]`, a `beta` outside
    /// `(0, 1]` or a `tau` that is not finite and greater than 1.
    fn validate<S>(&self) -> Result<(), Error<S>> {
        if !(0.0..=1.0).contains(&self.alpha) {
            return Err(Error::InvalidConfig {
                reason: "alpha must lie in [0, 1]",
            });
        }

        if !(self.beta > 0.0 && self.beta <= 1.0) {
            return Err(Error::InvalidConfig {
                reason: "beta must lie in (0, 1]",
            });
        }

        if !(self.tau.is_finite() && self.tau > 1.0) {
            return Err(Error::InvalidConfig {
                reason: "tau must be finite and greater than 1",
            });
        }

        Ok(())
    }
}

impl<B, O> From<Config<B, O>> for VpiRtdpConfig<B, O> {
    fn from(base: Config<B, O>) -> Self {
        Self::new(base)
    }
}

/// Solves an SSP from `initial` with VPI-RTDP, keeping a lower and an upper bound on every
/// stored state's optimal value.
///
/// VPI-RTDP (Sanner, Goetschalckx, Driessens and Shani 2009) differs from
/// [`brtdp`](crate::brtdp()) only in how trials pick states; bounds, dead-end handling,
/// guarantee, policy (greedy on the upper bound, with the fill pass after convergence),
/// events and errors are as documented there.
///
/// After backing up `s`, with `a*` the lower-greedy action, the trial picks a successor `t`:
///
/// 1. If some successor's gap exceeds `beta` times the initial gap of `initial`, it samples
///    `t` by `P(t) * gap(t)` as BRTDP does.
/// 2. Otherwise it computes, for every successor, the myopic value of perfect information
///    `VPI(t)`: values are taken as uniform between the bounds, and `VPI(t)` is the largest
///    expected gain over the actions `a != a*` from learning the true value of `t`, i.e.
///    the average over `v` in `[L(t), U(t)]` of
///    `max(0, E[Q(s, a*) | v] - E[Q(s, a) | v])`. If some `VPI(t)` is positive, it samples
///    `t` by `VPI(t)`.
/// 3. Otherwise, if the expected successor gap `sum P(t) * gap(t)` is at least the initial
///    state's gap divided by `tau`, it samples by gap (BRTDP's rule); else, with
///    probability `alpha`, it samples by gap if any gap is positive; else the trial ends.
///
/// # Deviations from the paper
///
/// The paper measures anytime policy quality and stops on a time limit. It deliberately
/// skips updates where the policy has settled but the bounds have not. This crate stops on
/// `U(initial) - L(initial) <= epsilon`, which pure VPI steering may never reach: a trial can
/// keep following a successor of positive VPI (a self-loop, say) whose backups no longer
/// change anything, while the states that hold the gap up have zero VPI. Two additions
/// restore BRTDP's convergence, at worst halving its speed:
///
/// - step 3 first applies BRTDP's rule: while the expected successor gap is at least the
///   initial state's gap divided by `tau`, the trial samples by gap;
/// - after a trial that did not shrink the initial state's gap, the next trial runs as a
///   plain BRTDP trial.
///
/// The paper maximizes rewards; here an action gains by having the lower expected cost.
/// `E[Q(s, a) | v]` is `Q(s, a)` under the bound midpoints `(L + U) / 2`, with `t` fixed at
/// `v`; actions with an infinite expected cost (a dead-end successor) gain nothing. A
/// successor whose bounds meet has `VPI = 0`.
///
/// # Errors
///
/// As for [`brtdp`](crate::brtdp()), with invalid thresholds reported as
/// [`Error::InvalidConfig`].
///
/// # Panics
///
/// Panics if more than `u32::MAX` states, actions or outcomes are stored.
///
/// # Example
///
/// A sure route for cost 3, or a coin flip for cost 1 that retries on failure (expected
/// cost 2):
///
/// ```
/// use stochastic_search::{Mdp, VpiRtdpConfig, Zero, heuristic_fn, vpi_rtdp};
///
/// struct Routes;
///
/// impl Mdp for Routes {
///     type State = bool;
///     type Action = &'static str;
///
///     fn actions(&self, _done: &bool, mut emit: impl FnMut(&'static str, f64)) {
///         emit("sure", 3.0);
///         emit("flip", 1.0);
///     }
///
///     fn outcomes(&self, _done: &bool, action: &&'static str, mut emit: impl FnMut(bool, f64)) {
///         if *action == "sure" {
///             emit(true, 1.0);
///         } else {
///             emit(true, 0.5);
///             emit(false, 0.5);
///         }
///     }
/// }
///
/// let upper = heuristic_fn(|_: &bool| Some(3.0));
/// let config = VpiRtdpConfig::default();
/// let solution = vpi_rtdp(&Routes, false, &|done: &bool| *done, Zero, upper, config)?;
///
/// assert!(solution.is_converged());
/// let (lower, upper) = (solution.initial_value(), solution.initial_upper_bound().unwrap());
/// assert!(lower <= 2.0 && 2.0 <= upper && upper - lower <= 1e-6);
/// assert_eq!(solution.initial_action(), Some(&"flip"));
/// # Ok::<(), stochastic_search::Error<bool>>(())
/// ```
pub fn vpi_rtdp<M, G, L, U, B, O>(
    mdp: &M,
    initial: M::State,
    goal: &G,
    lower: L,
    upper: U,
    config: VpiRtdpConfig<B, O>,
) -> SolveResult<M>
where
    M: Mdp,
    M::State: Clone + Eq + Hash,
    G: Goal<M::State>,
    L: Heuristic<M::State, f64>,
    U: Heuristic<M::State, f64>,
    B: Budget<Stats>,
    O: Observer,
{
    config.validate()?;
    let explorer = Explorer::Vpi(Vpi {
        alpha: config.alpha,
        beta: config.beta,
        tau: config.tau,
        threshold: 0.0,
        means: Vec::new(),
    });

    solve((mdp, initial, goal), lower, upper, config.base, explorer)
}

/// VPI-RTDP's per-run state.
pub(crate) struct Vpi {
    /// The probability of continuing a trial without value of information.
    alpha: f64,
    /// The fraction of the initial gap that counts as maximal uncertainty.
    beta: f64,
    /// BRTDP's trial-end ratio, for states without value of information.
    tau: f64,

    /// `beta` times the initial gap of the initial state; set once the run starts.
    threshold: f64,
    /// Reused buffer: the expected Q-value of every action of the current state under the
    /// bound midpoints.
    means: Vec<f64>,
}

impl Vpi {
    /// Fixes the gap threshold from the initial state's initial gap.
    pub(crate) fn set_threshold(&mut self, initial_gap: f64) {
        self.threshold = self.beta * initial_gap;
    }

    /// The trial-end ratio of the `tau` rule, also used by fallback trials.
    pub(crate) fn tau(&self) -> f64 {
        self.tau
    }
}

impl<M: Mdp, G, L, U, B, O> Run<'_, M, G, L, U, B, O> {
    /// VPI-RTDP's successor choice at `state` under its lower-greedy action at `index`.
    pub(crate) fn vpi_next(&mut self, state: Id, index: u32) -> Option<Id> {
        let Explorer::Vpi(vpi) = &self.explorer else {
            unreachable!("only VPI-RTDP runs choose by value of information");
        };

        let (alpha, tau, threshold) = (vpi.alpha, vpi.tau, vpi.threshold);

        let u = self.rng.next_unit();
        let (total, largest) = self.gap_weights(index);
        if largest > threshold {
            return self.pick_weighted(index, total, u);
        }

        let information = self.information_weights(state, index);
        if information > 0.0 {
            return self.pick_weighted(index, information, u);
        }

        self.gap_weights(index);
        if total > 0.0 && total >= self.gap(self.start) / tau {
            return self.pick_weighted(index, total, u);
        }

        let r = self.rng.next_unit();
        if r < alpha && total > 0.0 {
            return self.pick_weighted(index, total, u);
        }

        None
    }

    /// Fills [`Run::weights`] with `VPI(t)` for every outcome `t` of the action at `index`
    /// (the lower-greedy action of `state`) and returns their sum.
    fn information_weights(&mut self, state: Id, index: u32) -> f64 {
        let Explorer::Vpi(vpi) = &mut self.explorer else {
            unreachable!("only VPI-RTDP runs choose by value of information");
        };

        let mut means = core::mem::take(&mut vpi.means);
        means.clear();
        means.extend(
            self.space
                .actions_of(state)
                .map(|(action, entry)| entry.cost + self.expected_midpoint(action)),
        );

        let mut weights = core::mem::take(&mut self.weights);
        weights.clear();
        let (first, _) = self.space.node(state).actions;
        for outcome in self.space.outcomes(index) {
            weights.push(self.information(state, index - first, outcome.target, &means));
        }

        let total = weights.iter().sum();
        self.weights = weights;
        if let Explorer::Vpi(vpi) = &mut self.explorer {
            vpi.means = means;
        }

        total
    }

    /// The expected value of the successors of the action at `index` under the bound
    /// midpoints.
    fn expected_midpoint(&self, index: u32) -> f64 {
        self.space
            .outcomes(index)
            .iter()
            .map(|outcome| outcome.probability * self.midpoint(outcome.target))
            .sum()
    }

    /// The midpoint `(L + U) / 2` of the bounds of `id`.
    fn midpoint(&self, id: Id) -> f64 {
        f64::midpoint(self.space.node(id).value, self.upper[id as usize])
    }

    /// `VPI(t)` at `state` with greedy action number `greedy` (its position among the
    /// state's actions), given the expected Q-value `means` of every action.
    fn information(&self, state: Id, greedy: u32, t: Id, means: &[f64]) -> f64 {
        let (lower, upper) = (self.space.node(t).value, self.upper[t as usize]);
        if gap(lower, upper) <= 0.0 {
            return 0.0;
        }

        let midpoint = self.midpoint(t);
        let (first, _) = self.space.node(state).actions;
        let greedy_mean = means[greedy as usize];
        let greedy_probability = self.probability_of(first + greedy, t);
        let mut best = 0.0_f64;
        for (position, &mean) in means.iter().enumerate() {
            let position = u32::try_from(position).expect("action positions fit u32");
            if position == greedy || !mean.is_finite() {
                continue;
            }

            let slope = greedy_probability - self.probability_of(first + position, t);
            let gain = |v: f64| (greedy_mean - mean) + slope * (v - midpoint);
            best = best.max(average_positive_part(gain(lower), gain(upper)));
        }

        best
    }

    /// The probability of reaching `t` with the action at `index`, summed over repeated
    /// outcomes.
    fn probability_of(&self, index: u32, t: Id) -> f64 {
        self.space
            .outcomes(index)
            .iter()
            .filter(|outcome| outcome.target == t)
            .map(|outcome| outcome.probability)
            .sum()
    }
}

/// The average of `max(0, f)` over an interval on which the linear function `f` goes from
/// `at_start` to `at_end`.
fn average_positive_part(at_start: f64, at_end: f64) -> f64 {
    if at_start >= 0.0 && at_end >= 0.0 {
        f64::midpoint(at_start, at_end)
    } else if at_start <= 0.0 && at_end <= 0.0 {
        0.0
    } else {
        let (positive, negative) = (at_start.max(at_end), at_start.min(at_end));

        positive * positive / (2.0 * (positive - negative))
    }
}

#[cfg(test)]
mod tests {
    use super::average_positive_part;

    /// The closed form matches the triangle and trapezoid areas.
    #[test]
    fn average_positive_part_is_the_mean_gain() {
        assert_eq!(average_positive_part(2.0, 4.0), 3.0);
        assert_eq!(average_positive_part(-2.0, -4.0), 0.0);
        // Crosses zero at the midpoint: a triangle of height 2 over half the interval.
        assert_eq!(average_positive_part(2.0, -2.0), 0.5);
        assert_eq!(average_positive_part(-2.0, 2.0), 0.5);
    }
}
