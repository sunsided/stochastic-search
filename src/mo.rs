//! Multi-objective stochastic shortest path problems (MOSSPs) and the heuristic search
//! algorithms of Chen, Trevizan and Thiébaux (2023): MOLAO*, iMOLAO* and MOLRTDP, with
//! multi-objective value iteration (MOVI) as the exact reference.
//!
//! Every action has a cost vector instead of a scalar cost ([`MoMdp`]). No single policy is
//! optimal; the solution is a *convex coverage set* (CCS): one deterministic proper policy
//! per vertex of the lower convex hull of the expected cost vectors of all proper policies.
//! Every policy that minimises some positive weighting `w . V` of the objectives is a
//! vertex policy or a mixture of them; the stochastic mixtures, which fill the hull's
//! faces, are implied by the vertices and never listed.
//!
//! A value is a set of cost vectors. Backups take the bounded set sum of the successors'
//! sets and prune the union over actions to a CCS with a small linear program per vector.
//! The paper's upper bound `b` ([`Objectives::bound`]) caps every value: a sum that exceeds
//! `b` in some objective stands for an improper policy and becomes `b`, which is dropped
//! from the final answer. This makes the algorithms terminate when improper policies have
//! finite cost in some objectives (the paper's weak improper policy assumption).
//!
//! # Heuristics
//!
//! A heuristic is a `Heuristic<S, Vec<[f64; N]>>`: a set of vectors per state, `None` (or
//! an empty set) for a proven dead end. Admissible means: for every weight `w` in the
//! simplex, `min_{u in H(s)} w . u <= min_{v in V*(s)} w . v`. That is the bound the CCS
//! backups preserve, and it is weaker than the paper's set domination (Definition 4.1),
//! which the determinisation heuristic does not satisfy (see `docs/design.md`). Provided:
//!
//! - [`Zero`](crate::Zero): the blind heuristic `{0}`.
//! - [`IdealPoint`]: one single-objective heuristic per objective, combined into the one
//!   vector `[h_1(s), ..., h_N(s)]` (the paper's `H_ideal`).
//! - [`Abstraction`]: MOVI on a smaller MOSSP, looked up through a projection (the
//!   explicit-state form of the paper's `H_mossp`); over [`Determinized`] it solves the
//!   all-outcomes determinisation exactly, which bounds the paper's `H_mo` heuristics.
//!
//! # Example
//!
//! Figure 1 of the paper: from `s0`, action `a1` costs `[1, 0]` and action `a2` costs
//! `[0, 1]`; each reaches its goal with probability 0.5 and stays in `s0` otherwise.
//!
//! ```
//! use stochastic_search::{Config, MoMdp, Objectives, Zero, molrtdp};
//!
//! struct Figure1;
//!
//! impl MoMdp<2> for Figure1 {
//!     type State = &'static str;
//!     type Action = &'static str;
//!
//!     fn actions(&self, _state: &&'static str, mut emit: impl FnMut(&'static str, [f64; 2])) {
//!         emit("a1", [1.0, 0.0]);
//!         emit("a2", [0.0, 1.0]);
//!     }
//!
//!     fn outcomes(
//!         &self,
//!         _state: &&'static str,
//!         action: &&'static str,
//!         mut emit: impl FnMut(&'static str, f64),
//!     ) {
//!         emit(if *action == "a1" { "g1" } else { "g2" }, 0.5);
//!         emit("s0", 0.5);
//!     }
//! }
//!
//! let goal = |state: &&str| state.starts_with('g');
//! let objectives = Objectives::new([100.0, 100.0]);
//! let solution = molrtdp(&Figure1, "s0", &goal, Zero, objectives, Config::default())?;
//!
//! assert!(solution.is_converged());
//! let mut front: Vec<[f64; 2]> = solution.policies().map(|policy| *policy.value()).collect();
//! front.sort_by(|a, b| a[0].total_cmp(&b[0]));
//! assert_eq!(front.len(), 2);
//! assert!((front[0][1] - 2.0).abs() < 1e-4 && (front[1][0] - 2.0).abs() < 1e-4);
//! # Ok::<(), stochastic_search::Error<&str>>(())
//! ```

mod coverage;
mod heuristics;
mod lao;
mod lrtdp;
mod movi;
mod run;
mod simplex;
mod solution;
mod space;

pub use heuristics::{Abstraction, Determinized, IdealPoint};
pub use lao::{imolao_star, molao_star};
pub use lrtdp::molrtdp;
pub use movi::mo_value_iteration;
pub use solution::{MoPolicy, MoSolution, MoSolveResult};

use crate::error::Error;

/// A finite MDP whose actions cost a vector of `N` objectives.
///
/// With a [`Goal`](crate::Goal) it defines a multi-objective stochastic shortest path
/// problem. Identical to [`Mdp`](crate::Mdp) except for the cost type, and queried the same
/// way: once per state, never for goals.
///
/// # Example
///
/// A retry loop that trades time (objective 0) against fuel (objective 1):
///
/// ```
/// use stochastic_search::MoMdp;
///
/// struct Retry;
///
/// impl MoMdp<2> for Retry {
///     type State = bool;
///     type Action = &'static str;
///
///     fn actions(&self, _done: &bool, mut emit: impl FnMut(&'static str, [f64; 2])) {
///         emit("walk", [3.0, 0.0]);
///         emit("drive", [1.0, 2.0]);
///     }
///
///     fn outcomes(&self, _done: &bool, _action: &&'static str, mut emit: impl FnMut(bool, f64)) {
///         emit(true, 1.0);
///     }
/// }
/// ```
pub trait MoMdp<const N: usize> {
    /// A state; algorithms store it, so they ask for `Clone + Eq + Hash`.
    type State;
    /// An action label, returned in policies.
    type Action;

    /// Calls `emit` once for every action applicable in `state`, with its cost vector.
    ///
    /// Every component must be finite and non-negative. A state without applicable actions
    /// is a dead end.
    fn actions(&self, state: &Self::State, emit: impl FnMut(Self::Action, [f64; N]));

    /// Calls `emit` with every successor of applying `action` in `state` and its
    /// probability, as for [`Mdp::outcomes`](crate::Mdp::outcomes).
    fn outcomes(
        &self,
        state: &Self::State,
        action: &Self::Action,
        emit: impl FnMut(Self::State, f64),
    );
}

/// The objective space of a multi-objective run: the upper bound `b` and the coverage set
/// margin.
///
/// The struct is `#[non_exhaustive]`; build it with [`Objectives::new`].
///
/// ```
/// use stochastic_search::Objectives;
///
/// let objectives = Objectives::new([100.0, 100.0]).with_margin(1e-4);
/// # let _ = objectives;
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Objectives<const N: usize> {
    /// The paper's upper bound `b` (Assumption 1, the weak improper policy assumption):
    /// every proper policy's expected cost vector from every state is at most `b` in every
    /// objective (and not equal to it), and every improper policy's exceeds it in at least
    /// one. Finite and non-negative.
    ///
    /// Values are capped at `b`, which makes improper policies, dead ends and traps
    /// visible and keeps every run finite. Too small a bound cuts off proper policies; too
    /// large a one only slows down the climb of trap values.
    pub bound: [f64; N],
    /// The convex coverage set margin (the paper's `lambda`): a vector is kept only if some
    /// weight makes it better than every other vector of its set by more than this.
    /// Finite and non-negative. `None` uses `10 * epsilon`, as the paper's experiments do.
    ///
    /// Intermediate value sets contain points that sit just below a face of the hull and
    /// reach it only in the limit (values of truncated non-stationary policies); a margin
    /// in the order of the residual prunes them. Vertices closer than the margin to the
    /// hull of the others are pruned too, so the margin is also an accuracy setting.
    ///
    /// It is the main performance setting as well. The transient sets grow fast as the
    /// margin shrinks, above all with three or more objectives (see `docs/design.md`,
    /// section "Performance"). A tight [`Objectives::bound`] helps too, because a trap's
    /// value adds vertices until it climbs to the bound.
    pub margin: Option<f64>,
}

impl<const N: usize> Objectives<N> {
    /// Objectives with upper bound `bound` and the default margin.
    #[must_use]
    pub const fn new(bound: [f64; N]) -> Self {
        Self {
            bound,
            margin: None,
        }
    }

    /// Returns the same objectives with another [`Objectives::margin`].
    #[must_use]
    pub const fn with_margin(self, margin: f64) -> Self {
        Self {
            margin: Some(margin),
            ..self
        }
    }

    /// Checks the settings and returns the margin to use with `epsilon`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfig`] for no objectives, a bound component that is not
    /// finite and non-negative, or a margin that is not finite and non-negative.
    pub(crate) fn validate<S>(&self, epsilon: f64) -> Result<f64, Error<S>> {
        if N == 0 {
            return Err(Error::InvalidConfig {
                reason: "a multi-objective problem needs at least one objective",
            });
        }

        if !self.bound.iter().all(|b| b.is_finite() && *b >= 0.0) {
            return Err(Error::InvalidConfig {
                reason: "every component of the bound b must be finite and non-negative",
            });
        }

        let margin = self.margin.unwrap_or(10.0 * epsilon);
        if !(margin.is_finite() && margin >= 0.0) {
            return Err(Error::InvalidConfig {
                reason: "the coverage set margin must be finite and non-negative",
            });
        }

        Ok(margin)
    }
}
