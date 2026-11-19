//! # Stochastic search
//!
//! Planning under probabilistic outcomes: LAO*, iLAO*, RTDP, LRTDP, POMCP and DESPOT.
//!
//! ## Planned scope
//!
//! - AND/OR heuristic search: LAO*, iLAO*, CG-iLAO*
//! - Real-time dynamic programming: RTDP, LRTDP
//! - Belief space: POMCP, DESPOT
//!
//! ## Status
//!
//! Pre-release. Version 0.0.0 reserves the name; the API is not started yet and will change.
//!
//! Repository: <https://github.com/sunsided/stochastic-search>

#![no_std]

/// A finite Markov decision process.
pub trait Mdp {
    /// A state.
    type State;
    /// An action.
    type Action;

    /// Calls `emit` with every successor of applying `action` in `state`
    /// and its probability. The probabilities sum to 1.
    fn outcomes(&self, state: &Self::State, action: &Self::Action, emit: impl FnMut(Self::State, f64));
}
