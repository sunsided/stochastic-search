//! # Stochastic search
//!
//! Planning under probabilistic outcomes: value iteration, RTDP, labeled RTDP and iLAO* for
//! fully observable problems; POMCP and DESPOT for partially observable ones.
//!
//! A problem is a stochastic shortest path (SSP) problem: reach a goal state with minimal
//! expected total cost, where every action has a cost and a probability distribution over
//! successor states. It is described by small capability traits instead of one universal
//! problem trait: the MDP implements [`Mdp`], the goal test is a [`Goal`] and the initial
//! value estimate a [`Heuristic`]. Closures work for both (wrap heuristic closures in
//! [`heuristic_fn`]). Every algorithm takes
//! a [`Config`] and returns a [`Solution`]: a value function and a greedy policy, plus a
//! [`Status`] and [`Stats`].
//!
//! Partially observable problems are generative models ([`Pomdp`]): a step samples successor,
//! observation and reward from a `rand_core` generator. The online planners [`pomcp()`] and
//! [`despot()`] take a particle belief and return a [`Plan`] for one decision.
//!
//! The crate is `no_std` and needs only `alloc`.
//!
//! ## Algorithms
//!
//! - [`lrtdp()`]: labeled real-time dynamic programming (Bonet and Geffner 2003). Explores only
//!   the states a good policy visits; epsilon-consistent on them when it converges.
//! - [`rtdp()`]: plain RTDP (Barto, Bradtke and Singh 1995), with LRTDP's solved check as
//!   its stopping rule. The classic baseline.
//! - [`ilao_star()`]: improved LAO* (Hansen and Zilberstein 2001). Grows the best partial
//!   solution graph in depth-first passes; no sampling.
//! - [`value_iteration()`]: value iteration over every reachable state. The exact reference.
//! - [`pomcp()`]: partially observable Monte-Carlo planning (Silver and Veness 2010). UCT
//!   over action-observation histories with rollouts.
//! - [`despot()`]: anytime regularized DESPOT (Somani et al. 2013, Ye et al. 2017). Bound
//!   guided search over a tree of `K` determinized scenarios.
//!
//! ## Dead ends
//!
//! States from which no policy reaches a goal with probability 1 are valued per
//! [`DeadEnds`]: at infinity (the default, exact SSP semantics), or at a finite give-up
//! penalty that makes every input solvable.
//!
//! ## Events
//!
//! An [`Observer`] set with [`Config::with_observer`] receives [`Event`]s while a run
//! progresses: progress, proven dead ends, and suspected divergence (a heuristic signal,
//! never a verdict). The default observer `()` costs nothing.
//!
//! ## Example
//!
//! ```
//! use stochastic_search::{Config, Mdp, heuristic_fn, lrtdp};
//!
//! /// A die roll: action `roll` (cost 1) ends on a six; action `pay` (cost 4) ends always.
//! struct Dice;
//!
//! impl Mdp for Dice {
//!     type State = bool;
//!     type Action = &'static str;
//!
//!     fn actions(&self, _done: &bool, mut emit: impl FnMut(&'static str, f64)) {
//!         emit("roll", 1.0);
//!         emit("pay", 4.0);
//!     }
//!
//!     fn outcomes(&self, _done: &bool, action: &&'static str, mut emit: impl FnMut(bool, f64)) {
//!         match *action {
//!             "roll" => {
//!                 emit(true, 1.0 / 6.0);
//!                 emit(false, 5.0 / 6.0);
//!             }
//!             _ => emit(true, 1.0),
//!         }
//!     }
//! }
//!
//! let heuristic = heuristic_fn(|_: &bool| Some(1.0));
//! let solution = lrtdp(&Dice, false, &|done: &bool| *done, heuristic, Config::default())?;
//!
//! // Rolling takes 6 attempts in expectation, so paying 4 is cheaper.
//! assert_eq!(solution.initial_action(), Some(&"pay"));
//! assert!((solution.initial_value() - 4.0).abs() < 1e-5);
//! # Ok::<(), stochastic_search::Error<bool>>(())
//! ```
//!
//! ## Planned scope
//!
//! - AND/OR heuristic search: LAO* (the one-tip-at-a-time variant), CG-iLAO*
//! - Bounded RTDP (upper and lower bounds for an anytime gap)
//! - Belief tracking between POMDP decisions (POMCP's particle reinvigoration, tree reuse)
//!
//! See `docs/design.md` in the repository for the design and milestones.
//!
//! ## Status
//!
//! Pre-release. The API will change.
//!
//! Repository: <https://github.com/sunsided/stochastic-search>

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

mod budget;
mod config;
mod despot;
mod error;
mod ilao;
mod lrtdp;
mod meter;
mod model;
mod observer;
mod pomcp;
mod pomdp;
mod rng;
mod rtdp;
mod solution;
mod space;
mod value_iteration;

pub use budget::{Budget, Limits, Stats};
pub use config::{Config, DeadEnds};
pub use despot::{DespotConfig, despot};
pub use error::{Error, PROBABILITY_TOLERANCE};
pub use ilao::ilao_star;
pub use lrtdp::lrtdp;
pub use model::{Goal, Heuristic, HeuristicFn, Mdp, Zero, heuristic_fn};
pub use observer::{Event, Observer, ObserverFn, observer_fn};
pub use pomcp::{PomcpConfig, pomcp};
pub use pomdp::{Plan, Pomdp, RolloutPolicy, Step, UniformRandom};
pub use rng::unit_interval;
pub use rtdp::rtdp;
pub use solution::{Solution, SolveResult, Status};
pub use value_iteration::value_iteration;
