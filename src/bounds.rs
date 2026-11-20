//! The two-sided bounds run shared by BRTDP, FRTDP and VPI-RTDP: lower and upper bounds on
//! every stored state, trials that back both up, and the policy fill pass.
//!
//! The three algorithms differ only in how a trial picks the next state and when it stops,
//! which [`Explorer`] decides; see `docs/design.md`, section "Bounded RTDP".
//!
//! ```mermaid
//! graph TD
//!     A{"gap of s0 at most epsilon?"} -- no --> B[back up both bounds, act greedily on the lower bound]
//!     B --> C{"explorer picks a successor?"}
//!     C -- yes --> B
//!     C -- no --> E[back up the trial in reverse]
//!     E --> A
//!     A -- yes --> F[fill policy gaps on the upper-greedy envelope]
//!     F --> G[return both bounds and the upper-greedy policy]
//! ```

pub(crate) mod brtdp;
pub(crate) mod frtdp;
pub(crate) mod vpi;

use core::hash::Hash;
use core::mem;

use alloc::collections::VecDeque;
use alloc::vec;
use alloc::vec::Vec;

use crate::bounds::frtdp::Focus;
use crate::bounds::vpi::Vpi;
use crate::budget::{Budget, Stats};
use crate::config::Config;
use crate::error::Error;
use crate::meter::{Halt, Meter, finish};
use crate::model::{Goal, Heuristic, Mdp};
use crate::observer::Observer;
use crate::rng::SplitMix64;
use crate::solution::{Solution, SolveResult, Status};
use crate::space::backup::residual;
use crate::space::{Id, Kind, Problem, Space};

/// How a trial picks its next state and when it ends: the part that differs between the
/// bounded algorithms.
pub(crate) enum Explorer {
    /// BRTDP: sample by probability times gap; end when the summed weight is below the
    /// initial gap divided by `tau`.
    Brtdp {
        /// The trial-end ratio, finite and greater than 1.
        tau: f64,
    },
    /// FRTDP: follow the successor of highest priority, with an adaptive depth limit.
    Frtdp(Focus),
    /// VPI-RTDP: sample by gap where bounds are loose, else by the value of perfect
    /// information.
    Vpi(Vpi),
}

/// Solves with a bounded run under `explorer`: the common body of the public functions.
///
/// # Errors
///
/// As documented on [`brtdp`](crate::brtdp()).
pub(crate) fn solve<M, G, L, U, B, O>(
    problem: (&M, M::State, &G),
    lower: L,
    upper: U,
    config: Config<B, O>,
    explorer: Explorer,
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
    let (mdp, initial, goal) = problem;
    let mut run = Run::new(mdp, initial, goal, lower, upper, config, explorer)?;
    let status = finish(run.solve())?;
    if status == Status::Converged {
        run.fill_policy()?;
    }

    Ok(run.into_solution(status))
}

/// The state of one bounded run.
pub(crate) struct Run<'a, M: Mdp, G, L, U, B, O> {
    /// The problem being solved; its heuristic is the lower bound.
    problem: Problem<'a, M, G, L>,
    /// The initial upper bound of newly stored states.
    upper_heuristic: U,
    /// The states touched so far; [`Node::value`](crate::space::Node) is the lower bound
    /// and [`Node::policy`](crate::space::Node) the upper-greedy action.
    pub(crate) space: Space<M::State, M::Action>,
    /// The upper bound of every stored state, indexed by [`Id`]. Infinite only where the
    /// lower bound is infinite too, so gaps are never infinite or NaN.
    pub(crate) upper: Vec<f64>,
    /// Budget, counters and observer.
    meter: Meter<B, O>,
    /// The successor sampler.
    pub(crate) rng: SplitMix64,
    /// The gap at the initial state that ends the run.
    pub(crate) epsilon: f64,
    /// The step limit of one trial.
    max_trial_length: usize,
    /// The initial state.
    pub(crate) start: Id,
    /// How trials pick states.
    explorer: Explorer,
    /// The change of the lower bound at the most recent backup.
    pub(crate) last_change: f64,
    /// Whether the current trial runs as a plain BRTDP trial: FRTDP and VPI-RTDP switch to
    /// one after a trial of their own that did not shrink the initial state's gap.
    fallback: bool,
    /// Reused buffer: the states of the current trial, in visiting order.
    trail: Vec<Id>,
    /// Reused buffer: one sampling weight per outcome of the action being explored.
    pub(crate) weights: Vec<f64>,
}

impl<'a, M, G, L, U, B, O> Run<'a, M, G, L, U, B, O>
where
    M: Mdp,
    M::State: Clone + Eq + Hash,
    G: Goal<M::State>,
    L: Heuristic<M::State, f64>,
    U: Heuristic<M::State, f64>,
    B: Budget<Stats>,
    O: Observer,
{
    /// Validates `config` and sets up a run from `initial` with an empty space.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfig`] for an invalid configuration, and the estimate
    /// errors of [`Run::estimate_upper`] for `initial`.
    fn new(
        mdp: &'a M,
        initial: M::State,
        goal: &'a G,
        lower: L,
        upper_heuristic: U,
        config: Config<B, O>,
        explorer: Explorer,
    ) -> Result<Self, Error<M::State>> {
        config.validate()?;
        let mut problem = Problem {
            mdp,
            goal,
            heuristic: lower,
            cap: config.dead_ends.cap(),
        };

        let mut space = Space::new(problem.cap);
        let start = space.intern(&mut problem, initial)?;
        let mut run = Self {
            problem,
            upper_heuristic,
            space,
            upper: Vec::new(),
            rng: SplitMix64::new(config.seed),
            epsilon: config.epsilon,
            max_trial_length: config.max_trial_length,
            meter: Meter::new(config),
            start,
            explorer,
            last_change: 0.0,
            fallback: false,
            trail: Vec::new(),
            weights: Vec::new(),
        };
        run.estimate_new_states()?;
        if let Explorer::Vpi(vpi) = &mut run.explorer {
            vpi.set_threshold(gap(run.space.node(start).value, run.upper[start as usize]));
        }

        Ok(run)
    }

    /// Ends the run with `status` and hands both bounds to a [`Solution`].
    fn into_solution(mut self, status: Status) -> Solution<M::State, M::Action> {
        self.meter.sync(&self.space);

        Solution::new(self.space, self.start, status, self.meter.stats).with_upper(self.upper)
    }

    /// Runs trials until the gap at the initial state is at most epsilon.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    fn solve(&mut self) -> Result<(), Halt<M::State>> {
        while self.gap(self.start) > self.epsilon {
            self.meter.check(&self.space)?;
            self.meter.stats.trials += 1;
            self.trial()?;
        }

        Ok(())
    }

    /// Runs one trial: a walk from the initial state guided by the explorer, then backups
    /// in reverse.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    fn trial(&mut self) -> Result<(), Halt<M::State>> {
        let mut trail = mem::take(&mut self.trail);
        trail.clear();
        let gap_before = self.gap(self.start);
        let fallback = self.fallback;
        if let (Explorer::Frtdp(focus), false) = (&mut self.explorer, fallback) {
            focus.start_trial();
        }

        let result = self.walk(&mut trail).and_then(|()| {
            while let Some(state) = trail.pop() {
                self.backup(state)?;
            }

            Ok(())
        });
        self.trail = trail;
        // A fallback trial tracks no update quality, so it must not move FRTDP's depth limit.
        if let (Explorer::Frtdp(focus), false) = (&mut self.explorer, fallback) {
            focus.end_trial();
        }

        let is_brtdp = matches!(self.explorer, Explorer::Brtdp { .. });
        self.fallback = !is_brtdp && !self.fallback && self.gap(self.start) >= gap_before;

        result
    }

    /// Walks from the initial state, backing up every state on the way and recording it in
    /// `trail`, until the explorer ends the trial, a state has no lower-greedy action, or the
    /// length limit is reached.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    fn walk(&mut self, trail: &mut Vec<Id>) -> Result<(), Halt<M::State>> {
        let mut state = self.start;
        while trail.len() < self.max_trial_length {
            let depth = trail.len();
            trail.push(state);
            let Some(action) = self.backup(state)? else {
                break;
            };

            let Some(next) = self.next_state(state, action, depth) else {
                break;
            };

            state = next;
        }

        Ok(())
    }

    /// Asks the explorer for the successor of `state` under its lower-greedy `action`, at
    /// trial depth `depth` (0 at the initial state).
    fn next_state(&mut self, state: Id, action: u32, depth: usize) -> Option<Id> {
        if self.fallback {
            return self.brtdp_next(action, self.fallback_tau());
        }

        match &self.explorer {
            Explorer::Brtdp { tau } => {
                let tau = *tau;
                self.brtdp_next(action, tau)
            }
            Explorer::Frtdp(_) => self.frtdp_next(state, action, depth),
            Explorer::Vpi(_) => self.vpi_next(state, action),
        }
    }

    /// The `tau` of fallback BRTDP trials: VPI-RTDP's own, otherwise BRTDP's default.
    fn fallback_tau(&self) -> f64 {
        match &self.explorer {
            Explorer::Brtdp { tau } => *tau,
            Explorer::Vpi(vpi) => vpi.tau(),
            Explorer::Frtdp(_) => crate::BrtdpConfig::DEFAULT_TAU,
        }
    }

    /// Charges the budget, expands `state` if needed and backs up both bounds. Returns the
    /// action that is greedy on the lower bound, which trials follow; stores the
    /// upper-greedy one as the policy.
    ///
    /// The budget is charged before the expansion, so a state is expanded exactly when it
    /// has been backed up at least once (the fill pass relies on it).
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    fn backup(&mut self, state: Id) -> Result<Option<u32>, Halt<M::State>> {
        self.meter.charge_backup(&self.space)?;
        self.space.expand(&mut self.problem, state)?;
        self.estimate_new_states()?;

        let lower = self.space.bellman(state);
        let upper = {
            let bounds = &self.upper;
            self.space
                .bellman_under(state, |target| bounds[target as usize])
        };
        let node = self.space.node_mut(state);
        self.last_change = residual(node.value, lower.value);
        node.value = lower.value;
        node.policy = upper.policy;
        self.upper[state as usize] = upper.value;
        if matches!(self.explorer, Explorer::Frtdp(_)) {
            self.update_priority(state, lower.policy);
        }

        if self.meter.progress_due() {
            let value = self.space.node(self.start).value;
            let residual = self.gap(self.start);
            self.meter.emit_progress(&self.space, value, residual);
        }

        Ok(lower.policy)
    }

    /// After convergence, backs up every state of the upper-greedy policy's envelope that
    /// no trial backed up, until the envelope has an action wherever one exists (goals,
    /// dead ends and give-up states have none). Backups only tighten the bounds, so the
    /// guarantee is kept; the gap at the initial state is unchanged.
    ///
    /// Stops quietly when the budget runs out; the run then stays converged with the
    /// remaining gaps.
    ///
    /// # Errors
    ///
    /// Returns the [`Error`] of invalid MDP output or estimates in the newly expanded states.
    fn fill_policy(&mut self) -> Result<(), Error<M::State>> {
        let mut seen = vec![false; self.space.len()];
        let mut queue = VecDeque::from([self.start]);
        seen[self.start as usize] = true;
        while let Some(state) = queue.pop_front() {
            let node = self.space.node(state);
            if node.kind == Kind::Open && !node.expanded {
                match self.backup(state) {
                    Ok(_) => {}
                    Err(Halt::Budget) => return Ok(()),
                    Err(Halt::Invalid(error)) => return Err(error),
                }
            }

            let Some(action) = self.space.node(state).policy else {
                continue;
            };

            seen.resize(self.space.len(), false);
            for outcome in self.space.outcomes(action) {
                let target = outcome.target as usize;
                if !seen[target] {
                    seen[target] = true;
                    queue.push_back(outcome.target);
                }
            }
        }

        Ok(())
    }

    /// Estimates the upper bound (and, for FRTDP, the priority) of every state stored since
    /// the last call.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Run::estimate_upper`].
    fn estimate_new_states(&mut self) -> Result<(), Error<M::State>> {
        while self.upper.len() < self.space.len() {
            let id = Id::try_from(self.upper.len()).expect("interned indices fit an id");
            let bound = self.estimate_upper(id)?;
            self.upper.push(bound);
            let excess = self.gap(id) - self.epsilon / 2.0;
            if let Explorer::Frtdp(focus) = &mut self.explorer {
                focus.priority.push(excess);
            }
        }

        Ok(())
    }

    /// The initial upper bound of the stored state `id`: `0` at goals, the cap at proven dead
    /// ends, otherwise the upper estimate capped at the dead-end value.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidHeuristic`] for an estimate that is NaN, negative infinity or
    /// below the state's lower bound, and [`Error::UnboundedUpper`] for an infinite upper
    /// bound (`None` or infinity under an infinite cap) on a state with a finite lower bound.
    fn estimate_upper(&mut self, id: Id) -> Result<f64, Error<M::State>> {
        let node = self.space.node(id);
        let (kind, lower, cap) = (node.kind, node.value, self.problem.cap);
        if kind != Kind::Open {
            return Ok(lower);
        }

        let state = self.space.state(id);
        let bound = match self.upper_heuristic.estimate(state) {
            Some(u) if u.is_nan() || u == f64::NEG_INFINITY || u < lower => {
                return Err(Error::InvalidHeuristic {
                    state: state.clone(),
                });
            }
            Some(u) => u.min(cap),
            None => cap,
        };

        if bound.is_infinite() {
            return Err(Error::UnboundedUpper {
                state: state.clone(),
            });
        }

        Ok(bound)
    }
}

impl<M: Mdp, G, L, U, B, O> Run<'_, M, G, L, U, B, O> {
    /// The gap between the bounds of `id`.
    pub(crate) fn gap(&self, id: Id) -> f64 {
        gap(self.space.node(id).value, self.upper[id as usize])
    }

    /// Picks the outcome of the action at `index` whose bucket in [`Run::weights`] (one
    /// non-negative weight per outcome, summing to `total > 0`) contains `u * total`, for a
    /// uniform `u` in `[0, 1)`. Falls back to the last outcome of positive weight if rounding
    /// leaves the sample past the final bucket.
    pub(crate) fn pick_weighted(&self, index: u32, total: f64, u: f64) -> Option<Id> {
        let mut remaining = u * total;
        let mut last = None;
        for (outcome, &weight) in self.space.outcomes(index).iter().zip(&self.weights) {
            if weight > 0.0 {
                if remaining < weight {
                    return Some(outcome.target);
                }

                remaining -= weight;
                last = Some(outcome.target);
            }
        }

        last
    }

    /// Fills [`Run::weights`] with `P(s') * gap(s')` for every outcome of the action at
    /// `index` and returns their sum and the largest successor gap.
    pub(crate) fn gap_weights(&mut self, index: u32) -> (f64, f64) {
        let mut weights = mem::take(&mut self.weights);
        weights.clear();
        let mut largest = 0.0_f64;
        for outcome in self.space.outcomes(index) {
            let gap = self.gap(outcome.target);
            largest = largest.max(gap);
            weights.push(outcome.probability * gap);
        }

        let total = weights.iter().sum();
        self.weights = weights;

        (total, largest)
    }
}

/// The gap `upper - lower`, or `0` if the bounds meet or cross (including both infinite).
pub(crate) fn gap(lower: f64, upper: f64) -> f64 {
    if upper > lower { upper - lower } else { 0.0 }
}
