//! CG-iLAO* (Schmalz and Trevizan 2024): iLAO* over a partial SSP whose states hold only
//! the actions constraint generation has found necessary.
//!
//! ```mermaid
//! graph TD
//!     A[post-order DFS of the greedy policy from s0] --> B[partly expand fringe states with their greedy actions]
//!     B --> C[backups over the active actions, recording possible violations]
//!     C --> D[fix violated constraints: add actions, lower values]
//!     D --> E{no fringe, policy unchanged, residual below epsilon?}
//!     E -- no --> A
//!     E -- yes --> F{solved check on s0 over all actions passes?}
//!     F -- no --> A
//!     F -- yes --> G[return values and frozen policy]
//! ```

use core::hash::Hash;
use core::mem;

use alloc::vec::Vec;

use crate::budget::{Budget, Stats};
use crate::config::Config;
use crate::lrtdp::Run;
use crate::meter::{Halt, finish};
use crate::model::{Goal, Heuristic, Mdp};
use crate::observer::Observer;
use crate::solution::SolveResult;
use crate::space::backup::residual;
use crate::space::{Id, Kind};

/// Solves an SSP from `initial` with CG-iLAO*.
///
/// CG-iLAO* (Schmalz and Trevizan 2024, Algorithm 2) is [`ilao_star`](crate::ilao_star())
/// with partially expanded states. A fringe state of the greedy policy is expanded with its
/// greedy actions only (those whose Q-value under the current values is minimal); Bellman
/// backups consider only a state's *active* actions. The other actions stay out of the
/// partial SSP until a separation oracle finds their constraint `V(s) <= Q(s, a)` violated:
/// when a backup raises `V(s)`, the inactive actions of `s` are candidates; when it lowers
/// `V(s)`, every action leading into `s` is. Candidates with `V(s) > Q(s, a) + epsilon` are
/// added to the partial SSP and set `V(s) = Q(s, a)`. So, unlike other backup-based
/// algorithms, values can decrease. Iterations repeat until there is no fringe state in the
/// greedy envelope, the policy did not change and the largest residual is at most
/// [`Config::epsilon`] (the paper's termination test).
///
/// Two deviations make it robust to dead ends and to improper cycles among active actions,
/// which the paper's analysis does not cover. First, when a backup would raise `V(s)` by
/// more than epsilon, the inactive actions of `s` are checked right away, and those whose
/// Q-value lies below the raised value join the partial SSP before the value is written:
/// otherwise an infinite value from a dead-end successor spreads through active actions in
/// the same sweep, and once a cycle of states is infinite, no constraint `V(s) <= Q(s, a)`
/// is violated any more (both sides are infinite) and the run ends unsolvable although
/// escapes exist (found on a random SSP with the blind heuristic). Second, the paper's
/// backup loop runs until the residual is small, the policy
/// changes or the fringe is non-empty. When the active actions of some states form a cycle
/// that never reaches a goal (found on a random SSP with the determinization heuristic),
/// that loop never ends: the values climb without bound, and the inactive action that
/// escapes is only added by the constraint fix after the loop. Here the loop also ends after
/// a sweep that queued candidates.
///
/// This crate then runs the solved check of [`lrtdp`](crate::lrtdp()) on `initial`, over
/// all actions, so a converged run is epsilon-consistent on its envelope as for LRTDP.
/// LRTDP's `V(initial) <= V*(initial)` does not carry over: values can decrease here, so
/// they can end slightly above `V*` (the tests check `|V - V*|` against LRTDP's gap bound). By the paper's Theorem 2
/// the check passes; if it does not, every inactive action of the envelope becomes a
/// candidate and iterations resume. Each iteration counts in [`Stats::sweeps`](crate::Stats);
/// backups and the oracle's Q-value checks count in [`Stats::backups`](crate::Stats).
///
/// All actions and outcomes of an expanded state are still queried and cached (this crate's
/// state space caches whole expansions), so the saving is in Q-value computations, which is
/// what the paper measures, not in model calls or memory.
///
/// # Termination
///
/// As for [`ilao_star`](crate::ilao_star()): with the default [`Limits`](crate::Limits)
/// budget the call always returns, and under [`DeadEnds::Infinite`](crate::DeadEnds) the
/// doubling dead-end proof turns traps into dead ends.
///
/// # Errors
///
/// Returns an [`Error`](crate::Error) for an invalid [`Config`], or for invalid MDP output or
/// heuristic estimates in the states the run touches.
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
/// use stochastic_search::{Config, Mdp, Zero, cg_ilao_star};
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
/// let solution = cg_ilao_star(&Routes, false, &|done: &bool| *done, Zero, Config::default())?;
///
/// assert!(solution.is_converged());
/// assert!((solution.initial_value() - 2.0).abs() < 1e-5);
/// assert_eq!(solution.initial_action(), Some(&"flip"));
/// # Ok::<(), stochastic_search::Error<bool>>(())
/// ```
pub fn cg_ilao_star<M, G, H, B, O>(
    mdp: &M,
    initial: M::State,
    goal: &G,
    heuristic: H,
    config: Config<B, O>,
) -> SolveResult<M>
where
    M: Mdp,
    M::State: Clone + Eq + Hash,
    G: Goal<M::State>,
    H: Heuristic<M::State, f64>,
    B: Budget<Stats>,
    O: Observer,
{
    let mut run = Run::new(mdp, initial, goal, heuristic, config)?;
    let mut partial = Partial::default();
    let status = finish(run.solve_cg(&mut partial))?;

    Ok(run.into_solution(status))
}

/// The partial SSP of CG-iLAO*: which cached actions are active, and the candidates of the
/// separation oracle.
#[derive(Default)]
struct Partial {
    /// Per state id: whether the state's actions are registered below.
    registered: Vec<bool>,
    /// Per action arena index: whether the action is in the partial SSP.
    active: Vec<bool>,
    /// Per action arena index: the state the action belongs to.
    owner: Vec<Id>,
    /// Per state id: the arena indices of every cached action with the state as an outcome.
    predecessors: Vec<Vec<u32>>,
    /// The candidate constraints `Gamma`, as action arena indices, without duplicates.
    candidates: Vec<u32>,
    /// Per action arena index: whether the action is in [`Partial::candidates`].
    queued: Vec<bool>,
    /// The post-order traversal `E` of the current iteration.
    order: Vec<Id>,
    /// The depth-first stack of the traversal: a state and whether its children were pushed.
    stack: Vec<(Id, bool)>,
}

impl Partial {
    /// Queues every cached action that has `state` as an outcome.
    fn queue_predecessors(&mut self, state: Id) {
        let predecessors = mem::take(&mut self.predecessors[state as usize]);
        for &index in &predecessors {
            self.queue(index);
        }

        self.predecessors[state as usize] = predecessors;
    }

    /// Adds the action at `index` to the candidates unless it is there already.
    fn queue(&mut self, index: u32) {
        let slot = &mut self.queued[index as usize];
        if !*slot {
            *slot = true;
            self.candidates.push(index);
        }
    }
}

/// What the backups and constraint fixes of one iteration found.
#[derive(Debug, Clone, Copy, Default)]
struct Sweep {
    /// The largest change of a value.
    residual: f64,
    /// Whether some state's greedy action changed.
    policy_changed: bool,
}

impl<M, G, H, B, O> Run<'_, M, G, H, B, O>
where
    M: Mdp,
    M::State: Clone + Eq + Hash,
    G: Goal<M::State>,
    H: Heuristic<M::State, f64>,
    B: Budget<Stats>,
    O: Observer,
{
    /// Runs CG-iLAO* iterations until the initial state is solved.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    fn solve_cg(&mut self, partial: &mut Partial) -> Result<(), Halt<M::State>> {
        while !self.space.node(self.start).solved {
            self.meter.check(&self.space)?;
            self.meter.stats.sweeps += 1;
            let sweep = self.iterate(partial)?;
            self.maybe_prove_dead_ends(false);
            self.track_divergence();
            if !sweep.policy_changed
                && sweep.residual <= self.epsilon
                && !self.has_fringe(partial)
                && !self.check_solved(self.start, false)?
            {
                self.queue_inactive_actions(partial);
            }
        }

        Ok(())
    }

    /// One iteration: traversal, partial expansion, backups and constraint fixes.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out or the input is invalid.
    fn iterate(&mut self, partial: &mut Partial) -> Result<Sweep, Halt<M::State>> {
        self.register_expanded(partial);
        self.post_order(partial);
        let mut order = mem::take(&mut partial.order);
        let result = self.expand_fringe(&order, partial).and_then(|()| {
            let fringe = self.has_fringe(partial);
            let backups = self.cg_backups(&order, fringe, partial)?;
            let fixes = self.fix_constraints(partial)?;

            Ok(Sweep {
                residual: backups.residual.max(fixes.residual),
                policy_changed: backups.policy_changed || fixes.policy_changed,
            })
        });
        order.clear();
        partial.order = order;

        result
    }

    /// Partly expands the unexpanded open states of `order` with their greedy actions.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the input is invalid.
    fn expand_fringe(&mut self, order: &[Id], partial: &mut Partial) -> Result<(), Halt<M::State>> {
        for &state in order {
            if !self.space.node(state).expanded {
                self.space.expand(&mut self.problem, state)?;
                self.register_expanded(partial);
            }
        }

        Ok(())
    }

    /// Registers every expanded state not registered yet, including states other code
    /// expanded (the solved check): records owners and predecessors of its actions and
    /// activates the actions of minimal Q-value under the current values.
    fn register_expanded(&mut self, partial: &mut Partial) {
        partial.registered.resize(self.space.len(), false);
        partial.predecessors.resize_with(self.space.len(), Vec::new);
        let count = self.space.action_count();
        partial.active.resize(count, false);
        partial.owner.resize(count, 0);
        partial.queued.resize(count, false);
        for index in 0..self.space.len() {
            let state = Id::try_from(index).expect("interned indices fit an id");
            let node = self.space.node(state);
            if partial.registered[index] || !node.expanded || node.kind != Kind::Open {
                continue;
            }

            partial.registered[index] = true;
            self.register(state, partial);
        }
    }

    /// Registers the expanded `state`: see [`Run::register_expanded`]. Its greedy action
    /// becomes its policy, so the next traversal follows it.
    fn register(&mut self, state: Id, partial: &mut Partial) {
        let mut best = f64::INFINITY;
        for (index, _) in self.space.actions_of(state) {
            partial.owner[index as usize] = state;
            for outcome in self.space.outcomes(index) {
                partial.predecessors[outcome.target as usize].push(index);
            }

            best = best.min(self.space.q(index));
        }

        let mut policy = None;
        for (index, _) in self.space.actions_of(state) {
            // Equal to the minimum: `best` is the smallest of these very Q-values.
            if self.space.q(index) <= best {
                partial.active[index as usize] = true;
                policy = policy.or(Some(index));
            }
        }

        if best <= self.space.cap() {
            self.space.node_mut(state).policy = policy;
        }
    }

    /// Collects the open, unsolved states reachable from the initial state under the
    /// current policy into [`Partial::order`], in depth-first post-order. Unexpanded states
    /// are leaves.
    fn post_order(&mut self, partial: &mut Partial) {
        let epoch = self.next_epoch();
        let (order, stack) = (&mut partial.order, &mut partial.stack);
        order.clear();
        stack.clear();
        if !self.is_unsolved_open(self.start) {
            return;
        }

        self.space.node_mut(self.start).mark = epoch;
        stack.push((self.start, false));
        while let Some(frame) = stack.last_mut() {
            let (state, pushed) = *frame;
            frame.1 = true;
            let policy = self.space.node(state).policy;
            if pushed || !self.space.node(state).expanded || policy.is_none() {
                stack.pop();
                order.push(state);
                continue;
            }

            let action = policy.expect("checked above");
            for index in 0..self.space.outcomes(action).len() {
                let target = self.space.outcomes(action)[index].target;
                if self.is_unsolved_open(target) && self.space.node(target).mark != epoch {
                    self.space.node_mut(target).mark = epoch;
                    stack.push((target, false));
                }
            }
        }
    }

    /// Returns `true` if the greedy envelope of the initial state has an unexpanded state
    /// (the paper's `F`).
    fn has_fringe(&mut self, partial: &mut Partial) -> bool {
        self.post_order(partial);

        partial
            .order
            .iter()
            .any(|&state| !self.space.node(state).expanded)
    }

    /// The paper's `CG-BACKUPS`: sweeps over `order` with backups over active actions,
    /// queueing possible violations, until the residual is at most epsilon, the policy
    /// changed, or (with `fringe`) after one sweep. Unlike the paper, it also stops once a
    /// sweep queued candidates: when the active actions form a cycle that never reaches the
    /// goal, their values climb forever and only a constraint fix can add the escape.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out.
    fn cg_backups(
        &mut self,
        order: &[Id],
        fringe: bool,
        partial: &mut Partial,
    ) -> Result<Sweep, Halt<M::State>> {
        let mut sweep = Sweep::default();
        loop {
            sweep.residual = 0.0;
            for &state in order {
                let node = self.space.node(state);
                if node.kind != Kind::Open || !node.expanded {
                    continue;
                }

                let (change, changed) = self.backup_active(state, partial)?;
                sweep.residual = sweep.residual.max(change);
                sweep.policy_changed |= changed;
            }

            let candidates = !partial.candidates.is_empty();
            if sweep.residual <= self.epsilon || sweep.policy_changed || fringe || candidates {
                return Ok(sweep);
            }

            // Values climbing a trap cycle never settle; the doubling proof ends that.
            self.maybe_prove_dead_ends(false);
        }
    }

    /// Backs up `state` over its active actions and queues the constraints the change may
    /// violate: its inactive actions if the value rose by more than epsilon, the actions
    /// leading into it if it fell by more than epsilon. Returns the change and whether the
    /// greedy action changed.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out.
    fn backup_active(
        &mut self,
        state: Id,
        partial: &mut Partial,
    ) -> Result<(f64, bool), Halt<M::State>> {
        self.meter.charge_backup(&self.space)?;
        let old = self.space.node(state).value;
        let mut backup = self.active_bellman(state, partial);
        if backup.value - old > self.epsilon
            && self.activate_cheaper(state, backup.value, partial)?
        {
            backup = self.active_bellman(state, partial);
        }

        let node = self.space.node(state);
        let changed = node.policy != backup.policy;
        if backup.value - old > self.epsilon {
            for (index, _) in self.space.actions_of(state) {
                if !partial.active[index as usize] {
                    partial.queue(index);
                }
            }
        } else if old - backup.value > self.epsilon {
            partial.queue_predecessors(state);
        }

        let node = self.space.node_mut(state);
        node.value = backup.value;
        node.policy = backup.policy;
        if state == self.start {
            self.start_residual = residual(old, backup.value);
        }

        self.report_progress();

        Ok((residual(old, backup.value), changed))
    }

    /// The Bellman backup of `state` over its active actions.
    fn active_bellman(&self, state: Id, partial: &Partial) -> crate::space::backup::Backup {
        let space = &self.space;

        space.bellman_among(
            state,
            |s| space.node(s).value,
            |index| partial.active[index as usize],
        )
    }

    /// Before a backup raises `V(state)` to `raised`, activates every inactive action whose
    /// Q-value is below `raised - epsilon`, i.e. whose constraint the rise would violate.
    /// Returns whether one was activated. (This crate's addition, see [`cg_ilao_star`].)
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out; each Q-value counts as a backup.
    fn activate_cheaper(
        &mut self,
        state: Id,
        raised: f64,
        partial: &mut Partial,
    ) -> Result<bool, Halt<M::State>> {
        let (first, end) = self.space.node(state).actions;
        let mut activated = false;
        for index in first..end {
            if partial.active[index as usize] {
                continue;
            }

            self.meter.charge_backup(&self.space)?;
            if self.space.q(index) + self.epsilon < raised {
                partial.active[index as usize] = true;
                activated = true;
            }
        }

        Ok(activated)
    }

    /// The paper's `FIX-CONSTRS`: checks every candidate `(s, a)`; if
    /// `V(s) > Q(s, a) + epsilon`, activates `a`, sets `V(s) = Q(s, a)` and queues the
    /// actions leading into `s` as the next candidates.
    ///
    /// # Errors
    ///
    /// Returns [`Halt`] when the budget runs out.
    fn fix_constraints(&mut self, partial: &mut Partial) -> Result<Sweep, Halt<M::State>> {
        let candidates = mem::take(&mut partial.candidates);
        let mut sweep = Sweep::default();
        for &index in &candidates {
            partial.queued[index as usize] = false;
        }

        for &index in &candidates {
            let state = partial.owner[index as usize];
            if self.space.node(state).kind != Kind::Open {
                continue;
            }

            self.meter.charge_backup(&self.space)?;
            let q = self.space.q(index);
            let value = self.space.node(state).value;
            if value <= q + self.epsilon {
                continue;
            }

            partial.active[index as usize] = true;
            sweep.residual = sweep.residual.max(value - q);
            let node = self.space.node_mut(state);
            sweep.policy_changed |= node.policy != Some(index);
            node.value = q;
            node.policy = Some(index);
            partial.queue_predecessors(state);
        }

        Ok(sweep)
    }

    /// After a failed final check: queues every inactive action of every expanded state, so
    /// the next constraint fix checks them all.
    fn queue_inactive_actions(&mut self, partial: &mut Partial) {
        self.register_expanded(partial);
        for index in 0..partial.active.len() {
            if !partial.active[index] {
                partial.queue(u32::try_from(index).expect("action indices fit u32"));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::cg_ilao_star;
    use crate::{Config, Mdp, Zero};

    /// The SSP of the paper's Figure 1, with its monotonic heuristic: `s0 -a0-> s1`, `s1`
    /// has `a1` (to `s2`) and `a1'` (to `s3`), `s2 -a2-> g` (cost 3), `s3 -a3-> g` (cost 2).
    /// Costs: `a0` 1, `a1` 1, `a1'` 1. The optimal cost is `1 + 1 + 2 = 4` through `s3`.
    struct Figure1;

    impl Mdp for Figure1 {
        type State = u8;
        type Action = &'static str;

        fn actions(&self, state: &u8, mut emit: impl FnMut(&'static str, f64)) {
            match state {
                0 => emit("a0", 1.0),
                1 => {
                    emit("a1", 1.0);
                    emit("a1'", 1.0);
                }
                2 => emit("a2", 3.0),
                _ => emit("a3", 2.0),
            }
        }

        fn outcomes(&self, state: &u8, action: &&'static str, mut emit: impl FnMut(u8, f64)) {
            let next = match (*state, *action) {
                (0, _) => 1,
                (1, "a1") => 2,
                (1, _) => 3,
                _ => 4,
            };
            emit(next, 1.0);
        }
    }

    /// CG-iLAO* finds the cheaper branch even though the heuristic first prefers the other,
    /// so a value must decrease on the way (the paper's example).
    #[test]
    fn figure_one_is_solved_optimally() {
        let heuristic = [3.0, 2.0, 1.0, 2.0, 0.0];
        let h = crate::heuristic_fn(|s: &u8| Some(heuristic[usize::from(*s)]));
        let solution = cg_ilao_star(&Figure1, 0, &|s: &u8| *s == 4, h, Config::default())
            .expect("valid input");

        assert!(solution.is_converged());
        assert!((solution.initial_value() - 4.0).abs() < 1e-9);
        assert_eq!(solution.action(&1), Some(&"a1'"));

        let blind = cg_ilao_star(&Figure1, 0, &|s: &u8| *s == 4, Zero, Config::default())
            .expect("valid input");
        assert!((blind.initial_value() - 4.0).abs() < 1e-9);
    }
}
