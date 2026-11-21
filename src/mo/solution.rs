//! The result of a multi-objective run: a coverage set of deterministic policies
//! ([`MoSolution`], [`MoPolicy`]).

use core::fmt;
use core::hash::Hash;

use alloc::vec::Vec;

use crate::budget::Stats;
use crate::error::Error;
use crate::mo::MoMdp;
use crate::mo::coverage::{NO_ACTION, best_under};
use crate::mo::simplex::{best_weight, margin, uniform};
use crate::mo::space::Space;
use crate::solution::Status;
use crate::space::{Id, Kind};

/// The result of solving the multi-objective MDP `M` with `N` objectives: an
/// [`MoSolution`] or the [`Error`] that stopped the run.
pub type MoSolveResult<M, const N: usize> = Result<
    MoSolution<<M as MoMdp<N>>::State, <M as MoMdp<N>>::Action, N>,
    Error<<M as MoMdp<N>>::State>,
>;

/// One policy of the coverage set: its data, without the state space.
#[derive(Debug, Clone)]
struct PolicyData<const N: usize> {
    /// The expected cost vector at the initial state, from the run's value.
    value: [f64; N],
    /// A weight under which this policy beats the others of the set.
    weight: [f64; N],
    /// The action (arena index) of every state the policy reaches, sorted by state id.
    actions: Vec<(Id, u32)>,
}

/// A coverage set of deterministic policies over the states a run touched: one policy per
/// vector of the initial state's value.
///
/// For a converged run, the policy values are the vertices of the lower convex hull of the
/// expected cost vectors of all proper policies (the convex coverage set), up to epsilon
/// and the margin of [`Objectives`](crate::Objectives). Stochastic policies that mix two
/// vertex policies have the values on the hull face between them; they are not listed.
pub struct MoSolution<S, A, const N: usize> {
    /// The states, value sets and actions.
    space: Space<S, A, N>,
    /// The id of the initial state.
    initial: Id,
    /// How the run ended.
    status: Status,
    /// Counters of the run.
    stats: Stats,
    /// The policies, one per vector of the initial value.
    policies: Vec<PolicyData<N>>,
}

impl<S, A, const N: usize> MoSolution<S, A, N> {
    /// Wraps the space of a finished run and extracts one policy per vector of the initial
    /// state's value. A converged run whose initial value holds no vector besides `b` is
    /// [`Status::Unsolvable`]; a run stopped by its budget keeps
    /// [`Status::BudgetExhausted`], since its sets may still gain vectors.
    ///
    /// Each policy's value is its own expected cost vector, found by evaluating it to
    /// `epsilon / 1000`; improper policies and duplicates are dropped.
    pub(crate) fn new(
        space: Space<S, A, N>,
        initial: Id,
        status: Status,
        counters: Stats,
        epsilon: f64,
    ) -> Self {
        let status = if status == Status::Converged && space.node(initial).value.points.is_empty() {
            Status::Unsolvable
        } else {
            status
        };

        let policies = extract_policies(&space, initial, epsilon / 1000.0);

        Self {
            space,
            initial,
            status,
            stats: counters,
            policies,
        }
    }

    /// How the run ended.
    #[must_use]
    pub fn status(&self) -> Status {
        self.status
    }

    /// Returns `true` if the run met its convergence criterion.
    #[must_use]
    pub fn is_converged(&self) -> bool {
        self.status == Status::Converged
    }

    /// Counters of the run.
    #[must_use]
    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    /// The initial state.
    #[must_use]
    pub fn initial_state(&self) -> &S {
        self.space.state(self.initial)
    }

    /// The policies of the coverage set, one per vector of the initial value. Empty when
    /// the problem is unsolvable.
    pub fn policies(&self) -> impl ExactSizeIterator<Item = MoPolicy<'_, S, A, N>> {
        self.policies.iter().map(|data| MoPolicy {
            solution: self,
            data,
        })
    }

    /// The vectors of the stored value set of `id`, without `b`.
    pub(crate) fn vectors_of(&self, id: Id) -> impl Iterator<Item = &[f64; N]> {
        self.space.node(id).value.vectors()
    }

    /// The number of stored states.
    pub(crate) fn state_count(&self) -> usize {
        self.space.len()
    }

    /// The state of `id`.
    pub(crate) fn state_of(&self, id: Id) -> &S {
        self.space.state(id)
    }
}

impl<S: Hash + Eq, A, const N: usize> MoSolution<S, A, N> {
    /// The vectors of the value of `state`, without the bound `b`; `None` if the run never
    /// touched it.
    ///
    /// Outside the states the policies reach, values are only estimates.
    #[must_use]
    pub fn value(&self, state: &S) -> Option<Vec<[f64; N]>> {
        self.space
            .find(state)
            .map(|id| self.vectors_of(id).copied().collect())
    }
}

impl<S: fmt::Debug, A, const N: usize> fmt::Debug for MoSolution<S, A, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let front: Vec<&[f64; N]> = self.policies.iter().map(|data| &data.value).collect();
        f.debug_struct("MoSolution")
            .field("initial_state", self.initial_state())
            .field("front", &front)
            .field("status", &self.status)
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

/// One deterministic policy of an [`MoSolution`].
///
/// The policy is defined on the states it reaches from the initial state. At the initial
/// state it takes the action of its own vector; elsewhere the action whose vector minimises
/// [`MoPolicy::weight`] `. v` in the state's value, which is how a coverage set vertex
/// decomposes into its successors' vertices.
pub struct MoPolicy<'a, S, A, const N: usize> {
    /// The solution the policy belongs to.
    solution: &'a MoSolution<S, A, N>,
    /// The policy itself.
    data: &'a PolicyData<N>,
}

impl<'a, S, A, const N: usize> MoPolicy<'a, S, A, N> {
    /// The expected cost vector of the policy at the initial state: the policy evaluated
    /// on its own (iteratively, to a thousandth of epsilon), not the run's estimate.
    ///
    /// The two can differ even after convergence. Within the margin, a set may keep the
    /// value of a non-stationary policy (one that switches between vertex policies when it
    /// revisits a state) instead of the nearby vertex, and only a deterministic stationary
    /// policy can be returned.
    #[must_use]
    pub fn value(&self) -> &'a [f64; N] {
        &self.data.value
    }

    /// A weight (non-negative, summing to 1) under which the policy's value is the best of
    /// the coverage set, by the widest margin found.
    #[must_use]
    pub fn weight(&self) -> &'a [f64; N] {
        &self.data.weight
    }

    /// The number of states with an action.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.actions.len()
    }

    /// Returns `true` if the policy has no action anywhere (the initial state is a goal).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.actions.is_empty()
    }

    /// The states with an action and their actions, in the order the run stored them.
    pub fn iter(&self) -> impl Iterator<Item = (&'a S, &'a A)> {
        let space = &self.solution.space;

        self.data
            .actions
            .iter()
            .map(move |&(id, action)| (space.state(id), &space.action(action).action))
    }
}

impl<'a, S: Hash + Eq, A, const N: usize> MoPolicy<'a, S, A, N> {
    /// The policy's action in `state`; `None` at goals, dead ends and states the policy
    /// does not reach.
    #[must_use]
    pub fn action(&self, state: &S) -> Option<&'a A> {
        let space = &self.solution.space;
        let id = space.find(state)?;
        let index = self
            .data
            .actions
            .binary_search_by_key(&id, |&(state, _)| state)
            .ok()?;

        Some(&space.action(self.data.actions[index].1).action)
    }
}

/// Extracts one policy per vector of the initial state's value.
fn extract_policies<S, A, const N: usize>(
    space: &Space<S, A, N>,
    initial: Id,
    tolerance: f64,
) -> Vec<PolicyData<N>> {
    let points = &space.node(initial).value.points;
    let mut tableau = Vec::new();
    let mut others = Vec::new();

    let mut policies: Vec<PolicyData<N>> = Vec::new();
    for (index, point) in points.iter().enumerate() {
        others.clear();
        others.extend(
            points
                .iter()
                .enumerate()
                .filter(|&(other, _)| other != index)
                .map(|(_, other)| other.vector),
        );
        let (weight, gap) = best_weight(&point.vector, &others, &mut tableau);
        let weight = nudge_inward(weight, gap, &point.vector, &others);
        let actions = follow(space, initial, point.action, &weight);
        if space.node(initial).kind != Kind::Open {
            policies.push(PolicyData {
                value: point.vector,
                weight,
                actions,
            });
            continue;
        }

        let duplicate = policies.iter().any(|policy| policy.actions == actions);
        let Some(value) = evaluate(space, initial, &actions, &weight, tolerance) else {
            continue;
        };

        if !duplicate {
            policies.push(PolicyData {
                value,
                weight,
                actions,
            });
        }
    }

    policies
}

/// The expected cost vector of the policy `actions` (sorted by state) at `initial`; `None`
/// if it is improper: some state it reaches
/// has no action and is no goal, or cannot reach a goal under the policy.
///
/// Gauss-Seidel sweeps from the run's `weight`-best vectors until no component moves by
/// more than `tolerance`, at most [`MAX_EVALUATION_SWEEPS`] sweeps.
fn evaluate<S, A, const N: usize>(
    space: &Space<S, A, N>,
    initial: Id,
    actions: &[(Id, u32)],
    weight: &[f64; N],
    tolerance: f64,
) -> Option<[f64; N]> {
    let index_of = |state: Id| actions.binary_search_by_key(&state, |&(id, _)| id).ok();
    let closed = actions.iter().all(|&(_, action)| {
        space.outcomes(action).iter().all(|outcome| {
            space.node(outcome.target).kind == Kind::Goal || index_of(outcome.target).is_some()
        })
    });
    if !closed || !reaches_goal(space, actions, &index_of) {
        return None;
    }

    let mut values: Vec<[f64; N]> = actions
        .iter()
        .map(|&(state, _)| {
            let set = &space.node(state).value;
            best_under(set, weight).map_or([0.0; N], |best| set.points[best].vector)
        })
        .collect();
    for _ in 0..MAX_EVALUATION_SWEEPS {
        if sweep_policy(space, actions, &index_of, &mut values) <= tolerance {
            break;
        }
    }

    Some(values[index_of(initial)?])
}

/// The most sweeps of one policy evaluation.
const MAX_EVALUATION_SWEEPS: usize = 1_000_000;

/// One Gauss-Seidel sweep of policy evaluation; returns the largest change of a component.
fn sweep_policy<S, A, const N: usize>(
    space: &Space<S, A, N>,
    actions: &[(Id, u32)],
    index_of: &impl Fn(Id) -> Option<usize>,
    values: &mut [[f64; N]],
) -> f64 {
    let mut largest = 0.0_f64;
    for (row, &(_, action)) in actions.iter().enumerate() {
        let mut value = space.action(action).cost;
        for outcome in space.outcomes(action) {
            if let Some(index) = index_of(outcome.target) {
                for (component, add) in value.iter_mut().zip(&values[index]) {
                    *component += outcome.probability * add;
                }
            }
        }

        for (new, old) in value.iter().zip(&values[row]) {
            largest = largest.max(libm::fabs(new - old));
        }

        values[row] = value;
    }

    largest
}

/// Returns `true` if every state of the policy reaches a goal in the policy's graph, so
/// that a goal is reached with probability 1.
fn reaches_goal<S, A, const N: usize>(
    space: &Space<S, A, N>,
    actions: &[(Id, u32)],
    index_of: &impl Fn(Id) -> Option<usize>,
) -> bool {
    let mut good = alloc::vec![false; actions.len()];
    loop {
        let mut changed = false;
        for (row, &(_, action)) in actions.iter().enumerate() {
            if good[row] {
                continue;
            }

            let reaches = space
                .outcomes(action)
                .iter()
                .any(|outcome| index_of(outcome.target).is_none_or(|index| good[index]));
            if reaches {
                good[row] = true;
                changed = true;
            }
        }

        if !changed {
            return good.iter().all(|&g| g);
        }
    }
}

/// Moves `weight` towards the uniform weight while `v` keeps at least half its margin
/// `gap` over `others`.
///
/// The linear program's optimum often sits on a corner of the simplex, where objectives
/// with weight zero tie; further down the policy, such ties could pick a vector that is
/// worse in those objectives. An interior weight breaks them.
fn nudge_inward<const N: usize>(
    weight: [f64; N],
    gap: f64,
    v: &[f64; N],
    others: &[[f64; N]],
) -> [f64; N] {
    let center = uniform::<N>();
    let mut share = 0.5;
    for _ in 0..30 {
        let mut candidate = weight;
        for (w, c) in candidate.iter_mut().zip(&center) {
            *w = (1.0 - share) * *w + share * c;
        }

        if margin(&candidate, v, others) >= gap / 2.0 {
            return candidate;
        }

        share /= 2.0;
    }

    weight
}

/// Collects the actions of the policy that takes `first` at `initial` and the
/// `weight`-best vector's action everywhere else, over the states it reaches. Stops at
/// goals, dead ends, unexpanded states and states without a vector.
fn follow<S, A, const N: usize>(
    space: &Space<S, A, N>,
    initial: Id,
    first: u32,
    weight: &[f64; N],
) -> Vec<(Id, u32)> {
    let mut actions: Vec<(Id, u32)> = Vec::new();
    let mut seen = alloc::vec![false; space.len()];
    let mut stack = alloc::vec![initial];
    seen[initial as usize] = true;
    while let Some(state) = stack.pop() {
        let Some(action) = chosen_action(space, state, initial, first, weight) else {
            continue;
        };

        actions.push((state, action));
        for outcome in space.outcomes(action) {
            if !seen[outcome.target as usize] {
                seen[outcome.target as usize] = true;
                stack.push(outcome.target);
            }
        }
    }

    actions.sort_unstable();

    actions
}

/// The action of the policy in `state`, if it has one there.
fn chosen_action<S, A, const N: usize>(
    space: &Space<S, A, N>,
    state: Id,
    initial: Id,
    first: u32,
    weight: &[f64; N],
) -> Option<u32> {
    let node = space.node(state);
    if node.kind != Kind::Open || !node.expanded {
        return None;
    }

    let action = if state == initial {
        first
    } else {
        node.value.points[best_under(&node.value, weight)?].action
    };

    (action != NO_ACTION).then_some(action)
}
