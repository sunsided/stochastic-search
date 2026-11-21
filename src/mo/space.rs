//! The lazily built state space of the multi-objective algorithms: interned states with
//! value sets, and cached expansions with cost vectors.
//!
//! The scalar [`Space`](crate::space::Space) stores one `f64` per state and per action;
//! this one stores a [`ValueSet`] per state and a cost vector per action. Interning,
//! validation and the arena layout are the same.

use core::hash::{BuildHasher, Hash};
use core::mem;

use alloc::vec::Vec;

use hashbrown::{DefaultHashBuilder, HashTable};

use crate::error::Error;
use crate::mo::MoMdp;
use crate::mo::coverage::{NO_ACTION, Point, Scratch, ValueSet, prune};
use crate::model::{Goal, Heuristic};
use crate::space::expand::validate_distribution;
use crate::space::{Id, Kind, Outcome};

/// The inputs a multi-objective run solves.
pub(crate) struct Problem<'a, M, G, H, const N: usize> {
    /// The MDP being solved.
    pub(crate) mdp: &'a M,
    /// The goal test.
    pub(crate) goal: &'a G,
    /// The initial value estimate of newly interned states.
    pub(crate) heuristic: H,
    /// The paper's upper bound `b` (Assumption 1).
    pub(crate) bound: [f64; N],
    /// The convex coverage set margin (the paper's `lambda`).
    pub(crate) margin: f64,
}

/// Per-state data.
#[derive(Debug, Clone)]
pub(crate) struct Node<const N: usize> {
    /// What kind of state this is.
    pub(crate) kind: Kind,
    /// The current value. Never empty: a dead end holds only `b`.
    pub(crate) value: ValueSet<N>,
    /// Whether MOLRTDP labeled the state solved. Goals and dead ends start solved.
    pub(crate) solved: bool,
    /// Whether [`Node::actions`] is filled in.
    pub(crate) expanded: bool,
    /// The range of this state's actions in the action arena; empty until expanded.
    pub(crate) actions: (u32, u32),
    /// The epoch of the last graph search that visited this state.
    pub(crate) mark: u32,
}

/// One applicable action of an expanded state.
#[derive(Debug, Clone)]
pub(crate) struct ActionEntry<A, const N: usize> {
    /// The action label from the MDP.
    pub(crate) action: A,
    /// The cost vector; finite and non-negative.
    pub(crate) cost: [f64; N],
    /// The range of this action's outcomes in the outcome arena; never empty.
    pub(crate) outcomes: (u32, u32),
}

/// The interned states, their values and their cached expansions.
pub(crate) struct Space<S, A, const N: usize> {
    /// The states, indexed by [`Id`].
    states: Vec<S>,
    /// The per-state data, indexed by [`Id`].
    nodes: Vec<Node<N>>,
    /// Maps a state's hash to its id; only used for lookup, never iterated.
    index: HashTable<Id>,
    /// The hasher of [`Space::index`].
    hasher: DefaultHashBuilder,
    /// The actions of all expanded states, grouped by state.
    actions: Vec<ActionEntry<A, N>>,
    /// The outcomes of all cached actions, grouped by action.
    outcomes: Vec<Outcome>,
    /// The number of expanded states.
    expanded: u64,
    /// Reused buffer for the actions an expansion collects.
    scratch_actions: Vec<(A, [f64; N])>,
    /// Reused buffer for the outcomes an expansion collects.
    scratch_outcomes: Vec<(S, f64)>,
}

impl<S, A, const N: usize> Space<S, A, N> {
    /// Creates an empty space.
    pub(crate) fn new() -> Self {
        Self {
            states: Vec::new(),
            nodes: Vec::new(),
            index: HashTable::new(),
            hasher: DefaultHashBuilder::default(),
            actions: Vec::new(),
            outcomes: Vec::new(),
            expanded: 0,
            scratch_actions: Vec::new(),
            scratch_outcomes: Vec::new(),
        }
    }

    /// The number of interned states.
    pub(crate) fn len(&self) -> usize {
        self.nodes.len()
    }

    /// The node of `id`.
    pub(crate) fn node(&self, id: Id) -> &Node<N> {
        &self.nodes[id as usize]
    }

    /// The node of `id`, mutably.
    pub(crate) fn node_mut(&mut self, id: Id) -> &mut Node<N> {
        &mut self.nodes[id as usize]
    }

    /// All nodes, indexed by [`Id`].
    pub(crate) fn nodes_mut(&mut self) -> &mut [Node<N>] {
        &mut self.nodes
    }

    /// The state of `id`.
    pub(crate) fn state(&self, id: Id) -> &S {
        &self.states[id as usize]
    }

    /// The action entry at `index` in the action arena.
    pub(crate) fn action(&self, index: u32) -> &ActionEntry<A, N> {
        &self.actions[index as usize]
    }

    /// The arena indices of the actions of an expanded state.
    pub(crate) fn action_range(&self, id: Id) -> core::ops::Range<u32> {
        let (start, end) = self.node(id).actions;

        start..end
    }

    /// The outcomes of the action at `index` in the action arena.
    pub(crate) fn outcomes(&self, index: u32) -> &[Outcome] {
        let (start, end) = self.action(index).outcomes;

        &self.outcomes[start as usize..end as usize]
    }
}

impl<S, A, const N: usize> crate::meter::Counts for Space<S, A, N> {
    type State = S;

    fn states(&self) -> u64 {
        self.nodes.len() as u64
    }

    fn expanded(&self) -> u64 {
        self.expanded
    }
}

impl<S: Hash + Eq, A, const N: usize> Space<S, A, N> {
    /// Returns the id of `state`, if it was interned.
    pub(crate) fn find(&self, state: &S) -> Option<Id> {
        let hash = self.hasher.hash_one(state);

        self.index
            .find(hash, |&id| self.states[id as usize] == *state)
            .copied()
    }
}

impl<S: Hash + Eq + Clone, A, const N: usize> Space<S, A, N> {
    /// Returns the id of `state`, interning it with its initial value if it is new.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidHeuristic`] if the heuristic estimates a vector with a
    /// non-finite component.
    ///
    /// # Panics
    ///
    /// Panics if more than `u32::MAX` states are interned.
    pub(crate) fn intern<M, G, H>(
        &mut self,
        problem: &mut Problem<'_, M, G, H, N>,
        state: S,
        scratch: &mut Scratch<N>,
    ) -> Result<Id, Error<S>>
    where
        G: Goal<S>,
        H: Heuristic<S, Vec<[f64; N]>>,
    {
        let hash = self.hasher.hash_one(&state);
        if let Some(&id) = self
            .index
            .find(hash, |&id| self.states[id as usize] == state)
        {
            return Ok(id);
        }

        let (kind, value) = initial_value(problem, &state, scratch)?;
        let id = Id::try_from(self.nodes.len()).expect("at most u32::MAX states fit the space");
        self.nodes.push(Node {
            kind,
            value,
            solved: kind != Kind::Open,
            expanded: false,
            actions: (0, 0),
            mark: 0,
        });
        self.states.push(state);
        let (states, hasher) = (&self.states, &self.hasher);
        self.index
            .insert_unique(hash, id, |&other| hasher.hash_one(&states[other as usize]));

        Ok(id)
    }

    /// Expands `id` unless it is expanded already or never expanded (goals, dead ends).
    /// A state without actions becomes a dead end.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidCost`], [`Error::InvalidProbability`] or
    /// [`Error::ProbabilitySum`] for invalid MDP output at `id`, and
    /// [`Error::InvalidHeuristic`] for an invalid estimate of a new successor.
    pub(crate) fn expand<M, G, H>(
        &mut self,
        problem: &mut Problem<'_, M, G, H, N>,
        id: Id,
        scratch: &mut Scratch<N>,
    ) -> Result<(), Error<S>>
    where
        M: MoMdp<N, State = S, Action = A>,
        G: Goal<S>,
        H: Heuristic<S, Vec<[f64; N]>>,
    {
        let node = self.node(id);
        if node.expanded || node.kind != Kind::Open {
            return Ok(());
        }

        let mut raw = mem::take(&mut self.scratch_actions);
        raw.clear();
        problem
            .mdp
            .actions(self.state(id), |action, cost| raw.push((action, cost)));

        let start = arena_index(self.actions.len());
        let outcomes_start = self.outcomes.len();
        let result = self.cache_actions(problem, id, &mut raw, scratch);
        self.scratch_actions = raw;
        if let Err(error) = result {
            self.actions.truncate(start as usize);
            self.outcomes.truncate(outcomes_start);
            return Err(error);
        }

        let end = arena_index(self.actions.len());
        let node = self.node_mut(id);
        node.actions = (start, end);
        node.expanded = true;
        if start == end {
            node.kind = Kind::DeadEnd;
            node.value = ValueSet::dead_end();
            node.solved = true;
        }

        self.expanded += 1;

        Ok(())
    }

    /// Validates the collected actions of `id`, queries their outcomes and appends them.
    ///
    /// # Errors
    ///
    /// See [`Space::expand`].
    fn cache_actions<M, G, H>(
        &mut self,
        problem: &mut Problem<'_, M, G, H, N>,
        id: Id,
        raw: &mut Vec<(A, [f64; N])>,
        scratch: &mut Scratch<N>,
    ) -> Result<(), Error<S>>
    where
        M: MoMdp<N, State = S, Action = A>,
        G: Goal<S>,
        H: Heuristic<S, Vec<[f64; N]>>,
    {
        for (action, cost) in raw.drain(..) {
            if let Some(&bad) = cost.iter().find(|c| !(c.is_finite() && **c >= 0.0)) {
                return Err(Error::InvalidCost {
                    state: self.state(id).clone(),
                    cost: bad,
                });
            }

            let outcomes = self.cache_outcomes(problem, id, &action, scratch)?;
            self.actions.push(ActionEntry {
                action,
                cost,
                outcomes,
            });
        }

        Ok(())
    }

    /// Queries, validates and interns the outcomes of `action` in `id`, and returns their
    /// range in the outcome arena.
    ///
    /// # Errors
    ///
    /// See [`Space::expand`].
    fn cache_outcomes<M, G, H>(
        &mut self,
        problem: &mut Problem<'_, M, G, H, N>,
        id: Id,
        action: &A,
        scratch: &mut Scratch<N>,
    ) -> Result<(u32, u32), Error<S>>
    where
        M: MoMdp<N, State = S, Action = A>,
        G: Goal<S>,
        H: Heuristic<S, Vec<[f64; N]>>,
    {
        let mut raw = mem::take(&mut self.scratch_outcomes);
        raw.clear();
        problem
            .mdp
            .outcomes(self.state(id), action, |state, p| raw.push((state, p)));

        if let Err(defect) = validate_distribution(&raw) {
            self.scratch_outcomes = raw;
            return Err(defect.at(self.state(id).clone()));
        }

        let start = arena_index(self.outcomes.len());
        let mut result = Ok(());
        for (state, probability) in raw.drain(..) {
            match self.intern(problem, state, scratch) {
                Ok(target) => self.outcomes.push(Outcome {
                    target,
                    probability,
                }),
                Err(error) => {
                    result = Err(error);
                    break;
                }
            }
        }

        raw.clear();
        self.scratch_outcomes = raw;
        if let Err(error) = result {
            self.outcomes.truncate(start as usize);
            return Err(error);
        }

        Ok((start, arena_index(self.outcomes.len())))
    }
}

/// The kind and initial value of a new state: the zero vector at goals, only `b` where the
/// heuristic proves a dead end (`None` or no vector), else the pruned estimate.
///
/// # Errors
///
/// Returns [`Error::InvalidHeuristic`] for an estimate with a non-finite component.
fn initial_value<S: Clone, M, G, H, const N: usize>(
    problem: &mut Problem<'_, M, G, H, N>,
    state: &S,
    scratch: &mut Scratch<N>,
) -> Result<(Kind, ValueSet<N>), Error<S>>
where
    G: Goal<S>,
    H: Heuristic<S, Vec<[f64; N]>>,
{
    if problem.goal.is_goal(state) {
        return Ok((Kind::Goal, ValueSet::goal()));
    }

    let estimate = problem.heuristic.estimate(state).unwrap_or_default();
    if estimate.iter().flatten().any(|c| !c.is_finite()) {
        return Err(Error::InvalidHeuristic {
            state: state.clone(),
        });
    }

    if estimate.is_empty() {
        return Ok((Kind::DeadEnd, ValueSet::dead_end()));
    }

    let mut points: Vec<Point<N>> = estimate
        .into_iter()
        .map(|vector| Point {
            vector,
            action: NO_ACTION,
        })
        .collect();
    prune(&mut points, problem.margin, scratch);

    Ok((
        Kind::Open,
        ValueSet {
            points,
            bound: Vec::new(),
        },
    ))
}

/// Converts an arena length to a `u32` index.
///
/// # Panics
///
/// Panics if an arena outgrows `u32::MAX` entries.
fn arena_index(len: usize) -> u32 {
    u32::try_from(len).expect("at most u32::MAX actions and outcomes fit the arenas")
}
