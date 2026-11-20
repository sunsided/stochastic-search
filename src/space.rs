//! The explicit, lazily built state space shared by all algorithms.
//!
//! States are interned into dense [`Id`]s. The first time an algorithm needs a state's
//! actions, they are queried from the [`Mdp`](crate::Mdp), validated and cached in flat arenas, so
//! Bellman backups run over contiguous slices without calling back into the MDP.
//!
//! ```mermaid
//! graph LR
//!     Node -- "actions range" --> ActionEntry
//!     ActionEntry -- "outcomes range" --> Outcome
//!     Outcome -- "target id" --> Node
//! ```

pub(crate) mod backup;
mod dead_ends;
mod expand;

use core::hash::{BuildHasher, Hash};

use alloc::vec::Vec;
use hashbrown::{DefaultHashBuilder, HashTable};

use crate::error::Error;
use crate::model::{Goal, Heuristic};

/// A dense state index, assigned in insertion order.
pub(crate) type Id = u32;

/// The inputs an algorithm solves: the MDP, the goal test, the heuristic and the value cap.
pub(crate) struct Problem<'a, M, G, H> {
    /// The MDP being solved.
    pub(crate) mdp: &'a M,
    /// The goal test.
    pub(crate) goal: &'a G,
    /// The initial value estimate of newly interned states.
    pub(crate) heuristic: H,
    /// The value of a dead end and the cap on every value: infinity, or the give-up penalty.
    pub(crate) cap: f64,
}

/// What kind of state a node is, decided when it is interned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    /// A goal state: value 0, never expanded.
    Goal,
    /// A state the heuristic proved to be a dead end: value at the cap, never expanded.
    DeadEnd,
    /// Any other state; expanded on demand.
    Open,
}

/// Per-state data.
#[derive(Debug, Clone)]
pub(crate) struct Node {
    /// What kind of state this is.
    pub(crate) kind: Kind,
    /// The current value estimate. Never NaN, never negative infinity.
    pub(crate) value: f64,
    /// Whether LRTDP labeled the state solved. Goals and proven dead ends start solved.
    pub(crate) solved: bool,
    /// The greedy action, as an index into [`Space::actions`]. `None` at goals, dead ends,
    /// give-up states and states not backed up yet.
    pub(crate) policy: Option<u32>,
    /// Whether [`Node::actions`] is filled in.
    pub(crate) expanded: bool,
    /// The range of this state's actions in [`Space::actions`]; empty until expanded.
    pub(crate) actions: (u32, u32),
    /// The epoch of the last depth-first search that visited this state.
    pub(crate) mark: u32,
}

/// One applicable action of an expanded state.
#[derive(Debug, Clone)]
pub(crate) struct ActionEntry<A> {
    /// The action label from the MDP.
    pub(crate) action: A,
    /// The action cost; finite and non-negative.
    pub(crate) cost: f64,
    /// The range of this action's outcomes in [`Space::outcomes`]; never empty.
    pub(crate) outcomes: (u32, u32),
}

/// One successor of an action.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Outcome {
    /// The successor state.
    pub(crate) target: Id,
    /// The probability of reaching it; in `(0, 1]`.
    pub(crate) probability: f64,
}

/// The interned states, their nodes and their cached expansions.
pub(crate) struct Space<S, A> {
    /// The states, indexed by [`Id`].
    states: Vec<S>,
    /// The per-state data, indexed by [`Id`].
    nodes: Vec<Node>,
    /// Maps a state's hash to its id. Only used for lookup, never iterated, so its randomly
    /// seeded hasher cannot influence results.
    index: HashTable<Id>,
    /// The hasher of [`Space::index`].
    hasher: DefaultHashBuilder,
    /// The actions of all expanded states, grouped by state.
    actions: Vec<ActionEntry<A>>,
    /// The outcomes of all cached actions, grouped by action.
    outcomes: Vec<Outcome>,
    /// The value of a dead end and the cap on every value.
    cap: f64,
    /// The number of expanded states.
    expanded: u64,
    /// Reused buffer for the actions an expansion collects from the MDP.
    scratch_actions: Vec<(A, f64)>,
    /// Reused buffer for the outcomes an expansion collects from the MDP.
    scratch_outcomes: Vec<(S, f64)>,
}

impl<S, A> Space<S, A> {
    /// Creates an empty space whose values are capped at `cap`.
    pub(crate) fn new(cap: f64) -> Self {
        Self {
            states: Vec::new(),
            nodes: Vec::new(),
            index: HashTable::new(),
            hasher: DefaultHashBuilder::default(),
            actions: Vec::new(),
            outcomes: Vec::new(),
            cap,
            expanded: 0,
            scratch_actions: Vec::new(),
            scratch_outcomes: Vec::new(),
        }
    }

    /// The number of interned states.
    pub(crate) fn len(&self) -> usize {
        self.nodes.len()
    }

    /// The number of expanded states.
    pub(crate) fn expanded(&self) -> u64 {
        self.expanded
    }

    /// The node of `id`.
    pub(crate) fn node(&self, id: Id) -> &Node {
        &self.nodes[id as usize]
    }

    /// The node of `id`, mutably.
    pub(crate) fn node_mut(&mut self, id: Id) -> &mut Node {
        &mut self.nodes[id as usize]
    }

    /// All nodes, indexed by [`Id`].
    pub(crate) fn nodes_mut(&mut self) -> &mut [Node] {
        &mut self.nodes
    }

    /// The state of `id`.
    pub(crate) fn state(&self, id: Id) -> &S {
        &self.states[id as usize]
    }

    /// The action entry at `index` in the action arena.
    pub(crate) fn action(&self, index: u32) -> &ActionEntry<A> {
        &self.actions[index as usize]
    }

    /// The action entries of an expanded state, with their arena indices.
    pub(crate) fn actions_of(&self, id: Id) -> impl Iterator<Item = (u32, &ActionEntry<A>)> {
        let (start, end) = self.node(id).actions;

        (start..end).zip(&self.actions[start as usize..end as usize])
    }

    /// The outcomes of the action at `index` in the action arena.
    pub(crate) fn outcomes(&self, index: u32) -> &[Outcome] {
        let (start, end) = self.action(index).outcomes;

        &self.outcomes[start as usize..end as usize]
    }

    /// The number of entries in the action arena.
    pub(crate) fn action_count(&self) -> usize {
        self.actions.len()
    }

    /// The number of entries in the outcome arena.
    pub(crate) fn outcome_count(&self) -> usize {
        self.outcomes.len()
    }

    /// The value of a dead end and the cap on every value.
    pub(crate) fn cap(&self) -> f64 {
        self.cap
    }
}

impl<S: Hash + Eq, A> Space<S, A> {
    /// Returns the id of `state`, if it was interned.
    pub(crate) fn find(&self, state: &S) -> Option<Id> {
        let hash = self.hasher.hash_one(state);

        self.index
            .find(hash, |&id| self.states[id as usize] == *state)
            .copied()
    }
}

impl<S: Hash + Eq + Clone, A> Space<S, A> {
    /// Returns the id of `state`, interning it with its initial value if it is new.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidHeuristic`] if the heuristic estimates NaN or negative infinity
    /// for a new state.
    ///
    /// # Panics
    ///
    /// Panics if more than `u32::MAX` states are interned.
    pub(crate) fn intern<M, G, H>(
        &mut self,
        problem: &mut Problem<'_, M, G, H>,
        state: S,
    ) -> Result<Id, Error<S>>
    where
        G: Goal<S>,
        H: Heuristic<S, f64>,
    {
        let hash = self.hasher.hash_one(&state);
        if let Some(&id) = self
            .index
            .find(hash, |&id| self.states[id as usize] == state)
        {
            return Ok(id);
        }

        let node = initial_node(problem, &state)?;
        let id = Id::try_from(self.nodes.len()).expect("at most u32::MAX states fit an id");
        let (states, hasher) = (&self.states, &self.hasher);
        self.index
            .insert_unique(hash, id, |&other| hasher.hash_one(&states[other as usize]));
        self.states.push(state);
        self.nodes.push(node);

        Ok(id)
    }
}

/// Builds the node of a newly interned state from the goal test and the heuristic.
///
/// # Errors
///
/// Returns [`Error::InvalidHeuristic`] if the heuristic estimates NaN or negative infinity.
fn initial_node<M, G, H, S>(problem: &mut Problem<'_, M, G, H>, state: &S) -> Result<Node, Error<S>>
where
    S: Clone,
    G: Goal<S>,
    H: Heuristic<S, f64>,
{
    let (kind, value) = if problem.goal.is_goal(state) {
        (Kind::Goal, 0.0)
    } else {
        match problem.heuristic.estimate(state) {
            Some(h) if is_invalid_estimate(h) => {
                return Err(Error::InvalidHeuristic {
                    state: state.clone(),
                });
            }
            Some(h) if h < f64::INFINITY => (Kind::Open, h.min(problem.cap)),
            _ => (Kind::DeadEnd, problem.cap),
        }
    };

    Ok(Node {
        kind,
        value,
        solved: kind != Kind::Open,
        policy: None,
        expanded: false,
        actions: (0, 0),
        mark: 0,
    })
}

/// Returns `true` for estimates no value may take: NaN and negative infinity.
fn is_invalid_estimate(h: f64) -> bool {
    h.is_nan() || (h.is_infinite() && h < 0.0)
}
