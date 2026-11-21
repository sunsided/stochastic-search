//! Small table models decoded from fuzzer bytes, shared by the fuzz targets.
//!
//! Most decoded numbers come from a coarse grid inside the valid range, so the planners get
//! well-formed problems most of the time and run their real search. One number in eight is
//! raw `f64` bits instead (NaN, infinities, negatives, subnormals), and one probability list in
//! eight is left unnormalized, so the input validation is exercised too.

#![forbid(unsafe_code)]

use libfuzzer_sys::arbitrary::{Result, Unstructured};
use rand_core::Rng;
use stochastic_search::{Mdp, MoMdp, Pomdp, Step, unit_interval};

/// The most states a decoded model has; small models keep every run fast.
pub const MAX_STATES: usize = 6;

/// The most actions a decoded state (or POMDP) has.
pub const MAX_ACTIONS: usize = 3;

/// The most outcomes a decoded action has.
pub const MAX_OUTCOMES: usize = 3;

/// The number of steps of the grid that [`number`] draws from.
const GRID_STEPS: u8 = 16;

/// Draws a number in `[low, high]` on a grid of [`GRID_STEPS`] steps, or, one time in eight,
/// raw `f64` bits that may be anything.
///
/// # Errors
///
/// Fails only if `u` cannot produce the bytes, which ends the input.
pub fn number(u: &mut Unstructured<'_>, low: f64, high: f64) -> Result<f64> {
    if u.ratio(1, 8)? {
        return Ok(f64::from_bits(u.arbitrary()?));
    }

    let step = u.int_in_range(0..=GRID_STEPS)?;

    Ok(low + (high - low) * f64::from(step) / f64::from(GRID_STEPS))
}

/// Draws a state index below `states`.
///
/// # Errors
///
/// Fails only if `u` cannot produce the bytes.
pub fn state(u: &mut Unstructured<'_>, states: usize) -> Result<usize> {
    u.int_in_range(0..=states - 1)
}

/// Draws between 1 and [`MAX_OUTCOMES`] items with `item`, each with a probability.
///
/// The probabilities are positive grid weights normalized to sum to 1, except one time in
/// eight, when they are raw [`number`]s that may break every rule.
///
/// # Errors
///
/// Fails only if `u` cannot produce the bytes.
pub fn distribution<T>(
    u: &mut Unstructured<'_>,
    mut item: impl FnMut(&mut Unstructured<'_>) -> Result<T>,
) -> Result<Vec<(T, f64)>> {
    let count = u.int_in_range(1..=MAX_OUTCOMES)?;
    let raw = u.ratio(1, 8)?;
    let mut outcomes = Vec::with_capacity(count);
    for _ in 0..count {
        let value = item(u)?;
        let weight = if raw {
            number(u, 0.0, 1.0)?
        } else {
            f64::from(u.int_in_range(1..=GRID_STEPS)?)
        };
        outcomes.push((value, weight));
    }

    if !raw {
        let total: f64 = outcomes.iter().map(|(_, weight)| weight).sum();
        for (_, weight) in &mut outcomes {
            *weight /= total;
        }
    }

    Ok(outcomes)
}

/// One action of a [`Table`]: its cost and its successor distribution.
#[derive(Debug)]
pub struct Action<C> {
    /// The cost: `f64` for [`Mdp`], a cost vector for [`MoMdp`].
    pub cost: C,
    /// The successors and their probabilities; never empty.
    pub outcomes: Vec<(usize, f64)>,
}

/// An MDP over the states `0..actions.len()`, with actions as indices into a state's list.
#[derive(Debug)]
pub struct Table<C> {
    /// The actions of every state; a state without actions is a dead end.
    pub actions: Vec<Vec<Action<C>>>,
    /// The goal state, below `actions.len()`.
    pub goal: usize,
}

impl<C> Table<C> {
    /// Decodes a table of 1 to [`MAX_STATES`] states, drawing each action cost with `cost`.
    ///
    /// # Errors
    ///
    /// Fails only if `u` cannot produce the bytes.
    pub fn decode(
        u: &mut Unstructured<'_>,
        mut cost: impl FnMut(&mut Unstructured<'_>) -> Result<C>,
    ) -> Result<Self> {
        let states = u.int_in_range(1..=MAX_STATES)?;
        let goal = state(u, states)?;
        let mut actions = Vec::with_capacity(states);
        for _ in 0..states {
            let count = u.int_in_range(0..=MAX_ACTIONS)?;
            let mut list = Vec::with_capacity(count);
            for _ in 0..count {
                let cost = cost(u)?;
                let outcomes = distribution(u, |u| state(u, states))?;
                list.push(Action { cost, outcomes });
            }

            actions.push(list);
        }

        Ok(Self { actions, goal })
    }

    /// The number of states.
    #[must_use]
    pub fn len(&self) -> usize {
        self.actions.len()
    }

    /// Returns `true` if the table has no states; decoded tables never do.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }

    /// Returns `true` if `state` is the goal.
    #[must_use]
    pub fn is_goal(&self, state: usize) -> bool {
        state == self.goal
    }

    /// Calls `emit` with every successor of `action` in `state`.
    fn emit_outcomes(&self, state: usize, action: usize, mut emit: impl FnMut(usize, f64)) {
        for &(target, probability) in &self.actions[state][action].outcomes {
            emit(target, probability);
        }
    }
}

impl Mdp for Table<f64> {
    type State = usize;
    type Action = usize;

    fn actions(&self, state: &usize, mut emit: impl FnMut(usize, f64)) {
        for (index, action) in self.actions[*state].iter().enumerate() {
            emit(index, action.cost);
        }
    }

    fn outcomes(&self, state: &usize, action: &usize, emit: impl FnMut(usize, f64)) {
        self.emit_outcomes(*state, *action, emit);
    }
}

impl<const N: usize> MoMdp<N> for Table<[f64; N]> {
    type State = usize;
    type Action = usize;

    fn actions(&self, state: &usize, mut emit: impl FnMut(usize, [f64; N])) {
        for (index, action) in self.actions[*state].iter().enumerate() {
            emit(index, action.cost);
        }
    }

    fn outcomes(&self, state: &usize, action: &usize, emit: impl FnMut(usize, f64)) {
        self.emit_outcomes(*state, *action, emit);
    }
}

/// One possible result of a [`GenerativeTable`] step.
#[derive(Debug, Clone, Copy)]
pub struct Transition {
    /// The successor state.
    pub state: usize,
    /// The observation received.
    pub observation: u8,
    /// The reward collected.
    pub reward: f64,
    /// Whether the episode ends.
    pub terminal: bool,
}

/// A generative POMDP over the states `0..steps.len()` and the actions `0..actions`.
#[derive(Debug)]
pub struct GenerativeTable {
    /// The number of actions, at most [`MAX_ACTIONS`]; zero is invalid input.
    pub actions: usize,
    /// `steps[state][action]`: the transitions and their probabilities.
    pub steps: Vec<Vec<Vec<(Transition, f64)>>>,
}

impl GenerativeTable {
    /// The most observations a decoded POMDP emits.
    pub const MAX_OBSERVATIONS: u8 = 3;

    /// Decodes a POMDP of 1 to [`MAX_STATES`] states and 0 to [`MAX_ACTIONS`] actions, with
    /// rewards in `[-10, 10]`.
    ///
    /// # Errors
    ///
    /// Fails only if `u` cannot produce the bytes.
    pub fn decode(u: &mut Unstructured<'_>) -> Result<Self> {
        let states = u.int_in_range(1..=MAX_STATES)?;
        let actions = u.int_in_range(0..=MAX_ACTIONS)?;
        let mut steps = Vec::with_capacity(states);
        for _ in 0..states {
            let mut per_action = Vec::with_capacity(actions);
            for _ in 0..actions {
                per_action.push(distribution(u, |u| transition(u, states))?);
            }

            steps.push(per_action);
        }

        Ok(Self { actions, steps })
    }

    /// The number of states.
    #[must_use]
    pub fn states(&self) -> usize {
        self.steps.len()
    }
}

/// Draws one [`Transition`] into a state below `states`.
fn transition(u: &mut Unstructured<'_>, states: usize) -> Result<Transition> {
    Ok(Transition {
        state: state(u, states)?,
        observation: u.int_in_range(0..=GenerativeTable::MAX_OBSERVATIONS - 1)?,
        reward: number(u, -10.0, 10.0)?,
        terminal: u.ratio(1, 4)?,
    })
}

impl Pomdp for GenerativeTable {
    type State = usize;
    type Action = usize;
    type Observation = u8;

    fn actions(&self, mut emit: impl FnMut(usize)) {
        for action in 0..self.actions {
            emit(action);
        }
    }

    /// Samples a transition by inverting the cumulative probabilities. Unnormalized input
    /// falls through to the last transition, so every step returns one.
    fn step<R: Rng + ?Sized>(&self, state: &usize, action: &usize, rng: &mut R) -> Step<usize, u8> {
        let transitions = &self.steps[*state][*action];
        let mut remaining = unit_interval(rng);
        let mut chosen = transitions[transitions.len() - 1].0;
        for &(transition, probability) in transitions {
            if remaining < probability {
                chosen = transition;
                break;
            }

            remaining -= probability;
        }

        let step = Step::new(chosen.state, chosen.observation, chosen.reward);
        if chosen.terminal {
            step.terminal()
        } else {
            step
        }
    }
}
