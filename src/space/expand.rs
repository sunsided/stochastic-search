//! Expansion: querying a state's actions and outcomes from the MDP, validating them and
//! caching them in the arenas.

use core::hash::Hash;
use core::mem;

use alloc::vec::Vec;

use crate::error::{Error, PROBABILITY_TOLERANCE};
use crate::model::{Goal, Heuristic, Mdp};
use crate::space::{ActionEntry, Id, Kind, Outcome, Problem, Space};

impl<S: Hash + Eq + Clone, A> Space<S, A> {
    /// Expands `id` unless it is expanded already or never expanded (goals, dead ends).
    ///
    /// Successor states are interned with their initial values.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidCost`], [`Error::InvalidProbability`] or
    /// [`Error::ProbabilitySum`] for invalid MDP output at `id`, and
    /// [`Error::InvalidHeuristic`] for an invalid estimate of a new successor. A failed
    /// expansion leaves the state unexpanded.
    pub(crate) fn expand<M, G, H>(
        &mut self,
        problem: &Problem<'_, M, G, H>,
        id: Id,
    ) -> Result<(), Error<S>>
    where
        M: Mdp<State = S, Action = A>,
        G: Goal<S>,
        H: Heuristic<S, f64>,
    {
        let node = self.node(id);
        if node.expanded || node.kind != Kind::Open {
            return Ok(());
        }

        let mut raw_actions = mem::take(&mut self.scratch_actions);
        raw_actions.clear();
        problem.mdp.actions(self.state(id), |action, cost| {
            raw_actions.push((action, cost));
        });

        let start = arena_index(self.actions.len());
        let outcomes_start = self.outcomes.len();
        let result = self.cache_actions(problem, id, &mut raw_actions);
        self.scratch_actions = raw_actions;
        if let Err(error) = result {
            self.actions.truncate(start as usize);
            self.outcomes.truncate(outcomes_start);
            return Err(error);
        }

        let end = arena_index(self.actions.len());
        let node = self.node_mut(id);
        node.actions = (start, end);
        node.expanded = true;
        self.expanded += 1;

        Ok(())
    }

    /// Validates the collected actions of `id`, queries their outcomes and appends them to
    /// the arenas.
    ///
    /// # Errors
    ///
    /// See [`Space::expand`].
    fn cache_actions<M, G, H>(
        &mut self,
        problem: &Problem<'_, M, G, H>,
        id: Id,
        raw_actions: &mut Vec<(A, f64)>,
    ) -> Result<(), Error<S>>
    where
        M: Mdp<State = S, Action = A>,
        G: Goal<S>,
        H: Heuristic<S, f64>,
    {
        for (action, cost) in raw_actions.drain(..) {
            if !(cost.is_finite() && cost >= 0.0) {
                return Err(Error::InvalidCost {
                    state: self.state(id).clone(),
                    cost,
                });
            }

            let outcomes = self.cache_outcomes(problem, id, &action)?;
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
        problem: &Problem<'_, M, G, H>,
        id: Id,
        action: &A,
    ) -> Result<(u32, u32), Error<S>>
    where
        M: Mdp<State = S, Action = A>,
        G: Goal<S>,
        H: Heuristic<S, f64>,
    {
        let mut raw = mem::take(&mut self.scratch_outcomes);
        raw.clear();
        problem
            .mdp
            .outcomes(self.state(id), action, |state, p| raw.push((state, p)));

        if let Err(error) = validate_distribution(&raw) {
            self.scratch_outcomes = raw;
            return Err(error.at(self.state(id).clone()));
        }

        let start = arena_index(self.outcomes.len());
        let result = self.intern_outcomes(problem, &mut raw);
        raw.clear();
        self.scratch_outcomes = raw;
        if let Err(error) = result {
            self.outcomes.truncate(start as usize);
            return Err(error);
        }

        Ok((start, arena_index(self.outcomes.len())))
    }
}

impl<S: Hash + Eq + Clone, A> Space<S, A> {
    /// Interns the successors in `raw` and appends them to the outcome arena.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidHeuristic`] for an invalid estimate of a new successor.
    fn intern_outcomes<M, G, H>(
        &mut self,
        problem: &Problem<'_, M, G, H>,
        raw: &mut Vec<(S, f64)>,
    ) -> Result<(), Error<S>>
    where
        G: Goal<S>,
        H: Heuristic<S, f64>,
    {
        for (state, probability) in raw.drain(..) {
            let target = self.intern(problem, state)?;
            self.outcomes.push(Outcome {
                target,
                probability,
            });
        }

        Ok(())
    }
}

/// A distribution defect, before the state it belongs to is attached.
enum Defect {
    /// A probability is NaN or outside `(0, 1]`.
    Probability(f64),
    /// The probabilities do not sum to 1 within the tolerance.
    Sum(f64),
}

impl Defect {
    /// Turns the defect into an error at `state`.
    fn at<S>(self, state: S) -> Error<S> {
        match self {
            Self::Probability(probability) => Error::InvalidProbability { state, probability },
            Self::Sum(sum) => Error::ProbabilitySum { state, sum },
        }
    }
}

/// Checks that every probability lies in `(0, 1]` and that they sum to 1 within
/// [`PROBABILITY_TOLERANCE`].
fn validate_distribution<S>(outcomes: &[(S, f64)]) -> Result<(), Defect> {
    let mut sum = 0.0;
    for &(_, p) in outcomes {
        if !(p > 0.0 && p <= 1.0) {
            return Err(Defect::Probability(p));
        }

        sum += p;
    }

    if abs(sum - 1.0) > PROBABILITY_TOLERANCE {
        return Err(Defect::Sum(sum));
    }

    Ok(())
}

/// The absolute value; hand-rolled because `f64::abs` is not in `core` on the minimum
/// supported Rust version.
pub(crate) fn abs(x: f64) -> f64 {
    if x < 0.0 { -x } else { x }
}

/// Converts an arena length to a `u32` index.
///
/// # Panics
///
/// Panics if an arena outgrows `u32::MAX` entries.
fn arena_index(len: usize) -> u32 {
    u32::try_from(len).expect("at most u32::MAX actions and outcomes fit the arenas")
}
