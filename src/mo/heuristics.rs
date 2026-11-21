//! Multi-objective heuristics (Section 4 of Chen, Trevizan and Thiébaux 2023): the blind
//! heuristic, ideal points of single-objective heuristics, the all-outcomes determinisation
//! and value iteration on an abstraction.

use core::hash::Hash;

use alloc::vec::Vec;

use hashbrown::HashMap;

use crate::config::Config;
use crate::mo::movi::mo_value_iteration;
use crate::mo::{MoMdp, Objectives};
use crate::model::{Goal, Heuristic, Zero};

/// The blind multi-objective heuristic: the zero vector everywhere. Admissible for every
/// MOSSP.
impl<S, const N: usize> Heuristic<S, Vec<[f64; N]>> for Zero {
    fn estimate(&mut self, _state: &S) -> Option<Vec<[f64; N]>> {
        Some(alloc::vec![[0.0; N]])
    }
}

/// The ideal-point heuristic `H_ideal`: one single-objective heuristic per objective,
/// combined into the single vector `[h_1(s), ..., h_N(s)]`.
///
/// Each `h_i` must be admissible for objective `i` alone (the SSP whose costs are the
/// `i`-th components); then the ideal point is admissible, even in the set-domination
/// sense of the paper's Definition 4.1. The paper instantiates it with `h^max` on the
/// determinisation (`H^max_ideal`) and with SSP pattern databases (`H^pdb_ideal`); any
/// `Heuristic<S, f64>` works, including the ones written for
/// [`lrtdp`](crate::lrtdp()). A `None` (or infinite) estimate from any `h_i` proves a dead
/// end.
///
/// # Example
///
/// ```
/// use stochastic_search::{Heuristic, IdealPoint, heuristic_fn};
///
/// // The array needs one type: closures differ, function pointers do not.
/// type Estimate = fn(&u32) -> Option<f64>;
/// let distance: Estimate = |state| Some(f64::from(10 - state));
/// let blind: Estimate = |_| Some(0.0);
/// let mut ideal = IdealPoint([heuristic_fn(distance), heuristic_fn(blind)]);
///
/// assert_eq!(ideal.estimate(&4), Some(vec![[6.0, 0.0]]));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdealPoint<H, const N: usize>(pub [H; N]);

impl<S, H, const N: usize> Heuristic<S, Vec<[f64; N]>> for IdealPoint<H, N>
where
    H: Heuristic<S, f64>,
{
    fn estimate(&mut self, state: &S) -> Option<Vec<[f64; N]>> {
        let mut vector = [0.0; N];
        for (component, heuristic) in vector.iter_mut().zip(&mut self.0) {
            let estimate = heuristic.estimate(state)?;
            if estimate == f64::INFINITY {
                return None;
            }

            *component = estimate;
        }

        Some(alloc::vec![vector])
    }
}

/// The all-outcomes determinisation of an [`MoMdp`]: every outcome of every action becomes
/// a deterministic action with the action's cost vector.
///
/// Actions are labeled by their successor state. Solving it with [`Abstraction`] (and the
/// identity projection) gives the exact multi-objective determinisation heuristic, the
/// strongest of the paper's `H_mo` family (which approximates it with `h^max` and pattern
/// databases on a factored model).
///
/// The determinisation is admissible only in the scalarised sense of the
/// [module docs](crate::mo): from a state that moves at zero cost to `s1` or `s2` with
/// probability 0.5 each, where `s1` reaches the goal for `[1, 0]` and `s2` for `[0, 1]`,
/// the determinisation estimates `{[1, 0], [0, 1]}` but the optimal value is
/// `{[0.5, 0.5]}`, which neither vector dominates.
///
/// # Example
///
/// ```
/// use stochastic_search::{Abstraction, Determinized, Heuristic, MoMdp, Objectives};
///
/// struct Coin;
///
/// impl MoMdp<2> for Coin {
///     type State = u8;
///     type Action = ();
///
///     fn actions(&self, state: &u8, mut emit: impl FnMut((), [f64; 2])) {
///         emit((), if *state == 0 { [0.0, 0.0] } else if *state == 1 { [1.0, 0.0] } else { [0.0, 1.0] });
///     }
///
///     fn outcomes(&self, state: &u8, _action: &(), mut emit: impl FnMut(u8, f64)) {
///         if *state == 0 {
///             emit(1, 0.5);
///             emit(2, 0.5);
///         } else {
///             emit(3, 1.0);
///         }
///     }
/// }
///
/// let determinized = Determinized(&Coin);
/// let goal = |state: &u8| *state == 3;
/// let mut heuristic = Abstraction::new(&determinized, |state: &u8| *state, &goal, Objectives::new([100.0; 2]));
///
/// let mut estimate = heuristic.estimate(&0).unwrap();
/// estimate.sort_by(|a, b| a[0].total_cmp(&b[0]));
/// assert_eq!(estimate, vec![[0.0, 1.0], [1.0, 0.0]]);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Determinized<'a, M>(pub &'a M);

impl<M, const N: usize> MoMdp<N> for Determinized<'_, M>
where
    M: MoMdp<N>,
    M::State: Clone,
{
    type State = M::State;
    type Action = M::State;

    fn actions(&self, state: &M::State, mut emit: impl FnMut(M::State, [f64; N])) {
        self.0.actions(state, |action, cost| {
            self.0
                .outcomes(state, &action, |target, _| emit(target, cost));
        });
    }

    fn outcomes(&self, _state: &M::State, action: &M::State, mut emit: impl FnMut(M::State, f64)) {
        emit(action.clone(), 1.0);
    }
}

/// The abstraction heuristic: the exact multi-objective value of a smaller MOSSP, looked up
/// through a projection of the states.
///
/// The explicit-state form of the paper's `H^pdb_mossp`, which solves each pattern
/// database's projection with multi-objective topological value iteration: here, the first
/// estimate of a projected state runs [`mo_value_iteration`](crate::mo_value_iteration())
/// from it, and every value that run computes is cached. Over [`Determinized`] it gives the
/// determinisation heuristic.
///
/// Admissible (in the scalarised sense of the [module docs](crate::mo)) when the
/// abstraction is a relaxation: for every weight, the optimal weighted cost from
/// `project(s)` is at most the one from `s`. And when the bound of `objectives` is large
/// enough to contain the abstract optimal vectors. That bound caps the abstract problem,
/// not the concrete one: a determinisation's path costs can exceed every concrete expected
/// cost, so pass a bound above the abstract path costs, not the concrete `b`. A vector cut
/// off by the inner bound is missing from the estimate, which then overestimates for the
/// weights that favour it.
///
/// The estimate is never `None`. An abstract state without a vector under the inner bound,
/// an inner run that fails, and one that exhausts its budget all fall back to the zero
/// vector.
///
/// # Example
///
/// See [`Determinized`].
pub struct Abstraction<'a, M: MoMdp<N>, P, G, const N: usize> {
    /// The abstract MOSSP.
    mdp: &'a M,
    /// Maps a concrete state to its abstract state.
    project: P,
    /// The goal test of the abstract MOSSP.
    goal: &'a G,
    /// The bound and margin of the inner runs.
    objectives: Objectives<N>,
    /// The settings of the inner runs.
    config: Config,
    /// The value of every abstract state solved so far, without the bound `b`.
    table: HashMap<M::State, Vec<[f64; N]>>,
}

impl<'a, M: MoMdp<N>, P, G, const N: usize> Abstraction<'a, M, P, G, N> {
    /// An abstraction heuristic over `mdp` with goal `goal`, reached through `project`.
    /// The inner runs use `objectives` and the default [`Config`].
    pub fn new(mdp: &'a M, project: P, goal: &'a G, objectives: Objectives<N>) -> Self {
        Self {
            mdp,
            project,
            goal,
            objectives,
            config: Config::default(),
            table: HashMap::new(),
        }
    }

    /// Returns the same heuristic with another configuration for its inner runs.
    #[must_use]
    pub fn with_config(self, config: Config) -> Self {
        Self { config, ..self }
    }
}

impl<S, M, P, G, const N: usize> Heuristic<S, Vec<[f64; N]>> for Abstraction<'_, M, P, G, N>
where
    M: MoMdp<N>,
    M::State: Clone + Eq + Hash,
    P: FnMut(&S) -> M::State,
    G: Goal<M::State>,
{
    fn estimate(&mut self, state: &S) -> Option<Vec<[f64; N]>> {
        let abstract_state = (self.project)(state);
        if !self.table.contains_key(&abstract_state) {
            self.solve(abstract_state.clone());
        }

        match self.table.get(&abstract_state) {
            Some(value) if !value.is_empty() => Some(value.clone()),
            // No proper vector under the inner bound proves nothing about the concrete
            // problem (its costs can exceed the abstract bound); fall back to blind.
            _ => Some(alloc::vec![[0.0; N]]),
        }
    }
}

impl<M, P, G, const N: usize> Abstraction<'_, M, P, G, N>
where
    M: MoMdp<N>,
    M::State: Clone + Eq + Hash,
    G: Goal<M::State>,
{
    /// Solves the abstract MOSSP from `state` and caches every value of a converged run.
    fn solve(&mut self, state: M::State) {
        let Ok(solution) =
            mo_value_iteration(self.mdp, state, self.goal, self.objectives, self.config)
        else {
            return;
        };

        if solution.status() == crate::Status::BudgetExhausted {
            return;
        }

        for index in 0..solution.state_count() {
            let id = crate::space::Id::try_from(index).expect("ids fit u32");
            self.table
                .entry(solution.state_of(id).clone())
                .or_insert_with(|| solution.vectors_of(id).copied().collect());
        }
    }
}
