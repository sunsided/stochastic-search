//! Value iteration over the reachable state space, the reference oracle for the heuristic
//! search algorithms.

mod almost_sure;

use core::hash::Hash;

use crate::budget::Budget;
use crate::config::Config;
use crate::error::Error;

use crate::meter::{Halt, Meter, finish};
use crate::model::{Goal, Mdp, Zero};
use crate::solution::{Solution, SolveResult};
use crate::space::{Id, Kind, Problem, Space};

/// Solves an SSP from `initial` with Gauss-Seidel value iteration over every reachable state.
///
/// Enumerates all states reachable from `initial`, then sweeps them in discovery order until
/// the largest Bellman residual of a sweep is below [`Config::epsilon`]. Under
/// [`DeadEnds::Infinite`](crate::DeadEnds::Infinite), states without a policy that reaches a
/// goal with probability 1 are found first by graph analysis and valued at infinity, so the
/// sweeps converge whenever every action costs more than zero.
///
/// Meant as an exact reference on small problems: it stores the whole reachable space.
/// [`Config::seed`] and [`Config::max_trial_length`] are ignored.
///
/// # Errors
///
/// Returns an [`Error`] for an invalid [`Config`] or invalid MDP output in a reachable state.
///
/// # Panics
///
/// Panics if more than `u32::MAX` states, actions or outcomes are reachable.
///
/// # Example
///
/// ```
/// use stochastic_search::{Config, Mdp, value_iteration};
///
/// /// Walks right along `0..=3`; each step costs 1 and slips back in place with
/// /// probability 0.25.
/// struct Corridor;
///
/// impl Mdp for Corridor {
///     type State = u8;
///     type Action = ();
///
///     fn actions(&self, _state: &u8, mut emit: impl FnMut((), f64)) {
///         emit((), 1.0);
///     }
///
///     fn outcomes(&self, state: &u8, _action: &(), mut emit: impl FnMut(u8, f64)) {
///         emit(state + 1, 0.75);
///         emit(*state, 0.25);
///     }
/// }
///
/// let config = Config { epsilon: 1e-12, ..Config::default() };
/// let solution = value_iteration(&Corridor, 0, &|state: &u8| *state == 3, config)?;
///
/// // Three steps, each taking 1 / 0.75 attempts in expectation.
/// assert!((solution.initial_value() - 4.0).abs() < 1e-9);
/// # Ok::<(), stochastic_search::Error<u8>>(())
/// ```
pub fn value_iteration<M, G, B>(
    mdp: &M,
    initial: M::State,
    goal: &G,
    config: Config<B>,
) -> SolveResult<M>
where
    M: Mdp,
    M::State: Clone + Eq + Hash,
    G: Goal<M::State>,
    B: Budget,
{
    config.validate()?;
    let problem = Problem {
        mdp,
        goal,
        heuristic: &Zero,
        cap: config.dead_ends.cap(),
    };

    let mut space = Space::new(problem.cap);
    let start = space.intern(&problem, initial)?;
    explore(&mut space, &problem)?;
    if problem.cap.is_infinite() {
        exclude_dead_ends(&mut space);
    }

    let mut meter = Meter::new(config.budget);
    let status = finish(sweep_until_converged(
        &mut space,
        &mut meter,
        config.epsilon,
    ))?;
    meter.sync(&space);

    Ok(Solution::new(space, start, status, meter.stats))
}

/// Expands every state reachable from the interned states, in discovery order.
///
/// # Errors
///
/// Returns the first invalid MDP output.
fn explore<M, G>(
    space: &mut Space<M::State, M::Action>,
    problem: &Problem<'_, M, G, Zero>,
) -> Result<(), Error<M::State>>
where
    M: Mdp,
    M::State: Clone + Eq + Hash,
    G: Goal<M::State>,
{
    let mut next = 0;
    while next < space.len() {
        space.expand(problem, id_of(next))?;
        next += 1;
    }

    Ok(())
}

/// Turns every open state without a proper policy into a dead end valued at infinity.
fn exclude_dead_ends<S, A>(space: &mut Space<S, A>) {
    let proper = almost_sure::reaches_goal_almost_surely(space);
    for (index, has_proper_policy) in proper.into_iter().enumerate() {
        let node = space.node_mut(id_of(index));
        if !has_proper_policy && node.kind == Kind::Open {
            node.kind = Kind::DeadEnd;
            node.value = f64::INFINITY;
        }
    }
}

/// Runs Gauss-Seidel sweeps over the open states until a sweep's largest residual is below
/// `epsilon`.
///
/// # Errors
///
/// Returns [`Halt::Budget`] when the budget runs out.
fn sweep_until_converged<S, A, B: Budget>(
    space: &mut Space<S, A>,
    meter: &mut Meter<B>,
    epsilon: f64,
) -> Result<(), Halt<S>> {
    loop {
        let mut largest = 0.0_f64;
        for index in 0..space.len() {
            let id = id_of(index);
            if space.node(id).kind != Kind::Open {
                continue;
            }

            meter.charge_backup(space)?;
            largest = largest.max(space.update(id));
        }

        meter.stats.sweeps += 1;
        if largest < epsilon {
            return Ok(());
        }
    }
}

/// Converts a state index below [`Space::len`] to an id.
///
/// # Panics
///
/// Never panics for such indices, since interning caps the space at `u32::MAX` states.
fn id_of(index: usize) -> Id {
    Id::try_from(index).expect("interned indices fit an id")
}
