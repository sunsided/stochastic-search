//! Value iteration over the reachable state space, the reference oracle for the heuristic
//! search algorithms.

use core::hash::Hash;

use crate::budget::Budget;
use crate::config::Config;

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
/// sweeps converge whenever every action costs more than zero; if the initial state is one
/// of them, the status is [`Status::Unsolvable`](crate::Status::Unsolvable). Both the state
/// enumeration and the sweeps ask [`Config::budget`] before every step, so the default
/// limits stop runs on huge or infinite state spaces.
///
/// Meant as an exact reference on small problems: it stores the whole reachable space.
/// [`Config::seed`] and [`Config::max_trial_length`] are ignored.
///
/// # Errors
///
/// Returns an [`Error`](crate::Error) for an invalid [`Config`] or invalid MDP output in a reachable state.
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
    let mut problem = Problem {
        mdp,
        goal,
        heuristic: Zero,
        cap: config.dead_ends.cap(),
    };

    let mut space = Space::new(problem.cap);
    let start = space.intern(&mut problem, initial)?;
    let mut meter = Meter::new(config.budget);
    let status = finish(solve(&mut space, &mut problem, &mut meter, config.epsilon))?;
    meter.sync(&space);

    Ok(Solution::new(space, start, status, meter.stats))
}

/// Explores the reachable space, proves dead ends under infinite cost, and sweeps.
///
/// # Errors
///
/// Returns [`Halt`] when the budget runs out or the input is invalid.
fn solve<M, G, B>(
    space: &mut Space<M::State, M::Action>,
    problem: &mut Problem<'_, M, G, Zero>,
    meter: &mut Meter<B>,
    epsilon: f64,
) -> Result<(), Halt<M::State>>
where
    M: Mdp,
    M::State: Clone + Eq + Hash,
    G: Goal<M::State>,
    B: Budget,
{
    explore(space, problem, meter)?;
    if problem.cap.is_infinite() {
        space.mark_dead_ends();
    }

    sweep_until_converged(space, meter, epsilon)
}

/// Expands every state reachable from the interned states, in discovery order, asking the
/// budget before each expansion.
///
/// # Errors
///
/// Returns [`Halt`] when the budget runs out or the input is invalid.
fn explore<M, G, B>(
    space: &mut Space<M::State, M::Action>,
    problem: &mut Problem<'_, M, G, Zero>,
    meter: &mut Meter<B>,
) -> Result<(), Halt<M::State>>
where
    M: Mdp,
    M::State: Clone + Eq + Hash,
    G: Goal<M::State>,
    B: Budget,
{
    let mut next = 0;
    while next < space.len() {
        meter.check(space)?;
        space.expand(problem, id_of(next))?;
        next += 1;
    }

    Ok(())
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
