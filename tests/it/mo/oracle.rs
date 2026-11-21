//! An explicit multi-objective MDP given as tables, and a brute-force oracle over its
//! deterministic policies that shares no code with the crate.

use std::collections::VecDeque;

use stochastic_search::{MoMdp, MoSolution};

use crate::table::Table;

/// One action of an [`MoTable`] state.
#[derive(Debug, Clone)]
pub struct MoAction<const N: usize> {
    /// The cost vector.
    pub cost: [f64; N],
    /// The successors and their probabilities.
    pub outcomes: Vec<(usize, f64)>,
}

/// A multi-objective MDP over states `0..actions.len()`; actions are indices into a state's
/// list. A non-goal state without actions is a dead end.
#[derive(Debug, Clone)]
pub struct MoTable<const N: usize> {
    /// The actions of each state; never queried at the goal.
    pub actions: Vec<Vec<MoAction<N>>>,
    /// The goal state.
    pub goal: usize,
}

impl<const N: usize> MoTable<N> {
    /// The number of states.
    pub fn len(&self) -> usize {
        self.actions.len()
    }

    /// The scalar table of objective `objective` alone.
    pub fn projection(&self, objective: usize) -> Table {
        let actions = self
            .actions
            .iter()
            .map(|list| {
                list.iter()
                    .map(|action| crate::table::Action {
                        cost: action.cost[objective],
                        outcomes: action.outcomes.clone(),
                    })
                    .collect()
            })
            .collect();

        Table {
            actions,
            goal: self.goal,
        }
    }
}

impl<const N: usize> MoMdp<N> for MoTable<N> {
    type State = usize;
    type Action = usize;

    fn actions(&self, state: &usize, mut emit: impl FnMut(usize, [f64; N])) {
        for (index, action) in self.actions[*state].iter().enumerate() {
            emit(index, action.cost);
        }
    }

    fn outcomes(&self, state: &usize, action: &usize, mut emit: impl FnMut(usize, f64)) {
        for &(target, p) in &self.actions[*state][*action].outcomes {
            emit(target, p);
        }
    }
}

/// A one-objective view of a scalar [`Table`].
pub struct Scalar<'a>(pub &'a Table);

impl MoMdp<1> for Scalar<'_> {
    type State = usize;
    type Action = usize;

    fn actions(&self, state: &usize, mut emit: impl FnMut(usize, [f64; 1])) {
        for (index, action) in self.0.actions[*state].iter().enumerate() {
            emit(index, [action.cost]);
        }
    }

    fn outcomes(&self, state: &usize, action: &usize, mut emit: impl FnMut(usize, f64)) {
        for &(target, p) in &self.0.actions[*state][*action].outcomes {
            emit(target, p);
        }
    }
}

/// The exact expected cost vector from `initial` of the deterministic policy `policy`
/// (`None` where it has no action), or `None` if the policy is improper from `initial`.
pub fn evaluate<const N: usize>(
    table: &MoTable<N>,
    initial: usize,
    policy: impl Fn(usize) -> Option<usize>,
) -> Option<[f64; N]> {
    if initial == table.goal {
        return Some([0.0; N]);
    }

    let reachable = reachable(table, initial, &policy)?;
    if !reaches_goal(table, &reachable, &policy) {
        return None;
    }

    Some(solve_linear(table, &reachable, &policy, initial))
}

/// The non-goal states reachable from `initial` under `policy`, or `None` if one of them
/// has no action.
fn reachable<const N: usize>(
    table: &MoTable<N>,
    initial: usize,
    policy: &impl Fn(usize) -> Option<usize>,
) -> Option<Vec<usize>> {
    let mut seen = vec![false; table.len()];
    let mut order = Vec::new();
    let mut queue = VecDeque::from([initial]);
    seen[initial] = true;
    while let Some(state) = queue.pop_front() {
        if state == table.goal {
            continue;
        }

        order.push(state);
        let action = policy(state)?;
        for &(target, _) in &table.actions[state][action].outcomes {
            if !seen[target] {
                seen[target] = true;
                queue.push_back(target);
            }
        }
    }

    Some(order)
}

/// Returns `true` if every state of `reachable` reaches the goal in the policy's graph,
/// which in a finite chain means the goal is reached with probability 1.
fn reaches_goal<const N: usize>(
    table: &MoTable<N>,
    reachable: &[usize],
    policy: &impl Fn(usize) -> Option<usize>,
) -> bool {
    let mut good = vec![false; table.len()];
    good[table.goal] = true;
    loop {
        let mut changed = false;
        for &state in reachable {
            if good[state] {
                continue;
            }

            let action = policy(state).expect("reachable states have actions");
            if table.actions[state][action]
                .outcomes
                .iter()
                .any(|&(target, _)| good[target])
            {
                good[state] = true;
                changed = true;
            }
        }

        if !changed {
            return reachable.iter().all(|&state| good[state]);
        }
    }
}

/// Solves `V = C + P V` over `reachable` per objective by Gaussian elimination and returns
/// `V(initial)`.
fn solve_linear<const N: usize>(
    table: &MoTable<N>,
    reachable: &[usize],
    policy: &impl Fn(usize) -> Option<usize>,
    initial: usize,
) -> [f64; N] {
    let n = reachable.len();
    let index = |state: usize| reachable.iter().position(|&s| s == state);
    let mut matrix = vec![vec![0.0; n + N]; n];
    for (row, &state) in reachable.iter().enumerate() {
        let action = &table.actions[state][policy(state).expect("has an action")];
        matrix[row][row] += 1.0;
        for &(target, p) in &action.outcomes {
            if let Some(column) = index(target) {
                matrix[row][column] -= p;
            }
        }

        matrix[row][n..].copy_from_slice(&action.cost);
    }

    for column in 0..n {
        let pivot = (column..n)
            .max_by(|&a, &b| matrix[a][column].abs().total_cmp(&matrix[b][column].abs()))
            .expect("non-empty");
        matrix.swap(column, pivot);
        let scale = matrix[column][column];
        for cell in &mut matrix[column] {
            *cell /= scale;
        }

        for row in 0..n {
            if row != column {
                let factor = matrix[row][column];
                for k in 0..n + N {
                    matrix[row][k] -= factor * matrix[column][k];
                }
            }
        }
    }

    let row = index(initial).expect("initial is reachable");

    std::array::from_fn(|objective| matrix[row][n + objective])
}

/// The expected cost vectors from `initial` of every proper deterministic policy.
pub fn proper_values<const N: usize>(table: &MoTable<N>, initial: usize) -> Vec<[f64; N]> {
    let counts: Vec<usize> = table
        .actions
        .iter()
        .enumerate()
        .map(|(state, list)| {
            if state == table.goal {
                1
            } else {
                list.len().max(1)
            }
        })
        .collect();
    let mut choice = vec![0; table.len()];
    let mut values = Vec::new();
    loop {
        let policy = |state: usize| (!table.actions[state].is_empty()).then(|| choice[state]);
        if let Some(value) = evaluate(table, initial, policy) {
            values.push(value);
        }

        let mut digit = 0;
        loop {
            if digit == choice.len() {
                return values;
            }

            choice[digit] += 1;
            if choice[digit] < counts[digit] {
                break;
            }

            choice[digit] = 0;
            digit += 1;
        }
    }
}

/// The smallest `w . v` over `vectors`.
pub fn support<const N: usize>(vectors: &[[f64; N]], weight: &[f64; N]) -> f64 {
    vectors
        .iter()
        .map(|v| v.iter().zip(weight).map(|(a, b)| a * b).sum::<f64>())
        .fold(f64::INFINITY, f64::min)
}

/// The vertices of the lower-left convex hull of 2D vectors, sorted by the first
/// objective; points within `tolerance` (cross product) of an edge are not vertices.
pub fn hull(vectors: &[[f64; 2]], tolerance: f64) -> Vec<[f64; 2]> {
    let mut front: Vec<[f64; 2]> = vectors
        .iter()
        .copied()
        .filter(|v| {
            !vectors.iter().any(|u| {
                // Dominance up to rounding: evaluations of equal values differ in the last bits.
                u[0] <= v[0] + 1e-9
                    && u[1] <= v[1] + 1e-9
                    && (u[0] < v[0] - 1e-9 || u[1] < v[1] - 1e-9)
            })
        })
        .collect();
    front.sort_by(|a, b| a[0].total_cmp(&b[0]));
    front.dedup_by(|a, b| (a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9);

    let mut hull: Vec<[f64; 2]> = Vec::new();
    for point in front {
        while hull.len() >= 2 {
            let (a, b) = (hull[hull.len() - 2], hull[hull.len() - 1]);
            let cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]);
            if cross <= tolerance {
                hull.pop();
            } else {
                break;
            }
        }

        hull.push(point);
    }

    hull
}

/// Checks every policy of `solution`: it is proper, and its exact expected cost vector is
/// within `tolerance` of the value the solution reports.
pub fn assert_policies_evaluate<const N: usize>(
    table: &MoTable<N>,
    solution: &MoSolution<usize, usize, N>,
    initial: usize,
    tolerance: f64,
) {
    for policy in solution.policies() {
        let exact = evaluate(table, initial, |state| policy.action(&state).copied())
            .unwrap_or_else(|| panic!("policy for {:?} is improper", policy.value()));
        let error = exact
            .iter()
            .zip(policy.value())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            error <= tolerance,
            "policy value {:?}, exact {exact:?}",
            policy.value()
        );
    }
}
