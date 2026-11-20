//! Checks computed from the raw [`Table`] only, sharing no code with the crate.

use std::collections::VecDeque;

use stochastic_search::Solution;

use crate::table::Table;

/// The Q-value of `action` in `state` under the solution's values.
///
/// # Panics
///
/// Panics if a successor has no value in the solution.
pub fn q_value(
    table: &Table,
    solution: &Solution<usize, usize>,
    state: usize,
    action: usize,
) -> f64 {
    let entry = &table.actions[state][action];
    let expected: f64 = entry
        .outcomes
        .iter()
        .map(|&(target, p)| p * solution.value(&target).expect("successor has a value"))
        .sum();

    entry.cost + expected
}

/// The states reachable from the initial state when following the solution's policy,
/// in breadth-first order. Stops at goals and at states without a policy action.
pub fn policy_envelope(
    table: &Table,
    solution: &Solution<usize, usize>,
    initial: usize,
) -> Vec<usize> {
    let mut seen = vec![false; table.len()];
    let mut order = Vec::new();
    let mut queue = VecDeque::from([initial]);
    seen[initial] = true;
    while let Some(state) = queue.pop_front() {
        order.push(state);
        let Some(&action) = solution.action(&state) else {
            continue;
        };

        for &(target, _) in &table.actions[state][action].outcomes {
            if !seen[target] {
                seen[target] = true;
                queue.push_back(target);
            }
        }
    }

    order
}

/// Returns `true` if the policy reaches the goal with probability 1 from `initial`.
///
/// In a finite Markov chain, the goal is reached with probability 1 iff every state
/// reachable from `initial` can reach the goal. Checked on the policy's support graph.
pub fn policy_is_proper(table: &Table, solution: &Solution<usize, usize>, initial: usize) -> bool {
    let envelope = policy_envelope(table, solution, initial);
    let mut reaches = vec![false; table.len()];
    reaches[table.goal] = true;

    let mut changed = true;
    while changed {
        changed = false;
        for &state in &envelope {
            if reaches[state] {
                continue;
            }

            let Some(&action) = solution.action(&state) else {
                continue;
            };

            if table.actions[state][action]
                .outcomes
                .iter()
                .any(|&(t, _)| reaches[t])
            {
                reaches[state] = true;
                changed = true;
            }
        }
    }

    envelope.iter().all(|&state| reaches[state])
}

/// Exact-enough optimal values by plain Jacobi value iteration from the raw tables, for
/// problems where every state has a proper policy and costs are positive.
pub fn optimal_values(table: &Table) -> Vec<f64> {
    let mut values = vec![0.0; table.len()];
    loop {
        let next: Vec<f64> = (0..table.len())
            .map(|state| {
                if state == table.goal {
                    return 0.0;
                }

                table.actions[state]
                    .iter()
                    .map(|a| a.cost + a.outcomes.iter().map(|&(t, p)| p * values[t]).sum::<f64>())
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();

        let change = next
            .iter()
            .zip(&values)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        values = next;
        if change < 1e-12 {
            return values;
        }
    }
}

/// The all-outcomes determinization heuristic: the cheapest cost to the goal if the agent
/// could pick each action's outcome. Admissible and consistent; infinite where the goal is
/// unreachable.
pub fn determinization(table: &Table) -> Vec<f64> {
    let mut h = vec![f64::INFINITY; table.len()];
    h[table.goal] = 0.0;
    for _ in 0..table.len() {
        for state in 0..table.len() {
            if state == table.goal {
                continue;
            }

            for action in &table.actions[state] {
                let best = action
                    .outcomes
                    .iter()
                    .map(|&(t, _)| h[t])
                    .fold(f64::INFINITY, f64::min);
                h[state] = h[state].min(action.cost + best);
            }
        }
    }

    h
}

/// The expected cost of always taking action 0, by Jacobi policy evaluation from the raw
/// tables; infinity where action 0 does not reach the goal with probability 1.
///
/// States that cannot reach the goal under action 0 are fixed at infinity first, which
/// spreads to every state that reaches them. Iterating from 0 approaches the finite values
/// from below; the result is the last iterate plus 1, which is above the policy's true value
/// (the residual is at most 1e-12 and no expected step count in the random tables comes
/// near 10^12), and therefore an upper bound on the optimal value.
pub fn first_action_values(table: &Table) -> Vec<f64> {
    let reaches = reaches_goal_under_first_action(table);
    let mut values: Vec<f64> = reaches
        .iter()
        .map(|&r| if r { 0.0 } else { f64::INFINITY })
        .collect();
    loop {
        let next: Vec<f64> = (0..table.len())
            .map(|state| match table.actions[state].first() {
                _ if state == table.goal => 0.0,
                Some(a) if reaches[state] => {
                    a.cost + a.outcomes.iter().map(|&(t, p)| p * values[t]).sum::<f64>()
                }
                _ => f64::INFINITY,
            })
            .collect();

        let change = next
            .iter()
            .zip(&values)
            .filter(|(a, _)| a.is_finite())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        values = next;
        if change < 1e-12 {
            return values.into_iter().map(|v| v + 1.0).collect();
        }
    }
}

/// Marks the states from which the goal is reachable with positive probability when always
/// taking action 0.
fn reaches_goal_under_first_action(table: &Table) -> Vec<bool> {
    let mut reaches = vec![false; table.len()];
    reaches[table.goal] = true;
    let mut changed = true;
    while changed {
        changed = false;
        for state in 0..table.len() {
            let Some(first) = table.actions[state].first() else {
                continue;
            };

            if !reaches[state] && first.outcomes.iter().any(|&(t, _)| reaches[t]) {
                reaches[state] = true;
                changed = true;
            }
        }
    }

    reaches
}
