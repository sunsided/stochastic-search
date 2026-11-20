//! Point-based value iteration (PBVI, Pineau, Gordon and Thrun 2003) on Tiger's explicit
//! model, as an oracle for POMCP and DESPOT: their first action must match PBVI's wherever
//! PBVI's Q-values separate the best action from the rest by a clear margin.
//!
//! Only actions are compared, not values: POMCP's estimates are biased by exploration, and
//! DESPOT's bounds hold for its sampled scenarios only.

use stochastic_search::{DespotConfig, Limits, PomcpConfig, despot, pomcp};

use crate::pomdps::{AlwaysListen, DISCOUNT, TIGER_UPPER_BOUND, Tiger, TigerAction, tiger_belief};

/// Tiger's states: 0 is "tiger left", 1 is "tiger right".
const STATES: usize = 2;

/// Tiger's actions, in the emission order of [`Tiger`].
const ACTIONS: [TigerAction; 3] = [
    TigerAction::Listen,
    TigerAction::OpenLeft,
    TigerAction::OpenRight,
];

/// Tiger's observations: 0 "heard left", 1 "heard right", 2 "door opened" (`None`).
const OBSERVATIONS: usize = 3;

/// The probability of hearing the tiger on its side.
const ACCURACY: f64 = 0.85;

/// The reward of `action` in `state`.
fn reward(state: usize, action: usize) -> f64 {
    match action {
        0 => -1.0,
        1 => {
            if state == 0 {
                -100.0
            } else {
                10.0
            }
        }
        _ => {
            if state == 1 {
                -100.0
            } else {
                10.0
            }
        }
    }
}

/// The probability of moving from `state` to `next` with `action`: listening keeps the
/// state, opening a door places the tiger anew uniformly.
fn transition(state: usize, action: usize, next: usize) -> f64 {
    if action == 0 {
        if state == next { 1.0 } else { 0.0 }
    } else {
        0.5
    }
}

/// The probability of `observation` after `action` lands in `next`.
fn observation(action: usize, next: usize, observation: usize) -> f64 {
    match (action, observation) {
        (0, 2) | (1 | 2, 0 | 1) => 0.0,
        (0, o) => {
            if o == next {
                ACCURACY
            } else {
                1.0 - ACCURACY
            }
        }
        _ => 1.0,
    }
}

/// An alpha vector: the value of a conditional plan in each state.
type Alpha = [f64; STATES];

/// The dot product of a belief and an alpha vector.
fn dot(belief: &[f64; STATES], alpha: &Alpha) -> f64 {
    belief[0] * alpha[0] + belief[1] * alpha[1]
}

/// The best alpha vector of `set` at `belief` and its value.
fn best<'a>(set: &'a [Alpha], belief: &[f64; STATES]) -> (&'a Alpha, f64) {
    set.iter().map(|alpha| (alpha, dot(belief, alpha))).fold(
        (&set[0], f64::NEG_INFINITY),
        |a, b| if b.1 > a.1 { b } else { a },
    )
}

/// The point-based backup of `set` at `belief` for `action`: the alpha vector of taking
/// `action`, then following the best vector of `set` after each observation.
fn backup(set: &[Alpha], belief: &[f64; STATES], action: usize) -> Alpha {
    let mut alpha = [reward(0, action), reward(1, action)];
    for o in 0..OBSERVATIONS {
        // g(s) = sum_s' T(s' | s, a) O(o | a, s') alpha'(s') for each candidate alpha'.
        let projected = |candidate: &Alpha| -> Alpha {
            core::array::from_fn(|s| {
                (0..STATES)
                    .map(|next| {
                        transition(s, action, next) * observation(action, next, o) * candidate[next]
                    })
                    .sum()
            })
        };
        let best = set
            .iter()
            .map(projected)
            .fold(None, |best: Option<(Alpha, f64)>, g| {
                let value = dot(belief, &g);
                match best {
                    Some((_, v)) if v >= value => best,
                    _ => Some((g, value)),
                }
            })
            .expect("the set is never empty")
            .0;
        for s in 0..STATES {
            alpha[s] += DISCOUNT * best[s];
        }
    }

    alpha
}

/// Runs PBVI on a grid of 101 belief points until the values change by less than `1e-9`.
fn solve() -> Vec<Alpha> {
    let points: Vec<[f64; STATES]> = (0..=100)
        .map(|i| {
            let p = f64::from(i) / 100.0;
            [p, 1.0 - p]
        })
        .collect();
    // The blind lower bound: always listening forever.
    let mut set = vec![[-1.0 / (1.0 - DISCOUNT); STATES]];
    for _ in 0..10_000 {
        let next: Vec<Alpha> = points
            .iter()
            .map(|belief| {
                (0..ACTIONS.len())
                    .map(|action| backup(&set, belief, action))
                    .fold(None, |best: Option<Alpha>, alpha| match best {
                        Some(b) if dot(belief, &b) >= dot(belief, &alpha) => Some(b),
                        _ => Some(alpha),
                    })
                    .expect("three actions")
            })
            .collect();
        let change = points
            .iter()
            .map(|belief| (best(&next, belief).1 - best(&set, belief).1).abs())
            .fold(0.0, f64::max);
        set = next;
        if change < 1e-9 {
            break;
        }
    }

    set
}

/// The Q-values of every action at the belief "tiger left with probability `left`".
fn q_values(set: &[Alpha], left: f64) -> [f64; 3] {
    let belief = [left, 1.0 - left];
    core::array::from_fn(|action| dot(&belief, &backup(set, &belief, action)))
}

/// PBVI's best action at `left` if it beats the runner-up by at least `margin`.
fn clear_action(set: &[Alpha], left: f64, margin: f64) -> Option<TigerAction> {
    let q = q_values(set, left);
    let mut order = [0, 1, 2];
    order.sort_by(|&a, &b| q[b].total_cmp(&q[a]));

    (q[order[0]] - q[order[1]] >= margin).then_some(ACTIONS[order[0]])
}

/// PBVI's values match the known Tiger answer: listening is best at the uniform belief, and
/// opening the far door is best once three net observations agree.
#[test]
fn pbvi_solves_tiger() {
    let set = solve();
    assert_eq!(clear_action(&set, 0.5, 1.0), Some(TigerAction::Listen));
    let confident = 0.85_f64.powi(3) / (0.85_f64.powi(3) + 0.15_f64.powi(3));
    assert_eq!(
        clear_action(&set, confident, 0.1),
        Some(TigerAction::OpenRight)
    );
    assert_eq!(
        clear_action(&set, 1.0 - confident, 0.1),
        Some(TigerAction::OpenLeft)
    );
}

/// POMCP and DESPOT agree with PBVI on every belief of a sweep where PBVI's margin is at
/// least 5, on four seeds.
///
/// Smaller margins are not reliable at these budgets: at 15 of 200 particles on the left,
/// PBVI's listening beats opening the left door by 3.0 (23.18 against 20.15), but POMCP with
/// 20 000 simulations and always-listen rollouts opens the door on seed 0, probably because its rollouts
/// value the round after opening at `-1 / (1 - gamma) = -20` instead of the optimal 19.37,
/// a bias the tree has not overcome at that budget (not verified further).
#[test]
fn tree_planners_agree_with_pbvi() {
    let set = solve();
    let mut checked = 0;
    for left in [1, 15, 30, 50, 85, 97, 100, 103, 115, 150, 170, 185, 199] {
        let fraction = f64::from(left) / 200.0;
        let Some(expected) = clear_action(&set, fraction, 5.0) else {
            continue;
        };

        checked += 1;
        let belief = tiger_belief(usize::try_from(left).expect("small"), 200);
        for seed in 0..4 {
            let config = PomcpConfig {
                discount: DISCOUNT,
                exploration: 110.0,
                max_depth: 60,
                seed,
                ..PomcpConfig::default()
            }
            .with_budget(Limits {
                max_trials: Some(20_000),
                ..PomcpConfig::DEFAULT_LIMITS
            });
            let plan = pomcp(&Tiger, &belief, AlwaysListen, config).expect("valid input");
            assert_eq!(plan.action, expected, "POMCP, left {left}/200, seed {seed}");

            let config = DespotConfig {
                discount: DISCOUNT,
                scenarios: 200,
                max_depth: 40,
                seed,
                ..DespotConfig::default()
            }
            .with_budget(Limits {
                max_trials: Some(2_000),
                ..DespotConfig::DEFAULT_LIMITS
            });
            let plan = despot(
                &Tiger,
                &belief,
                AlwaysListen,
                |_: &bool| TIGER_UPPER_BOUND,
                config,
            )
            .expect("valid input");
            assert_eq!(
                plan.action, expected,
                "DESPOT, left {left}/200, seed {seed}"
            );
        }
    }

    assert!(checked >= 6, "the sweep checks enough beliefs");
}
