//! Random small SSPs for the property tests.

use crate::table::{Action, Table};

/// A small xorshift generator, so the tables depend only on the proptest-drawn seed.
struct XorShift(u64);

impl XorShift {
    /// Creates a generator; the state is forced non-zero.
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    /// Returns a value in `0..bound`.
    fn below(&mut self, bound: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;

        usize::try_from(self.0 % bound as u64).expect("below a usize bound")
    }

    /// Returns an integer-valued float in `low..=high`.
    fn float(&mut self, low: usize, high: usize) -> f64 {
        f64::from(u32::try_from(low + self.below(high - low + 1)).expect("small"))
    }
}

/// Which kind of random SSP to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    /// Every state has a proper policy.
    Proper,
    /// Extra trap states (a costly self-loop, or no actions at all) that some actions may
    /// enter; the chain actions avoid them, so every non-trap state has a proper policy.
    Traps,
}

/// Builds an SSP over `states` regular states (the last one is the goal), plus up to two
/// trap states for [`Shape::Traps`]. State `0` is the initial state. Costs are integers in
/// `1..=10`.
///
/// Action 0 of state `i` reaches `i + 1` with positive probability and otherwise stays
/// among the regular states, so following it reaches the goal with probability 1.
pub fn ssp(states: usize, seed: u64, shape: Shape) -> Table {
    let mut rng = XorShift::new(seed);
    let traps = if shape == Shape::Traps { 2 } else { 0 };
    let total = states + traps;
    let goal = states - 1;

    let mut actions = Vec::with_capacity(total);
    for state in 0..states {
        if state == goal {
            actions.push(Vec::new());
            continue;
        }

        let mut list = vec![chain_action(&mut rng, state, states)];
        for _ in 0..rng.below(3) {
            list.push(random_action(&mut rng, total));
        }

        actions.push(list);
    }

    if traps > 0 {
        actions.push(vec![random_action_into(&mut rng, &[states])]);
        actions.push(Vec::new());
    }

    Table { actions, goal }
}

/// An action that advances from `state` to `state + 1` with positive probability and
/// otherwise moves among the regular states.
fn chain_action(rng: &mut XorShift, state: usize, states: usize) -> Action {
    let mut targets = vec![state + 1];
    for _ in 0..rng.below(3) {
        targets.push(rng.below(states));
    }

    random_action_into(rng, &targets)
}

/// An action with one to three uniformly random successors among `0..total`.
fn random_action(rng: &mut XorShift, total: usize) -> Action {
    let targets: Vec<usize> = (0..=rng.below(3)).map(|_| rng.below(total)).collect();

    random_action_into(rng, &targets)
}

/// An action over the given successors with random positive weights and a random cost.
fn random_action_into(rng: &mut XorShift, targets: &[usize]) -> Action {
    let weights: Vec<f64> = targets.iter().map(|_| rng.float(1, 10)).collect();
    let total: f64 = weights.iter().sum();

    Action {
        cost: rng.float(1, 10),
        outcomes: targets
            .iter()
            .zip(&weights)
            .map(|(&t, &w)| (t, w / total))
            .collect(),
    }
}
