//! Benchmark POMDPs for the belief-space planners: Tiger and a small `RockSample`.

use rand_core::Rng;
use stochastic_search::{Pomdp, RolloutPolicy, Step, unit_interval};

/// The discount factor both benchmarks are usually solved with.
pub const DISCOUNT: f64 = 0.95;

/// An action of [`Tiger`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TigerAction {
    /// Listen for the tiger: reward -1, hears the correct side with probability 0.85.
    Listen,
    /// Open the left door: -100 if the tiger is there, +10 otherwise; then a new round.
    OpenLeft,
    /// Open the right door, likewise.
    OpenRight,
}

/// The Tiger problem (Kaelbling, Littman and Cassandra 1998). The state is `true` if the
/// tiger is behind the left door; observations are `Some(true)` for hearing it on the left,
/// `Some(false)` on the right, and `None` after opening a door. Opening a door starts a new
/// round with the tiger placed uniformly at random, as in the original.
pub struct Tiger;

impl Pomdp for Tiger {
    type State = bool;
    type Action = TigerAction;
    type Observation = Option<bool>;

    fn actions(&self, mut emit: impl FnMut(TigerAction)) {
        emit(TigerAction::Listen);
        emit(TigerAction::OpenLeft);
        emit(TigerAction::OpenRight);
    }

    fn step<R: Rng + ?Sized>(
        &self,
        left: &bool,
        action: &TigerAction,
        rng: &mut R,
    ) -> Step<bool, Option<bool>> {
        let opened_left = match action {
            TigerAction::Listen => {
                let correct = unit_interval(rng) < 0.85;
                return Step::new(*left, Some(*left == correct), -1.0);
            }
            TigerAction::OpenLeft => true,
            TigerAction::OpenRight => false,
        };

        let reward = if opened_left == *left { -100.0 } else { 10.0 };

        Step::new(unit_interval(rng) < 0.5, None, reward)
    }
}

/// An action of [`RockSample`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RockAction {
    /// Move one cell north; no effect at the border.
    North,
    /// Move one cell south; no effect at the border.
    South,
    /// Move one cell east; leaving the grid on the east side ends the episode with +10.
    East,
    /// Move one cell west; no effect at the border.
    West,
    /// Sample the rock under the agent: +10 if good (it turns bad), -10 if bad or no rock.
    Sample,
    /// Check rock `i`: observe its quality, correct with probability `(1 + eta) / 2` where
    /// `eta = 2^(-distance / 20)`.
    Check(usize),
}

/// A state of [`RockSample`]: the agent position and the quality of every rock.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RockState {
    /// The agent's column, `0..size`.
    pub x: i32,
    /// The agent's row, `0..size`.
    pub y: i32,
    /// `good[i]` is whether rock `i` is good.
    pub good: Vec<bool>,
}

/// `RockSample` (Smith and Simmons 2004) on a `size` by `size` grid. Observations are
/// `Some(good)` after a check and `None` otherwise.
pub struct RockSample {
    /// The side length of the grid.
    pub size: i32,
    /// The rock positions.
    pub rocks: Vec<(i32, i32)>,
}

impl RockSample {
    /// The reward and successor of a move by `(dx, dy)`.
    fn moved(&self, state: &RockState, dx: i32, dy: i32) -> Step<RockState, Option<bool>> {
        let (x, y) = (state.x + dx, state.y + dy);
        if x >= self.size {
            return Step::new(state.clone(), None, 10.0).terminal();
        }

        let mut next = state.clone();
        if (0..self.size).contains(&x) && (0..self.size).contains(&y) {
            next.x = x;
            next.y = y;
        }

        Step::new(next, None, 0.0)
    }

    /// The reward and successor of sampling at the agent position.
    fn sampled(&self, state: &RockState) -> Step<RockState, Option<bool>> {
        let rock = self.rocks.iter().position(|&r| r == (state.x, state.y));
        let mut next = state.clone();
        let reward = match rock {
            Some(i) if state.good[i] => {
                next.good[i] = false;
                10.0
            }
            _ => -10.0,
        };

        Step::new(next, None, reward)
    }

    /// The noisy observation of checking `rock`.
    fn checked<R: Rng + ?Sized>(
        &self,
        state: &RockState,
        rock: usize,
        rng: &mut R,
    ) -> Step<RockState, Option<bool>> {
        let (rx, ry) = self.rocks[rock];
        let distance = f64::from((rx - state.x).pow(2) + (ry - state.y).pow(2)).sqrt();
        let efficiency = (-distance / 20.0).exp2();
        let correct = unit_interval(rng) < f64::midpoint(1.0, efficiency);

        Step::new(state.clone(), Some(state.good[rock] == correct), 0.0)
    }
}

impl Pomdp for RockSample {
    type State = RockState;
    type Action = RockAction;
    type Observation = Option<bool>;

    fn actions(&self, mut emit: impl FnMut(RockAction)) {
        for action in [
            RockAction::North,
            RockAction::South,
            RockAction::East,
            RockAction::West,
            RockAction::Sample,
        ] {
            emit(action);
        }

        for rock in 0..self.rocks.len() {
            emit(RockAction::Check(rock));
        }
    }

    fn step<R: Rng + ?Sized>(
        &self,
        state: &RockState,
        action: &RockAction,
        rng: &mut R,
    ) -> Step<RockState, Option<bool>> {
        match *action {
            RockAction::North => self.moved(state, 0, 1),
            RockAction::South => self.moved(state, 0, -1),
            RockAction::East => self.moved(state, 1, 0),
            RockAction::West => self.moved(state, -1, 0),
            RockAction::Sample => self.sampled(state),
            RockAction::Check(rock) => self.checked(state, rock, rng),
        }
    }
}

/// `RockSample` on a 3 by 3 grid with one rock under the agent's start `(0, 1)`. The exit is
/// three moves east.
pub fn one_rock() -> RockSample {
    RockSample {
        size: 3,
        rocks: vec![(0, 1)],
    }
}

/// A particle belief for [`one_rock`] with the agent at the start and the rock good in
/// `good` of `total` particles.
pub fn one_rock_belief(good: usize, total: usize) -> Vec<RockState> {
    (0..total)
        .map(|i| RockState {
            x: 0,
            y: 1,
            good: vec![i < good],
        })
        .collect()
}

/// A Tiger belief with the tiger on the left in `left` of `total` particles.
pub fn tiger_belief(left: usize, total: usize) -> Vec<bool> {
    (0..total).map(|i| i < left).collect()
}

/// An upper bound on the discounted return of [`RockSample`] from `state`: every good rock
/// sampled and the exit reached, without discounting.
pub fn rock_upper_bound(state: &RockState) -> f64 {
    let good = state.good.iter().filter(|&&g| g).count();

    10.0 * f64::from(u32::try_from(good).expect("few rocks")) + 10.0
}

/// An upper bound on the discounted return of [`Tiger`]: +10 at every step.
pub const TIGER_UPPER_BOUND: f64 = 10.0 / (1.0 - DISCOUNT);

/// The usual Tiger rollout policy: always listen. Random rollouts open doors at an expected
/// -45 per step, which buries the first step's difference under rollout noise.
pub struct AlwaysListen;

impl RolloutPolicy<bool> for AlwaysListen {
    fn action<R: Rng + ?Sized>(&mut self, _state: &bool, _actions: usize, _rng: &mut R) -> usize {
        0
    }
}

/// The usual `RockSample` default policy: head east to the exit.
pub struct HeadEast;

impl RolloutPolicy<RockState> for HeadEast {
    fn action<R: Rng + ?Sized>(
        &mut self,
        _state: &RockState,
        _actions: usize,
        _rng: &mut R,
    ) -> usize {
        2
    }
}
