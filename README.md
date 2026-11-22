# stochastic-search

[![crates.io](https://img.shields.io/crates/v/stochastic-search.svg)](https://crates.io/crates/stochastic-search)
[![docs.rs](https://img.shields.io/docsrs/stochastic-search)](https://docs.rs/stochastic-search)
[![CI](https://github.com/sunsided/stochastic-search/actions/workflows/ci.yml/badge.svg)](https://github.com/sunsided/stochastic-search/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sunsided/stochastic-search/graph/badge.svg)](https://codecov.io/gh/sunsided/stochastic-search)
[![MSRV](https://img.shields.io/crates/msrv/stochastic-search)](https://crates.io/crates/stochastic-search)
[![License](https://img.shields.io/crates/l/stochastic-search)](https://github.com/sunsided/stochastic-search#license)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)

Planning under probabilistic outcomes: LAO\*, iLAO\*, CG-iLAO\*, RTDP, LRTDP, BRTDP, FRTDP,
VPI-RTDP, POMCP, DESPOT, and the multi-objective MOLAO\*, iMOLAO\* and MOLRTDP.

When actions have uncertain outcomes (a wheel slips, a request times out), a single path is
no longer a plan: the crate solves stochastic shortest
path problems (SSPs) and returns a value function with a greedy policy that says what to do
in every state the plan can reach. Problems are described through a small `Mdp` trait with
callbacks, so states are generated on demand rather than enumerated up front; dead ends can
be treated as infinitely or finitely expensive. Partially observable problems use a separate
generative POMDP model with the online planners POMCP and DESPOT, and problems with several
cost dimensions get the multi-objective algorithms of Chen, Trevizan and Thiébaux. The crate
is `no_std` with `alloc`, and value iteration serves as the reference that the other
algorithms are tested against.

Part of the [planning and search crates](https://sunsided.github.io/planning-and-search/); see also this crate's [overview page](https://sunsided.github.io/planning-and-search/crates/stochastic-search/).

## Who this is for

- Robotics and autonomy work where actions have uncertain outcomes and the planner needs a
  policy with a convergence guarantee (LRTDP, iLAO\*) or with bounds on how far it is from
  optimal (BRTDP, FRTDP, VPI-RTDP).
- Online decision making under partial observability, e.g. a robot or game agent acting on
  noisy sensor readings, where POMCP or DESPOT plan from a particle belief and reuse their
  search tree between decisions.
- Planning with several competing costs (time against risk, energy against wear), where the
  multi-objective algorithms return the convex coverage set of policies instead of one
  weighted compromise.
- Researchers and students comparing MDP heuristic search algorithms on the same model and
  test oracle.
- If every action has exactly one outcome, use
  [heuristic-search](https://github.com/sunsided/heuristic-search) (or
  [multiobjective-search](https://github.com/sunsided/multiobjective-search) for several
  costs); it is simpler and much faster. Goal tests written against
  [search-core](https://github.com/sunsided/search-core) work in both, and heuristics carry
  over once their cost type is `f64`.

## Installation

```sh
cargo add stochastic-search
```

Or in `Cargo.toml`:

```toml
[dependencies]
stochastic-search = "0.1"
```

## Scope

- Value iteration, RTDP, labeled RTDP (LRTDP), classic LAO*, iLAO* and CG-iLAO* for stochastic
  shortest path problems, with infinite or finite-penalty dead ends
- Bounded RTDP (BRTDP), focused RTDP (FRTDP) and VPI-RTDP with lower and upper
  bounds and an upper-greedy policy; upper bounds must be finite, so infinite dead ends need
  a lower bound that proves them (or use a penalty)
- A generative POMDP model with online planners POMCP and anytime regularized
  DESPOT (Ye et al. 2017) over particle beliefs, with tree reuse between decisions, a
  particle filter with reinvigoration, POMCP exploration calibration, progress events and
  dedicated tree counters
- Multi-objective SSPs (`MoMdp<N>`, cost vectors) with the algorithms of
  Chen, Trevizan and Thiébaux (2023): MOLAO*, iMOLAO* and MOLRTDP, with multi-objective
  value iteration as the reference. They return the convex coverage set of deterministic
  proper policies under the weak improper policy assumption (bound `b`). Heuristics: blind,
  ideal point of single-objective heuristics, and MOVI on an abstraction or on the
  all-outcomes determinisation. The paper's factored heuristics (`h^max`, comax, PDBs) are
  not implemented, because the crate has no factored model

- Shared traits: `Goal`, `Heuristic`, `heuristic_fn` and `Budget` come from
  `search-core` and are re-exported here; `Budget` bounds are `Budget<Stats>`. `Zero` is `search-core`'s as well (it covers `f64`); the
  multi-objective algorithms use the local `ZeroVector`.

See [docs/design.md](https://github.com/sunsided/stochastic-search/blob/main/docs/design.md) for the model and its guarantees.

## Example

An SSP is an `Mdp` (actions with costs, outcomes with probabilities), a goal test and a
heuristic; every algorithm returns a value function and a greedy policy:

```rust
use stochastic_search::{Config, Mdp, heuristic_fn, lrtdp};

/// A die roll: action `roll` (cost 1) ends on a six; action `pay` (cost 4) ends always.
struct Dice;

impl Mdp for Dice {
    type State = bool;
    type Action = &'static str;

    fn actions(&self, _done: &bool, mut emit: impl FnMut(&'static str, f64)) {
        emit("roll", 1.0);
        emit("pay", 4.0);
    }

    fn outcomes(&self, _done: &bool, action: &&'static str, mut emit: impl FnMut(bool, f64)) {
        match *action {
            "roll" => {
                emit(true, 1.0 / 6.0);
                emit(false, 5.0 / 6.0);
            }
            _ => emit(true, 1.0),
        }
    }
}

let heuristic = heuristic_fn(|_: &bool| Some(1.0));
let solution = lrtdp(&Dice, false, &|done: &bool| *done, heuristic, Config::default())?;

// Rolling takes 6 attempts in expectation, so paying 4 is cheaper.
assert_eq!(solution.initial_action(), Some(&"pay"));
assert!((solution.initial_value() - 4.0).abs() < 1e-5);
# Ok::<(), stochastic_search::Error<bool>>(())
```

The crate is `no_std` and needs only `alloc`.

## Minimum supported Rust version

Rust 1.85 (edition 2024). The MSRV applies to the library; tests and benchmarks may need a
newer toolchain. It may be raised in a minor release when a dependency or a language
feature calls for it.

## Contributing

See [CONTRIBUTING.md](https://github.com/sunsided/stochastic-search/blob/main/CONTRIBUTING.md).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/sunsided/stochastic-search/blob/main/LICENSE-APACHE))
- MIT license ([LICENSE-MIT](https://github.com/sunsided/stochastic-search/blob/main/LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion
in this work by you, as defined in the Apache-2.0 license, shall be dual licensed as above,
without any additional terms or conditions.
