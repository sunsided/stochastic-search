# stochastic-search

[![crates.io](https://img.shields.io/crates/v/stochastic-search.svg)](https://crates.io/crates/stochastic-search)
[![docs.rs](https://img.shields.io/docsrs/stochastic-search)](https://docs.rs/stochastic-search)
[![CI](https://github.com/sunsided/stochastic-search/actions/workflows/ci.yml/badge.svg)](https://github.com/sunsided/stochastic-search/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sunsided/stochastic-search/graph/badge.svg)](https://codecov.io/gh/sunsided/stochastic-search)
[![MSRV](https://img.shields.io/crates/msrv/stochastic-search)](https://crates.io/crates/stochastic-search)
[![License](https://img.shields.io/crates/l/stochastic-search)](https://github.com/sunsided/stochastic-search#license)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)

Planning under probabilistic outcomes: LAO*, iLAO*, CG-iLAO*, RTDP, LRTDP, BRTDP, FRTDP, VPI-RTDP, POMCP, DESPOT, and
the multi-objective MOLAO*, iMOLAO* and MOLRTDP.

## Scope

- Implemented: value iteration, RTDP, labeled RTDP (LRTDP), classic LAO*, iLAO* and CG-iLAO* for stochastic
  shortest path problems, with infinite or finite-penalty dead ends
- Implemented: bounded RTDP (BRTDP), focused RTDP (FRTDP) and VPI-RTDP with lower and upper
  bounds and an upper-greedy policy; upper bounds must be finite, so infinite dead ends need
  a lower bound that proves them (or use a penalty)
- Implemented: a generative POMDP model with online planners POMCP and anytime regularized
  DESPOT (Ye et al. 2017) over particle beliefs, with tree reuse between decisions, a
  particle filter with reinvigoration, POMCP exploration calibration, progress events and
  dedicated tree counters
- Implemented: multi-objective SSPs (`MoMdp<N>`, cost vectors) with the algorithms of
  Chen, Trevizan and Thiébaux (2023): MOLAO*, iMOLAO* and MOLRTDP, with multi-objective
  value iteration as the reference. They return the convex coverage set of deterministic
  proper policies under the weak improper policy assumption (bound `b`). Heuristics: blind,
  ideal point of single-objective heuristics, and MOVI on an abstraction or on the
  all-outcomes determinisation. The paper's factored heuristics (`h^max`, comax, PDBs) are
  not implemented, because the crate has no factored model

- Shared traits: `Goal`, `Heuristic`, `heuristic_fn` and `Budget` come from
  `search-core` and are re-exported here. Breaking: `Budget` bounds are
  now `Budget<Stats>`. `Zero` is `search-core`'s as well (it covers `f64`); the
  multi-objective algorithms use the local `ZeroVector`.

See [docs/design.md](https://github.com/sunsided/stochastic-search/blob/main/docs/design.md) for the model, guarantees and milestones.

## Installation

The crate is not on crates.io yet beyond a name placeholder; use a git dependency until
the first release:

```toml
[dependencies]
stochastic-search = { git = "https://github.com/sunsided/stochastic-search" }
```

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

## Status

Pre-release. Version 0.0.0 reserves the name; the API will change.

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
