# stochastic-search

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
  now `Budget<Stats>`; `Zero` stays local because it is the `f64` heuristic.

See [docs/design.md](docs/design.md) for the model, guarantees and milestones.

## Status

Pre-release. Version 0.0.0 reserves the name; the API will change.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
