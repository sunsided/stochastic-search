# stochastic-search

Planning under probabilistic outcomes: iLAO*, RTDP, LRTDP, BRTDP, POMCP and DESPOT.

## Scope

- Implemented: value iteration, RTDP, labeled RTDP (LRTDP) and iLAO* for stochastic
  shortest path problems, with infinite or finite-penalty dead ends
- Implemented: bounded RTDP (BRTDP) with lower and upper bounds; upper bounds must be
  finite, so infinite dead ends need a lower bound that proves them (or use a penalty)
- Implemented: a generative POMDP model with one-shot online planners POMCP and anytime
  regularized DESPOT (Ye et al. 2017) over particle beliefs
- Planned: one-tip LAO*, CG-iLAO*, FRTDP; belief tracking and tree reuse between
  POMDP decisions

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
