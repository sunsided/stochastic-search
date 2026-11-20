# stochastic-search

Planning under probabilistic outcomes: iLAO*, RTDP, LRTDP, POMCP and DESPOT.

## Scope

- Implemented: value iteration, RTDP, labeled RTDP (LRTDP) and iLAO* for stochastic
  shortest path problems, with infinite or finite-penalty dead ends
- Implemented: a generative POMDP model with one-shot online planners POMCP and anytime
  regularized DESPOT (Ye et al. 2017) over particle beliefs
- Planned: one-tip LAO*, CG-iLAO*, Bounded RTDP; belief tracking and tree reuse between
  POMDP decisions

See [docs/design.md](docs/design.md) for the model, guarantees and milestones.

## Status

Pre-release. Version 0.0.0 reserves the name; the API will change.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
