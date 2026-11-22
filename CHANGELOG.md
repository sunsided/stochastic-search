# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2](https://github.com/sunsided/stochastic-search/compare/v0.1.1...v0.1.2) - 2026-11-22

### Changed

- Link the crate to its page on the planning and search site

## [0.1.1](https://github.com/sunsided/stochastic-search/compare/v0.1.0...v0.1.1) - 2026-11-21

### Changed

- Shorten the README scope list

## [0.1.0](https://github.com/sunsided/stochastic-search/releases/tag/v0.1.0) - 2026-11-21

### Added

- Add CONTRIBUTING.md
- Add README badges and absolute links, and trim the package
- Add cargo-fuzz targets for the SSP, multi-objective and POMDP planners
- Add property tests for the particle filter and the POMDP planners
- Add REUSE licensing metadata
- Add cargo-deny, Dependabot and typos
- Add coverage reporting with Codecov
- Add the Taskfile and the CI workflows
- Add multi-objective SSPs: MOVI, MOLAO*, iMOLAO* and MOLRTDP
- Add CG-iLAO*
- Add classic one-tip LAO*
- Add bounded RTDP with finite upper bounds
- Add anytime regularized DESPOT
- Add a generative POMDP model and one-shot POMCP
- Add plain RTDP and iLAO* on the shared LRTDP run
- Add runtime events and a doubling trigger for dead-end proofs
- Add MDP traits, value iteration and labeled RTDP
- Add design document

### Changed

- Prepare the 0.1.0 release
- Update hashbrown requirement from 0.16 to 0.17 in the cargo-major group across 1 directory ([#1](https://github.com/sunsided/stochastic-search/pull/1))
- Describe the released scope and audience in the README
- Update hashbrown requirement
- Stop tracking Cargo.lock
- Depend on search-core from crates.io instead of a sibling path
- Extract the re-rooting checks of a DESPOT test into helpers
- Adopt the strict lint table and the stricter clippy thresholds
- Render mermaid diagrams in rustdoc and on docs.rs
- Track Cargo.lock
- Require search-core 0.3
- Report each multi-objective policy's own evaluated value
- Use search-core's Zero instead of a local copy
- Keep budget stops and capped abstractions from claiming dead ends
- Require search-core 0.2
- Require search-core 0.1
- Document the multi-objective algorithms against the paper and bench them
- Bench LAO*, tree reuse and the particle filter
- Refuse to plan after an unseen DESPOT history
- Make the tree planner and belief filter configs non-exhaustive
- Tighten the reuse gate, the PBVI checks and the fallback claims
- Cross-check POMCP and DESPOT against a PBVI oracle on Tiger
- Reuse DESPOT trees between decisions
- Reuse POMCP trees, calibrate exploration and track beliefs
- Report tree planner progress to an observer
- Count tree planner work in TreeStats
- Test the fill pass and align docs with LAO* and the bounded algorithms
- Share the bounds run and add FRTDP and VPI-RTDP
- Publish the heuristic contract and budget schedule in rendered docs
- Test the upper-greedy policy bound and document BRTDP policy holes
- Adopt search-core for Goal, Heuristic and Budget
- Pin DESPOT values, reject xi = 1 and update shared docs
- Document runtime events and the doubling trigger
- Test rarely entered traps and runtime events
- Lower the default state cap to one million and document state memory
- Test that the default budget is finite
- Keep solved states out of dead-end marking and tighten status docs
- Document termination, dead-end proofs and the heuristic shape
- Test termination, dead-end proofs and stateful heuristics
- Bound default runs and prove dead ends in LRTDP
- Test the literal tolerance to value iteration and bad-heuristic policies
- Document when LRTDP needs a budget to terminate
- Align the design document with the implementation
- Describe the implemented scope in the README
- Test value iteration and LRTDP on fixtures and random SSPs
- Disambiguate algorithm links in the crate docs
- Reserve the crate name with a 0.0.0 placeholder

### Removed

- Drop redundant explicit targets from doc links
