---
applyTo: "**/*.rs"
---

# Rust code

- Every module (`//!`), type, field, variant, function, constant and macro has a doc
  comment, private and `pub(crate)` items included. Docs state intent, invariants, units and
  error/panic conditions (`# Errors`, `# Panics`, `# Safety`). They do not restate the name.
- No `mod.rs`: modules with children are `foo.rs` plus `foo/`. Integration tests live under
  `tests/it/`.
- No separator comments (`// -----`, `// =====`). If a file needs sections, split it into
  submodules.
- Leave a blank line after the closing `}` of a block statement when another statement
  follows, and between logical phases of a function. No lumped code.
- One level of abstraction per function. Long loops (simulations, game ticks, solvers) are
  short drivers that call documented step functions. If a block needs a "what" comment,
  extract it into a function named after that comment.
- `clippy::pedantic` clean. Suppression only with a scoped
  `#[expect(lint, reason = "...")]`. Never `#[allow]`, and never crate-level suppression.
- No `unsafe` unless the crate opts out of `#![forbid(unsafe_code)]` with a documented
  reason. Each unsafe block has one operation and a `// SAFETY:` comment.
- The crate is `#![no_std]` and needs only `alloc`. Flag any `std` use outside tests and
  benches, and float math that bypasses `libm`.
- Inputs are validated, never trusted: NaN or negative costs, probabilities outside `(0, 1]`
  or not summing to 1 within `PROBABILITY_TOLERANCE`, NaN heuristics and non-finite rewards
  return an `Error`, never a panic. Every loop is bounded by the `Budget` (`Limits` by
  default). The fuzz targets in `fuzz/` check exactly this; extend them with new planners.
- `Goal`, `Heuristic`, `heuristic_fn`, `Budget` and `Zero` come from `search-core` and are
  re-exported; do not redefine them here.
- A new algorithm or guarantee updates the crate docs in `src/lib.rs`, the README "Scope"
  list and `docs/design.md`, and gets a property test against value iteration or the oracle
  in `tests/it/`.
- Mermaid diagrams in doc comments must match the code they describe. Update them when the
  structure changes.
