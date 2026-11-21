---
applyTo: ".github/workflows/**,Taskfile.dist.yaml,**/Taskfile.dist.yaml,.pre-commit-config.yaml"
---

# CI, Taskfile and hooks stay in sync

- The `ci` task in `Taskfile.dist.yaml` mirrors the jobs in `.github/workflows/ci.yml`. A
  change that adds, removes or renames a CI job, or changes its flags, must make the same
  change to `task ci` and the task it calls, and the other way round. Flag any drift in review.
- Local hooks in `.pre-commit-config.yaml` call Taskfile tasks. They must not duplicate
  commands.
- Every `uses:` is pinned to a full 40-character commit SHA, with a `# vX.Y.Z` comment. Reject
  tag or branch refs.
- `dtolnay/rust-toolchain` is pinned to its `stable` or `nightly` branch commit and always
  gets an explicit `toolchain:` input, so the toolchain does not depend on which branch commit
  Dependabot pins.
- The workflow sets `permissions: {}` at the top, and each job sets the least privilege it
  needs. Every `actions/checkout` sets `persist-credentials: false`.
- No `pull_request_target`. No `${{ github.event.* }}` or other untrusted expressions
  directly in `run:`. Pass them through `env:`.
- Do not rename `publish-crates-io.yml` or its `crates-io` environment. crates.io trusted publishing is
  bound to both. Deploy jobs run in their own tag-restricted environment, and `needs:` a
  verification job.
- CI jobs: `branch`, `fmt`, `clippy`, `style`, `doc`, `test` (Linux only), `coverage`,
  `deny`, `msrv`, `semver`, `fuzz`, plus `actionlint` and `zizmor` in `actions-lint.yml`.
- `msrv` checks the library only (`cargo hack check --rust-version --all-features`, no
  `--all-targets`): dev-dependencies may need a newer Rust. Do not add `--all-targets` back.
- `fuzz/` is a separate nightly workspace. Its tasks live in `fuzz/Taskfile.dist.yaml`,
  included as `fuzz:`; CI's `fuzz` job mirrors `task fuzz:build` and `task fuzz:lint`.
- `Cargo.lock` is tracked, but no step uses `--locked`: CI builds against the sibling
  `search-core` at its default branch, whose version may be ahead of the lock.
- `search-core` is a sibling path dependency (`../search-core`). Every job that runs cargo
  checks this repository out with `path: stochastic-search`, then calls
  `./stochastic-search/.github/actions/checkout-siblings`, and sets
  `defaults.run.working-directory: stochastic-search`. Actions that take a path (rust-cache
  `workspaces`, cargo-deny and semver-checks `manifest-path`, codecov `working-directory`,
  release-plz `manifest_path`/`config`) get it explicitly. A new sibling path dependency adds
  a step to that composite action.
- `cargo fmt` runs without `--all`, which would also format the sibling path dependencies.
- release-plz jobs are gated on `vars.RELEASE_APP_CLIENT_ID`, so they stay dormant until the
  release App is configured.
- `scripts/check-branch.sh` is the single definition of allowed branch names. It is used by
  the Taskfile, the hook and the CI `branch` job.
- New jobs that should gate merges must also be added to branch protection. Mention that in
  the PR.
