# stochastic-search: design

Status: draft on branch `draft/initial`; milestones 1, 2 (RTDP, LAO*, iLAO*, BRTDP, FRTDP, VPI-RTDP) and 4 (POMCP and DESPOT with
tree reuse and belief tracking) implemented.

## Goal and scope

`stochastic-search` plans under probabilistic action outcomes. A caller describes a
finite Markov decision process (MDP) through small capability traits and gets back a value
function and a greedy policy.

In scope, in milestone order (see [Milestones](#milestones)):

- reference: value iteration (VI) on the reachable state space, used as a test oracle;
- real-time dynamic programming: RTDP, labeled RTDP (LRTDP), and with two-sided bounds
  bounded RTDP (BRTDP), focused RTDP (FRTDP) and VPI-RTDP;
- AND/OR heuristic search: LAO*, improved LAO* (iLAO*), CG-iLAO*;
- belief space: POMCP, DESPOT, over a separate generative POMDP model.

Out of scope: deterministic graph search (`heuristic-search`), factored SAS+ models
(`sas-plus`), PDDL front ends. Probabilistic PDDL (PPDDL, RDDL) grounding belongs to a
front end, not here.

Constraints:

- `no_std` + `alloc`. Time limits come from the caller through a budget.
- No dependency on `sas-plus`: its transitions are deterministic and cannot express a
  distribution over successors. The only ecosystem dependency is `search-core`, for the
  shared `Goal`, `Heuristic` and `Budget` traits (see [Cross-crate needs](#cross-crate-needs)).
- External dependencies: `hashbrown` (state interning), `libm` (`sqrt`, `log` under
  `no_std`), `rand_core` without default features (the generator trait of the POMDP
  model).
- `forbid(unsafe_code)`, pedantic clippy, docs on every item.

## Model: stochastic shortest path (SSP)

The primary setting is the stochastic shortest path problem (Bertsekas and Tsitsiklis
1991):

- a finite set of states `S`, an initial state `s0`, and a set of goal states `G`;
- for each non-goal state `s`, a finite set of applicable actions `A(s)`;
- for each applicable action, a cost `c(s, a) >= 0` and a distribution `P(s' | s, a)` over
  successor states;
- goal states are absorbing and cost-free: planning stops on reaching one.

The optimal value `V*(s)` is the minimal expected total cost to reach a goal. It is the
least fixed point of the Bellman equation

```text
V(s) = 0                                         if s in G
V(s) = min_{a in A(s)} Q(s, a)                   otherwise
Q(s, a) = c(s, a) + sum_{s'} P(s' | s, a) * V(s')
```

A policy is *proper* if it reaches a goal with probability 1. The standard SSP assumptions
are that a proper policy exists and that every improper policy has infinite expected cost.
The second holds when every non-goal action has a strictly positive cost. Zero-cost cycles
break it: value iteration from 0 and LRTDP with an admissible heuristic then converge to
the least fixed point, which can undercut the cost of every proper policy. The crate
documents this as an assumption, it does not detect it.

Discounted MDPs are optional. A discount `gamma < 1` is an SSP in disguise (every action
moves to an absorbing goal with probability `1 - gamma`), so it can be added later as an
adapter over the `Mdp` trait without touching the algorithms.

Costs and values are `f64`, not generic over `heuristic-search`'s `Cost` trait: values
are expectations, which are fractional even when every cost is an integer, so an integer
cost type would be converted at the first backup anyway.

### Traits

```rust
pub trait Mdp {
    type State;   // Clone + Eq + Hash where an algorithm stores states
    type Action;  // Clone where a policy is returned
    fn actions(&self, state: &Self::State, emit: impl FnMut(Self::Action, f64));
    fn outcomes(&self, state: &Self::State, action: &Self::Action,
                emit: impl FnMut(Self::State, f64));
}

// From search-core, re-exported at the crate root:
pub trait Goal<S> { fn is_goal(&self, state: &S) -> bool; }               // + impl for Fn(&S) -> bool
pub trait Heuristic<S, C = u32> { fn estimate(&mut self, state: &S) -> Option<C>; }
// + impls for &mut H, Box<H>, and HeuristicFn<F> (closures via heuristic_fn)
pub trait Budget<T> { fn exhausted(&mut self, stats: &T) -> bool; }        // + impl for FnMut(&T) -> bool

// Local:
pub struct Zero; // Heuristic<S, f64>, estimates 0 everywhere
```

- `actions` emits every applicable action with its cost `c(s, a)`. Cost per state and
  action is the SSP standard; outcome-dependent costs fold into it as their expectation.
- `outcomes` keeps the callback form of the 0.0.0 placeholder: the successors of one action
  and their probabilities. An implementation needs no allocation and no iterator type.
- `Goal` and `Heuristic` are separate values, not methods on the MDP, exactly as in
  `heuristic-search` (both are `search-core`'s). One MDP can be solved for many goals and heuristics. `Heuristic<S, C>`
  has the same shape as there: `estimate` takes `&mut self` so a heuristic can keep a
  workspace, algorithms take the heuristic by value, and callers pass `&mut h` to keep it.
  A blanket impl for closures would overlap with the one for `&mut H` (`&mut F` is itself
  a closure), so closures go through `heuristic_fn`. `Goal` keeps `&self` and a blanket
  closure impl. This crate uses `C = f64`, which closures infer without annotations.
- `Heuristic::estimate` returning `None` means a proven dead end, as in `heuristic-search`.
  `Some(f64::INFINITY)` means the same.

### Dead ends

A dead end is a non-goal state from which no policy reaches a goal with probability 1.
Three kinds show up: states without applicable actions, states the heuristic marks with
`None`, and states whose every action risks leaving the goal behind forever. The caller
picks one of two semantics through `DeadEnds`:

- `DeadEnds::Infinite` (default, the "fragile" SSP semantics). Dead ends cost infinity, so
  any action that reaches one with positive probability has infinite Q-value. The answer
  is exact when a proper policy from `s0` exists. If none exists, the run ends with
  `Status::Unsolvable`, an infinite `V(s0)` and no action at `s0`. The third kind of dead
  end is found by graph analysis (see [Dead-end proofs](#dead-end-proofs)): by value
  iteration on the full space before sweeping, and by LRTDP on the stored space whenever
  trials get cut off. Without that analysis, LRTDP's values in a trap cycle would rise
  forever.
- `DeadEnds::Penalty(d)` (the fSSPUDE semantics of Kolobov, Mausam and Weld 2012). Every
  non-goal state gets an implicit "give up" action that costs `d` and ends planning. Values
  are capped at `d`, every policy is proper, and all algorithms converge on any
  input. A state whose value is `d` has no action in the returned policy.

The finite penalty is the robust choice for domains with avoidable traps; the infinite one
matches textbook SSP results and is what the oracle tests compare.

## Probability representation

Probabilities are `f64`. Each emitted probability must be finite and in `(0, 1]`, and the
outcomes of one action must sum to `1` within `PROBABILITY_TOLERANCE = 1e-9`. Violations
are reported as errors (with the offending state), not silently normalized. Duplicate
successors in one outcome list are allowed and act as their summed probability.

Exact rationals were considered and rejected for now:

- With rational inputs `V*` is rational (it solves the linear system of the optimal
  policy), but iterative algorithms only reach it in the limit, and the denominators of the
  iterates grow with every backup. Exact iteration would be slow without being exact;
  an exact answer needs a linear solve, which is a different algorithm.
- The convergence tests are epsilon-based anyway (residual <= epsilon), which absorbs the
  `1e-16` relative error of an `f64` backup.
- An input format with decimal probabilities (`0.1`) is not exactly representable in
  binary either way; the tolerance covers it.

A case for rationals would be exact policy verification (probability-1 reachability is a
graph property, not a numeric one, so it does not need them). If that case comes up, it
fits as a separate checker, not as a type parameter on every algorithm.

## Value function storage

Algorithms share one explicit, lazily built state space (`space` module):

- states are interned once into a `Vec<S>` and addressed by dense `u32` ids; a
  `hashbrown::HashTable<u32>` maps a state to its id without storing the state twice;
- per state: value, solved label, frozen greedy action, a visit stamp for depth-first
  searches;
- the expansion of a state (its actions, costs and outcome lists) is cached in flat arenas
  the first time an algorithm needs it, after validation. Backups then run over contiguous
  `(successor id, probability)` slices without calling back into the MDP.

The hash table is only used for lookup, never iterated. Ids follow insertion order, so
results do not depend on the randomly seeded hasher.

Memory grows with the number of states an algorithm touches: all reachable states for
value iteration, the visited envelope for LRTDP.

## Bellman backups and residuals

A backup at a non-goal state computes `min_a Q(s, a)` from the cached expansion, takes the
minimum with the penalty under `DeadEnds::Penalty`, and records the minimizing action
(ties break towards the action emitted first). The residual is `|V_new(s) - V_old(s)|`,
with `inf - inf` defined as `0` instead of NaN. NaN cannot enter values: costs,
probabilities and heuristic values are validated on input, and `NaN > epsilon` would
otherwise label a state solved.

### Convergence criterion

A value function is *epsilon-consistent* at `s` if the residual at `s` is at most `epsilon`.
LRTDP labels a state solved once every state reachable from it under the greedy policy is
epsilon-consistent (Bonet and Geffner 2003, `checkSolved`). The default epsilon is `1e-6`.

The guarantee, stated precisely:

- for every state `s` in the solved envelope of `s0` (states reachable from `s0` under the
  returned policy `pi`), `|V(s) - Q(s, pi(s))| <= epsilon` with the final values. The
  policy is the greedy action *frozen at labeling time*, and solved states and their
  policy successors are never updated again, so this holds even for inconsistent or
  inadmissible heuristics.
- the full Bellman residual `|V(s) - min_a Q(s, a)| <= epsilon` holds on the envelope when
  the heuristic is consistent (Bellman updates never lower a value), because later updates
  then only raise the Q-values of non-policy actions.
- epsilon-consistency is not `|V(s0) - V*(s0)| <= epsilon`. With an admissible heuristic,
  `V(s0) <= V*(s0)`, and if every action costs at least `c_min > epsilon`, then
  `V*(s0) - V(s0) <= epsilon * V(s0) / (c_min - epsilon)`. This follows from
  `V_pi(s) - V(s) <= epsilon * E_pi[steps from s]` and `E_pi[steps] <= V_pi / c_min`; the
  same condition makes `pi` reach a goal (or, under a penalty, a give-up state) with
  probability 1; under a penalty, `c_min` also bounds the penalty (a goal-free closed cycle
  cannot be epsilon-consistent when costs exceed epsilon). For a tolerance `t` on
  `|V(s0) - V*(s0)|`, pick `epsilon <= t * c_min / (V + t)` for an upper estimate `V` of
  `V*(s0)`. The tests assert both the bound and a tolerance picked this way.

The `lrtdp` API documentation states the same guarantee under "Guarantee". All of it
applies only to `Status::Converged`.

### Policy extraction

The result type `Solution` answers `value(&s)` and `action(&s)`. For LRTDP, `action` is the
frozen greedy action of solved states; for unsolved states (only when a budget stopped the
run) it is the last greedy action recorded. Value iteration records the greedy action of
its final sweep. `action` returns `None` at goals, dead ends, give-up states and states the
algorithm never touched. BRTDP records the action greedy on its upper bound at each
state's last backup; `upper_bound(&s)` returns that bound (and `None` for the other
algorithms). A fill pass after convergence backs up the envelope states no trial backed
up, so the envelope has actions wherever they exist (unless the budget runs out). The pass
is not bounded by the trials' work: under dense noise the upper-greedy envelope is every
reachable state.

## Algorithms

### Value iteration (milestone 1, oracle)

Gauss-Seidel value iteration over all states reachable from `s0`. Under `DeadEnds::Infinite`
it first computes the states that reach a goal with probability 1 under some policy (the
standard greatest-fixed-point pass: repeatedly drop states that cannot reach a goal through
actions whose every outcome stays in the candidate set). States outside that set get
infinite value, and VI runs from `0` on the rest, which converges under the positive-cost
assumption. Sweeps stop when the largest residual is below epsilon. It exists to check
LRTDP; it is not meant for large problems.

### RTDP and LRTDP (milestones 1 and 2)

RTDP (Barto, Bradtke and Singh 1995) runs trials from `s0`: back up the current state, take
the greedy action, sample a successor, repeat until a goal. With an admissible heuristic
the values on the states it keeps visiting converge to `V*`, but it has no stopping rule.

LRTDP (Bonet and Geffner 2003) adds labels. After a trial, it walks the trial back and
calls `checkSolved` on each state: a depth-first search over the greedy graph that stops at
solved states and at states whose residual exceeds epsilon. If no residual exceeded
epsilon, every state it visited is labeled solved; otherwise all of them are backed up in
reverse visiting order. A trial ends early on a solved state. The run ends when `s0` is
solved.

Additions to the paper's pseudocode:

- a trial longer than `Config::max_trial_length` steps is cut off and unwound through
  `checkSolved` as usual. Aborting is sound because only `checkSolved` labels, and the
  values backed up along the way steer later trials elsewhere. This keeps a trap cycle
  under `DeadEnds::Infinite` from hanging a trial;
- under `DeadEnds::Infinite`, the dead-end proof below runs on the stored graph after a
  cut-off trial and whenever the backup count has doubled since the last analysis (which
  catches traps that trials rarely enter but solved checks keep backing up), in both
  cases at most once per `|S| + |outcomes|` backups, so its cost stays proportional to
  the search effort;
- a `Budget` checked before every backup and trial stops the run; the result then reports
  `Status::BudgetExhausted` and carries the values so far;
- states with no finite-valued action stop a trial (they are dead ends and get labeled by
  `checkSolved`).

### Dead-end proofs

The standard almost-sure reachability analysis (a greatest fixed point): start with all
states as candidates, keep the candidates that reach a target with positive probability
through *safe* actions (every outcome a candidate), repeat until nothing changes. Each
round is a backward search over predecessor lists, linear in the stored outcomes. States
left out have no proper policy and become dead ends valued at infinity.

On a partially expanded space (LRTDP), unexpanded states count as targets. That makes the
analysis optimistic and therefore sound: a state is only marked if it cannot reach a goal
whatever the unexpanded part looks like. On a fully expanded space (value iteration) it is
exact. Solved states are never marked, so frozen policies stay untouched even when the
cost assumptions fail (with costs above epsilon a solved envelope contains no goal-free
closed cycle anyway).

### Termination

The default `Config` always returns: its budget, `Limits::DEFAULT`, allows `10^8` backups
and `10^6` stored states (about 250 MB for typical states: roughly `38 + size_of::<S>()`
bytes per state, `16 + size_of::<A>()` per action, `16` per outcome). An exhausted budget gives
`Status::BudgetExhausted`, never a verdict. `Limits::unlimited()` removes the limits. Without limits, a run returns on every
finite SSP with positive costs whose traps the dead-end proof finds; it may not return on
an infinite state space or with zero-cost cycles.

A distinct "diverged" status was considered: a value exceeding a provable upper bound
would show divergence. Under a penalty the bound is the penalty itself and values are
capped, so nothing diverges; under infinite cost no finite bound is known in general. The
dead-end proof replaces it with a stronger answer: `Status::Unsolvable` when the initial
state has no proper policy. Suspected divergence is reported as an event instead (next
section), never as a status.

### Events

An `Observer` (passed as `Config::observer`, swapped with `Config::with_observer`)
receives `Event`s. The trait has one method with a no-op default and an associated
`ENABLED` constant; `()` is the default observer with `ENABLED = false`, so every event
computation sits behind a compile-time constant and vanishes. Closures work through
`observer_fn`, as heuristics do through `heuristic_fn`. Events never change values,
policies or the status.

- `Progress { backups, states, initial_value, residual }` every `Config::progress_interval`
  backups (default `10^6`). `residual` is the initial state's last value change (LRTDP,
  RTDP, iLAO*) or the largest residual so far in the current sweep (value iteration).
- `DeadEndsProven { count }` after every dead-end analysis, `count` newly proven states.
- `SuspectedDivergence { value, growth, backups }`, LRTDP, RTDP and iLAO*. After each
  trial (iLAO*: pass) the initial value is checked. If it rose in each of the last `Config::divergence_window`
  trials (default 16) and the last rise is at least half the first rise of that window,
  the event fires with the growth over the window, and the window starts over; a trial
  without a rise resets it. Converging runs usually rise by shrinking amounts, a trap by a
  steady amount. It is a heuristic: a solvable problem whose values climb slowly and
  steadily to a large optimum (a retry loop with success probability 0.001, tested)
  triggers it too, and a divergent run whose growth fluctuates may not.

### Seeded deterministic sampling

Trials sample successors with an internal SplitMix64 generator seeded from
`Config::seed`. An own generator (a dozen lines) rather than a `rand` generator:

- the sequence is fixed by this crate, not by a dependency's version, so a seed reproduces
  a run across releases;
- no dependency beyond `rand_core`'s trait, `no_std` without features.

The generator implements `rand_core`'s `TryRng` (hence `Rng`), because the POMDP model
draws from a caller-visible generator; the trait fixes the interface, not the sequence.
`unit_interval` converts generator output to `[0, 1)` for model authors, since
`rand_core` has no float helpers.

Sampling draws `u` in `[0, 1)` with 53 random bits and walks the cumulative probabilities
of the cached outcome list in emission order. If rounding leaves `u` past the last bucket,
the last outcome is taken. Together with insertion-ordered ids and first-wins tie
breaking, one seed gives bit-identical values, policies and statistics.

### LAO* and iLAO* (milestone 2)

LAO* (Hansen and Zilberstein 2001) is AO* for cyclic AND/OR graphs: keep an explicit
graph and the best partial solution graph; expand a tip of the best solution graph; run
value (or policy) iteration on the ancestors of the expanded states; repeat until the best
solution graph has no unexpanded non-goal tips, then run a convergence test. iLAO* from the
same paper expands all tips of the best solution graph in one depth-first traversal and
backs up states in postorder during that same traversal. Both are implemented: `ilao_star`
needs no ancestor sets and is the variant the paper recommends; `lao_star` is the classic
one-tip variant, kept as the reference and for comparisons.

`lao_star`, on the same LRTDP run: each step collects the best partial solution graph
(depth-first preorder over greedy arcs, stopping at tips), expands its first tip, collects
the tip's ancestors along greedy arcs inside that graph (reverse arcs sorted by target,
then breadth-first from the tip) and sweeps them, tip first, until the residual is below
epsilon. Without tips, it sweeps the solution graph and runs `checkSolved` on `s0` once a
sweep's residual is below epsilon, as iLAO* does. Every sweep counts in `Stats::sweeps`
and may trigger the doubling dead-end proof. Rebuilding the graph per expansion makes it
quadratic in the solution size, which is why the paper improved it.

Implementation, on the LRTDP run (`space`, meter, dead-end proofs and events shared):

- a pass is an explicit-stack depth-first search from `s0` over the stored greedy actions,
  with `Node::mark` epochs. An unexpanded state is a tip: it is expanded and backed up, and
  the pass does not descend below it. Other states are backed up in postorder;
- the paper's convergence test is a pass that expands nothing and whose largest residual is
  below epsilon. A postorder backup can still flip a greedy action towards states the pass
  did not visit, so a settled pass is followed by `checkSolved` on `s0` (no repair
  backups); only a passing check ends the run. The result is closed under the final
  policy and has LRTDP's guarantee verbatim;
- under `DeadEnds::Infinite`, values in a trap rise with every pass; the doubling trigger of
  the dead-end proof catches them as in LRTDP;
- passes count in `Stats::sweeps`.

### Bounded RTDP (milestone 2)

BRTDP (McMahan, Likhachev and Gordon 2005) keeps a lower bound `L` (the node value, from
the heuristic) and an upper bound `U` (a `Vec<f64>` parallel to the node ids, so `Node`
stays at 32 bytes) and backs up both in the same step. Trials act greedily on `L` and sample
the successor `s'` with weight `P(s' | s, a) * (U(s') - L(s'))`; a trial ends when the total
weight is below `(U(s0) - L(s0)) / tau`, then backs up its states in reverse. The run stops
when `U(s0) - L(s0) <= epsilon`. `tau` is `BrtdpConfig::tau`, default 10: the paper used
"tau between 10 and 100", and the low end favours short trials.

The bounds run (`src/bounds.rs`) is shared by BRTDP, FRTDP and VPI-RTDP; an `Explorer`
enum decides the next state of a trial and when it ends. Each algorithm takes its own
config type (`BrtdpConfig`, `FrtdpConfig`, `VpiRtdpConfig`) wrapping the shared `Config` as
`base`, so algorithm-specific parameters do not accumulate on `Config`. After a converged
run, a fill pass backs up (breadth-first from `s0`) every unexpanded state of the
upper-greedy envelope, so the returned policy covers its envelope; the budget is charged
before each expansion, which makes "expanded" equal "backed up at least once".

FRTDP (Smith and Simmons 2006), translated from rewards to costs (the paper's optimistic
bound is our lower bound): priorities `prio(s) = min(gap(s) - epsilon / 2,
max_s' P(s') * prio(s'))` under the lower-greedy action; a trial follows the successor of
highest `P * prio`, ends when `gap(s) - epsilon / 2 <= 0` or at depth `D`; `D` starts at
`D0 = 10` and grows by `kD = 1.1` after a trial whose late updates (depth above `D / kD`)
had an average quality `delta * W` at least that of the early ones (paper's values). The
paper outputs the pessimistic-greedy policy, which is our upper-greedy one. The deterministic
choice can stall (found on a random SSP: the bounds stopped changing for 10^7 backups,
probably because a low-probability successor holding the gap up is never the
highest-priority one), so after an FRTDP or VPI-RTDP trial that did not
shrink the gap of `s0`, the shared run makes the next trial a plain BRTDP trial.

VPI-RTDP (Sanner, Goetschalckx, Driessens, Shani 2009), in costs: values are uniform between
the bounds; `VPI(t)` is the largest, over `a != a*`, average of
`max(0, E[Q(a*) | v_t] - E[Q(a) | v_t])` for `v_t` uniform in `[L(t), U(t)]`, which is
linear in `v_t` and integrates in closed form. A trial samples by `P * gap` while some
successor's gap exceeds `beta` times the initial gap of `s0` (the paper's "95% of the
maximum possible bound", our reading), else by `VPI`, else with probability `alpha = 0.001`
by gap, else ends. Two additions (documented on `vpi_rtdp`) make it terminate on the gap
criterion, which the paper does not use (it stops on time): before the `alpha` rule,
BRTDP's `tau` rule continues the trial by gap; and a trial that did not shrink the gap of
`s0` is followed by one plain BRTDP trial (the shared fallback above). Without them, a random SSP was found where VPI
steering cycled forever on a self-loop of positive VPI while the state holding the gap up
had zero VPI.

Dead ends are the obstacle: an infinite upper bound never becomes finite through backups
on a cycle (`c + p * 0 + (1 - p) * inf = inf`), so gap-weighted sampling and the stopping
test would be meaningless. The chosen formulation:

- under `DeadEnds::Penalty(d)` (fSSPUDE), the upper estimate is capped at `d` and `None`
  means `d`, so every upper bound is finite. `U = d` is a fixed point of the capped Bellman
  operator, hence a monotone upper bound;
- under `DeadEnds::Infinite`, every stored state needs a finite upper estimate unless the
  lower heuristic proves it a dead end (`None`); otherwise the run fails with
  `Error::UnboundedUpper` naming the state. Proven dead ends have `L = U = inf` and gap 0,
  so trials never sample them. Since every stored state that may have a finite value has a
  finite (valid) upper bound, no dead-end proof is needed. An avoidable dead end the lower
  heuristic misses is rejected, not avoided; use a penalty for such problems.

Invariant: `U(s) = inf` implies `L(s) = inf` (both bounds are backed up together, and an
infinite upper Q-value needs a successor with infinite upper bound), so gaps are never
infinite or NaN. Gaps are clamped at 0 when invalid bounds cross. Initial estimates with
`U < L` are rejected as `Error::InvalidHeuristic`.

Guarantee: with an admissible lower and a valid upper heuristic, `L <= V* <= U` on every
stored state at any time, converged or not. The policy is greedy on `U` at each state's
last backup; with a monotone `U` (which backups preserve) it satisfies
`U(s) >= Q_U(s, pi(s))`. The trial follows the `L`-greedy action, as in the paper.

### CG-iLAO* (milestone 3)

CG-iLAO* (Schmalz and Trevizan 2024) restricts each backup to the actions that can still be
optimal under the current values, adding actions to a state only when they could become
greedy (action pruning in the spirit of "constraint generation"). It needs per-state lower
bounds on Q-values and an extensible action list, which the arena layout must allow.
Exact algorithm details are to be checked against the paper before milestone 3.

### Belief space (milestone 4)

POMCP (Silver and Veness 2010) and DESPOT (Somani, Ye, Hsu and Lee 2013; Ye et al. 2017)
plan in belief space with sampled observations. They use a separate model, not `Mdp`:

```rust
pub trait Pomdp {
    type State; type Action; type Observation;
    fn actions(&self, emit: impl FnMut(Self::Action));        // same in every state
    fn step<R: Rng + ?Sized>(&self, state: &Self::State, action: &Self::Action,
                             rng: &mut R) -> Step<Self::State, Self::Observation>;
}
pub struct Step<S, O> { pub state: S, pub observation: O, pub reward: f64, pub terminal: bool }
pub trait RolloutPolicy<S> { fn action<R: Rng + ?Sized>(&mut self, state: &S,
                                                         actions: usize, rng: &mut R) -> usize; }
```

- A generative model, because both planners only sample. Rewards are maximized with
  discount `gamma`, as in both papers, instead of the SSP's costs: the benchmark problems
  (Tiger, RockSample) are stated that way and an SSP view would need a goal.
- Actions do not depend on the state: the agent cannot tell states apart. Planners refer to
  actions by emission position.
- Rollout and default policies, and DESPOT's upper bound, are separate values, as the
  heuristic is for SSPs. `UniformRandom` is the knowledge-free policy. In Tiger at depth
  60 it buries the first step's difference under rollout noise (random opens cost 45 in
  expectation); the tests use the usual domain policies (always listen, head east).
- Beliefs are particle slices with equal weights. `pomcp` and `despot` are one-shot:
  belief in, `Plan { action, value, status, stats }` out, with `TreeStats` counters
  (simulations, nodes, backups, steps, depth) and `Event::TreeProgress` for an observer.
- Between decisions: `Pomcp` keeps its tree, and `advance(action, observation)` copies the
  subtree of `T(hao)` breadth-first into a fresh arena (root stays 0; an unexpanded child
  is expanded), as in the paper. `Despot::advance` re-roots likewise (this crate's design;
  the papers rebuild). DESPOT's stored numbers are relative to the root (K in the weights,
  depth in the discount powers and the depth limit, lambda, fixed flags, the horizon end of
  every default simulation), so re-rooting re-simulates the default policy of every kept
  node and recomputes every bound; random streams keep absolute depths. A test checks that
  every re-rooted leaf equals a node built fresh under the new root. The new root holds
  only the child's scenarios.
- `update_belief` is POMCP's rejection particle filter: non-terminal successors whose
  observation matches are kept; reinvigoration applies a caller transform (states are
  opaque) to fill the shortfall and add requested extras; zero accepted particles is
  `Error::BeliefDepleted`. `pomcp_exploration` measures the paper's `c = R_hi - R_lo`;
  `PomcpConfig::with_reward_range` gives the closed-form bound.
- Budgets reuse `Budget` over `Stats`: `trials` counts simulations or explorations,
  `states` tree nodes, `backups` value updates. Default limits are finite (`10^4` POMCP
  simulations, `10^3` DESPOT explorations). The `Observer` is not wired into them yet.
- Errors reuse `Error`: `InvalidReward`, `NoActions`, `EmptyBelief`, `InvalidConfig`, and
  `InvalidHeuristic` for a NaN or infinite DESPOT upper bound.

POMCP: UCB1 (`Q + c * sqrt(ln N / n)`, untried actions first) over histories; a new history
is added and evaluated by one rollout; the return is averaged into every action node on the
path. Observation children are found by linear scan with `==`. The recommendation is the
root action with the highest average, not the UCB score. The exploration constant must be
scaled to the return range by the caller.

DESPOT follows the anytime algorithm of Ye et al. (2017, Algorithms 1 to 6), which the
2017 paper calls the anytime DESPOT algorithm (whether it matches the 2013 paper's
"AR-DESPOT" is unverified):

- `K` scenarios: a start state drawn from the belief, and a random stream per depth. The
  step of scenario `k` at depth `d` uses a fresh generator seeded from `(seed, k, 2d)`, the
  default policy's choice there one from `(seed, k, 2d + 1)`. Streams do not depend on the
  action or on the order of expansions, which is what determinizes a scenario;
- per node: `U(b)` (average upper bound, raised to `L0(b)` if below), `L0(b)` (average
  default-policy return), `l0 = |Phi_b| / K * gamma^depth * L0`, and the RWDU bounds
  `mu0 = max(l0, |Phi_b| / K * gamma^depth * U - lambda)` and `ell0 = l0`;
- backup: `rho(b, a) = gamma^depth * sum R / K - lambda`,
  `mu(b) = max(l0, max_a rho + sum mu(child))`, the same for `ell`, and
  `U(b) = max_a [avg R + gamma * sum |Phi_c| / |Phi_b| * U(c)]`;
- explore: action by `mu(b, a)`, child by excess uncertainty
  `E(b) = mu - ell - |Phi_b| / K * xi * (mu(b0) - ell(b0))`; stop when `E <= 0`, at a
  pruned node, or beyond depth `D` (then `MakeDefault`: `U = L0`, `mu = ell = l0`, fixed);
  default policy simulations take their last step at depth `D`, like the tree;
- prune: a node is blocked if an ancestor `b'` has
  `|Phi_b'| / K * gamma^depth(b') * (U(b') - L0(b')) <= lambda * (depth - depth(b'))`;
  blocked nodes on the path are fixed to the default policy bottom-up;
- recommendation: the root action with the best `ell(b0, a)`, or the default policy's
  action (in the first scenario) if `l0(b0)` beats every action.

Tests check `ell <= mu` at every node, a non-increasing root `mu`, a non-decreasing root
`ell` without regularization, identical trees for one seed, and a one-node tree under a
huge `lambda`.

## Error handling

Algorithms return `Result<Solution, Error<S>>` (`SolveResult<M>`). A solution's `Status` is
`Converged`, `BudgetExhausted` or `Unsolvable`. Errors carry the state where the input went
wrong:

- `Error::InvalidCost`: a cost is negative, NaN or infinite;
- `Error::InvalidProbability`: a probability is NaN or outside `(0, 1]`;
- `Error::ProbabilitySum`: an action's outcomes do not sum to 1 within the tolerance
  (including an action with no outcomes);
- `Error::InvalidHeuristic`: the heuristic returned NaN or negative infinity, or a BRTDP
  upper estimate is NaN, negative infinity or below the lower estimate;
- `Error::UnboundedUpper`: under infinite dead ends, BRTDP's upper estimate is infinite at
  a state the lower estimate does not prove a dead end;
- `Error::InvalidConfig`: epsilon not finite and positive, a penalty not finite and
  non-negative, a zero trial length, or a BRTDP `tau` not finite and above 1.

Negative heuristic values are allowed (they are admissible, only uninformative), and
overestimates are allowed (they only void the optimality bound). Neither panics.

## Testing strategy

- Closed-form fixtures, solved by value iteration, LRTDP, RTDP and iLAO*: a retry loop (`V = c / p`), a choice between a safe and a risky
  action, dead ends under both semantics, a goal initial state.
- An oracle that does not share code with the crate: test-side Bellman residuals and
  probability-1 reachability are computed from the generator's raw tables.
- Property tests on random small SSPs where every state has a proper policy and costs are
  strictly positive (an action that advances along a chain with positive probability):
  - VI matches the closed-form fixtures and is used as `V*`;
  - LRTDP with an admissible heuristic satisfies `V(s0) <= V*(s0)` and the gap bound from
    [Convergence criterion](#convergence-criterion);
  - the policy reaches a goal with probability 1: every state reachable from `s0` in the
    policy's support graph reaches a goal in that graph;
  - epsilon-consistency on the policy envelope (full residual with a consistent heuristic,
    policy residual with any);
  - two runs with the same seed give identical values, policies and statistics.
  - BRTDP brackets `V*` (value iteration from every state) with `L <= V* <= U` on every
    stored state, converged or cut off by a small budget, under infinite dead ends (blind
    or determinization lower bound, the always-action-0 policy value plus 1 as upper
    bound, with avoidable traps proven by the determinization) and under a penalty (no
    upper estimate); converged runs have a gap of at most epsilon at `s0`.
- Robustness: overestimating, negative, `Some(inf)`, false `None` and NaN heuristics under
  a budget; no panic, no hang, NaN reported as an error; the frozen-policy residual holds
  for any heuristic.
- Termination: unavoidable trap cycles end as `Unsolvable` with the default configuration
  (both algorithms), including a trap entered with probability `1e-9`; infinite state
  spaces stop at the state limit, for value iteration's enumeration too.
- Events: an unsolvable ring reports suspected divergence before its budget runs out; a
  slow solvable retry loop may report it but converges, with or without an observer to the
  same bits; progress follows its interval; the default observer is zero-sized and
  disabled at compile time.
- Belief space: Tiger and a one-rock RockSample, with the expected first action and its
  margin derived by hand in each test, over four seeds, for POMCP and DESPOT; determinism
  per seed; input validation; DESPOT's bound invariants as unit tests.
- Proptest case counts are fixed in code (64), so `cargo test` stays fast.

## Milestones

1. Model traits, explicit state space, value iteration, LRTDP, tests. (this draft)
2. Plain RTDP and iLAO*, sharing `space` and the LRTDP run. (done) Bounded RTDP with
   finite upper bounds, FRTDP and VPI-RTDP on a shared bounds run (done). One-tip LAO*
   (done).
3. CG-iLAO*; policy evaluation utilities (exact evaluation of a fixed policy by linear
   solve or iteration); discounted MDP adapter.
4. Belief-space trait set; POMCP; DESPOT; belief tracking and tree reuse between
   decisions (done). Next: decide whether this becomes `pomdp-search`.

## Cross-crate needs

- `Goal`, `Heuristic`, `heuristic_fn`, `HeuristicFn` and `Budget` come from `search-core`
  (adopted per its `MIGRATION.md`) and are re-exported at the crate root, so
  `stochastic_search::Heuristic` and friends keep working and a heuristic or budget
  written for `heuristic-search` works here. `Zero` (for `f64`), `Limits`, `Stats` and
  `Mdp` stay local. Breaking changes of the adoption:
  - `Budget` is now `Budget<Stats>`: code naming the trait in bounds writes
    `B: Budget<Stats>` and `impl Budget<Stats> for MyBudget`. Closures and `Limits` are
    unaffected.
  - `Heuristic` gains a default type parameter (`C = u32`); this crate's algorithms still
    require `Heuristic<S, f64>`, which closures infer, so call sites are unchanged.
  - Importing `search_core::Zero` next to `stochastic_search::Zero` clashes; the local one
    is the `f64` heuristic.
- `heuristic-search` is generic over `Cost`; this crate fixes `f64`. A shared heuristic
  trait must therefore stay generic in `C`, and integer-cost heuristics need a conversion
  adapter (`u32 -> f64` is lossless).
- `planning-heuristics` could provide determinization-based heuristics (all-outcomes
  determinization plus a classical heuristic), which are admissible for SSPs. That needs a
  deterministic view of an MDP, which is an adapter in this crate, not a `sas-plus`
  dependency.
- A probabilistic PDDL front end (in `pddl-rs` or `miniplan`) would implement `Mdp`.

## References

- Barto, Bradtke, Singh. *Learning to act using real-time dynamic programming.* Artificial
  Intelligence 72, 1995.
- Bertsekas, Tsitsiklis. *An analysis of stochastic shortest path problems.* Mathematics of
  Operations Research 16(3), 1991.
- Bonet, Geffner. *Labeled RTDP: Improving the convergence of real-time dynamic
  programming.* ICAPS 2003.
- Hansen, Zilberstein. *LAO\*: A heuristic search algorithm that finds solutions with
  loops.* Artificial Intelligence 129, 2001.
- Schmalz, Trevizan. *Efficient constraint generation for stochastic shortest path
  problems.* AAAI 2024. (Venue and details to verify before milestone 3.)
- Sanner, Goetschalckx, Driessens, Shani. *Bayesian real-time dynamic programming.* IJCAI
  2009.
- Smith, Simmons. *Focused real-time dynamic programming for MDPs: squeezing more out of a
  heuristic.* AAAI 2006.
- McMahan, Likhachev, Gordon. *Bounded real-time dynamic programming: RTDP with monotone
  upper bounds and performance guarantees.* ICML 2005.
- Kolobov, Mausam, Weld. *A theory of goal-oriented MDPs with dead ends.* UAI 2012.
- Silver, Veness. *Monte-Carlo planning in large POMDPs.* NeurIPS 2010.
- Somani, Ye, Hsu, Lee. *DESPOT: Online POMDP planning with regularization.* NeurIPS 2013.
- Ye, Somani, Hsu, Lee. *DESPOT: Online POMDP planning with regularization.* JAIR 58,
  2017.
- Kaelbling, Littman, Cassandra. *Planning and acting in partially observable stochastic
  domains.* Artificial Intelligence 101, 1998. (Tiger)
- Smith, Simmons. *Heuristic search value iteration for POMDPs.* UAI 2004. (RockSample)
