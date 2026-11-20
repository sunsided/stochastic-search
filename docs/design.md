# stochastic-search: design

Status: draft, written with milestone 1 on branch `draft/initial`.

## Goal and scope

`stochastic-search` plans under probabilistic action outcomes. A caller describes a
finite Markov decision process (MDP) through small capability traits and gets back a value
function and a greedy policy.

In scope, in milestone order (see [Milestones](#milestones)):

- reference: value iteration (VI) on the reachable state space, used as a test oracle;
- real-time dynamic programming: RTDP, labeled RTDP (LRTDP);
- AND/OR heuristic search: LAO*, improved LAO* (iLAO*), CG-iLAO*;
- later, belief space: POMCP, DESPOT, over a separately defined belief and observation
  model.

Out of scope: deterministic graph search (`heuristic-search`), factored SAS+ models
(`sas-plus`), PDDL front ends. Probabilistic PDDL (PPDDL, RDDL) grounding belongs to a
front end, not here.

Constraints:

- `no_std` + `alloc`. Time limits come from the caller through a budget.
- No dependency on `sas-plus`: its transitions are deterministic and cannot express a
  distribution over successors. No dependency on other ecosystem crates yet.
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

pub trait Goal<S> { fn is_goal(&self, state: &S) -> bool; }               // + impl for Fn(&S) -> bool
pub trait Heuristic<S, C> { fn estimate(&mut self, state: &S) -> Option<C>; }
// + impls for &mut H, Box<H>, Zero, and HeuristicFn<F> (closures via heuristic_fn)
pub struct Zero; // estimates 0 everywhere
```

- `actions` emits every applicable action with its cost `c(s, a)`. Cost per state and
  action is the SSP standard; outcome-dependent costs fold into it as their expectation.
- `outcomes` keeps the callback form of the 0.0.0 placeholder: the successors of one action
  and their probabilities. An implementation needs no allocation and no iterator type.
- `Goal` and `Heuristic` are separate values, not methods on the MDP, exactly as in
  `heuristic-search`. One MDP can be solved for many goals and heuristics. `Heuristic<S, C>`
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
  same condition makes `pi` reach a goal with probability 1 (a goal-free closed cycle
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
algorithm never touched.

## Algorithms

### Value iteration (milestone 1, oracle)

Gauss-Seidel value iteration over all states reachable from `s0`. Under `DeadEnds::Infinite`
it first computes the states that reach a goal with probability 1 under some policy (the
standard greatest-fixed-point pass: repeatedly drop states that cannot reach a goal through
actions whose every outcome stays in the candidate set). States outside that set get
infinite value, and VI runs from `0` on the rest, which converges under the positive-cost
assumption. Sweeps stop when the largest residual is below epsilon. It exists to check
LRTDP; it is not meant for large problems.

### RTDP and LRTDP (milestone 1: LRTDP)

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
- after a cut-off trial under `DeadEnds::Infinite`, the dead-end proof below runs on the
  stored graph, at most once per `|S| + |outcomes|` backups so its cost stays
  proportional to the search effort;
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
exact. Marking cannot invalidate a solved label: a solved, epsilon-consistent envelope
with costs above epsilon contains no goal-free closed cycle.

### Termination

The default `Config` always returns: its budget, `Limits::DEFAULT`, allows `10^8` backups
and `10^7` stored states. An exhausted budget gives `Status::BudgetExhausted`, never a
verdict. `Limits::unlimited()` removes the limits. Without limits, a run returns on every
finite SSP with positive costs whose traps the dead-end proof finds; it may not return on
an infinite state space or with zero-cost cycles.

A distinct "diverged" status was considered: a value exceeding a provable upper bound
would show divergence. Under a penalty the bound is the penalty itself and values are
capped, so nothing diverges; under infinite cost no finite bound is known in general. The
dead-end proof replaces it with a stronger answer: `Status::Unsolvable` when the initial
state has no proper policy.

### Seeded deterministic sampling

Trials sample successors with an internal SplitMix64 generator seeded from
`Config::seed`. An own generator (a dozen lines) rather than `rand`:

- the sequence is fixed by this crate, not by a dependency's version, so a seed reproduces
  a run across releases;
- no dependency, `no_std` without features.

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
backs up states in postorder during that same traversal, which is the variant worth
implementing first. Both reuse `space` and the backup code.

### CG-iLAO* (milestone 3)

CG-iLAO* (Schmalz and Trevizan 2024) restricts each backup to the actions that can still be
optimal under the current values, adding actions to a state only when they could become
greedy (action pruning in the spirit of "constraint generation"). It needs per-state lower
bounds on Q-values and an extensible action list, which the arena layout must allow.
Exact algorithm details are to be checked against the paper before milestone 3.

### Belief space (later)

POMCP (Silver and Veness 2010) and DESPOT (Somani, Ye, Hsu and Lee 2013) plan in belief
space with sampled observations. They need an observation model and a belief
representation that the MDP traits do not have. They get a separate trait set
(`Pomdp: actions, sample transition, sample observation`) defined then; if that set grows
large it may move into its own crate. Nothing in milestone 1 depends on it.

## Error handling

Algorithms return `Result<Solution, Error<S>>` (`SolveResult<M>`). A solution's `Status` is
`Converged`, `BudgetExhausted` or `Unsolvable`. Errors carry the state where the input went
wrong:

- `Error::InvalidCost`: a cost is negative, NaN or infinite;
- `Error::InvalidProbability`: a probability is NaN or outside `(0, 1]`;
- `Error::ProbabilitySum`: an action's outcomes do not sum to 1 within the tolerance
  (including an action with no outcomes);
- `Error::InvalidHeuristic`: the heuristic returned NaN or negative infinity;
- `Error::InvalidConfig`: epsilon not finite and positive, a penalty not finite and
  non-negative, or a zero trial length.

Negative heuristic values are allowed (they are admissible, only uninformative), and
overestimates are allowed (they only void the optimality bound). Neither panics.

## Testing strategy

- Closed-form fixtures: a retry loop (`V = c / p`), a choice between a safe and a risky
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
- Robustness: overestimating, negative, `Some(inf)`, false `None` and NaN heuristics under
  a budget; no panic, no hang, NaN reported as an error; the frozen-policy residual holds
  for any heuristic.
- Termination: unavoidable trap cycles end as `Unsolvable` with the default configuration
  (both algorithms); infinite state spaces stop at the state limit, for value iteration's
  enumeration too.
- Proptest case counts are fixed in code (64), so `cargo test` stays fast.

## Milestones

1. Model traits, explicit state space, value iteration, LRTDP, tests. (this draft)
2. Plain RTDP (with a trial budget, no labels), LAO* and iLAO*, sharing `space`; Bounded
   RTDP or FRTDP as upper-bound variants if a consumer needs an anytime gap.
3. CG-iLAO*; policy evaluation utilities (exact evaluation of a fixed policy by linear
   solve or iteration); discounted MDP adapter.
4. Belief-space trait set; POMCP; DESPOT. Decide then whether that becomes `pomdp-search`.

## Cross-crate needs

- `heuristic-search` defines `Heuristic<S, C>` and `Goal<S>` with the same shape as here.
  Once both crates are stable, the traits should move to one place (`search-core` per the
  architecture's rule, since two crates then need the same public trait) so a heuristic
  written for one works with the other.
- `heuristic-search::Budget` and `Stats` have the same role as this crate's `Budget`; a
  shared budget trait would let callers reuse their clock-based budget.
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
- Kolobov, Mausam, Weld. *A theory of goal-oriented MDPs with dead ends.* UAI 2012.
- Silver, Veness. *Monte-Carlo planning in large POMDPs.* NeurIPS 2010.
- Somani, Ye, Hsu, Lee. *DESPOT: Online POMDP planning with regularization.* NeurIPS 2013.
