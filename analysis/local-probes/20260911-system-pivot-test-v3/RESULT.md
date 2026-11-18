# System-level pivot test: comparison unresolved after provider timeout

The requested matched live comparison did not complete. V3 pair A began with the
critic arm and stopped on an OpenRouter read timeout before its baseline arm. Pair B
was not started, as required by the frozen cards. No retry, fallback, Kaggle launch,
or deployed-runtime mutation occurred.

## Usable critic-arm prefix

| Measure | Observation |
|---|---:|
| Session before stop | 324.08s |
| Successful Actor calls | 18 |
| Successful critic calls | 9 |
| Executed environment actions | 1 UP |
| Levels completed | 0 |
| First action | 151.89s |
| First warranted correction | 157.98s, after 1 action |
| Three-decision retention | unresolved; no later executed decision |
| False pivots | 1 |
| Critic request time | 36.52s |
| Confirmed response cost | $0.166696 |
| Failed-request reservation | $0.06515592 unresolved |

All successful paid Actor requests contained the three predeclared evidence markers.
The compact seed stayed in the cacheable prefix, replayed all seven historical grids,
and included the mistaken claim and measured transitions. Frozen source hashes passed.
The repaired harness therefore reached the intended epistemic boundary before the
external failure.

The Actor predicted that UP would move the dense color-9 block two rows. The real
successor moved it from rows37-39 to32-34: five rows. Actor call23 explicitly corrected
the vertical displacement to five. The scorer credits that correction without asking
whether the Actor or critic deserves authorship.

The same response asserted that the color-12 band stayed at rows35-36 and proposed
encoding it as static. The actual successor moved it to rows30-31. This is counted as
one false pivot. Actor25 then corrected that separate mistake, but it did not execute
another action before the timeout. Repeated statements without later decisions do not
satisfy the precommitted retention criterion.

## Critic behavior

Six reviews passed delivery validation: five no_concern and one insufficient_context.
Three concern responses failed the output contract and were not delivered. One rejected
concern correctly noticed the Actor's color-12 misreading, after the Actor had already
made it. Another asserted that the replayed state might be an unrelated reset and told
the Actor to discard the preserved table; that objection ignored the host restart
evidence. Frequent review therefore supplied review opportunities, but this prefix
does not demonstrate that they produced the warranted movement correction or prevented
the simultaneous false pivot.

Critic calls used17.5% of successful model-request time and11.3% of total observed
session time. This is measured overhead, not an estimate of net harm: a completed
comparison could still show that review saves more time elsewhere.

## Why the comparison is unresolved

Request28, an Actor call after the first action, timed out while reading the provider
response. It has a preserved request and no response. The reservation cannot be
settled from the available evidence. The card therefore marks the critic arm
operationally invalid and prohibits starting its baseline or pair B. There is no
numeric seconds-saved result and no treatment ranking.

## Earlier invalid preparations

- V1 replayed the right environment state but embedded seven full grid pairs in a
  178,964-byte volatile prompt block. The model never received the seed. That run was
  stopped and preserved as invalid.
- V2 reduced the seed to3,287 bytes and verified its presence initially, but left it
  volatile. Context truncation evicted it at baseline call60. The fail-closed boundary
  stopped before paid inference; placement of reservation before that boundary left
  an unresolved reservation. No arm comparison was made.
- V3 moved the same compact seed to the cacheable prefix and checked evidence before
  spend reservation. Twenty-four offline tests passed. Every successful Actor request
  retained the seed; its failure was the later provider read timeout.

These are distinct failures. V1 and V2 diagnosed and repaired our harness. V3 confirms
that repair on the live request path but cannot answer the causal comparison because
the matched arm is absent.

## Evidence

Pair evidence is under `pair-a/`: the immutable card, CONFIG.json, STARTED.json,
requests/responses, full action grids, review contexts, snapshots, transport error,
GRADES.json, SCORE.json, AUDIT.json and driver log. `pair-b/` is prepared but unstarted.
The compact prompt and full replay evidence remain in the v2 fixture directory.

The scorer deliberately gives independent Actor correction full credit, accepts an
unchanged action when its decision rationale retains the correction, requires three
later decisions, counts relapse/false pivots, and refuses to calculate paired benefit
when an arm is missing or invalid. Its focused suite and the associated input/delivery
checks passed24 tests before V3 launched. The frozen deployed runtime still verifies
at SHA `061369c045635581ac5e4d9fea99d532722f3a3e5d0c4ce93b4b75475a7876ea`.

## Decision

The testing machinery is now capable of measuring the requested whole-system outcome,
and the live prefix demonstrates one fast-enough correction plus one false pivot.
It does not establish that frequent criticism improved correction speed. A new paired
run is warranted only under a new card after provider availability is acceptable;
the existing cards cannot be relaunched or silently extended.
