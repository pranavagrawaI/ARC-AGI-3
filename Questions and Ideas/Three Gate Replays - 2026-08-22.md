# Three gates and matched continuation — 2026-08-22

## Judgment

Candidate retention and the immediate goal lifecycle pass their preserved-boundary
host-contract tests. Job-scoped Builder allocation does **not** survive its cheapest
discriminator as an explanation of the selected timeout: that pass was already
explicitly dynamics-scoped and all 28 calls were mechanics work.

Do not implement a job allocator. The matched continuation now supports using the
higher-scoring retained candidate as the next Builder's starting point, but it
contradicts the stronger hope that another generic mechanics pass would gain score:
both pinned-route arms gained 0/16 within their caps.

## Gate 1 — turn-18 candidate retention: PASS

Raw result:
`../analysis/candidate-retention/20260822-turn18/result.json`

The replay reconstructed checkpoint call `000491`, applied the exact candidate first
visible at call `000492`, supplied all 28 preserved Builder tool results, and exercised
the new rollback path.

- source deadline observed at level 0, turn 18;
- canonical SHA restored byte-exactly to
  `060648964ad3943f0083650a8b5fa382986a18a7c59e440db17ba1d959788905`;
- candidate SHA retained as
  `4f78ef3b371d3e8f4d106710d19d03f94de6c60cfa399776818714fa1c5a86df`;
- contemporaneous verifier result retained: 8/16 evaluated transitions, 0.50 exact
  accuracy, first divergence at turn 4 RIGHT;
- evidence frontier contains 19 public turn records;
- a fresh Builder tool surface listed the index and read the candidate exactly;
- publication and planner eligibility both remained false.

Execution: zero provider-model calls and zero ARC calls.

## Gate 2 — selected typed-goal checkpoint: PASS

Raw result:
`../analysis/goal-lifecycle/20260822-selected-checkpoint/result.json`

The replay reconstructed selected checkpoint call `000006`, immediately after the
historical revision-1 goal proposal, and supplied scripted responses through the
current lifecycle.

```text
preserved UP attempt -> blocked: request objective review
objective review -> inconclusive, with live alternative named
second UP attempt -> blocked: resolve review
update_goal -> revision 2, retained with a discriminator
preserved UP attempt -> permitted
```

Canonical `world_model.py` remained byte-identical. This establishes host-contract
enforcement at the exact failure boundary. Because responses were scripted, it does
not establish natural model uptake or gameplay benefit.

Execution: zero provider-model calls and zero ARC calls.

## Gate 3 — job-scoped Builder allocation: CONTRADICTED AS CAUSE

Bounded retrospective analysis card (the source trace had already been inspected, so
this is not a blind prospective prediction):
`../analysis/builder-allocation/20260822-job-scope/EXPERIMENT-CARD.md`

Raw result:
`../analysis/builder-allocation/20260822-job-scope/result.json`

The preserved request was already explicitly `cause: dynamics`. Argument-only
classification assigned all 28 calls to mechanics:

- 4 context calls;
- 19 mechanics probes;
- 5 mechanics model edits;
- 0 objective, verifier-diagnosis, or planner jobs.

Exact-transition accuracy remained 0.00 through call 26 and first rose to 0.50 at
call 27. Any simple call cap below 27 would have discarded the only positive candidate.
Therefore mixed jobs did not cause this timeout, and a typed job allocator is not the
next repair. This result is local to the selected pass; it does not prove job scope is
never useful.

## Gate 4 — continuation: preservation supported; further gain contradicted in this pass

Frozen card and valid result:

- `../analysis/builder-continuation/20260822-matched-coreweave/EXPERIMENT-CARD.md`
- `../analysis/builder-continuation/20260822-matched-coreweave/result.json`
- `../analysis/builder-continuation/20260822-matched-coreweave/trajectory-analysis.json`

The first attempt is preserved separately under `20260822-matched/`. It is invalid
for causal comparison because automatic routing sent its 28 calls through CoreWeave,
Novita, and ModelRun in unequal arm distributions. Its descriptive result must not be
combined with the valid run.

The replacement pinned `coreweave/bf16` with fallbacks disabled. All 27 completed
provider calls reported CoreWeave; prompt, tools, task, evidence frontier, verifier,
caps, and nonpublication were matched; zero ARC actions occurred. It cost $0.05353172.

| Arm | Start | Verifier observations during edits | Final | Calls |
| --- | ---: | --- | ---: | ---: |
| retained candidate | 8/16 | call 6: 0/16; call 7: 8/16 | 8/16 | 14 |
| canonical | 0/16 | call 10: 0/18; call 11: 0/16 | 0/16 | 13 |

The frozen result map calls this `retained_better`: the retained arm ended 8/16 while
canonical ended 0/16. The narrower scientific reading matters more. Retention
preserved an earned advantage through a fresh Builder pass, including recovery after
a destructive edit, but neither arm gained exact-transition score. The retained
arm's first divergence also widened from 3 to 24 claimed-cell differences while its
aggregate score stayed flat. Therefore this supports **continuation selection**, not
more generic Builder allocation and not a claim of productive improvement.

## Licensed next action

Add an explicit, host-auditable continuation-selection step:

```text
higher contemporaneously verified retained score
  -> start the next Builder from that immutable candidate
  -> keep it noncanonical and planner-ineligible
  -> rollback remains available after every edit
```

Do not pay for a longer generic mechanics pass. The next research experiment, if
needed, must target the unresolved turn-4 shape/phase representation and must measure
score gain over the 8/16 baseline, not merely final absolute score.

## Gate 5 — reasoning-on repetition: performance prediction contradicted at fixed time

Frozen card and assessment:

- `../analysis/builder-continuation/20260822-matched-coreweave-reasoning/EXPERIMENT-CARD.md`
- `../analysis/builder-continuation/20260822-matched-coreweave-reasoning/card-assessment.json`

Prediction frozen before execution: retained would finish above 8/16, with canonical
above 0/16 as a secondary result. The only client-context delta was
`reasoning: {enabled: true}`; both arms again used pinned CoreWeave.

Both arms reached the precommitted 240-second stop without editing their models:

| Arm | Reasoning-off control | Reasoning-on | Exposed reasoning |
| --- | --- | --- | ---: |
| retained | 8/16, complete in 222 s, 14 calls | 8/16, timeout, 9 completed calls | 3,060 tokens |
| canonical | 0/16, complete in 207 s, 13 calls | 0/16, timeout, 6 completed calls | 3,296 tokens |

The copied generic harness labels any timeout invalid. The frozen card instead names
timeout as a stop and invalidates only condition drift, so `card-assessment.json`
records the card-governed result: the accuracy prediction was contradicted in this
time-matched repetition. Reasoning was successfully exposed on all 15 completed
responses, but reduced throughput enough that neither arm reached an edit. This does
not establish that reasoning is generally harmful or settle a call-matched contrast.

Reasoning remains mandatory for observability. Do not buy a longer run merely to seek
a positive result. First use the newly preserved reasoning, tool arguments, and probe
outputs to locate why thousands of reasoning tokens produced no consequential edit.

## Superseded planned action

Run one matched continuation experiment:

```text
A: canonical 0/16 model at the same turn-18 frontier
B: retained 8/16 candidate at the same frontier
task: mechanics repair only
measure: exact score gained per additional Builder call
```

This action was executed above. The automatic-route attempt failed validity, and the
pinned-route replacement supplied the admissible result.
