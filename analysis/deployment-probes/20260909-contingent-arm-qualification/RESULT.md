# Contingent controller: arm isolation and assessor qualification

The controller candidate was packaged as a launchable notebook whose four arms
did not vary the controller. This probe makes the controller the isolated factor,
and in doing so found three defects that would each have wasted a Kaggle run.

Contract `8b5defcb53e1c5b3c211bdd6902f04474803ec0d8e3a3f3ec8730519368fc9f3`.
Donor designer-player-discovery-20260908-v2 unchanged at `cce5829f…`.

## What was wrong before

| Defect | Where | Consequence |
| --- | --- | --- |
| No arm varied the controller | `probe.ipynb` arms a/b/c/d | `grep -c contingent probe.ipynb` = 0; the interface was unconditionally on in all four arms, a uniform confound |
| Verifier described another candidate | `verify_candidate.py` | asserted slug `designer-player-reasoning-20260907`, arm `mindset`, `reasoning enabled:true`, 32768 tokens; also unimportable — no working pre-flight gate existed |
| Donor slug reused | `kernel-metadata.json` | would have pushed v2 over `structural-self-test-deep-20260909`, whose v1 is a different candidate with its own writeup |
| New factor stripped before the worker | `shared_local_solver.py:53` | `ARC_CONTINGENT_ARM` absent from `WORKER_ENVIRONMENT_KEYS`; the off arm silently ran the treatment |
| Assessor demanded reasoning | `assess_candidate.py:33` | reasoning-off run records `reasoning_tokens 0`, `reasoning_present False` on every call → `reasoning not established` |
| Assessor pinned the action limit to 30 | `assess_candidate.py:48` | `adapter.py:169` writes the configured limit, 10 → `wrong budgets` |
| Arm absent from run output | `nooa_worker.py:146` | both arms wrote `discovery_arm a` and nothing else; a reader could not tell a real control from a dropped one |

An eighth defect was introduced by the repair itself and caught by a peer
session's card test. `solver_agent.py:781` recorded
`reasoning_present = bool(getattr(response, "reasoning", None))`, and
`bool("\n\n")` is true, so a thinking-off Qwen3 turn emitting an empty
`<think>` block reports reasoning. The first repair gated on that observation and
would have invalidated a correct arm. The card makes the reasoning-off *state* an
invalidation condition, not observed tokens, so the gate now reads the declared
`ARC_LLM_EXTRA_BODY` from the worker's own `model_config` and reports observed
reasoning as a measurement. `reasoning_present` now strips before testing.

The three assessor defects were fatal together: `assess_candidate.py` ends
`raise SystemExit(result['status']!='PASS')` and each arm cell calls it under
`subprocess.run(..., check=True)`. Kernel version 1 would have failed at the end
of arm `off`, and arm `on` would never have run.

## Discriminating evidence

The env-strip defect was not visible in any rendered copy of the prompt. It was
caught by running the actual adapter, transport, host, dispatcher and CodeAct
once per arm against a deterministic loopback model and asserting on the request
the worker actually sent (`test_contingent_arm.py`, `seen[0]`):

| Arm | Controller tools in system prompt | Instructions | Packet key | Model calls for two actions |
| --- | --- | --- | --- | ---: |
| `off` | absent | absent | absent | 2 |
| `on` | present | present | present | 1 |

Before the allowlist entry, the `off` arm's prompt still contained both tools
while nothing errored anywhere. That is the failure mode worth remembering: a
control arm that is not a control, in a run that looks clean.

The assessor defects were confirmed by calling `assess()` directly on evidence
matching this run's configuration — `INVALID`, errors exactly
`['reasoning not established', 'wrong budgets']`.

Real `launch.json` files written by the real worker in both arms now carry the
factor:

    off: {'solver_arm': 'loop', 'discovery_arm': 'a', 'contingent_arm': 'off'}
    on:  {'solver_arm': 'loop', 'discovery_arm': 'a', 'contingent_arm': 'on'}

## Repairs

`ARC_CONTINGENT_ARM` (`off`/`on`, default `on`) gates the tool surface, the
prompt sentences, the decision-packet key and the dispatcher fast path, and the
methods fail closed on a direct call. The variable is on the worker allowlist.
The worker records the arm it saw from inside the worker process, so a future
strip reports `on` and the assessor rejects the mislabelled arm. The assessor
now takes `--reasoning`, `--contingent-arm`, `--action-limit` and
`--admission-seconds` from the cell that configured them instead of constants.
The notebook is two arms on a fresh slug; `verify_candidate.py` asserts this
package and passes with zero network calls.

## Validation

124 tests pass (`pytest3.log`, `junit3.xml`), including the real subprocess
lifecycle, the notebook-extracted package checks, per-arm request inspection,
and `test_assessment.py`: a clean reasoning-off arm passes, a dropped factor
fails, an absent arm record fails, reasoning contradicting the arm fails in both
directions, the old 30-action constant fails, a declared reasoning state that contradicts
the arm fails in both directions, and an empty think block (`reasoning_tokens 3`,
`reasoning_present True`) passes a reasoning-off arm with the tokens surfaced as a
measurement.

The generalisable test, from the peer session: ask of every gate not "is this
correct?" but "does the card state this rule?" A gate can be defensible in
isolation and still not be a stop the experiment authorised. Both defects of this
class passed the first test and failed the second.

These establish configuration, isolation and execution. They are not evidence of
model quality: the model is scripted, and no Qwen call was made.

## Gate audit against producers

A peer session's sharper test — where a gate polices an observation, check the
request or configuration that produced it before checking the card, because a
gate contradicting its own producer is wrong regardless of the card — was applied
to every abort site in `assess_candidate.py`. One had already failed it (the
reasoning gate, above). The remaining five do not:

| Gate | Producer | Verdict |
| --- | --- | --- |
| stop reason in `win`/`wall_budget`/`step_budget` | `adapter.py:156-159` emits exactly those three and no others | consistent |
| `action_limit` equals the configured limit | `adapter.py:169` writes `_max_actions_from_env()`, i.e. `TYCHO_KAGGLE_MAX_ACTIONS`, which the cell now passes to the assessor | consistent |
| `executed_solver_actions` equals transitions | `adapter.py:245` increments once per emitted decision, and `choose_action` returns one action per call, including a contingent continuation | consistent |
| integrity rows required | `integrity_guard.py:24` writes `GUARD_INSTALLED` at construction, so a clean run always has one | consistent |
| `contingent_arm` matches the cell | the worker writes what its own process observed | consistent |
| reasoning state | reads the declared `ARC_LLM_EXTRA_BODY` | consistent after repair |

This is a negative result and is recorded as one: no change was made on its
account. The audit's value is that the three checks most likely to have drifted —
the stop-reason set, the decision/transition identity and the integrity-row
requirement — were each confirmed against the code that produces them rather than
assumed.

## Which repairs the evidence actually exercised

A peer session observed the mirror image of this morning's defect: a fix seen
only alongside successes looks validated and is not. Applied here:

| Repair | Exercised by evidence? |
| --- | --- |
| `ARC_CONTINGENT_ARM` on the worker allowlist | Yes. Before it, the off arm's real request still carried both tools; after, it does not |
| Tool/prompt/packet/dispatcher gating | Yes. Per-arm inspection of the request the worker sent |
| Assessor reads the configured action limit | Yes. The old constant fires on real v1 evidence; the new check passes it |
| Assessor gates the declared reasoning state | Yes, in the same way |
| Fresh slug | Yes. The donor's kernel and its write-up are intact |
| `verify_candidate.py` rewrite | Yes. The inherited one could not even import |
| `contingent_arm` in `launch.json` | Partly. Verified in real local `launch.json` files; not yet on the Kaggle path, since v1 predates it |
| `reasoning_present` strips whitespace | **No.** All 106 real thinking-off calls reported no reasoning at all, so the empty-think case never arose. The repair rests on the peer session's loopback observation, not on any evidence from this run |

The last row is the honest one. It is correct — `bool("\n\n")` is true and the old
record would have lied — but nothing here has exercised it, and it should not be
counted among the repairs this run validated.

## Envelope facts measured on the real stack

From v1's off arm, which is the same engine configuration v2 runs:

- Served context is `context_len=262144`, matching the configured window. A peer
  session reported a 32,768 ceiling; that is real but belongs to a different
  stack — `ANALYZER_CONTEXT = 32_768` hardcoded in its donor's `serving_setup.py`
  and passed as `--max-model-len`, with no environment override. An earlier note
  here called it "its own probe's setting", which was wrong and was asserted
  without reading the source. It does not constrain this candidate.
- Real Actor prompts ran 6,341 to 30,264 tokens, median 19,545, over 92 Actor
  calls. Peak is 12% of the served context, so there is ample headroom for the
  `on` arm's additional instructions. No call ended in `length`; all 106 ended
  `tool_calls` or `stop`.
- `max_prefill_tokens=16384` with `--chunked-prefill-size 2048`: a scheduling
  parameter, not a prompt cap, and 30k prompts completed under it.
- GPU memory is the real constraint on this stack, not context. A peer run peaked
  at 96,628 MiB of 97,887. Nothing here changes `--mem-fraction-static 0.85`, but
  any future context or KV change is an OOM risk rather than a tuning knob.

## Limits

Version 3 raises the budget to twenty actions per arm. The time budget moved with
it out of necessity, not caution: v1's off arm committed ten actions in 1,045.65s
of gameplay, so twenty project to ~2,091s and the unchanged 1,800s admission
would have stopped the arm at `wall_budget` near action seventeen — the action
bump alone would have bought nothing. Admission is 3,000s, launcher timeout
3,600s, worst case 7,450s of the 10,800s envelope. `ACTIONS_PER_ARM`,
`ADMISSION_SECONDS` and `LAUNCHER_TIMEOUT_SECONDS` now live once in
`prepare_candidate.py`; the notebook, `LAUNCH.json`, the assessor arguments and
the tests all derive from them.

Twenty actions may still be too few for any contingent plan to arise. A run whose
`on` arm proposes nothing is uninformative about the controller, not evidence
against it, and must be reported that way. The pace projection comes from a
single ten-action arm and the `on` arm's pace may differ. Plan estimates remain model
conjectures, not calibrated probabilities or RHAE. No score-improvement claim,
submission or promotion follows from this probe.

## The predicted failure, verified against a real run

Kernel version 1 was stopped from the Kaggle UI after 23 minutes, but its `off`
arm had already completed gameplay: `official launcher passed` at t=1428s,
`step_budget` after 10 actions of a 10-action limit, 1,045s of a 1,800s
admission horizon, 106 model calls. Its outputs are retained in `v1-output/`
(pulled before version 2 was pushed, because `kaggle kernels output` returns only
the latest version).

That evidence confirms the predicted abort on a real run rather than a synthetic
one. Version 1's assessor applied to it yields exactly the two predicted errors:
all 106 calls report `reasoning_tokens 0` and `reasoning_present False`, so
`any(...)` is false and `reasoning not established` fires; `action_limit` is 10
against the pinned constant 30, so `wrong budgets` fires. Version 1 would have
exited nonzero here and the `on` arm would never have run.

The repaired assessor over the same real evidence reports neither. Its only
errors are `lifecycle gates failed` (the run was cancelled mid-flight, so
containment never closed) and `wrong worker identity/arm` (the evidence carries
v1's contract `69ce2421…`, not the current one) — both correct readings of a
cancelled predecessor.

Two incidental facts from the same evidence. Real Qwen on this pinned stack
reported no reasoning at all across 106 thinking-off calls, so the empty-think
hazard did not materialise here; the repair stands on the peer session's
observation of it elsewhere, not on this run. And the `off` arm produced no
`contingent_events.jsonl` and no contingent activity in its IPC, which is
consistent with the gate holding on the Kaggle path but is not proof, since the
model need not have attempted the interface. The per-arm request inspection
remains the discriminating evidence.

## Disposition

Kernel version 1 (contract `69ce2421…`) was pushed before the assessor defects
were known and is superseded before its result; it is left to reach its terminal
status rather than deleted, so its logs remain inspectable. The corrected package
was pushed as version 2 after version 1 reached `CANCEL_ACKNOWLEDGED`. Version 2
entered `QUEUED` on push, so its 10,800-second envelope begins at `RUNNING`, not
at push.

A claim that it was queued *behind* another kernel holding the account's GPU slot
was recorded here and is withdrawn. Verified at 12:39:50Z: the other kernel read
`COMPLETE` while this one read `QUEUED`, concurrently — the slot was empty. So
`QUEUED` on push is simply the initial state and says nothing about contention,
and the earlier seven-minute queue that appeared to support serialization is
equally explained by baseline scheduler latency. The serialization hypothesis is now
**falsified**, not open: v2 was still `QUEUED` at 12:44:22Z, 7.4 minutes after a
12:37:00Z push, with the other kernel `COMPLETE` for the whole interval. That
meets and exceeds the seven-minute queue that appeared to demonstrate contention,
under a demonstrably empty slot, so baseline scheduler latency explains both and
contention explains neither. The discriminating measurement was queue *duration*
under an empty slot, not whether `QUEUED` appears. The error was mine in a specific way worth naming: I asserted the other
kernel was still running at my push time from a status message eleven minutes
old, without issuing the one command that would have checked it, and used it to
strengthen a hypothesis rather than to test one.
