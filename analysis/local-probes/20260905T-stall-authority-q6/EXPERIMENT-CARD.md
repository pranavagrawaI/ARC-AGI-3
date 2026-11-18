# q6 — stall authority, repaired host contract

Status: **unlaunched**. Amendments below precede any q6 run. The original draft
is preserved in `implementation/EXPERIMENT-CARD-before-repair.md`.

Question: Does interrupting a repeated, unsuccessful policy for an authored
single-action authority review reduce subsequent stalls?
Posture: targeted repair of q4's stalled execution, not a search experiment.
Contrast: one pinned candidate, `ARC_STALL_REVIEW=0` control versus `=1` treatment.
Three independent runs per arm, interleaved C T C T C T. No cross-platform pooling.

## Held fixed

Game ls20, markdown agent, interactive-game-solver skill, visual off, thinking
off, empty effort ladder, model and sampling settings identical within the arm.
Kaggle uses the existing pinned Qwen FP8/SGLang route; local OpenRouter runs use
`qwen/qwen3.8-27b`. Those are separate experiments, not interchangeable replicates.
The candidate retains the q3c actor and host evidence contract: re-entry history,
prediction scoring, interruption at no-impact/level/terminal boundaries, and
winning-frame evidence for the scribe. q4's separate goal-candidate treatment is
not included: the verified base is q3c, as recorded in CHASSIS.json.

`deployment/kaggle/q6-nooa-candidate/CHASSIS.json` pins actor, worker, headless,
shared modules, adapter/runtime, native harness, scorer and this card. Both arms
must have the same contract hash. `freeze_candidate.py --check` must pass before
packaging; each worker verifies its bundled files and records `launch.json`.
Historical candidates, before snapshots and prior scorers stay unchanged.

## Trigger and clock

One implementation, `patched-source/after/stall_contract.py:detect_stall`, is used
by the live actor, replay and scorer. Parameters remain window=6, min_actions=25,
similarity=0.55. The historical sweep still separates in all 27 settings, with
4.89x at the frozen operating point.

Evaluate once per state, on the **first attempted submission**. Compare its
current rationale with the preceding six committed batches, using all lowercase
alphanumeric tokens. Detect if no level progress occurred since the first of
those six, total session steps >=25, maximum Jaccard >=0.55, and an available base
action was absent from those six submitted action lists. This preserves the
historical current-rationale/six-predecessor definition. Record the raw attempt
in `ipc/policy_attempts.jsonl`, including turn, step, rationale and proposed
batch. Attempts rejected by another gate still belong to this clock. This is a
pre-launch clarification: the draft's after-execution implementation used a
different window and tokenization and is discarded.

Control logs `STALL_DETECTED` and continues through existing gates. Treatment
logs the same event and refuses ordinary submission until a resolution commits.
The rejection hands back repetition evidence, untried actions and the current
hypotheses file without parsing belief IDs.

## Treatment

The Actor calls `resolve_stall` in the **already running CodeAct invocation**:

```python
resolve_stall(
    disposition="continue" | "probe" | "switch",
    previous_basis=["..."], active_basis=["..."],
    action="one legal action", rationale="...",
    abandon_if=None, predictions=None,
)
```

Both bases are non-empty lists of non-empty strings. Continue requires a
non-empty falsifier. Probe requires a dictionary with at least two distinct,
non-empty predicted outcomes keyed by alternative. Switch requires unequal
bases. A successful call commits exactly one action and ends the turn. Existing
re-entry/scribe gates still apply. No separate review invocation is scheduled;
there is no review-count cap. Eight consecutive rejected resolution calls write
ABORT. Bases, falsifiers and predictions are stored in full.

The executing host records the action's consequence from raw frames before
publishing the next state. At the next ordinary invocation the Actor calls
`finish_stall(active_basis_after=[...], rationale="...")` before further play.
The host records whether the authored basis changed; it does not decide which
belief is true or require the Actor to change it.

Host-checkable falsifiers: `move`, `no_impact`, `level_up`, `terminal`,
`grid_changed`, `grid_unchanged`, or `levels_completed == N`, `>= N`, `> N`.
Other prose is retained and evaluates to null. Null continue falsifiers count as
P12 misses. No interpretation of arbitrary prose or game-specific semantics.

Trace: `STALL_DETECTED`, `AUTHORITY_REVIEWED`, `CONSEQUENCE_OBSERVED`,
`AUTHORITY_RESOLVED`, joined by stall_id, turn and env step in authority.jsonl.
Resolution is recorded immediately after its consequence; K=20 escape is computed
later by the scorer, not fabricated in an early resolution event.

A terminal/budget-end action may have no subsequent Actor invocation. Its
consequence is still recorded by the host. Missing post-action authoring is
explicitly censored if no further action follows; it never earns authority-change
credit. Playing on without authoring is invalid. This removes the draft's
assumption that every last action necessarily has a following invocation.

## Predictions and measurements

`score_q6.py` reads complete JSONL artifacts. It validates frame continuity,
published-state correspondence, attempt/batch coverage, executed action prefixes,
review shape and the correspondence between recomputed detections and logged
interventions. It never uses authority verdict fields to award a pass.

| Metric | Definition | Prediction |
|---|---|---|
| P9 | No next recomputed stall within 20 executed actions | Every treatment run exceeds every control run |
| P9b | Recomputed stalls per 100 actions | Report, including prevention |
| P10 | Continue whose active_basis equals the previous_basis authored at the next recurring stall, within K | <=0.50 |
| P11 | Detection-to-committed-review action latency | Structurally zero; nonzero invalidates |
| P12 | True continue falsifiers followed by changed authored basis; null continue falsifiers added as misses | 1.00 |
| P1 | Multi-action batches with frame-recomputed prediction accuracy >=0.6 | >=0.50 |
| P2 | Completed levels with model-written consolidation | 1.00 |
| P3 | Re-entry attempts / all action attempts, as q4 | <=0.05 |
| P4 | Waste share from frames, as q4 | <=0.15 |

P1–P4 preserve q4's full-run HUD classification and denominators. Online host
classification remains sequential, as in q3c. The two clocks are reported as such.

For P9, observed recurrence within K is a failure even near the cap. No recurrence
with fewer than K follow-up actions is **censored**, never an escape, including
wins. Require >=3 evaluable stall events in each of three runs per arm; otherwise
P9 is INCONCLUSIVE. Report total detections and censoring separately. The draft's
0.26 historical escape estimate included unobserved tails and is descriptive only;
it is not a calibrated baseline for this corrected endpoint.

P10 uses the basis governing at the next stall, not a temporary change immediately
after the reviewed action. Empty P10/P12 denominators are N/A and supply no
positive evidence; they do not demonstrate failure. Missing P1/P2 opportunities
leave their hold checks INCONCLUSIVE. Complete separation has one-sided rank
probability 1/20 under exchangeable runs; fixed launch order cannot rule out
platform/time drift. Preserve model/platform/config evidence for that assumption.

Completion, RHAE, dispositions, authority-change latency, force advances,
truncations, model calls, wall time and available spend are reported. Unavailable
values remain null, never zero. RHAE is not a gate. Runtime integrity is checked
before calls: authored/stored bytes and complete printed note readbacks at the
final request. Partial, unused or evicted reads are UNOBSERVED, not an integrity
pass. A detected mutation latches ABORT and preserves raw evidence.

## Cap, stop and launch prerequisites

Six runs, sequential. Kaggle: one game per kernel, 200 Solver decisions per game,
4 h wall per kernel, 14 h total arm wall; GPU quota must cover the planned arm.
Official bootstrap/GAME_OVER RESET counters are separate and remain recorded.
There is no additional 60-turn Kaggle cap: the official seam stops on its action
budget; the old draft's local launcher turn cap is not silently assumed there.
Local alternative: 60 agent turns, 200 env steps, 600 s turn timeout, 4 h wall,
$5/run and $30 arm cap; remaining credit >=$30 before any launch. Last observed
credit was $0.16. Choose one platform before launch and keep it fixed.

GAME_OVER creates a fresh worker under the existing adapter episode contract.
A process failure mid-episode fails closed; it does not silently restart or replay
unobserved actions. Session histories reset at that boundary. Score a game by
aggregating its sessions, with follow-up censored at each session end; never join
an earlier session's policy window to a new worker's empty history.

Abort on wall/spend/action cap, reasoning tokens, integrity failure, or eight
consecutive rejected resolutions. Unexpected worker failure invalidates the run.
Do not launch a partial arm to use up remaining credit. No concurrency.

Pre-launch: CPU qualification and package checks pass; scorer and card hashes are
frozen; one separately-carded repaired-host pre-flight establishes stall incidence
and runtime at the intended FP8 budget. Budget/credit/quota sizing is a launch
prerequisite, not answered by scripted tests. No launch is authorized by this card.

## Invalid if and result map

Invalid: missing/malformed evidence, noncontiguous frames, missing attempt
coverage, incorrect or missing treatment joins, nonzero review latency, control
receiving review, silent continuation after an unresolved consequence, integrity
mutation, reasoning tokens, fewer than 15 agent turns per game, or mismatched
contract/platform/config between arms. Scripted qualification can never PASS an
experimental arm. Missing K=20 follow-up or too few events is inconclusive.

- P9 separates, P10 <=.50 where applicable, P12 holds where applicable, P1–P4 hold:
  retain; bounded search gets its own next card.
- P9 separates but continuation fails: review may help by drift; do not claim
  successful recommitment. Next card tightens authoring requirements.
- No separation, repeated continuation: delete the typed review; test the
  single-action interruption alone on a new card.
- No separation with mixed dispositions: arbitration is insufficient; search is
  the next candidate.
- Incidence collapses without progress: compare action diversity with rationale
  diversity before treating detector avoidance as recovery.
- Too few qualifying events: inconclusive; re-derive on the acquired traces under
  a new card, without enlarging this arm after seeing results.
- P1/P2 collapse: restore the frozen chassis rather than adding another gate.
- Invalid: preserve artifacts; no interpretation, new run ID/card for a retry.

Outputs: one run directory per local run, or official worker session directories
under the Kaggle probe output, with launch.json, ipc/*.jsonl, note files and stop
artifacts. `score_q6.py --output SCORE.json RUN...` recomputes the report.

Standing decisions: assist beats refusal at lifecycle boundaries; no WikiSkill or
autonomous maintainer; search unchanged until q6; no game-specific priors; do not
rewrite prior scorers; capabilities count only when live Actor calls reach them.

Each Kaggle process receives one run identity retained across worker restarts; each worker receives a fresh session identity. The scorer rejects repeated run identities or reused session identities across replicates, including copied artifacts. Explicitly overriding ARC_Q6_RUN_ID must retain a unique identity per independent kernel run.
