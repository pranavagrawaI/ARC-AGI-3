# Epistemic questioner public run v1 result

## Verdict

The frozen Kaggle notebook completed one valid public-only run. The deployment,
request lifecycle, and critic contract all worked. The combined solver completed
zero levels across LS20, S5I5, and CD82.

This run supports a narrow conclusion: the event-driven critic can produce
source-bound epistemic questions during live play, but the current intervention
does not make the Actor pivot reliably or quickly enough to solve these games.
There is no control arm, so the run does not identify the critic's causal effect.

## Operational validity

- Kaggle kernel: `pranavagrawai/arc3-epistemic-questioner-20260912`, version 1,
  status `COMPLETE`.
- Frozen contract SHA-256:
  `8532c0a7d18fbbf7d4fa4afd583a868cbfe347d03977053533864d2cb9e105d9`.
- Local and downloaded notebook cell sources matched exactly.
- Result and containment checks passed; root return code was zero and shutdown
  was TERM-only.
- All three games received the frozen 600-second model-admission allocation.
- 184 Solver requests plus one warmup access were observed; all 185 completed
  successfully. Reasoning and retries remained off.
- Server-marker scan found no OOM, request deletion, traceback, or HTTP 4xx/5xx.
- GPU use increased from 23.16 to 23.78 hours: 0.62 hours, within the one-hour cap.

The local Kaggle-status monitor experienced DNS failures while the remote kernel
continued running. The recovered status, downloaded output, and notebook-owned
lifecycle records establish that this was a monitoring-path failure, not a model
transport or notebook-runtime failure.

## Gameplay

| Game | Actions executed | Actions submitted | Actor calls | Critic calls | Levels | Stop |
|---|---:|---:|---:|---:|---:|---|
| LS20 | 17 | 17 | 63 | 8 | 0 | model admission budget |
| S5I5 | 18 | 30 | 45 | 7 | 0 | wall budget |
| CD82 | 12 | 12 | 54 | 7 | 0 | model admission budget |
| **Total** | **47** | **59** | **162** | **22** | **0** | |

The distinction between submitted and executed actions matters on S5I5: two
batches were interrupted after prediction mismatches, so 30 proposed actions
produced 18 environment actions.

## Critic behavior

- 21 of 22 reviews produced a valid, nonempty, source-bound question. One LS20
  response exceeded the word budget and was recorded as invalid rather than
  injected.
- Trigger distribution was 13 `eight_actor_calls`, five
  `three_actions_observed`, and four `prediction_mismatch`. No level advanced and
  no observed sequence reached the sequence trigger.
- Only three of 22 critic requests were followed by `submit_actions` in the
  Actor's immediately next model response. Twenty had some later observed action;
  the mean delay from review completion to that action was 30.6 seconds.
- Critic calls consumed 401.2 seconds, 21.7% of the three game-session wall time.
  Their mean duration was 18.24 seconds. They read 1,515,619 prompt tokens and
  received zero cached prompt tokens. The critic is concise in output, but it is
  not fast on the live full-context path.

The Actor-call timing includes time spent inside nested critic review and must not
be added to critic time to estimate total session duration.

## What the questions changed

The strongest case was S5I5. From critic 3 onward, the critic repeatedly challenged
the inference that filling the bottom bar was the objective. After critic 6 and a
prediction mismatch, the Actor explicitly treated bar completion as unverified and
clicked a displayed object. This was a real local pivot. It arrived after thirteen
executed actions and did not produce objective progress. The Actor later submitted
another ten-click bar-filling batch before probing a second object.

On LS20, the critic challenged both the bar-depletion proxy and whether the moving
block could enter the apparent box. One question caused a direct reversible UP
probe. Later, however, the Actor committed to alternating LEFT and RIGHT to erode
the bar despite never linking depletion to level completion. The critic identified
the gap but did not close the commitment quickly.

On CD82, several questions helped the Actor reconsider whether the red-outlined
shape moved, rotated, or sheared, and one objective question preceded a click probe.
Other reviews drifted into factual mechanics verification, such as asking whether
LEFT rotates the shape and why two SPACEBAR transitions differed by one cell. Those
are useful game questions, but they do not provide the orthogonal epistemic review
we intended.

## Decision

Do not promote this candidate on the strength of this run. Keep the source-binding,
bounded schema, invalid-output handling, and event-driven cadence. The next design
must address two demonstrated boundaries:

1. A critic question is advisory. The Actor can spend many more CodeAct calls
   inspecting, defer the challenged commitment, or resume it without explicitly
   closing the question.
2. Sending the critic the entire growing context makes later reviews progressively
   expensive and yielded no prefix-cache benefit. Fewer reviews did not make the
   critic cheap enough for frequent pivot control.

Further prompt expansion is not licensed by this result. A discriminating next
test should make the pending epistemic question an explicit state the Actor must
answer with a narrower commitment or a cheap environment probe, while giving the
critic a compact evidence and commitment record instead of replaying the full
transcript. That change needs its own controlled card.

## Evidence

- `AUDIT.json`: executable audit and descriptive metrics.
- `audit.py`: read-only assertions used to produce the audit.
- `raw/qualification/solver-workspaces/*/ipc/critic_reviews.jsonl`: questions and
  source bindings.
- `raw/qualification/solver-workspaces/*/ipc/model_calls.jsonl`: live requests,
  timing, token use, and continuations.
- `raw/qualification/solver-workspaces/*/ipc/actions.jsonl`: submitted decisions.
- `raw/qualification/game-progress.jsonl`: executed actions and level outcome.
- `PRELAUNCH.json`, `push.log`, `postrun-status.log`, and `postrun-quota.log`:
  frozen identity, launch, terminal state, and resource accounting.
