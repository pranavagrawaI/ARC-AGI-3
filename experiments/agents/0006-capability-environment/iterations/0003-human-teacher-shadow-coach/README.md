# Iteration 0003 — human teacher → shadow coach

Status: **Q0 rejected by human review**.

## Decision

Sparse human notes are a diagnostic teacher, not an inference-time dependency of the
admissible contestant. The immediate question is whether their function can be reproduced
by an autonomous coach with the same public prefix.

## Failure selected from iteration 0002

The actor did not represent objective uncertainty. It invented absent user intent, then
spent its attention on perception and mechanism details inside that unsupported premise.
At frame 26 it wrote:

> The user wants to fill the bottom row of the grid.

No user had supplied that objective. Objective-result falsification did not propagate
upstream: action 6 remained 44/44, the same tautological objective was reinstated eight
times, and zero levels completed.

## Unifying model

0006 is not five independent fixes for perception, memory, mechanism learning, objective
discovery, and meta-cognition. It is one layer-arbitration problem:

```text
represent current unknowns
→ identify which layer blocks the next consequential decision
→ choose the cheapest discriminator
→ act or compute
→ propagate contradiction back to the responsible layer
```

A sparse human question may supply this missing control signal. This iteration tests that
claim before building a unified controller.

## Frozen source point

The source is the first actor request after environment step 26 in rejected v2:

```text
run: iterations/0002-bounded-objective-falsifier/artifacts/live-treatment-cd82-v2
request: request-payloads/000173.json
request SHA-256: e3769325db307143d5d74a72a9f2d454a90a1dc5725ddfdc14a0c6e5fa286055
action step to decide: 27
public state: NOT_FINISHED, levels_completed=0
legal actions: 1, 2, 3, 4, 5, 6
prior environment actions: action 6 on steps 1–26
archived next decision: action 6
```

The source image and prefix are admissible public evidence from the contestant's own run.
No successor after step 26 may enter any branch request.

## Human-note contract

The human authored exactly one question. It is frozen verbatim in
[`human-note.json`](human-note.json):

> have we tried everything to come to a final conclusion, or do we have more of the world
> we don't understand?

The note has 21 words. Its artifact SHA-256 is
`6ab7db6de14635069dc7a4a6964f4df31b2ad3dd8144ff552932be6e27e42dc6`.
The broader note contract is:

- at most 25 whitespace-delimited words;
- question form;
- no coordinates, regions, colors, shapes, counts, object labels, or visual features;
- no action ID, candidate action, tool command, or location;
- no mechanism, objective, correctness, progress, or outcome claim;
- no remembered game-specific information;
- may ask about unknowns, assumptions, alternatives, discrimination, layer choice, loops,
  or whether evidence licenses commitment.

The note is fallible guidance under the existing labeled `human_coach` channel. It is not
environment truth and makes the branch supervised/non-admissible.

## Four-call Q0 probe

All new calls use the same actor model, provider pin, sampling, tools, image, and frozen
prefix unless the branch definition changes them. The archived response is contextual
evidence, not the stochastic control.

1. **N — new no-note actor call.** Exact source request, reissued under the new generation's
   provider pin.
2. **H — human-note actor call.** N plus the exact frozen labeled human note.
3. **C — autonomous coach call.** A separate coach sees only the same public prefix through
   step 26 and the note contract. It must emit one question of at most 25 words and no
   answer.
4. **S — shadow-note actor call.** N plus C's frozen output under an explicitly autonomous
   `shadow_coach` channel. It must not masquerade as human advice. This channel-label
   difference is necessary to test an admissible autonomous replacement; it remains a
   possible causal explanation if H succeeds and S fails.

Hard ceiling: four paid calls. No environment action is executed. No later v2 response or
transition is visible to C, H, or S. Raw requests/responses are retained locally and ignored
from Git.

## Human semantic gates

Review N, H, and S blind to branch label. A response passes only if all are true:

1. **Unknown modeled:** it explicitly distinguishes unresolved objective knowledge from
   observed local mechanism evidence.
2. **Upstream layer reopened:** it reconsiders objective discovery instead of spending the
   response only on color/location perception inside the inherited theory.
3. **Live alternative or discriminator:** it states a competing explanation or chooses a
   computation/action whose outcomes distinguish objective theories.
4. **No invented principal:** it does not manufacture user intent or treat the coach note as
   a goal.

Action-class change is a useful diagnostic, not an automatic pass requirement: action 6
could still be the cheapest discriminator if the response explains why. Pixel/coordinate
work is not itself a failure; it fails when it consumes the decision without reducing the
identified uncertainty.

## Decision table

| Observation | Licensed next move |
|---|---|
| H fails | Kill this intervention point; sparse coaching potency is not reproduced here. |
| H passes, S fails | Human contributes semantic judgment not reproduced by the current autonomous coach; inspect the functional delta before architecture. |
| H and S pass | Build one full-loop autonomous layer-arbitration treatment, then test memory/Python as execution bottlenecks. |
| H and S both change prose but not the next discriminator | Reject as another form-compliance treatment. |

This Q0 probe establishes at most immediate functional potency at one decision. It does not
establish gameplay benefit, autonomous RHAE improvement, or generality across games.

## Prepared generation

Real preflight `q0-preflight-v1` completed without a model call:

```text
source request:  e3769325db307143d5d74a72a9f2d454a90a1dc5725ddfdc14a0c6e5fa286055
human note:      6ab7db6de14635069dc7a4a6964f4df31b2ad3dd8144ff552932be6e27e42dc6
no-note request: 42b3b793df417253f8fce139e38d9ef4a18be3100b82901dc23486b02d1e0df6
human request:   fedc2d5d7a081a46f446b8d57f4feb5183bab895aab4ef9e0adeb6ce929c230f
provider:        deepinfra only; fallback disabled
responses:       absent
```

The CLI and fake-endpoint tests verify successful four-call closure, source-hash refusal,
prepare-only zero-call behavior, honest shadow labeling, and persisted failed-transport
evidence. Paid execution creates a fresh output directory; it does not mutate this
preflight generation.

### Failed generation `q0-live-v1`

The approved DeepInfra-only generation attempted one call and stopped:

```text
stage:        no_note_actor
provider:     deepinfra only
HTTP status:  405
provider:     DeepInfra
error:        gemma-4-31B-it-turbo does not accept image input
request:      42b3b793df417253f8fce139e38d9ef4a18be3100b82901dc23486b02d1e0df6
response:     dd2b76be6a31c575b3807017cf2150d689358fead065463c1fe7dee78cdac591
```

No branch completion was produced. This attempt cannot compare N, H, or S and is not
retried or overwritten.

### Replacement preflight `q0-preflight-v2-modelrun`

The archived source response identifies ModelRun as the provider that successfully served
the exact image-plus-tools request. A new zero-call preflight pins the current endpoint tag
`modelrun/fp4`:

```text
no-note request: db40af6cb732b103e551363073c1a068a61e20215a208abeae42f587c572dc36
human request:   62cc2896ad77940b45f77d8045db67c3c373bafe38e0a95d14d764e268bbb3f0
provider:        modelrun/fp4 only; fallback disabled
responses:       absent
```

Changing the provider identity creates a new generation and requires fresh paid execution
approval.

### ModelRun execution and verdict

`q0-live-v2-modelrun` attempted three calls and stopped before the shadow actor:

| Stage | Result | Reported cost |
|---|---|---:|
| No-note actor | Repeated action 6 | `$0.00720874` |
| Human-note actor | Requested Python pixel sampling | `$0.00438939` |
| Shadow coach | 512 reasoning tokens; no content; `finish_reason=length` | `$0.00700458` |

Total reported cost was `$0.01860271`.

Human review rejected the late one-question intervention. The note changed the immediate
move from another click to more perception work, but it did not expose objective
uncertainty, reopen objective discovery, or stop inventing user intent. The requested
Python sampled colors from objects already inside the fixed theory. See
[`human-review.json`](human-review.json).

This rejection does not decide early coaching, adaptive dialogue, or a repaired autonomous
coach. No next mechanism is selected.
