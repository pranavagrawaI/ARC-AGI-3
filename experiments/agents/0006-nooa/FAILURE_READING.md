# Reading the recorded failures

Offline analysis of every run under `artifacts/runs/`. No provider calls. Three
measures, all computed from artifacts already on disk: the pre-registered inert
action rate, the model's own outcome self-classification scored against
`host/commitment_controller.py::classify`, and the per-crossing prompt cost.

## 1. Inert action rate

`inert_action_rate.py`, unmodified. An ls20 transition changes exactly 50
play-area cells or none.

| run | condition | model | inert |
|---|---|---|---:|
| control-44-20260803T080119Z | none, control arm | Gemma | 34/43 = 79% |
| treatment-44-20260803T080119Z | none, competence packet | Gemma | 34/44 = 77% |
| known-world-rules-l1-44-v1 | rules | Gemma | 30/43 = 70% |
| known-world-executable-l1-44-v1 | executable | Gemma | 29/43 = 67% |
| supervised-…-l1-44-v9 | rules_supervised | Gemma | 9/13 = 69% |
| supervised-…-l1-44-v3 | rules_supervised | Gemma | 22/38 = 58% |
| supervised-…-preferred-l1-44-v1 | rules_supervised | Gemma | 1/7 = 14% |
| controller-probe-rules_controller-l1-44-v1 | rules_controller | Gemma | 0/17 = 0% |
| unassisted-none-…-luna-openai-l1-44-v1 | none | Luna | 9/44 = 20% |
| known-world-rules-…-luna-openai-l1-44-v3 | rules | Luna | 3/27 = 11% |

Observation: unscaffolded Gemma wastes roughly four of every five actions.
Knowledge conditions move it a little. Luna wastes one in five unassisted and
one in nine with rules. The `rules_controller` run recorded zero inert actions
in 17 — but see §2 before reading that as competence.

## 2. The model cannot tell what its own action did

`classify()` exists in the controller as a scoring instrument and is never shown
to the model live: the host supplies a neutral `changed_cells` count and requires
the model to name `LAST_OUTCOME` itself. Scoring the model's label against the
mechanical one:

| run | n | agreement | confusion (reported → truth) |
|---|---:|---:|---|
| controller-probe-rules_controller-l1-44-v1 | 16 | **0%** | blocked→moved ×16 |
| supervised-…-l1-44-v3 | 37 | 30% | special_event→moved ×15, moved→blocked ×10 |
| supervised-…-l1-44-v9 | 12 | 33% | moved→blocked ×8 (all after the first four) |
| supervised-…-open-l1-44-v1 | 7 | 100% | — |
| supervised-…-preferred-l1-44-v1 | 7 | 86% | moved→blocked ×1 |

Observation: in the two longest controller/supervised runs the model's account of
its own last action is worse than chance over three labels. v9 is correct for its
first four action points and wrong for its last eight.

## 3. Why: a misbinding the controller then made permanent

In `controller-probe-rules_controller-l1-44-v1`, `PLAYER_BINDING` is
`white plus at (32, 21)` for all 17 action points. That cell is constant for the
whole run under either coordinate reading — `frames[32][21]` is colour 0 at every
step, `frames[21][32]` is colour 4 at every step.

Meanwhile every transition moves a 5×5 block by five cells: the changed cells
form two disjoint 5×5 clusters (e.g. step 0→1, rows 40-49 × cols 34-38; step
1→2, rows 40-44 × cols 34-43).

Interpretation: the model bound the player to a static object, then correctly
observed that its object never moved, and reported `blocked` 16 times while the
real avatar walked. `CURRENT_APPROX_POSITION` never updates. The same frozen
position appears in supervised v3, stuck at `32, 21` from action point 2 to 43
while `LAST_OUTCOME` says `moved`.

The controller rules `R1_BINDING_PERSISTS` and `R2_NO_RESCAN_ON_MOVE` exist to
stop the model rewriting its identity claim on every move. Here they locked in a
wrong one. The one hard guarantee the scaffold provides made the initial
perceptual error irreversible.

Falsification that would overturn this: show that the 50-cell two-cluster change
is an animation rather than the avatar translating, or that a white plus at the
declared cell exists and is the player.

## 4. Rejection taxonomy

25 `competence_rejected` records across all runs.

| class | n | what it is |
|---|---:|---|
| supervision handoff shape | 9 | Gemma cannot emit `FOLLOW <token>: …` / valid JSON |
| controller identity rules | 11 | R2 ×5, R3 ×3, R8 ×2, R1 ×1 — re-describing the player after a move |
| competence schema | 3 | reconciliation misuse, missing transition, trivially-true assertions |
| substantive contradiction | 3 | Gemma disputing the supervisor's stated player position |

Observation: 20 of 25 rejections are the model failing to *state* something in
the required form, not failing to choose an action. The 3 contradictions are
Gemma disputing Luna's positional claim; whether Gemma or Luna was right is
unresolved and worth checking, since it bears on whether the supervisor's
positional assertions are reliable enough to bind.

## 5. The tax, measured

Median input tokens per provider crossing, and model rounds per action point.

| run | condition | model | median in | rounds/action |
|---|---|---|---:|---:|
| control-44 | none, control arm | Gemma | 12,309 | 1.14 |
| known-world-rules | rules, control arm | Gemma | 17,262 | 2.52 |
| treatment-44 | none, competence packet | Gemma | 30,662 | 2.18 |
| controller-probe-rules_controller-v1 | rules_controller | Gemma | 38,265 | 2.44 |
| supervised-…-l1-44-v3 | rules_supervised | Gemma | 47,792 | 1.93 |
| luna unassisted | none | Luna | 25,222 | 1.32 |
| luna rules | rules | Luna | 24,400 | 1.32 |

Observation: the full scaffold takes Gemma's prompt from 12.3k to 47.8k — 3.9× —
and roughly doubles deliberation rounds. Luna's rounds/action is 1.32 in both of
its conditions and does not rise with the scaffold. The two Gemma conditions with
the largest prompts are also the two with the worst outcome self-classification.

## 6. Known measurement defect

`competence.py::_changed_cells_by_layer` returns `None` when the before/after
layer counts differ, and ls20 intermittently emits 6-layer or 2-layer frames. In
the Luna unassisted run this silently voided 8 of 44 predictions as
`inconclusive`. Comparing the rendered layer, or `changed_cells_by_layer`,
recovers them.

## 7. The three contradictions, adjudicated

A minimal host tracker settles these mechanically. Method: changed play-area
cells form a 5×10 or 10×5 bounding box; split into two 5×5 halves; the half whose
`before` content is non-background is the source. Absolute position is recovered
by template search for the avatar patch (`12,12 / 9,9,9`).

Validation: displacement is exactly 5 cells every time, and destination at step t
equals source at step t+1 — continuity 17/17, 4/4, 7/7 on the three runs checked.

Note a trap in the obvious implementation: `before[A] == after[B]` and
`before[B] == after[A]` are *both* true for a translation, so the first-match
test always reports down/right. Direction must be resolved by identifying which
patch holds the avatar, not by half ordering.

| case | Luna claimed | Gemma claimed | ground truth | verdict |
|---|---|---|---|---|
| l1-3-v1 ap3 | (41, 32) | (37, 32) | rows 35-39, cols 34-38 | **Gemma right on row, Luna a full block off** |
| l1-44-v9 ap13 | row 35, col 31 | row 35, col 32 | rows 35-39, cols 34-38 | both row-correct; dispute was ±1 column |
| open-v1 ap9 | row 26, col 17 | row 31, col 17 | rows 30-34, cols 19-23 | **Gemma right on row, Luna one move stale** |

Gemma's binding description in case 3 — "orange-blue vertical block" — matches the
template exactly.

Two consequences.

**Luna launders Gemma's number.** In case 2 Luna's `RELEVANT_EVIDENCE` reads "The
current state estimates the player's position as row 35, column 31" — that is
Gemma's own `CURRENT_APPROX_POSITION` quoted back. Luna is not measuring position
from pixels; it is repeating the state block. So the supervised condition has a
circularity: Gemma's positional error returns to Gemma as supervisory evidence,
and case 2's "contradiction" is Gemma disputing its own number by one column.

**The recovery path exists but does not compose.** `R1`/`R2` permit rebinding when
a `CONTRADICTION` is named, so the controller is not a pure consistency trap. But
in l1-3-v1 Gemma contradicted correctly at ap3, then changed `PLAYER_BINDING` at
ap4 and was killed by `R1_BINDING_PERSISTS: changed with no CONTRADICTION`. The
licence is scoped to a single turn and does not persist across the reacquisition
it is supposed to enable.

## 8. Open: a systematic column bias in both models

In all three cases both models get the row right or nearly right and under-report
the column by 2-3 cells (31/32 against a true 34-38; 17 against a true 19-23).
`p9/annotated_frame.py` draws column labels at `(x+1, 2)` and row labels at
`(2, y+1)`, both anchored on the tick, with no visible offset error. So this needs
a direct test — render a frame with a known object and ask each model to read its
coordinates — rather than a code fix. If real, it affects every run in the record.

## 9. The image the model is given is not the image it is told about

Verified offline on `unassisted-none-treatment-luna-openai-l1-44-v1`, crossing 0.

The attached `input_image` is ~7.6 kB. On disk, `worker/model-frame.png` is
**472×472** (the annotated frame: 64×7 + 24) and
`worker/workspace/current_frame.png` is **512×512**, 2.4 kB. The attached image
is the annotated one.

The JSON in the same message declares:

```json
"rendered_image": {"path": "current_frame.png", "shape": [512, 512, 3],
                   "source_grid_scale": 8,
                   "annotation": {"cell_pixels": 7, "margin_pixels": 24, ...}}
```

Three of those four facts are false about the image actually in the message. The
cause is `host/competence_runner.py:632-637`: the base `representations` block is
copied and `annotation` is merged into `rendered_image`, but `path`, `shape` and
`source_grid_scale` are left describing the unannotated 512×512 render.

This is present in every treatment-arm run in the record, including the two Luna
runs that cleared level 1.

**Hypothesis it may explain.** A model doing pixel arithmetic under the declared
scale 8 on an image drawn at scale 7 reports approximately `true × 7/8`:

| true column span | ×7/8 | reported |
|---|---:|---|
| 34-38 (centre 36) | 31.5 | 31 (Luna), 32 (Gemma) |
| 19-23 (centre 21) | 18.4 | 17 (both) |

Rows do not fit this at all — true centre 37 reported as 35 and 37. A possible
reading is that row labels down the left margin are legible and read directly,
while top-margin column labels are 6-px digits over 7-px cells and crowded, so
the column is computed instead of read — using contaminated metadata. One
observation is unexplained by any version: `PLAYER_BINDING: white plus at
(32, 21)` in the controller run, where the column fits 7/8 of the true 36 but the
row 21 matches nothing (true rows 45-49).

**Status: the metadata defect is established; the explanation is not.** The
coordinate test distinguishes them. The model-visible contract is corrected after
these runs in `host/competence_runner.py`; no historical artifact was rewritten.

## 10. Comparative status

Every treatment-arm spatial result in this record is contaminated by the false
image contract in §9. Those runs remain evidence of behavior under the interface
the models actually received, but they are not valid evidence for a clean spatial
comparison with the control arm. Later work must not silently pool them with runs
made under the corrected contract.

The experiment is complete as a diagnostic. It established the tracker defect and
its corrected source-template implementation (§7), the false image contract (§9),
and one concrete failure of R1 recovery to compose across turns (§7). It does not
establish that the 0006 treatment improves the frozen 0002-NOOA worker.
