# Vision readback on qwen/qwen3.8-27b via OpenRouter — the prediction failed

52 requests, 180 s, **$0.10**. Three pinned public `ls20` frames (levels 0, 3, 5),
`reasoning: {enabled: false}` verified on every no-thinking arm (0 reasoning tokens, 48/48),
`text_think` reasoning 618/291/219. Zero contract violations.

## 1. The image is delivered losslessly. The model still cannot read cells from it.

Prompt-token cost, differenced against the identical image-free request:

| render scale | image | expected if 1 token/cell | observed | verdict |
|---|---|---:|---:|---|
| 6 | 384×384 | 144 | 146 | as predicted |
| 16 | 1024×1024 | 1,024 | 1,026 | as predicted |
| 32 | 2048×2048 | 4,096 | **4,098** | **one merged visual token per ARC cell** |

Tycho's `vision.py` arithmetic (patch 16 × spatial-merge 2 = 32 px/token) is confirmed exactly,
with two tokens of image framing. No downscaling at 4,194,304 pixels.

*(My `one_token_per_cell` check demanded `== 4096` and so reported `false` on a result that
confirms the claim. That was a defect in the check, now allowing a framing allowance.)*

## 2. And it does not help. Per-cell readback is at chance from the image.

Chance floor = answering each frame's most common sampled colour: 0.333 / 0.333 / 0.250.

| condition | cells | palette | bbox | mean | completion tokens |
|---|---:|---:|---:|---:|---:|
| `text_think` (incumbent) | **12/12 = 1.000** | 1.000 | 0.000 | 0.667 | 2,828 / 3 req |
| `text_nothink` | 21/36 = 0.583 | 0.869 | 0.652 | 0.702 | 563 / 9 req |
| `both32_nothink` | 21/36 = 0.583 | **1.000** | 0.652 | **0.745** | 564 / 9 req |
| `image32_nothink` | 13/36 = 0.361 | 0.715 | 0.355 | 0.477 | 599 / 9 req |
| `image16_nothink` | 12/36 = 0.333 | 0.545 | 0.437 | 0.439 | 2,551 / 9 req |
| `image06_nothink` | 8/36 = 0.222 | 0.681 | 0.063 | 0.322 | 569 / 9 req |

**Every image-only arm is at or below the chance floor on cells, including the lossless one.**
The card predicted `image32` would beat `text_nothink` by ≥2× and exceed 0.75. It scored 0.361
against text's 0.583. The prediction is falsified.

## 3. The failure is coordinate indexing, not colour fidelity

Frame 0, truth `[5, 4, 4, 0, 4, 1, 4, 12, 5, 11, 8, 5]`:

```
image32   [5, 4, 2, 0, 3, 2, 2,  2, 5,  5, 5, 5]
text      [5, 4, 4, 3, 4, 3, 4,  3, 5, 11,11, 5]
think     [5, 4, 4, 0, 4, 1, 4, 12, 5, 11, 8, 5]   exact
```

The image arm reports grey `2` where the truth is orange `12` (255,133,27), yellow `11`
(255,220,0) and red `8` (249,60,49) — vivid, unambiguous colours, not adjacent greys. It is not
misreading colour; it is not sampling the named coordinate at all, and collapses toward the
field colour. Bounding boxes confirm this: `image32` frame 0 gives `[45,38,45,41]` against truth
`[45,34,46,38]` — right row band, wrong columns. Good gross localisation, no exact indexing.

At scale 6 it gives up entirely, returning `[1,1,62,62]` and `[0,0,63,63]` — the whole grid.

## 4. Two findings that cut against my earlier report

**The ASCII grid is not unreadable to this model.** `text_think` scored 12/12 on cells for
219–618 reasoning tokens per request. My reference-study report told a story in which hand-parsing
the grid is where V15's tokens died. V15's failure was an open-ended comprehension task with eight
tools, not a bounded readback — the grid format is not intrinsically the blocker I claimed.

**The pipe format induces a one-block column offset.** Both text arms returned columns 42–46 for a
truth of 34–38 on frame 0 — exactly +8, one `|`-delimited column group. That is a concrete,
cheap-to-test defect in the incumbent representation, and a smaller intervention than switching
perception channels.

## 5. Consequence for the Kaggle probe

The V16 vision probe is substantially devalued. Its gating question was whether *our* SGLang
admits 4,194,304 pixels without downscaling. Even answered yes — and the model's own processor
demonstrably does not downscale — per-cell readback on this executor is at chance. Getting the
same 4,098 tokens out of SGLang would not change that. **Do not re-push it as-is.**

## 6. Claim ceiling

Three frames of one public game; 36 cell judgments per condition, not independent within a frame;
one bbox and one palette call per frame. Differences among the image scales are inside noise. This
is **hosted `qwen/qwen3.8-27b` (Alibaba/Ionstream)**, not our `Qwen3.8-27B-FP8` on pinned SGLang —
same family and scale, different weights and serving. It measures readback, not gameplay, and says
nothing about whether an image helps an agent decide, only that it does not deliver exact cells to
this model.
