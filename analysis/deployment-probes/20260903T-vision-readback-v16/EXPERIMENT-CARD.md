# vision-readback-v16

Question: On the exact qualified Kaggle route (pinned `Qwen/Qwen3.8-27B-FP8` @ `017b9c7a…`,
patched R8-L3B SGLang, Triton attention, PyTorch sampling), does a 64×64 public `ls20` frame
rendered at 32 px/cell reach the model as 4,096 visual tokens without downscaling, and does the
served model recover sampled cell colours, the palette, and one object's bounding box more
accurately from that image than from the pipe-delimited text grid V15 actually sent?

Posture: bounded perception qualifier on three recorded frames of one public game. No gameplay,
RHAE, level-completion, or decision-quality claim.

Contrast: V15 (Kaggle kernel 13, 2026-09-03) used this same model route, backends, warmup, and
TERM-only lifecycle, and its runtime warmup already sent one 256×256 PNG `image_url` and received
a correct `take_actions` tool call in 8.103 s. V15 did not test this question: after that warmup
the Solver sent the Actor only `spaced_grid8` text (9,555 chars for the turn-0 frame), and Actor
calls 1, 2 and 5 each consumed the full 16,000 completion tokens as reasoning and emitted no tool
call. V16 changes exactly three things and nothing else in the route: (1) the outer runtime gains
a `--vision-readback-probe` mode that runs a perception probe against the ready server in place of
`--official-launcher-probe`, keeping model-hash verification, SGLang launch, warmup, request-count
reconciliation, containment and teardown unchanged; (2) the notebook drops the Solver wheelhouse,
the ARC-AGI-3-Agents launcher, the public-environment mount, and the submission cell, none of which
this question needs; (3) three `ls20` decision frames are pinned as a hashed asset. No game is
played, no environment action is committed, and no competition source is attached.

Frames: records 0 (level 0), 100 (level 3), and 300 (level 5) of the recorded winning trajectory
`recordings/ae20a3e4-…/ls20-9607627b-b7f0efe3-….jsonl`, frozen as
`frames.json` SHA-256 `fd4031afec63975cd5bc98e8cc32445f54b510ac927aa486708447ca02b2d21d`.
Ground truth is computed from the frame arrays, never from a model answer. Frame 0's bounding-box
truth for colour 12 is rows 45–46, cols 34–38, which independently matches the region V15's
`run_python` call reported.

Conditions (input modality × generation contract), all at `tool_choice=required`, temperature 0:

| condition | grid text | image | thinking | max_tokens | frames |
|---|---|---|---|---|---|
| `text_nothink` | yes | — | off | 2,048 | 3 |
| `image06_nothink` | — | 384×384 | off | 2,048 | 3 |
| `image16_nothink` | — | 1024×1024 | off | 2,048 | 3 |
| `image32_nothink` | — | 2048×2048 | off | 2,048 | 3 |
| `both32_nothink` | yes | 2048×2048 | off | 2,048 | 3 |
| `text_think` (incumbent) | yes | — | **on** | **16,000** | 1 |

Tasks per frame: `cells` (12 coordinates spread across the frame, mixing the dominant colour, the
three rarest, the colour-11 bar rows V15's actor misread as terrain, and the corners), `palette`
(the distinct colours present), `bbox` (the inclusive box of the rarest colour with ≥ 4 cells).
Guessing the single most common sampled colour scores 4/12, 4/12 and 3/12 on the three frames, so
the chance floor for `cells` is 0.25–0.33.

Prediction: the scale-32 render costs about 4,096 prompt tokens above the identical image-free
request, matching one merged visual token per ARC cell; scale 16 costs about 1,024 and scale 6
about 144. `image32_nothink` mean `cells` accuracy exceeds `text_nothink` by at least 2× and
exceeds 0.75, at under one tenth of `text_think`'s completion tokens. `text_nothink` scores near
the chance floor, because without reasoning tokens the model cannot count column blocks.
`text_think` scores well but spends two to three orders of magnitude more completion tokens per
answer. The known risk against this prediction is that palette indices 3, 4 and 5 are adjacent
greys (102/51/0), so per-cell colour may be legible as pixels yet not resolvable to an index.

Measure: for each render scale, prompt tokens with and without the image and their difference
(`visual_token_measurement`); per request, the condition, frame, task, prompt/completion/reasoning
tokens, finish reason, elapsed seconds, the tool arguments returned, the ground truth, and the
score (`cells` positional accuracy, `palette` set equality and Jaccard, `bbox` exact match and
IoU); per condition, mean accuracy, exact count, total completion tokens and total elapsed;
SGLang `server.log` including any image-resize or `max_pixels` message; `backend-facts.json`;
`readiness.json`; warmup facts; the reconciliation of probe-reported request counts against the
server access log; the R8-L3B lifecycle, containment and teardown record.

Cap: one RTX PRO 6000 notebook; three pinned frames of one public game; 53 chat completions
(1 runtime warmup + 52 probe: 48 readback, 4 token measurement); zero environment actions; 1,200 s
startup; 5,400 s probe wall clock; internet disabled; no competition source attached; financial
spend not applicable.

Stop: stop when all 52 probe requests have been issued or the 5,400 s probe deadline is reached,
whichever comes first; abort on model or runtime hash drift, competition-rerun mode, a backend that
does not resolve to Triton/PyTorch, a request count that does not reconcile with the server access
log, a frame asset whose hash or schema differs, model-server error, or containment/lifecycle
failure.

Invalid if: the installed checkpoint or runtime hashes differ from the frozen manifest; the pinned
frame asset hash differs from the value above; any request omits `tool_choice=required` or the
declared `max_tokens`; a no-thinking condition is sent without the exact
`{"chat_template_kwargs": {"enable_thinking": false}}` override, or `text_think` is sent with it;
the probe's reported request count differs from SGLang's successful chat-completion accesses;
ground truth is read from anything other than the pinned frame arrays; any game is played, any
environment action is committed, or a submission is produced.

Result map:
- Scale 32 measures ≈ 4,096 visual tokens **and** `image32_nothink` beats `text_nothink` by ≥ 2×
  on `cells` at ≥ 0.75 → the multimodal channel is qualified for this route; issue a card that
  feeds rendered frames to the Actor and re-costs the decision loop against V15's 1,561 s
  time-to-first-action.
- Scale 32 measures ≈ 4,096 visual tokens but every image arm scores at or near the chance floor →
  the tokens arrive and the mapping fails. Do not conclude the model cannot see; issue a card
  varying the *rendering* (cell gridlines, a high-contrast palette, a colour legend swatch) before
  abandoning the channel.
- Scale 32 measures materially fewer than 4,096 visual tokens → this SGLang deployment downscales
  the render, and `lossless_cells` is false on our route regardless of the accuracy numbers.
  Diagnose the serving-side limit (`max_pixels`, image-processor config, tokenizer truncation) from
  the preserved `server.log` and issue a new card. Do not report an accuracy comparison as a
  perception claim.
- `both32_nothink` clearly beats both `image32_nothink` and `text_nothink` → the image is an aid,
  not a replacement; the follow-on card must keep the text and re-cost the combined prompt rather
  than assume the text can be dropped.
- Request-option drift, unreconciled request counts, lifecycle or containment failure, or ambiguous
  evidence → repair the defect and issue a new card.

Do not create a competition submission.

Outputs: local candidate `deployment/kaggle/s0-vision-readback-v16-candidate`; frozen card,
candidate hashes, and reconciliation under `analysis/deployment-probes/20260903T-vision-readback-v16`;
exact remote source and output downloaded before interpretation.

Claim ceiling: a passing run establishes only what this route charges for a rendered 64×64 ARC
frame at each tested scale, and how accurately the served model recovers sampled cells, the
palette, and one bounding box from each input modality on three recorded `ls20` frames. It
establishes nothing about gameplay, RHAE, level completion, other games, other frames, or whether
an image-fed actor decides better.
