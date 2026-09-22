# qwen38-qualify — Duck alone on Qwen3.8-27B-FP8 (Step 1 recipe)

Frozen before push, 2026-09-22.

## Question
Does Duck, unchanged except for the checkpoint, run on `jakobbrggen/qwen3-8-27b-fp8-hf-snapshot`
under Duck's vLLM 0.19.0 wheelhouse on Kaggle's RTX PRO 6000, and clear L1 on sb26/su15?

## Why this is worth one kernel
Both checkpoints are `Qwen3_5ForConditionalGeneration` with byte-identical `text_config`
(diffed from the two Kaggle datasets' config.json on 2026-09-22). 3.8 uses the official
`quant_method: fp8` (dynamic e4m3, vision excluded) instead of vrfai's compressed-tensors static
FP8. Unknowns only a run can settle: 3.8's chat template (8952 B vs 7764 B) against the
`qwen3_coder` tool parser on vLLM 0.19.0, and the 3.8 vLLM version floor. 3.8 has only ever been
served here through SGLang (NOOA).

## Tree
Branch `qwen38-submission` from the submitted tree `04038a4`. Diff: model dataset + served name
(`inference/framework/kaggle.py`, `configs/inference.json`), kaggle-duck cap 132 -> 128 min
(Makefile; informational, the CLI passes the cap), and the notebook templates now keep the
soft deadline in a real submission (graceful cancel at budget - 10 min instead of a hard kill).
Sampling unchanged (T 0.6, top-p 0.95, top-k 20, thinking on): the checkpoint is the only variable.

## Recipe (= Step 1, `runs/step1-duck-alone` in duck-harness)
5 games (sb26, su15, lf52, ft09, ls20) x 2 passes, 600 s/game, concurrency 10, 1 wave,
experiment cap 75 min, analyzer timeout 900 s, NvidiaRtxPro6000, no critic env.

## Kill rule (any one -> do not proceed to a commit run on 3.8)
- vLLM fails to load the checkpoint or the API smoke test fails.
- Tool calls fail to parse (transcripts show `tool_calls_recovered_from_markup: yes` or
  `finish_reason` != `tool_calls` as the norm).
- 0/4 L1 on sb26/su15 (3.6 baseline: 4/4).

## Pass -> next
Commit run of the exact submission bytes: 25 public games, 28-way, 128 min cap, share notebook,
then submit. Also read: s/turn at 10-way vs 3.6's 46 s, output tokens/turn, reasoning share.

## Prediction (written before the run)
Loads and parses (same architecture, standard FP8, full snapshot): 80 %. L1 on sb26/su15
>= 3/4: 60 %. Turn latency within 25 % of 3.6's: 70 %.
