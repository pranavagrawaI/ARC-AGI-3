# Staged commit run — fires ONLY if qwen38-low passes its frozen rule

Gate (from `runs/qwen38-low/CARD.md`, frozen before the run):
1. median s/turn **< 46.3** (3.6 on the same recipe), AND
2. levels **>= 4** over 10 episodes.

Both must hold. If either fails: do not commit-run, do not submit, keep the slot.
User decision 2026-09-22: "3.8 @ low only if it passes the rule."

## Command (25 public games, the exact bytes that would be submitted)
```bash
cd ~/projects/arc-agi-3-reference/duck-qwen38/ARC3-Inference
source ~/.cache/duck-step1-env.sh
unset DUCK_CRITIC_ENABLED DUCK_CRITIC_MODE DUCK_CRITIC_TRIGGER DUCK_CRITIC_PERIOD
export DUCK_CHAT_TEMPLATE_KWARGS='{"preserve_thinking": true, "reasoning_effort": "low"}'
# model pins default to 3.8; DUCK_MODEL_DATASET_SOURCE / DUCK_SERVED_MODEL_NAME would select 3.6
uv run --no-sync --offline inference-taaf-run \
  --environments-dir /kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files \
  --run-name duck-qwen38-low-submission --experiment-dir ../runs/qwen38-low-submission \
  --max-runtime-minutes 120 --max-experiment-runtime-minutes 540 \
  --n-passes 1 --concurrent-jobs 28 --agent duck-harness --model local --analyzer-timeout 900 \
  --deployment-target kaggle --no-deployment-wait \
  --kaggle-duck-public-harness --kaggle-make-share-version \
  --kaggle-username pranavagrawai \
  --kaggle-kernel-slug taaf-duck-qwen38-low-submission --kaggle-kernel-title taaf-duck-qwen38-low-submission \
  --kaggle-dataset-ref pranavagrawai/taaf-kaggle-source-duck-qwen38-low-submission \
  --kaggle-accelerator NvidiaRtxPro6000 \
  --kaggle-dataset-version-message 'qwen38-low submission: duck on Qwen3.8-27B-FP8 @ reasoning_effort=low, 28-way, 120 min/game'
```

## Pre-push checks (dry run first)
- bundle carries `qwen3-8-27b` and `reasoning_effort": "low"`, 0 `DUCK_CRITIC_*` keys
- pickled per-game cap **7200 s**, target `max_runtime_s` **32400**
- guard string `Applies in a real submission` present in the bundled notebook
- Kaggle token expiry covers push + ~2.3 h run + submit (refreshed 18:14Z -> 2026-09-23 06:14Z)

## Clock (unchanged, and why)
120 min x 4 waves + ~10 min startup = 490 min against the 540 min limit, 50 min slack.
Cap deliberately NOT raised: every level in the 2026-09-18 commit run landed by minute 78
(`analysis/level-timing-20260918.py`), so tail minutes buy nothing and slack protects against
the startup overrun that killed submission 56303234.

## Bar for the commit run itself
2026-09-18 on 3.6: 13/25 games with >=1 level, **17 levels**, TAAF 1.90 -> public 0.67.
Below ~17 levels, do not spend the submission slot: it cannot beat what is already on the board.

## Then
`kaggle competitions submit arc-prize-2026-arc-agi-3 -k pranavagrawai/taaf-duck-qwen38-low-submission -v 1 -f submission.parquet -m "..."`
