# Commit-run recipe (prepared 2026-09-22, NOT launched — gated on qualification passing)

Run only if `runs/qwen38-qualify/RESULT.md` records a PASS against the CARD.md kill rule.
This run qualifies the exact bytes that would be submitted.

## Clock arithmetic (verified against `inference.framework.run._wave_count`)
| | commit run (25 public) | rerun (110 hidden) |
|---|---|---|
| waves at 28-way | 1 | 4 |
| 128 min/game | 128 min | 512 min |
| + ~10 min startup | 138 min | 522 min |
| Kaggle limit | 540 | 540 |
| slack | — | 18 min |

Kernel `max_runtime_s` = 540 min = 32400 s. With the guard now active in submission, the soft
deadline fires at start + 8 h 50 m, leaving 10 min for teardown and `submission.parquet`
instead of a platform hard kill. The per-game cap is passed explicitly, so it does not change
when the gateway swaps in the 110-game list at runtime.

## Command
```bash
cd ~/projects/arc-agi-3-reference/duck-qwen38/ARC3-Inference
source ~/.cache/duck-step1-env.sh
unset DUCK_CRITIC_ENABLED DUCK_CRITIC_MODE DUCK_CRITIC_TRIGGER DUCK_CRITIC_PERIOD
uv run --no-sync --offline inference-taaf-run \
  --environments-dir /kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files \
  --run-name duck-qwen38-submission --experiment-dir ../runs/qwen38-submission \
  --max-runtime-minutes 128 --max-experiment-runtime-minutes 540 \
  --n-passes 1 --concurrent-jobs 28 --agent duck-harness --model local --analyzer-timeout 900 \
  --deployment-target kaggle --no-deployment-wait \
  --kaggle-duck-public-harness --kaggle-make-share-version \
  --kaggle-username pranavagrawai \
  --kaggle-kernel-slug taaf-duck-qwen38-submission --kaggle-kernel-title taaf-duck-qwen38-submission \
  --kaggle-dataset-ref pranavagrawai/taaf-kaggle-source-duck-qwen38-submission \
  --kaggle-accelerator NvidiaRtxPro6000 \
  --kaggle-dataset-version-message 'qwen38 submission: duck alone on Qwen3.8-27B-FP8, 28-way, 128 min/game' \
  --slurm-gpu B200 --slurm-gpu-count 1 --slurm-time 06:00:00
```
No `--game` (all 25 bundled public games). `--kaggle-make-share-version` matches the
2026-09-18 submission, which used the share notebook; per memory, the share notebook overrides
`bm.games` with all 25 envs, which is what a commit run wants.

Dry-run first (`--kaggle-dry-run`) and check: 3.8 dataset attached, 0 `DUCK_CRITIC_*` keys,
pickled cap 7680 s, target `max_runtime_s` 32400, guard string present in the bundled notebook.

## Read against
2026-09-18 commit run on 3.6: 13/25 games with ≥1 level, 17 levels, TAAF 1.90, ~57–59k generated
tokens/game at cap. Submission ref 56355127 scored 0.67 % public from those bytes.
A 3.8 commit run below ~17 levels is not worth the submission slot.
