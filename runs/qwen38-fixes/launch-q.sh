#!/usr/bin/env bash
# qwen38-fixes arm Q (fixes + questioner); MTP=1 adds MTP. Usage: launch-fm.sh [--kaggle-dry-run]
set -euo pipefail
cd "$(dirname "$0")/../../ARC3-Inference"
source ~/.cache/duck-step1-env.sh
export KAGGLE_API_TOKEN="$(python3 -c "import json;print(json.load(open('/home/pranav/.kaggle/credentials.json'))['access_token'])")"
export KAGGLE_KEY="$KAGGLE_API_TOKEN"
unset DUCK_CRITIC_TRIGGER DUCK_CRITIC_PERIOD DUCK_CHAT_TEMPLATE_KWARGS DUCK_VLLM_SPECULATIVE_CONFIG
export LOCAL_ANALYZER_YIELD_SECONDS=300
export DUCK_PROMPT_GAMEOVER_RETRY=1
export DUCK_PROMPT_LEVELUP_REVIEW=1
if [ "${MTP:-1}" = 1 ]; then export DUCK_VLLM_SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":3}'; fi
export DUCK_CRITIC_ENABLED=1 DUCK_CRITIC_MODE=questioner DUCK_CRITIC_PERIOD_TURNS=2 DUCK_CRITIC_QUESTIONER_EFFORT=none
uv run --no-sync --offline inference-taaf-run \
  --environments-dir /kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files \
  --run-name duck-qwen38-fixes-q --experiment-dir ../runs/qwen38-fixes/q${1:+-dryrun} \
  --max-runtime-minutes 120 --max-experiment-runtime-minutes 540 \
  --n-passes 1 --concurrent-jobs 28 --agent duck-harness --model local --analyzer-timeout 900 \
  --deployment-target kaggle --no-deployment-wait \
  --kaggle-duck-public-harness --kaggle-make-share-version \
  --kaggle-username pranavagrawai \
  --kaggle-kernel-slug taaf-duck-qwen38-fixes-q --kaggle-kernel-title taaf-duck-qwen38-fixes-q \
  --kaggle-dataset-ref pranavagrawai/taaf-kaggle-source-duck-qwen38-fixes-q \
  --kaggle-accelerator NvidiaRtxPro6000 \
  --kaggle-dataset-version-message 'qwen38-fixes Q: fixes + questioner (period 2, thinking off)' "$@"
