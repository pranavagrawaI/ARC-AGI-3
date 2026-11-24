#!/usr/bin/env bash
# Reproduce the full OLMo-2-0425-1B cognitive-map de-risking suite.
# Set PYTHON=/path/to/python to use a specific virtualenv.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"

"$PY" -m pytest -q tests/test_core.py
"$PY" -u exp0_sanity.py
"$PY" -u forward_all.py
"$PY" -u diag_alignment.py
"$PY" -u exp1_convergence.py

for part in main ctrl01 traj; do
  for hmm in Wing Mess3 Arch; do
    [ "$part" = "ctrl01" ] && [ "$hmm" = "Mess3" ] && continue
    "$PY" -u exp2_probes.py "$part" "$hmm"
  done
done

"$PY" -u exp3_cross_symbol.py transfer
"$PY" -u exp3_cross_symbol.py switch

"$PY" -u exp4_train.py 4A 1e-4
"$PY" -u exp4_train.py 4B 2e-5
"$PY" -u exp4_ckpt_probe.py 4A 0 10 25 50 100
"$PY" -u exp4_ckpt_probe.py 4B 10 25 50 100
"$PY" -u exp4_eval.py base
"$PY" -u exp4_eval.py 4A 10
"$PY" -u exp4_eval.py 4A 100
"$PY" -u exp4_eval.py 4B 100
"$PY" -u diag_top_tokens.py

"$PY" -u merge_results.py
"$PY" -u tables.py exp0 exp1 exp2 exp3 exp4
