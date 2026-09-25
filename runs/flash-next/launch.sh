#!/usr/bin/env bash
# Push arms K and KF (see CARD.md). Dataset pranavagrawai/duck-flash-next-kf-src is already uploaded.
set -euo pipefail
cd "$(dirname "$0")"
for arm in k kf; do echo "== $arm"; kaggle kernels push -p "$arm/kernel"; done
date -u
