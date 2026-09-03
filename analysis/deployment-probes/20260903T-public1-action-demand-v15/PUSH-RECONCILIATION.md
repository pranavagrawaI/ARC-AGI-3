# V15 push reconciliation (immediate post-push)

## Push
- Kernel version **13** of `pranavagrawai/prototype-tycho-nooa-public-3-full-timing` pushed
  2026-09-03 (`kaggle kernels push`).
- Local candidate: `deployment/kaggle/s0-public1-action-demand-v15-candidate`.
- Pre-push local hashes:
  - `s0_public3_full_timing.ipynb` = `f82dbe526d52da7c306c0e5432b4e38b0cd05f16ffcba1d0dd5c999ee3c1f754`
  - `local_solver.py` = `c4011c52133071df54812b3461d1397801b5ab04e875e3f60cb6ba5c9b60b36c`
  - `public25_launcher_probe.py` = `68889987a76c84fd83dcf05b73f8425994dbcaffe2576ad01bcc61a2dff9512c`

## Immediate remote readback (kernel version 13)
- Pulled remote source + metadata immediately after push.
- **All 4 notebook cell sources byte-identical to the frozen candidate.**
  - concat-source SHA-256 (local == remote) = `b166773d67f6d4025e1aa9d0da41d6cd40abfcfce6692da421e74c6851ab91bf`
- Metadata: wheelhouse `pranavagrawai/tycho-nooa-solver-wheelhouse-cp312-v11`; `enable_gpu=true`;
  `machine_shape=NvidiaRtxPro6000`; `enable_internet=false`;
  `competition_sources=["arc-prize-2026-arc-agi-3"]`.
- Status after push: **RUNNING**.

## V15 fix carried on the wire
- Cell 1 embeds the candidate `local_solver.py` (base64) whose `WORKER_ENVIRONMENT_KEYS` now
  contains `ARC_LLM_ACTION_DEMAND_MAX_TOKENS` and `ARC_LLM_ACTION_DEMAND_DISABLE_THINKING`.
- Cell 3 sets `ARC_LLM_ACTION_DEMAND_MAX_TOKENS='2048'` and
  `ARC_LLM_ACTION_DEMAND_DISABLE_THINKING='1'` (unchanged from V14).

## Pre-launch verification (Linux)
- V15 candidate tests: 28 passed.
- `deployment/kaggle/tests/` full: 37 failed / 129 passed — identical failure set to baseline
  (pre-existing evidence gaps), zero new failures.
- Focused Solver tests with `TYCHO_SANDBOX_RUNTIME=host`: 76 passed.
- V14 candidate tests still 26 passed.
- Ruff clean on changed lines.

## Status
- Awaiting terminal state. On terminal: download output + source, confirm the worker
  `decision_envelope` records `action_demand_max_output_tokens=2048` /
  `action_demand_disable_thinking=true`, reconcile the forced-request parameters, batch releases,
  official recording, and lifecycle, then write TERMINAL-RECONCILIATION.
