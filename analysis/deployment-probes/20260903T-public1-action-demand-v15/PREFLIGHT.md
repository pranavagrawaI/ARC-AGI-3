# V15 candidate — pre-flight (not launched)

## What changed from V14

Three edits, all inside `deployment/kaggle/s0-public1-action-demand-v15-candidate/`; the shared
`deployment/kaggle/runtime/local_solver.py` and its test are untouched (many frozen candidates pin
its old hash, and those pins are already mutually inconsistent).

1. **`local_solver.py`** — new self-contained candidate copy of the shared runtime adapter with
   `ARC_LLM_ACTION_DEMAND_MAX_TOKENS` and `ARC_LLM_ACTION_DEMAND_DISABLE_THINKING` added to
   `WORKER_ENVIRONMENT_KEYS`. This is the V14 root-cause fix: the worker now receives both vars.
   SHA-256 `c4011c52133071df54812b3461d1397801b5ab04e875e3f60cb6ba5c9b60b36c`.
2. **`public25_launcher_probe.py`** — `summarize_solver_sessions()` `expected_metadata`
   `decision_envelope` now includes `action_demand_max_output_tokens` (int) and
   `action_demand_disable_thinking` (bool), so it reconciles against the values the v11 worker
   already records in `worker-metadata.json` instead of raising
   "Solver decision envelope differs". SHA-256
   `68889987a76c84fd83dcf05b73f8425994dbcaffe2576ad01bcc61a2dff9512c`.
3. **`build_notebook.py`** — `LOCAL_SOLVER` points at the candidate copy; the `validate()` pin and
   `PROBE_SHA256` are updated; cell 0 documents the repair. SHA-256
   `55fd14853ba525be0691a22fa5be021cfaa2a1624f121996d4553dd66a9eca29`.

Tests:
- `tests/test_v15_local_solver_env.py` (new) — asserts `worker_environment()` forwards both keys
  and that the candidate copy differs from the shared runtime by exactly those two additions.
- `tests/test_action_demand_v15_candidate.py` — card path updated to the V15 dir; adds assertions
  that `local_solver.py` forwards the keys and the probe expects them.

Unchanged: `exact_request_outer_runtime.py` (`f33212d8…`, warmup already sends 2048/no-thinking),
`kernel-metadata.json` (`915cc9c9…`, same private notebook target), wheelhouse dataset
(`pranavagrawai/tycho-nooa-solver-wheelhouse-cp312-v11`), all env vars in cell 3, all caps.

## Frozen candidate hashes

```
55fd14853ba525be0691a22fa5be021cfaa2a1624f121996d4553dd66a9eca29  build_notebook.py
f33212d83b690df38c167eecaa6ae7aa3b20b4ee7979c502c64e5f358eea00d8  exact_request_outer_runtime.py
68889987a76c84fd83dcf05b73f8425994dbcaffe2576ad01bcc61a2dff9512c  public25_launcher_probe.py
c4011c52133071df54812b3461d1397801b5ab04e875e3f60cb6ba5c9b60b36c  local_solver.py
915cc9c94c56980ed6c9a637d5b2402bdad4c46f968824e56a98f2f1b0d6e197  kernel-metadata.json
f82dbe526d52da7c306c0e5432b4e38b0cd05f16ffcba1d0dd5c999ee3c1f754  s0_public3_full_timing.ipynb
```
(also in `deployment/kaggle/s0-public1-action-demand-v15-candidate/candidate-hashes.preupload.sha256`)

## Verification completed (Linux, solver venv)

- V15 candidate tests: **28 passed**.
- `deployment/kaggle/tests/` full: **37 failed, 129 passed** — byte-identical failure set to the
  pre-change baseline (all pre-existing: missing historical evidence artifacts and stale
  cross-candidate pins). **Zero new failures; +28 passing.**
- Focused Solver tests (`test_action_batch`, `test_llm`, `test_worker`) with
  `TYCHO_SANDBOX_RUNTIME=host`: **76 passed**.
- V14 candidate tests still **26 passed** (V14 dir untouched).
- Notebook rebuild is deterministic and every code cell compiles.
- Ruff on the changed V15 files: clean except the two pre-existing `TRY004` nits inherited verbatim
  from the V14 probe (lines 402, 430 — untouched code).
- Notebook embeds the fixed `local_solver.py` in cell 1; cell 3 sets both env vars.

## Not done

- Not pushed to Kaggle. The V15 card grants no permission; launch needs explicit authorization
  and a GPU-quota check immediately before.
- Nothing committed to git yet.
