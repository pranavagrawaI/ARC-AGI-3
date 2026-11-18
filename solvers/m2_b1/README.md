# M2 B1

A bundle of small changes on top of `../m2_evidence`. Each is expected to be worth only 2–5% alone. They are measured together because a single 4% effect is invisible under our noise (sd ≈ 11 levels per public-25 pass).
**CPU-qualified (`qualify.py`, 10 checks); GPU smoke and replicate pair under `experiment/`.**
The model, quantization, serving, context budget, scheduler and the notebook's TRUE_SUBMISSION branch are dfranzen's, unchanged. The only difference from the control notebook is the overlay cell.

## Changes beyond M2 evidence

| Change | Why (offline evidence) | Knob (B1 default) |
|---|---|---|
| **Pin the current level's opening turns through every trim**: request-time, commit-time (token budget and assistant-turn cap) and the overflow fallback. The pin covers up to 4 user-message segments and at most 12,288 tokens, and is restored in chronological order. | In the control run, stuck levels with a sibling solve had lost their own opening from context by the end in 28/35. The action that later solved the level had usually first been tried within the level's first ~15 actions. On real requests the pin forms in 22/23 games (median 9.6K tokens). Causality is not shown (`analysis/hidden-draws-20261010/PROBES-20261010.md`). | `ARC3_PIN_LEVEL_OPENING_TOKENS=12288` (0 disables) |
| **Static per-turn rules stated once in the system prompt** instead of in every opener. The opener keeps a one-line call to act. | A median 13.8% of a late request was these lines repeated in every historical opener. Moving them costs nothing in cache: the system prompt never moves. | `ARC3_STATIC_TURN_RULES=1` (only with `ARC3_MEMORY_SECTIONS=off`) |
| **Sandbox allows `time`**; the **tool-call ceiling is 60 s** (was a hard 30). The overlay sets `LOCAL_ANALYZER_TOOL_TIMEOUT=60`. | `time` was blocked 11 times in one run. The 30 s cap ended searches in 15% of stuck long levels against 6% of solved ones. | env / code |
| **Mechanism counters**: `artifacts/<game>_b1.json` records requests, pinned requests, trims with the pin, fallbacks, maximum pin tokens, and the effective rules and timeout. | Lets a null result be read: did the pin fire? | — |

Every pin path assumes trimming drops only from the front, which holds in this configuration (no control pruning, no image window). If an output is not a suffix of its input, the unpinned result is used and counted as a fallback.
When nothing outside the pin can be dropped, the overflow fallback reverts to ordinary dropping. That is the last resort for a request that still does not fit.

## Verify

```bash
PYTHONDONTWRITEBYTECODE=1 /home/pranav/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12 solvers/m2_b1/qualify.py
```

This rebuilds, runs the inherited m2_evidence checks (unit tests, trace replay, package install, donor receipts, evidence gate), and adds:
`tests/gate_b1.py`: every mechanism on and off through the real ToolAgent, including the cache-stable `[system, pin]` prefix;
`tests/replay_pin.py`: the pin on real control-run requests, checking structure, tool pairing, order and budget. It uses the tycho_nooa venv for the harness dependencies.

No improvement is claimed. See `experiment/smoke-1` and `experiment/pair-1` for the cards and results.
