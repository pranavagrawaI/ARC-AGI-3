# M2 memory candidate

Latest local candidate, built on the preserved `../m2_basics` notebook and its
four reproduced bug fixes, bounded sandbox transport, and animation history.
**CPU-qualified; not uploaded or GPU/gameplay-qualified.** The raw dfranzen
submission remains the incumbent; the local campaign report records 23.87 hidden
for submission 56747536. That score belongs to the raw copy, not this candidate.

Notebook: `kernel/m2-memory.ipynb`. Source: `src/inference`. Hashes: `manifest.json`.
The model, quantization, speculative decoding, context budget, scheduler, and
public-25 control settings are inherited unchanged from dfranzen. The notebook
includes a verified source overlay installed before harness imports.

## Changes beyond M2 Basics

- Explicit per-game data checkpoints: `save_data`, `load_data`, `list_data`,
  `delete_data`. A saved record includes value, source step, source level, and
  encoded byte count. Data survives Python errors/timeouts, reset, level change,
  and prompt trimming, but not a host restart or a new game. It is never silently
  installed into the next call's global variables.
- JSON-only values, with lossy tuple/set/non-string-key conversions rejected.
  Limits are 64 items, 256 KiB each, 1 MiB total. Replacement validates before
  mutation. Reads are detached copies. The host commits each save immediately;
  a later failure does not roll it back. A lost acknowledgement can coexist with
  a committed save; inspect `list_data`/`load_data` before recomputing.
- Cached frame `.rows`, numeric `.grid`, numeric-color `.counts`, and
  `.cells(color)` in row/column order. These are computed lazily per FrameView,
  so untouched historical boards incur no indexing work. Counts return a copy;
  rows, grids and coordinate sequences are tuples. A saved frame remains a
  snapshot after `action()` refreshes `current_frame`.
- `HistoryEntryView.step`/`.level` aliases and animation `.items()`/`.values()`
  support natural inspection calls observed failing in the control traces.
- API documentation in the model's tool prompt. Existing reasoning instructions
  and the S1 economics intervention are not added.

Data persistence is an explicit API, so it does not fix unchanged code that
expects arbitrary globals to survive. The model must choose what to save and
encode non-JSON structures explicitly. This is bounded storage, not a claim that
we have solved memory selection or eliminated NameErrors. Cached views remove
model-written parsing code; they have not demonstrated faster GPU inference.

## Reproduce offline qualification

```bash
PYTHONDONTWRITEBYTECODE=1 /home/pranav/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12 solvers/m2_memory/qualify.py
```

The command rebuilds locally, runs the inherited and new behavioral regressions,
replays selected real calls, tests installation/idempotence/drift rejection,
checks both donor receipt repairs, verifies the entire parent candidate remains
unchanged, and records artifact hashes. No network, model, or GPU is used.

`evidence/trace-replay.json` preserves source hashes, original snippets,
adaptations, omitted code and fixture boundaries:

- tr87 calls 7–8: reproduces missing `tops`; explicit JSON save/load plus cached
  rows gives identical stdout to uninterrupted execution. Tuple-key encoding is
  an intentional adaptation, not an unchanged-code replay.
- bp35 call 61: the unmodified first statement retrieves the exact archived
  action-52 board. The rest has an unrelated non-advancing loop and is excluded.
- dc22 call 175: the whole original `.items()` inspection runs with a deterministic
  animation fixture. The original animation payload was not recovered.

Tests exercise actual sandbox subprocesses. Agent session/dispatch methods are
extracted unchanged and run with deterministic external adapters. They establish
specific contracts, not full model/engine integration or absence of all bugs.

## Qualification after quota reset

1. Smoke-test this exact artifact on the retained serving stack before spending
   a full comparison budget. Verify startup/source hashes, real tool parsing,
   action/reset receipts, checkpoint use and recovery, animation access,
   cancellation and shutdown. Save raw failures; do not automatically relaunch.
2. If operationally valid, run repeated public-25 comparisons with the raw control
   at identical scheduled wall-time budgets, settings and termination rules.
   Candidate 3 repeats + 1 fresh control is a provisional allocation; ~9 GPU-hours
   is an estimate, not a demonstrated power calculation. Include all outcomes.
3. Keep levels/score at equal resource budgets as primary outcomes. Report time
   to first action, tokens, cache/throughput, tool failures, save/load usage,
   repeated computation, and unfinished games as mechanism diagnostics. Levels
   per million tokens is useful but is not a substitute for fixed-budget score.
4. Preserve the incumbent on operational failure, harm, or an inconclusive result.
   A small positive result needs confirmation; two historical controls cannot
   establish a reliable noise distribution. Existing 100-minute transcript cuts
   approximate event times and should not be presented as exact action cutoffs.

No zero-bug guarantee or improved-score claim is made. The next uncertainty is
whether the real model uses these APIs effectively without throughput loss.
