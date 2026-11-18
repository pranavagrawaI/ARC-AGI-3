# M2 basics

A local candidate built on the dfranzen Milestone 2 solver. It fixes four
reproduced correctness defects and combines specific, compatible infrastructure
improvements from the three donors. **CPU-qualified; not yet GPU/gameplay-qualified.**

The preserved control and audit artifacts remain under
`analysis/m2-winners-20261001/`. Development source lives in `src/inference/`.
The generated private notebook is `kernel/m2-basics.ipynb`; it has not been
uploaded, submitted, or run on Kaggle.

## What is combined

| Source | Retained contribution | Reason |
|---|---|---|
| dfranzen | Model/serving combination, context policy, cache-aware admission, action guards, frame differences, reusable helpers | Keep the working solver as the integration baseline. |
| lordhansolo | Deadline-aware pipe I/O, child cleanup, and preservation of previously committed code after interrupted calls | A failed calculation must not destroy useful code or strand a subprocess. Transport helpers are adapted from his runner. |
| sirikilohit M111 | Bounded per-action animation history | Intermediate evidence remains accessible after subsequent actions and automatic reset. Implemented directly in state serialization instead of notebook monkey patches. |

This is a selection of compatible mechanisms, not a claim that every donor
feature is beneficial. Lordhansolo's entire module/test workspace and
sirikilohit's resource allocator are not transplanted. There is one serving
stack, one scheduler, and one helper-retention contract. The only prompt addition
documents the new animation API.

## Correctness contracts

1. **The host owns action receipts.** Mutating an `action()` return value cannot
   alter whether the harness records execution, progress, or terminal state.
2. **An interrupted call preserves previously retained code.** Missing retention
   metadata is not an empty snapshot. Game actions already executed remain
   executed; only unfinished code-cache updates are discarded. Explicit helper
   deletion still commits. Ordinary Python exceptions preserve completed eligible
   definitions, matching the donor's partial-execution behavior.
3. **Retained dependencies must mean what they meant when accepted.** Helpers
   using rebound builtin names are rejected, along with dependent wrappers.
   Normal builtin use and explicit observation arguments continue to work.
4. **Prompts distinguish a tool call from a game action.** An inspection call
   clears an earlier `no_tool_call` reason while retaining the historical action
   outcome as context.
5. **Pipe I/O has a deadline, and children are reaped on every exit.** This covers
   initial writes, replies, malformed protocol, EOF, timeouts, and broken pipes.
   The synchronous game callback still owns its own deadline: killing a thread
   in the middle of an engine mutation would create a different correctness bug.
6. **Missing evidence is labelled.** `transition.frames` holds intermediate
   frames for the last eight animated actions, up to 32 per action.
   `animation_info` exposes sampling, original intermediate count, retained
   indices, and eviction. Empty evicted data does not mean no animation occurred.
   The pre/post boards and existing `last_animation_frames` API remain available.

## Verification

Use the same Python minor/patch release as the archived notebooks:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/pranav/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12 -m unittest discover -s solvers/m2_basics/tests -v
python3 solvers/m2_basics/build.py
PYTHONDONTWRITEBYTECODE=1 /home/pranav/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12 solvers/m2_basics/tests/check_package.py
PYTHONDONTWRITEBYTECODE=1 /home/pranav/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12 solvers/m2_basics/tests/check_donor_receipts.py lordhansolo
PYTHONDONTWRITEBYTECODE=1 /home/pranav/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12 solvers/m2_basics/tests/check_donor_receipts.py sirikilohit
```

The four original regressions can be checked against the archived source:

```bash
PYTHONDONTWRITEBYTECODE=1 M2_SOURCE_ROOT="$PWD/analysis/m2-winners-20261001/audit/dfranzen-repo/ARC3-Inference" PYTHONPATH="$PWD/solvers/m2_basics/tests" /home/pranav/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12 -m unittest test_basics.AuditRegression -v
```

That command is expected to fail four tests. See `evidence/baseline-red.log` and
`evidence/candidate-tests.log`. Tests run actual Python sandbox subprocesses.
HTTP/vision package initialization is bypassed. Host methods are extracted
unchanged from their selected source with AST; deterministic adapters replace
observations or the game engine. They do not exercise the neural model or GPU.

The animation integration test executes the actual `_execute_action` and
`_execute_auto_reset` methods, writes/reads runtime state, and checks that the
fatal action's evidence and automatic reset remain distinct. It uses a small
fake engine, not an official game replay.

## Packaging and donor repairs

`build.py` is local-only. It writes a diff, hashes, and a private notebook based on
the completed all-25 dfranzen control. The notebook applies a checked source
overlay immediately after the original harness patch, before imports and
unpickling. Unexpected source hashes or already-imported harness modules cause
an explicit failure. Installation is idempotent.

The model inputs, runtime launcher, serving flags, context budget, scheduling,
and control's game/runtime settings are preserved. The generated notebook is a
combined candidate, so any future score change cannot identify which individual
graft caused it without ablations.

`donor_fixes/` contains tested receipt-only patches and the corresponding fixed
runner files for lordhansolo and sirikilohit. Lord's patch targets his source
bundle runner. Siri's patch targets the active M90 embedded runner; replace the
decoded `_W90_SANDBOX_SRC` value in his notebook when using it, not the earlier
superseded runner. The original downloaded notebooks have not been overwritten.

## Next qualification

The next discriminating step is one bounded GPU smoke test of this exact
notebook: startup, source identity, real tool parsing, action/reset receipts,
helper recovery, animation access, and clean shutdown. If it passes, compare
the candidate with the preserved dfranzen control at equal public-25 budgets.
Record serving/IPC overhead as well as score; retained animation payloads add
bounded serialization work. Preserve raw transcripts, action receipts, source
hashes, and incomplete outcomes. Do not infer superiority from CPU tests.

GPU numerics, speculative decoding, KV/Mamba recovery, full cancellation, and
long-run concurrent behavior remain unqualified. No zero-bug or improved-score
claim is made. See `NOTICE.md` for sources and attribution.
