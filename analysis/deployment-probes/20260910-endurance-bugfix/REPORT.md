# Endurance baseline repairs

The baseline is the operationally qualified public endurance v2, Kaggle version
6: notebook `submission-20260910-v7` and runtime `submission-runtime-20260910-v4`.
`BASELINE.json` records the identities verified before editing. The successor is
notebook v8 with runtime v5. The original runtime, notebook, run card, and raw
endurance evidence remain unchanged.

## Reproduced defects and corrections

| Defect | Reproduction against runtime v4 | Correction in runtime v5 |
| --- | --- | --- |
| Python cell deadline is advisory for synchronous loops | A real Actor executes `while True: pass`. With its cell allowance accelerated to 0.3s, the baseline reaches the generic decision timeout and has no cell-timeout record. | The Actor atomically publishes cell start/end metadata. The separate protocol process monitors it while awaiting actions, records a timeout, kills and reaps the Actor, and returns an execution failure. The normal allowance remains 60s. |
| Shape changes are mismeasured | Appending/removing unchanged-prefix rows is classified as no impact; resized grids can report zero aligned pixel changes. Row-frequency tracking can retain the old shape. | Compare complete grids for equality, reset row-frequency history on resize, and report `diff_pixels=null` when shapes differ. Host evidence and the Actor trajectory now agree with `last_transition()`. |
| Configured model context window is ignored | `ARC_LLM_CONTEXT_WINDOW=262144` still yields `client.context_window is None`. | Populate NOOA's context metadata using its registry field and validate that the window exceeds the output allowance. Keep this metadata out of provider request kwargs. |
| Successful submission does not reset rejection streak | Seven rejected batches, one successful submission, then one rejection on the next turn trigger the eight-rejection abort. | Exclude all rejections from turns already committed. The new turn still aborts after eight consecutive rejections of its own. |

The before-fix failures are retained in `before-unit.txt`, `before-actor.txt`,
and `before-timeout.txt`. The timeout fixture retains its local request and
process evidence in `before-timeout-work/` and `after-timeout-work/`.
`repair.diff` contains the bounded source changes against v4.

## Verification

Final result: **116 passed** (78 focused checks, 34 real-entrypoint checks, and
4 contingent-worker checks). `VERIFIED.json` records runtime contract
`566adcfaf1a6107b7e86caa045aed3548853054e334a88d3f07d4ea49b49399b`.

Run from the repository root:

```bash
deployment/kaggle/.venv/bin/python deployment/kaggle/submission-20260910-v8/verify_repairs.py
```

The verifier checks the unchanged baseline identities, the successor manifest,
unit and Actor-method regressions, actual offline and local HTTP-gateway Agent
entrypoints, request failures, Python-cell failure, budget stops, process
retirement, contingent continuations, and notebook payload identity. Results are
written to `verify-*.txt` and, only on success, `VERIFIED.json`.

The pinned Python environment lacks pytest-asyncio. A small package-local
`conftest.py` runs these plain coroutine tests with `asyncio.run` and their
existing synchronous fixtures. No package was installed and the worker's NOOA
installation was not changed.

Three inherited step-check assertions expected eager successor publication.
They now test the already-qualified admission contract: `observe()` preserves
transitions without publishing a new Actor state, and `decide()` publishes it.
This is a test correction, not a change to the admission policy.

## Limits and disposition

These repairs do not establish improved solving. They are not demonstrated
causes of the three zero-level endurance outcomes. The cell watchdog is an
execution-failure boundary, not a recovery mechanism: partial cell writes can
exist when the Actor is retired. Existing earlier host deadlines still apply.
The context fix restores model-window metadata; it does not introduce a new
summarization or context-eviction policy or prove arbitrary-length safety.

The current renderer has no pipe-delimited column groups, and the current typed
experiment validator has already removed the earlier lexical alignment guard.
Neither old defect was reintroduced as a repair here. The saved CD82 helper is
model-generated evidence; editing that historical file would not repair future
helper generation, so it remains untouched.

Epistemic redesign is pending. No remote model request, new GPU experiment,
upload, or competition submission was performed. Runtime v5's inherited launch
readiness is explicitly disabled. Use notebook v8 for the repaired package;
inherited donor-screen notebooks and cards are not its launch instructions.
