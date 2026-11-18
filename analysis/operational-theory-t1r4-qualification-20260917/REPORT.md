# T5 — qualified and frozen T1R4; independent T6 required

The exact T3 treatment code sequence now passes its previously failing fifth cell
through the real Actor/CodeAct boundary. T5 made zero provider or Kaggle calls.
T3 remains INVALID_RESULT; neither its artifacts nor T4's independent audit changed.
This is local machinery qualification, not a new construction or efficacy result.

`T5-COMPLETION.json` binds the frozen card, manifest, successful checks, raw fixture
archive, before/after evidence and all predecessor inventories. The source is
`../operational-theory-t1r4-20260917/`; its files and directories are now read-only.
Independent T6 review and new user authorization remain outstanding. No approval
or paid-release record has been authored by this implementation task.

## Cause and correction

Pinned NOOA wrapped each cell in an async function, then excluded all names
beginning with `_` from both local capture and persistence declarations. T3's
fourth cell imported `_json`; the fifth legitimately reused it and failed before
reading the canonical schema. The sixth reimported it. This was an executed
contract defect, separate from the model's incomplete construction trajectory.

`cell_state.py` determines model binding ownership using Python's symbol table and
brackets actual CodeAct execution with a ContextVar. `actor_execute.py.txt` vendors
the exact corrected ActorRuntime method; `runtime-boundary.json` pins its original
method and installed module. Only model-owned names enter capture, irrespective
of underscore prefix or callable/module type. The old `_gvar` loop temporary no
longer collides with a valid model binding. The installed dependency is untouched.
`runtime-boundary.diff` shows every change to the vendored function.

Host call inputs are distinguished from public module convenience imports: ordinary
`import json`, aliases, and model-local rebinding remain valid. Reserved runtime
names, attempts to replace host inputs, direct private attributes, named reflection
builtins and runtime imports receive local rejection before execution. Missing
free underscore bindings receive pre-execution feedback; other runtime NameErrors
report that prior effects remain. There is no hidden execution, model retry,
request refund or extra allowance. Trusted prefill exemption uses allocator-owned
IDs, not a provider-controlled spelling. The tests include provider lookalike IDs.
These Python checks enforce the exposed cell contract; they are not a general
proof of Python sandbox security. OS isolation and unchanged evidence/helper
brokers remain the authority boundary.

Qualification also exposed a producer shutdown race after successful construction:
the worker sent `done`, the broker stopped reading, and a background evidence read
could block process exit. A forced cancellation-time read reproduced this in
`shutdown-before-r4/` (done recorded, empty read marker, TimeoutExpired, reaped).
The worker now cancels and joins registered jobs while the broker is available,
then sends done. `shutdown-pass-after/` records marker `2`, return code 0 and reaping.
Absolute deadlines, termination limits, request/accounting rules and caps are unchanged.

## Decisive evidence

- `fail-before/REPRODUCTION.json`, `python-output.txt`, raw requests/responses and
  worker source: the six preserved code cells reproduced cell 5's `_json` NameError
  before the correction. `BASELINE-SOURCE-PROOF.json` proves the staged worker files
  matched frozen T1R3. Two canonical reads occurred (cells 4 and 6).
- `pass-after/EXACT-REPLAY.json`, `python-output.txt`, raw requests/responses and
  host events: all six code strings and original tool-call IDs replayed offline;
  cell 5 succeeded and three schema reads occurred (cells 4, 5 and 6). This same
  sequence still makes no action proposal; no successful counterfactual is claimed.
- `final-checks-r3/CHECKS.json` and `raw-fixtures.tar.gz`: final stable-source
  qualification, including each command, exit code, log hash and source digest.
- `UNCHANGED-SEMANTICS.json`: unchanged workflow skill, canonical theory/schema,
  prefix, evidence/helper authority, transport/request contract and prior tests.
- `preserved-before.json` and `preserved-after.json`: identical file sets and bytes
  for 5,374 predecessor/run/audit files. All four prior manifests and every donor
  digest still verify. `SOURCE-DELTA.json` and `candidate.diff` describe only the
  fresh candidate's changes; generated outputs are outside its manifest.

## Final checks

| Suite | Passed |
|---|---:|
| Package: 56 prior + 27 new regression/adversarial cases | 83 |
| Prior transport | 56 |
| Prior Actor | 18 |
| Current runnable runtime | 160 |
| Total: all 290 prior + 27 new | 317 |

Ruff, format, transcript-free canonical replay, request parity, helper/evidence
integrity, deadlines/reaping, receipt accounting, source stability and predecessor
inventories passed. The two pre-existing collection exclusions remain
`test_assessment.py` and `test_packaging.py` (missing `prepare_candidate`), exactly
as in the prior 290-test baseline. No remote/model qualification is claimed.

Failed work was retained, not overwritten: exploratory focused runs found a trusted
prefill classification mistake and a missing fixture directory; `final-checks/`
found two overly broad public-import rejections; `final-checks-r2/` found the
shutdown race. Earlier shutdown probe attempts used an unregistered channel or
jail-denied marker path; `shutdown-before-r4/` is the valid causal reproduction.
The final rejection tests include a positive writable-path control, so a missing
attack marker is not merely a filesystem-denial proxy. Only `final-checks-r3/`
qualifies the final frozen source.

The replacement card keeps one ordered pair, the same recorded CD82 prefix and
contrast, Qwen3.8-27B, 6 calls per arm/12 total, 45-second request and 240-second arm/
480-second pair limits, matched token caps, a $2 admission ceiling and no retry or
fallback. It explicitly replaces invalid T3 under a new identity. Paid output must
be external and fresh; a candidate-hash claim prevents another launch under a
second timestamp. This card grants no launch authority.
