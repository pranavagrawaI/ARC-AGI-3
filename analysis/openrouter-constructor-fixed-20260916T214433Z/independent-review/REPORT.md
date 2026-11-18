# Independent exact-byte review — PASS

Task: OR-CONSTRUCTOR-SCHEMA-FIX-REPEAT. Package review only; no credentials, remote API calls, paid model calls, implementation edits or production execution. All review outputs are under independent-review/.

## Binding and decision

**PASS** for inventory SHA-256 `bba4d707c6c5c6edae7e818ca68ed8b0c969066832509b4acd2500f29b16276a`, endpoint `reka/fp8`. Candidate bytes must remain unchanged. This decision is the independent-review gate only; fresh Hermes artifacts and explicit PM release remain required. Carried-forward sole-key permission is not permission for this reviewer to access credentials. Existing R10 HOLD remains intact.

Key bindings:

- Card: `f55191c64d25d54b2877228a71bffa720d30d5651e587ac225e32c4c4cf7f9f5`
- engine/constructor.py: `86c56f21027bf6840c071db4700b8c04af87555b3804e114ece213493d6a3fbd`
- fixed-interface.diff: `f03ecaa6244433eff7f744d24428b4f7ce9fc33d8e222f9be590a9afb8314e00`
- CANDIDATE-DELTA.diff: `d1f6aec5638beb75bdccbbe909041d8bffdc5870285491b1dbd350258014b443`

BINDINGS.json contains the full candidate inventory and independent preservation results. All inventoried bytes match. All **1195** files in PRIOR-EXPERIMENT-HASHES.json match their prior-experiment paths; all **70** R10 manifest sources match, with manifest hash `9310d597434699539041d342346e630f18ffa295d6f16a4a10205a2229134be9`. Checks were repeated at completion. No new live/ directory exists.

## Scope and schema review

The substantive runtime change is confined to copied engine/constructor.py: full nested schema, field/type/key diagnostics, and insertion of actual-frontier bounds before dispatch. run_pair.py changes only the fresh task identity in its two approval predicates; make_inventory.py likewise changes task identity. Existing transport.py, protocol.py, auth.py, test_pair.py, sealed evaluators/runners, original messages and prefix remain byte-identical to the prior candidate. No change to source conversion, source syntax handling, reasoning policy, request caps, provider selection, money reservation, timeout, promotion, or once-only controls was found.

`work_item_schema()` and `_validate_work_item()` now agree on the nine required keys, nested extra-key prohibition, operation/reasoning enums, nonblank text, nonempty reference arrays bounded to the actual visible frontier, and nonempty dependency strings. The schema description identifies the required nesting. Runtime injection uses case.frontier=1 here; it does not merely leave an unbounded template in the outgoing request. Both arms expose identical schema.

Feedback now names arguments.work_item and its actual type, identifies missing/extra keys, distinguishes absent nesting from double wrapping, and identifies indexed field/type/range failures. Invalid data is not coerced. The previous empty-string/flattened error is no longer reduced to a generic field-set message. The model must submit a new valid call under unchanged budgets. No historical error, solution, or game-semantic diagnosis enters the live initial messages.

### Explicit parity limit — not a new blocker

Do not describe this as universal JSON-Schema/Python-parser equivalence. JSON Schema treats integral `1.0` as an integer; inherited Python validation rejects the parsed float. This is disclosed in SCHEMA-TRACE.md and independently reproduced. The host remains authoritative and supplies an indexed integer/type error; it does not coerce the value. Preserving host acceptance was part of this task. Standard schema integer semantics cannot express JSON number spelling, so this disclosed lexical-number limit is not grounds for an unrelated validator-policy change.

The unusual inherited acceptance of boolean true as step 1 is also preserved explicitly by the schema's const:true alternative and actual sealed-boundary test. False and out-of-range integers remain rejected. Review applies to the bound frontier-1 candidate; it does not qualify a frontier-zero scenario.

An independent 193-case differential check found identical old/new validator acceptance across representative JSON types and field mutations. Its only schema-versus-host disagreements were `[1.0]` in evidence_refs and completion_checks, as disclosed. This is bounded evidence, not an exhaustive proof over all JSON or all provider regex/schema implementations.

## Real-boundary validation

Independent runs:

- Original plus new candidate suite: **56 passed in 35.44s**, tests.log.
- Reviewer additions: **3 passed in 1.44s**, reviewer-tests.log.
- check_candidate.py: verified exact inventory, 15 unchanged donor bindings and step-1 RIGHT visible packet; no consumed state.

The 56-test suite exercises actual loopback HTTP → constructor → sealed inspection → malformed work-item feedback in the next outgoing request → valid work item → sealed initialize/predict/assimilate and ABI-only freeze. It covers wrong types, absent/empty/flattened/double nesting, missing/extra keys, enums, references and dependencies. Retained raw responses verify no coercion. Existing transport/reasoning/accounting/containment/lifecycle tests pass unchanged.

Reviewer test_review.py additionally loads the **exact malformed arguments from prior off call 003 and on call 005** as local fixtures. Both receive the new object/type/nesting diagnostic, retain original malformed bytes, and recover through a subsequent valid object to real sealed ABI evaluation. This fixture-only historical input is not added to live prompts. The third reviewer test compares inherited validator acceptance and independently checks schema behavior with jsonschema.

Commands, from repository root (use fresh basetemps if repeating):

```bash
PYTHONDONTWRITEBYTECODE=1 unshare --user --map-auto --map-root-user deployment/kaggle/.venv/bin/python -B -m pytest -q -p no:cacheprovider analysis/openrouter-constructor-fixed-20260916T214433Z/test_pair.py analysis/openrouter-constructor-fixed-20260916T214433Z/test_work_item.py --basetemp=analysis/openrouter-constructor-fixed-20260916T214433Z/independent-review/basetemp
PYTHONDONTWRITEBYTECODE=1 unshare --user --map-auto --map-root-user deployment/kaggle/.venv/bin/python -B -m pytest -q -p no:cacheprovider analysis/openrouter-constructor-fixed-20260916T214433Z/independent-review/test_review.py --basetemp=analysis/openrouter-constructor-fixed-20260916T214433Z/independent-review/reviewer-basetemp
PYTHONDONTWRITEBYTECODE=1 deployment/kaggle/.venv/bin/python -B analysis/openrouter-constructor-fixed-20260916T214433Z/check_candidate.py
```

## Interpretation and handoff

This qualifies the local interface repair and preserved controls. It does not demonstrate hosted compliance with the larger schema, rule out provider-side tool serialization, or predict a successful live repair. A provider rejection remains a consumed, fail-closed outcome under the unchanged transport. No preflight, retry, endpoint substitution, or relaxed schema validation is authorized by PASS.

The card correctly treats schema plus feedback as a joint intervention in a second descriptive pair. It cannot isolate their individual contribution or make a controlled checkpoint-only comparison against the earlier pair. Sealed ABI success remains unscored, with no dynamics/gameplay efficacy or R10b claim.

Next step belongs to Hermes/PM: release the exact reviewed bytes under the fresh task identity and carried credential scope. No implementation or paid execution was performed by this reviewer.
