# Operational Theory — T1 implementation and local qualification

No OpenRouter/provider calls, Kaggle calls, external writes, credential discovery,
commits, or changes to frozen/shared source are part of T1. All new files are here.
The control imports the current competition Actor without modifying its bytes.
The implementation is ready for independent review; this document grants no launch
authority and makes no scientific efficacy claim.

The mechanism is an Actor-callable workflow backed by one host-owned, versioned
Operational Theory. The same current Actor class is instantiated in both arms.
Treatment attaches trusted host methods to that instance before any model code runs;
NOOA's existing model-side dynamic-method guard remains intact. Its original class
API documentation and system prefix remain unchanged. The appended context explicitly
advertises `self.workflow_skill()`, `operational_cycle`, `operational_hash`, and
`register_operational_artifact`. `submit_actions` delegates to the incumbent checks
only after the proposed action matches a prepared canonical decision. The host broker
checks that preparation again before advancing the recorded evidence frontier.

The store lives outside the Actor's mount namespace. The isolated Actor has read-only
runtime/source, its own writable scratch, sterile environment, separate PID/network
namespaces and one inherited JSON pipe to the host. It has no credentials, other arm,
future evidence, canonical files or repository access. The host owns model transport,
usage reservations, canonical publication, result acceptance and cleanup. All model
requests, including incumbent critic/investigation requests, share one arm purse.

## Requirements and files

| Requirement | Implementation | Reproduction |
|---|---|---|
| Callable reusable workflow | `workflow-skill.md`, `actor.py`, `worker.py` | isolated actual Actor test prints skill, attempts skip, receives repair, then completes |
| One canonical asset | `operational-theory.schema.json`, `theory.py` | lineage, schema/ref/hash rejection tests; five accepted fixture versions |
| Mechanical-only validation | `theory.validate`, `Store.call` | deliberately fictitious fixture claim is accepted structurally; no truth oracle |
| Preserve control/full surface | `bootstrap.py`, current frozen Actor import, `worker.py` | control has no canonical store or workflow injection; native methods remain in actual requests |
| Exact shared initial request | `client.py`, `transport.py` | stable framework prefill IDs; remove only treatment context and compare entire first-request digest; wire schemas/settings also equal |
| Enforce cycle and bounded repair | `theory.py`, host broker in `supervisor.py` | skipped preparation/decision/revision rejected; real malformed action repaired; fourth treatment format error ends incomplete |
| Persist across turns/replay | `canonical/lineage.jsonl`, `decisions.jsonl`, `completed.jsonl`, `EVIDENCE.json`, `replay.py` | actual second Actor turn revises same lineage; independent store reload and replay without transcript |
| Immutable evidence/procedure bindings | `PREFIX-BINDING.json`, `evidence/`, archive in `canonical/artifacts/` | unknown refs/hash drift rejected; changed archived executable rejected on replay |
| Real prefix, staged observation | `experiment.py` | R10a CD82 step 1 RIGHT; step 2 DOWN revealed after first proposal, never attributed to proposed action |
| Isolated A/B driver, no launch | `run_pair.py`, `supervisor.py`, `worker.py` | real Actor in bwrap, local HTTP fixtures, exact-identity authorization gates, one-shot live directory |
| Caps/receipts/no retry | `transport.py`, `run_pair.reconcile` | missing usage stops pair with nonzero unresolved exposure; sixth-call cap; matched-pair receipt reconciliation; slow 200/429 absolute cutoff |
| Frozen experiment identity | `EXPERIMENT-CARD.md`, `identity.py`, `MANIFEST.json` | manifest verification before each future arm and before/after final verification |
| Local qualification | `test_operational.py`, `fixtures.py`, `verify.py` | reproducible script below, raw fixture requests/responses/host events retained |

`protocol.py` and the message projection in `local_wire.py` are copied from the prior
fixed-interface OpenRouter harness. `transport.py` retains its reservation, routing,
usage and no-retry contract, with scoped changes: control/treatment names, both
reasoning off, Actor request contract, matched first-request digest, ordinary budget
exhaustion, and a nested absolute request timer. The current Actor's full surface is
not replaced with the constructor-only tool schema. The prior constructor loop and
synthetic source are not placed in either paid arm's prompt.

No shared source changes were necessary. No prior approval, live directory, frozen
identity, ledger or consumption marker is reused. The memory-derived R10 guidance
was verified against current artifacts; it did not authorize a launch.

## Evidence and metadata

`PREFIX-BINDING.json` contains the original exact path/hash and comparison wire hash.
The selected first line is byte-identical to the prior OpenRouter CD82 prefix.
`evidence/continuation.jsonl` is the second recorded transition from that same source.
Both are primary real evidence; fixtures are only machinery tests. The Actor sees
only its admitted prefix. Proposal actions do not execute, so the recorded DOWN
transition is always labeled historical, including if the Actor happens to propose DOWN.

`metadata/selected.json`, `fixed-endpoints.raw`, `fixed-endpoint-refresh.json`, and
`endpoints.receipt.json` are copied preserved public discovery, not current T1 network
observations. Exact qwen/qwen3.8-27b and reka/fp8 are pinned. T3 must supply refreshed
metadata; the gate rejects identity/price/capability drift rather than substituting.
The cost amounts in fixture receipts are synthetic test values, not actual spend.

## Baselines and verification

An initial combined invocation of the historical constructor and join suites had
2 failures / 72 passes because their top-level `constructor`/`host` imports collided.
Isolated reruns pass: constructor 56 tests; integration 18 tests. Do not combine these
suites in one interpreter. The current competition runtime's full collection has two
pre-existing errors (`test_assessment.py`, `test_packaging.py`: missing
`prepare_candidate`). Its remaining complete runnable suite passes 160 tests. T1 does
not repair these historical packaging tests or import a substitute package to hide them.
Original logs are `baseline*.txt`; full-suite commands/results are in `qualification/final/CHECKS.json`.
A final projection regression (provider ID resembling a framework prefill ID) was
reproduced in `opaque-id-before.txt`, then repaired by tracking provider-owned IDs.
The resulting entire new suite and static checks are recorded in
`qualification/identity-final/CHECKS.json`; unchanged adjacent suites retain their
full-suite qualification. This last delta changes only client ID ownership and its
regression, not the Actor, store, transport budgets, or prior source bytes.

From repository root, with a fresh output directory each time:

```bash
PYTHONDONTWRITEBYTECODE=1 LITELLM_LOCAL_MODEL_COST_MAP=True deployment/kaggle/.venv/bin/python -B analysis/operational-theory-t1-20260917/verify.py --output analysis/operational-theory-t1-20260917/qualification/reviewer
```

This script runs the entire new suite and three adjacent suites in separate
`unshare --user --map-auto --map-root-user` interpreters, then Ruff lint and format
checks. It clears credential variables and uses loopback HTTP plus the real current
Actor. It does not call the provider or credential loader. Every check retains its
exact command, exit status and log digest. The known full-collection errors above
remain explicitly disclosed; the two offending files are excluded from the runnable
runtime invocation. Reviewers may reproduce the unfiltered baseline separately.

The fixture pair is retained at
`qualification/final/new-tmp/test_pair_budget_reconciliatio0/`:
control/treatment `*-raw/` contain provider-shaped raw receipts, their arm directories
contain worker IPC and host events, and `ACCOUNTING.json` reconciles six fixture
requests. Treatment has a five-version lineage, two replayable decisions and one
fixture-authored counterevidence revision. This proves interface behavior, not CD82
model truth or Qwen behavior.

Replay independently, without reading request transcripts:

```bash
PYTHONDONTWRITEBYTECODE=1 deployment/kaggle/.venv/bin/python -B analysis/operational-theory-t1-20260917/replay.py analysis/operational-theory-t1-20260917/qualification/final/new-tmp/test_pair_budget_reconciliatio0/treatment
```

## T2 review and first unresolved boundary

Verify the manifest; rerun verification to a fresh directory. Inspect actual raw
first requests for full shared-prefix/tool/evidence parity and ensure only the
appended treatment block differs. Inspect skip rejection and later successful repair,
canonical lineage and independent replay, unknown usage and deadline receipts, and
sandbox denial tests. Confirm the common workflow rubric in the card is applied to
native control assets rather than equating its lack of the treatment schema with
failure. Path-specific completion booleans alone are not a comparative statistic.

The first unresolved boundary is independent review of the exact frozen implementation
and experiment contract. After that, live endpoint compatibility and an actual model's
ability to construct/revise within the caps remain untested. T1 has no provider result.
No package reviewer/release approval is self-authored here.

## Exact prepared T3 command — do not run in T1

T2's `T2-REVIEW.json` must contain role `independent-reviewer`, decision `PASS`, task
`operational-theory-T1`, and the SHA256 of MANIFEST.json. T3's owner supplies
`T3-AUTHORIZATION.json` with role `user`, decision `AUTHORIZE_T3`, the same task and
manifest hash, plus `metadata_sha256` binding `T3-ENDPOINTS.json` (fresh raw public
endpoint response). These filenames are reserved external inputs, not generated
by preparation. T3 requires OPENROUTER_API_KEY in the host environment; never place
it in a command line, artifact or worker environment.

```bash
PYTHONDONTWRITEBYTECODE=1 LITELLM_LOCAL_MODEL_COST_MAP=True unshare --user --map-auto --map-root-user deployment/kaggle/.venv/bin/python -B analysis/operational-theory-t1-20260917/run_pair.py --execute --review analysis/operational-theory-t1-20260917/T2-REVIEW.json --authorization analysis/operational-theory-t1-20260917/T3-AUTHORIZATION.json --metadata analysis/operational-theory-t1-20260917/T3-ENDPOINTS.json
```

This is one control-then-treatment pair. Any existing `live/` blocks another attempt.
Missing usage/cost, provider/HTTP failure, integrity/isolation failure or deadline
stops the pair. There is no ambiguous-request retry or fallback. No Kaggle or external
submission path exists in the driver. Credentials remain entirely outside the Actor.
