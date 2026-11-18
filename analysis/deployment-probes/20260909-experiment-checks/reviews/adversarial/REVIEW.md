# Independent preservation, CLI and regression-test review

Candidate: current untracked `tools/experiment_checks.py`,
`tools/tests/test_experiment_checks.py`, and `tools/EXPERIMENT-CHECKS.md`.
Review only; implementation and existing tests were not edited.

## Findings

### P2: malformed nested evidence aborts the audit and loses its report

Locations: `tools/experiment_checks.py:168` and `:215` (also `:161` and `:235`).

`Evidence.obj`/`rows` validate only the outer object. A valid JSON object with
`runtime: null` reaches `.get` on null. A batching action with `actions: null`
reaches `len(None)`. Both CLI invocations exit 1 with a Python traceback and no
JSON, although exit 1 is documented to mean a found contradiction/invalidator.
The first reproduction includes a positively identified OOM, which never reaches
the user as a report because unrelated malformed provenance aborts reporting.
Comparable malformed epoch/usage values reach unchecked comparisons/member calls.

Smallest fix: validate the nested fields consumed by each command, add malformed
values to unresolved evidence, and continue constructing the report so existing
invalidators retain precedence. Exercise these paths through the CLI, checking
both JSON and exit status: invalid with OOM, unresolved without an invalidator.
Do not broadly convert all programming exceptions into evidence failures.

### P2: preservation silently omits unsupported filesystem entries

Location: `tools/experiment_checks.py:279`–`:294`.

The copy loop only handles directories and regular files. A FIFO named
`server.log` passes preflight, is silently skipped, and yields a completed
manifest and `status: supported` with zero files. Sockets and other special
entries have the same unchecked branch. This contradicts the unqualified
successful evidence-snapshot claim; the destination contains neither the
entry nor an omission record. No attempt should read the FIFO (that can block).

Smallest fix: reject every non-directory/non-regular-file entry during preflight,
as already done for symlinks, with an explicit path/type error. Add a FIFO control
that proves rejection occurs before the destination is created.

## Verification and coverage

`uv run --no-project --with pytest --with numpy --with jsonschema python -m pytest tools/tests/test_experiment_checks.py -q`
passed all 28 tests locally. The system Python alone has no pytest installed;
the documented uv command succeeds.

`python3 analysis/deployment-probes/20260909-experiment-checks/reviews/adversarial/reproduce.py`
reproduced both findings entirely in temporary directories:

```
OOM plus malformed provenance: 1 json_report= False AttributeError: 'NoneType' object has no attribute 'get'
FIFO preservation: status=supported, files=0
FIFO manifest: files={}
Malformed action row: 1 json_report= False TypeError: object of type 'NoneType' has no len()
```

The existing tests genuinely check the targeted OOM signature, missing DONE,
cleanup failure, altered request hash, temporal uncertainty, and real probe/gate
functions. The negative controls assert the specific invalidator, so unrelated
missing evidence does not alone make them pass. However, malformed-evidence
coverage currently stops at JSON syntax/outer shape. Preservation coverage
contains the ordinary-copy and existing-destination cases, but no unsupported
entry or mid-copy-failure control. CLI coverage contains only a missing-card
unresolved case, with no invalid, supported, malformed-input or output-collision
control. These coverage gaps explain how the reproduced defects survived 28
passing tests; the count does not establish those untested behaviors.

No paid runs, helper execution, external messages, or shared implementation edits.
