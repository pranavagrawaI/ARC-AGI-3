# R4 constructor outcome replay (read-only diagnostic)

## Outcome

Both exact captured submissions fail the frozen generic sealed evaluator. The first failure is remotely observed; the second verdict below is **local reconstruction, not a retrieved remote second verdict**. This does not demonstrate a containment malfunction or an otherwise ABI-valid candidate rejected erroneously. It does expose concrete omissions/mismatches in the constructor-facing contract. Changing that contract or expanding the runtime would be a changed intervention, not an authorized trivial retry.

| Submission | Evidence and actual result |
|---|---|
| Call 7 | Remote call-8 tool feedback reports `initialize` RPC failure: candidate module import raises `ModuleNotFoundError: No module named 'copy'`. Local sealed replay reproduces this, including identical initialize argument digest. Candidate function body never starts; prediction and assimilation are not reached. |
| Call 8 | Exact unmodified replacement source initializes and predicts locally. Frozen evaluator rejects before assimilation with `ValueError: commitments and declared_dependencies have bad types`. `commitments` is a list, but evaluator requires a dict; dependencies is correctly a list. This is an actual executed local verdict, not an inference promoted to remote evidence. |

Both captured work items pass `_validate_work_item(..., frontier=1)` locally. Source SHA256:
- Call 7: `f300d2559c5212a20288475e55cbf73455f178652acb6a26c492f51485501142`
- Call 8: `544dca83d27f261c480ef932f6b030bf7580cd816849e27f2601e06cb2ca8414`

## Exact boundaries and contract

The active path is `host.bound_constructor` replacing the donor evaluator with `evaluate_visible.evaluate_visible_abi`, not the LS20 semantic observer in `integration_boundary.evaluate_candidate`. The generic evaluator calls real `CandidateRPC` / `SealedBoundary` / `qualified_runner.py`. No source, runtime allowlist, sandbox, or prompt was changed. Local execution used Python 3.12.14 under mapped-root user namespace, followed by the runner's actual chroot, uid/gid 65534, capability drop, resource limits, and native seccomp allowlist. Local execution is not claimed to reproduce the remote OS/image byte-for-byte.

`containment_base.py:21-24,76-100` deliberately copies a restricted runtime. `copy.py` is absent (`copyreg.py` is a different module). A separate synthetic probe through the same sealed entrypoint actually imported `json`, `collections`, `math`, and `os`; `copy` and `random` raised ModuleNotFoundError. This is not a claim that all standard-library imports work, nor that importable `os` escapes containment. Import availability can also come from modules preloaded by the trusted runner.

The **captured** system prompt says `initialize -> JSON state`, `predict -> nonempty list of branches`, and that each branch has `state`, `commitments`, `declared_dependencies`; it never specifies commitments as an object, the initialize/assimilate object restriction, a singleton-only branch restriction, or an import allowlist. The submission tool only requests complete Python source, without a nested output schema. The prompt explicitly says not to assume game-specific `block`/`bar_left` targets. Call 7 starts in commit phase with zero inspections remaining, so the advertised `/evidence/case.json` inspection is not available during these captured requests. Call 8 feedback identifies only the import error, not the latent commitments error.

Thus, list commitments are a concrete violation of the **implemented** ABI (`evaluate_visible.py:39-41`) but not of an explicitly stated commitments type in the captured prompt. The first import failure is permitted fail-closed rejection under the frozen restricted runtime, not proof a promised general Python stdlib was broken. The undisclosed import surface and output-type restrictions are real contract-specification gaps; do not describe these as an unambiguous model refusal to follow a fully specified ABI.

Separately, the inherited prompt promises informative joint commitments, ancestry enforcement, and no credit for empty/all-unknown predictions. The active generic evaluator instead enforces one branch, dictionary commitments and list dependencies, supplies `{"status":"unscored"}` to assimilation, has no required targets, and does not score commitment values. This is a concrete prompt/checker mismatch, but it is **not the cause of these two observed/reproduced failures** and is not evidence of prospective model support. Its explicit result claim is `ABI_VISIBLE_REPLAY_ONLY_NOT_SUPPORTED_MODEL`.

## Prediction content and scope

Call 8's actual sealed prediction returns `[{"action":"RIGHT","cell":[61,13],"expected":3}]` as commitments. Both sources contain the same list shape; call 7's later error remains latent because its import fails first. An additional sealed, off-evaluator-path call confirms call 8 assimilation returns an object; it is labeled `assimilate_off_path_only`, not a passed validation or remote execution.

The public initial/current grids and sole RIGHT step expose 52 changed cells, saved in `observed-grid-deltas.json`. Consequently the submission's claim that all other observed cells remain unchanged is false even on the visible evidence (including another changed cell at [62,13]). This is a diagnostic comparison, **not** a semantic verdict emitted by the generic evaluator. No future or hidden observations were used.

## Reproduction and artifacts

From repository root:

```sh
unshare --user --map-auto --map-root-user deployment/kaggle/.venv/bin/python -B verification-r4/outcome-replay/replay.py > verification-r4/outcome-replay/execution.stdout.json
```

Executed successfully with exit code 0; assertions verify the remote first-source binding, copy failure, second exact error, and second RPC sequence `initialize,predict`. The script extracts both exact source strings and work items from `raw-http/07.body` and `08.body`; it does not execute candidate code in the parent interpreter.

`reconstructed-one-visible-step.json` joins real call-7 `initial_observation`, `current_observation`, and the sole step-1 RIGHT summary using `ActorPrefixCase`. This is a diagnostic case reconstruction, **not** a recovered full `/evidence/case.json` or transitions artifact. `remote-first-feedback.json` preserves the actual remote first verdict. `results.json`, `rpc-diagnostic-call{7,8}.json`, source/submission files, captured messages, synthetic import probe, and `input-hashes.json` provide reproducible inputs and outputs. All authored artifacts are confined to this directory.

The raw request text inspected supplies no concrete final constructor-session artifact prefix; none is invented. No remote second verdict or final per-constructor artifact was recovered here. No provider calls, relaunch, budget increase, repairs, or frozen deployment edits occurred.
