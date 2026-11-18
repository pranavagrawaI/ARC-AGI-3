# Result: bounded construction ended without a candidate

**Classification: `INTEGRATION_VALID_BUDGET_EXHAUSTED`.** Exactly one authorized OpenRouter episode was executed. There was no retry, endpoint substitution, second episode, or runtime deployment.

## Actual episode

| Measure | Observed |
|---|---:|
| Model | `qwen/qwen3.8-27b` |
| Completed model requests | 4 |
| Python inspection attempts / successful inspections | 4 / 4 |
| Candidate submissions | 0 |
| Requested completion-token allowances | 8,192 |
| Provider-reported completion tokens used | 961 |
| Provider-reported prompt tokens | 45,083 |
| Provider-reported cost, summed from response usage | $0.008152822 |
| Reasoning requested / reported | disabled / zero tokens |
| Episode elapsed time | 27.142 seconds |

The model inspected visible summaries and initial/current grids, compared changed cells, inspected colors 9 and 12, then rechecked the meter. It did not select and submit a work item, produce an executable candidate, or request additional evidence. Consequently **no model-produced candidate was frozen or evaluated on the suffix**. No claim of improved prediction, model-building competence, or prospective support follows.

The stop was the predeclared **requested-token** cap, not actual token consumption: the request allowances were 3,072, 2,048, 2,048, and 1,024. This conservative accounting exhausted the allowance while the model was still inspecting. The prompt did not explicitly expose remaining numeric budgets, and the harness did not reserve a final submission turn. These are diagnostic limitations of this episode, not evidence that the model cannot construct a useful model with a different protocol. No protocol change or retry was made after launch.

## Isolation repair and verified machinery

- Trusted observer, branch-local joint assessment, temporal ancestry, and source/belief version identity now remain in the host process.
- Every candidate ABI invocation starts a fresh bubblewrap process. Only candidate source, an ABI-only runner, and the current invocation's arguments are mounted. Neither the visible replay bundle nor historical suffix observations nor trusted verifier code are present in that realm.
- Visible replay predicts and freezes commitments before host assessment, then passes only the current observation to assimilation. Sealed rollout builds all action-conditioned forecast layers before host suffix assessment; no suffix assimilation occurs and rejected ancestry cannot resurrect.
- Output pipes are drained incrementally and execution is killed at the byte limit, rather than capturing unbounded output first. Time, address-space, CPU, file-size, and file-descriptor bounds are applied.
- Recognized failed inspection/submission attempts consume their attempt budget before parsing/execution. Request snapshots are detached from subsequent messages. Exact tool calls and rendered results are durably appended; request/response JSON, raw response bodies, and attempt records are retained.
- Prelaunch numeric checks now explicitly require incumbent compatible depth **2** and first contradiction at step **8**, rather than accepting any truthy value.

### Actual verification

`PRELAUNCH.json` records all four process checks and eight explicit semantic checks passing:

- Reference SHA-256 manifest: passed.
- Reference synthetic tests: **28 passed**.
- Local integration/adversarial tests: **16 passed**.
- Dry run: passed, zero paid calls.
- Incumbent passes visible steps 1–5, supports two suffix transitions, and contradicts step 8.
- Host-only witness passes all three suffix transitions. This is an evaluator control, **not** the model's output.

The new tests exercise denied evidence/verifier access, monkeypatched and printed forged assessments, output floods, failed-attempt accounting, detached request snapshots, prediction-before-assimilation, joint-branch compatibility, and temporal ancestry. These establish tested isolation properties, not a proof against every kernel or sandbox exploit.

`audit.py` read the completed artifacts without rerunning the model or evaluator. **All 18 audit checks passed**, reconciling exact tool calls/results, request/response snapshots and hashes, raw response bodies, usage sums, model identity, reasoning, caps, absence of a candidate/suffix evaluation, and unchanged prelaunch source/card hashes.

## Scope and interpretation

This remains a narrow independent integration with singleton observable commitments per branch, not a full integration of the reference prototype's finite-domain prediction and informativeness machinery. Dependency lists are declarations observed during execution, not a static completeness proof. The reference's successful tests do not imply that all its features are implemented here.

LS20 steps 6–8 are historical evidence already known to the experiment designers, withheld only from this constructor. They are not an untouched scientific holdout. The model never reached their evaluation in this episode.

Permitted next action under the card: preserve the bounded negative construction result and inspect these protocol limitations. Any changed budget, prompt, model, or new execution requires a new identity and separate authorization. No runtime files or unrelated worktree changes were intentionally edited.

## Artifacts

All paths below are relative to:

`/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/model-construction-real-prefix-20260913/`

- `RESULT.json`, `RESULT.md`: machine and human outcomes.
- `AUDIT.json`, `audit.py`: audited evidence and reproducible offline audit.
- `EXPERIMENT-CARD.md`, `STARTED.json`, `PRELAUNCH.json`: immutable card, launch identity/source hashes, and actual prelaunch outputs.
- `raw/requests/00.json` through `03.json`: detached request objects.
- `raw/responses/00.json` through `03.json`, and matching `.body` files: provider responses and original bodies.
- `raw/request_attempts.jsonl`, `raw/model_calls.jsonl`: attempt ledger and artifact hashes.
- `raw/tool_events.jsonl`, `raw/transcript.json`, `raw/construction.json`: exact inspection calls/results and final construction state.
- `sandbox_eval.py`, `constructor.py`, `run.py`, `verify.py`: repaired implementation and verification.
- `launch_once.py`: launcher parsing only the authorized key entry, without sourcing or logging other environment values.
- `tests/test_isolation.py`: added adversarial/contract regression tests.

`raw/FROZEN.json`, `raw/frozen_candidate.py`, and `raw/sealed_suffix_evaluation.json` do not exist because the model submitted no candidate.
