# V14 terminal result

Kernel version **12** of `pranavagrawai/prototype-tycho-nooa-public-3-full-timing`, launched 2026-09-03,
elapsed **1308.3 s**, terminal Kaggle status **ERROR**.

## Result

The run **did not test the V14 question**. It ended after **one** reconciled `ls20` action
(`UP` -> official `ACTION1`) when the forced action-demand request (Actor call 5, sequence 1)
consumed all 16,000 completion tokens as reasoning, ended `finish_reason=length`, and emitted
no tool call. The official launcher thread raised
`LocalSolverError: No action was committed within 0 native tool steps, and none was committed
when take_actions was the only tool left`, the launcher exited rc=71, and the outer probe marked
`request_path_status=invalid`, `failure_classification=pre_request_failure:official_launcher:RuntimeError`.

This is the **same non-commitment failure class as V13** (V13 call 9), reached earlier — at
sequence 1 instead of sequence 9.

## Why V14 answered nothing

The V14 mitigation was **never active in the Solver worker**. The forced request (Actor call 5)
was persisted with `tool_choice="required"` (correct), `max_tokens=16000` (should be `2048`), and
**no** `extra_body` at all (should carry `chat_template_kwargs.enable_thinking=false`).
The worker's own `worker-metadata.json` decision envelope confirms it:

```json
"action_demand_disable_thinking": false,
"action_demand_max_output_tokens": null
```

Both should have been `true` / `2048`.

### Root cause (confirmed)

`deployment/kaggle/runtime/local_solver.py`
(SHA-256 `517572979803525af52e7a6f2066fdee2eb3ce5923fd2780c5b185cb52b06795`; byte-identical to the
copy archived in wheelhouse v10 and v11) filters the Solver worker subprocess environment through a
hardcoded allowlist, `WORKER_ENVIRONMENT_KEYS` (lines 52-88). That set does **not** contain
`ARC_LLM_ACTION_DEMAND_MAX_TOKENS` or `ARC_LLM_ACTION_DEMAND_DISABLE_THINKING`.

Chain:
- notebook cell 3 sets both vars in `os.environ` — **OK**
- `public25_launcher_probe.official_launcher_environment()` forwards full `os.environ` to the
  official launcher subprocess (`env={**os.environ, ...}`) — **OK** (the probe's own
  `int(os.environ["ARC_LLM_ACTION_DEMAND_MAX_TOKENS"])` at line 528 did not `KeyError`)
- `local_solver.worker_environment()` rebuilds the child env as
  `{key: values[key] for key in WORKER_ENVIRONMENT_KEYS if key in values}` — **strips both vars here**
- `tycho_nooa.agent` reads a clean env -> `generation_action_demand_max_tokens()` returns `None`,
  `generation_action_demand_disable_thinking()` returns `False`
- `tycho_nooa.strategy` (lines 350-361) sends the ACTION_DEMAND_REQUEST with `max_tokens=16000`
  and thinking enabled -> model reasons to the cap -> no tool call -> `LocalSolverError`

The Solver wheel itself (v11) is correct: `llm.py` reads the env vars, `agent.py` builds the
config, `strategy.py` applies both the token cap and the no-thinking `extra_body` **when the
config carries them**. Only the env never reached it.

### Secondary defect (would also block a green run)

`public25_launcher_probe.summarize_solver_sessions()` builds
`expected_metadata["decision_envelope"]` (lines 745-758) **without** the
`action_demand_disable_thinking` / `action_demand_max_output_tokens` keys, while the v11 worker
always writes them into `worker-metadata.json`. The equality check
`observed_metadata != expected_metadata` therefore raises
`RuntimeError: Solver decision envelope differs for ls20` regardless of the primary fix. This
produced the run's `solver_sessions_error`; it was not the cause of the launcher failure but must
be repaired for a V15 run to reconcile.

### Test gap

The V14 candidate tests verify (a) `build_notebook.py` source contains the env-var assignments and
(b) `strategy` applies the config when handed it. Nothing exercises the
`local_solver.worker_environment()` allowlist boundary, so the drop was invisible in the
`26 passed` / `76 passed` local runs.

## What did reconcile

- Notebook cell sources on Kaggle kernel v12 were byte-identical to the frozen candidate
  (concat-source SHA-256 `084946e26897cbc522a867618ca656d70ac09bb17e7152ee083fc1f270f617ba`).
- Wheelhouse dataset bound as `pranavagrawai/tycho-nooa-solver-wheelhouse-cp312-v11`.
- Sequence 0: 4 exploratory Actor calls (full 8-tool surface, `tool_choice` unset). Calls 1-4
  emitted `ls, ls, run_python` / `run_python` / `write_file, write_file` / `take_actions`.
  Call 4 submitted a 1-action batch (`UP`, `stop_on_no_change=true`, `model_calls_avoided=0`).
- The single released action `UP` matches official recording `ACTION1`
  (`recorded_action_results=1`, `action_ids={"0": 1}`), `final_state=NOT_FINISHED`,
  `final_levels_completed=0`.
- Sequence 1: 1 forced Actor call (`tools=['take_actions']` only), `finish_reason=length`,
  `completion_tokens=16000`, `reasoning_tokens=16000`, no tool call -> failure.
- Warmup: 1 successful chat-completion access, 1 ready marker.
- Lifecycle / containment: `containment_passed=true`, `r8_l3b_gate.lifecycle_passed=true`,
  `teardown_log_accepted=true`, teardown `forbidden_count=0`. `request_path_passed=false`.
- **Zero competition submissions.** No scorecard-submission markers in the kernel log;
  `enable_internet=false`; competition-rerun guard present in cell 4.

## Claim ceiling

- V14 establishes **nothing new** about whether the Kaggle SGLang route honours a 2,048-token
  no-thinking forced commitment request. That request was never sent.
- The unified Solver still has one model-facing action path; its guarded batch machinery released
  one action that reconciled to the official recording.
- The V13 -> V14 delta (Solver wheel v10 -> v11, plus notebook env vars) did not change observable
  behaviour because the env vars were stripped before the worker started.
- No public-25 performance result. No competition result.

## Result-map disposition

Per the V14 card: *"request-option drift or ambiguous evidence -> repair the transport or
measurement defect and issue a new card."* This is a packaging/measurement defect. **Repair and
issue a V15 card.** Do not reinterpret V14 as a transport finding.

## Machine-readable evidence

`TERMINAL-RECONCILIATION.json` in this directory. Raw Kaggle output and source were downloaded
before interpretation.
