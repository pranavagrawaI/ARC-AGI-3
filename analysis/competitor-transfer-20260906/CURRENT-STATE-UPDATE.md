# Current solver update after the 100-action run

Implementation-order update: [Simon candidate plan](simon-candidate/PLAN.md) brings exact scene measurement, the compact packet and bounded experimental planning ahead of automatic repair generation. The historical run findings below remain unchanged.

This supersedes the earlier REVIEW.md statement that q6-summary-boundary-20260906 had not launched. It records inspected state and proposals, not implementation changes.

## Live state

Candidate contract remains `24c291bc6956a96dce361a52a713c0b3c4e4208da7e464d735d2b310834976a9`; current source files match the manifest. Downloaded session: `analysis/deployment-probes/20260906-q6-summary-boundary/raw/nooa-join-public3/solver-workspaces/ls20-8md_11v1`.

Raw IPC recomputation: 100 executed actions, 40 action batches, 153 Actor calls, two summarizer calls, zero completed levels, final state NOT_FINISHED. Actor model wall time is 2979.78 seconds; summarizer calls total 111.14 seconds. Peak recorded prompt size is 107615 tokens. The final recorded feedback per observed commitment has 23 matched and 16 mismatched results; this is not a gameplay accuracy measure. Sixteen distinct committed source hashes occur. The final action's feedback need not be delivered before cleanup.

Per-batch mismatch remains delayed: turn 8 mismatches at step 9 and continues through 10/11; turn 17 mismatches at 42 and continues at 43; turn 28 mismatches at 71 and continues at 72/73.

The official launcher returned 0. The outer probe marked ERROR/74 because nooa_public_probe.py:183 requires the obsolete identity q6-stall-authority. Corrected manifest-based validation in the independent exit-review passes saved-data checks, including 155 solver + one warmup reconciliation. Original ERROR and raw evidence remain unchanged. Read `../deployment-probes/20260906-q6-summary-boundary/exit-review/RESULT.md` and `REPLAY.json`; earlier interpreter-finalization speculation in RESULT-run.md was explicitly superseded.

## Configuration is not necessarily enforcement

`build_notebook.py:246-249` sets ARC_LLM_CONTEXT_WINDOW=32768, ARC_LLM_MAX_TOKENS=4096, ARC_LLM_RETRIES=0 and timeout180. `headless.py:106` calls build_llm() without arguments. `chassis/arc_llm.py:53-116` does not read the context, output-token or retry environment settings; it does use its default timeout180 and extra_body thinking switches. Do not call 4096 or zero retries effective limits. Largest observed completion in this run was 2571; that does not establish a cap.

Headless's TruncationConfig(max_context_tokens=32000) governs system/context block rendering, not the complete event history. Installed NOOA's context_blocks/renderer.py passes only system blocks to that truncator. The default token-budget summarizer triggers from client-reported prompt usage after a turn, at 80% of discovered model context or a 100000 fallback. The exact selected threshold is not established by these launch variables. This explains why a 32k configuration can coexist with >100k recorded requests.

## Architecture disposition

| Component | Current behavior | Proposed disposition |
|---|---|---|
| Official adapter and LocalSolver transport | Public observations/actions, per-action reconciliation, episode lifecycle and game budgets | Keep; integrate already-tested manifest-validator repair into next candidate |
| Model/serving | Fixed Qwen3.8-27B FP8 on SGLang; thinking disabled | Keep weights for comparisons; explicitly wire and verify effective budgets; MTP remains separate |
| State delivery | File-backed latest state, queue wakes/deduplication, synchronous actor read API | Keep; add compact packet assembled from existing records |
| Observation | Full settled grid, code conversion, optional PNG off; worker omits projected animation/terminal-frame detail | Preserve raw access; fix stale animation documentation; evaluate richer delivery separately |
| Actor/CodeAct | One game-reasoning agent writes/executes Python and submits actions | Keep; shorten routine workflow and clarify provisional nature of helpers |
| Helpers/world model | Mutable encode/predict code, named schema, one active helper frozen at submit | Keep partial models; retain candidate identities and explicit scope; add optional objective conjectures |
| Planning | Actor-authored search in generated code, no built-in general goal solver | Keep actor judgment; trial a bounded search helper when useful, not a persistent state graph |
| Prediction checks | Frozen self-encoder consistency, latest batch only | Extend with common raw-observable checks and past-transition replay |
| Action gates/worker queue | Legal action/reentry/consolidation/coarse and executable gates; batch stops on exact no-change, terminal or level | Add per-step prediction interruption; unify duplicate prediction payloads only while preserving their checks |
| Markdown memory | Per-session hypotheses/step log, periodic curation, required level consolidation with fallback | Keep; record belief changes rather than duplicate raw history; do not delete successful-level learning |
| Context summarizer | Background LLM compression, now exercised twice with verified serialization | Keep initially; measure semantic preservation; fix full-request budgeting |
| Stall review | Rationale-similarity heuristic, >=25 steps, six-batch window, untried action; blocking treatment off | Keep off; later compare prediction/failed-progress evidence against wording-based trigger |
| Integrity and evidence | Byte/readback guards, source hashes, exact transitions and model accounting | Keep; don't confuse serialization integrity with truth of notes |
| Candidate comparator (new) | Absent as an automatic runtime operation | Bounded CPU evaluation of old + up to two candidate repairs; actor retains selection authority |
| Offline learning corpus (new) | Logs exist, reusable split/evaluation not automatic | Convert admissible traces into prospective repair cases; no oracle labels or unsupported prose as training truth |

The compact packet, common observable checks and objective declarations are extensions of current state/prediction/helper interfaces, not extra LLM agents. The candidate comparator is a new computational helper. The offline corpus is outside the gameplay loop. Any runtime execution of candidate code must remain inside the solver computation boundary; the official game process must not import actor-authored helpers.

## Revised sequence

First integrate the known probe identity repair and make intended request budgets explicit and observable. Measure behavioral effects of budget changes separately. Then implement and test per-action interruption. Next add decision-relevant observable/objective checks and a bounded repair comparison; compare prospective behavior under equal budgets. Trial cheaper context/bookkeeping independently. Build the corpus from these episodes and only then consider weight training. No new neural world model, latent bridge, permanent ensemble or builder is proposed in this sequence.
