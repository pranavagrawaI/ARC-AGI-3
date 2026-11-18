# Endpoint judgment screen, 2026-10-13

The tool is implemented and qualified offline. It runs the existing dfranzen ToolAgent and Python sandbox through the real TAAF game path, starting from the 40 verified stuck-level prefixes. It accepts an OpenAI-compatible base URL, exact model ID, and API-key environment variable. No real endpoint screen, model training, Kaggle launch, or competition submission was performed. No dedicated free-provider key was configured, so even the optional connectivity check was omitted. See [key-presence.json](evidence/key-presence.json).

The deliverables are [screen.py](screen.py), [worker.py](worker.py), [RESULT.json](RESULT.json), and the dated provider catalogue [ENDPOINTS.md](ENDPOINTS.md) with [providers.json](providers.json). Source and environment hashes accompany each run in `config.json`.

## What was verified

| Check | Observed result | Evidence |
|---|---|---|
| Replay to the saved start level and frame hash | 40/40 match | [starts-verification.json](evidence/starts-verification.json), reproduced by [audit_inputs.py](audit_inputs.py) |
| RESET preserves start level and entry frame | 40/40 match | [starts-verification.json](evidence/starts-verification.json) |
| First system prompt and Python tool schema against A0 r1 | 40/40 system prompts and 40/40 tool schemas equal | [harness-fidelity-all.json](evidence/harness-fidelity-all.json) |
| First user observation against A0 r1 | 40/40 text and 40/40 decoded image pixel arrays equal | [observation-fidelity-all.json](evidence/observation-fidelity-all.json), reproduced by [audit_fidelity.py](audit_fidelity.py) |
| Complete offline mock through the native actor/tool/engine loop | 40/40 complete, 26 sibling fixtures solve, 14 starts without fixtures reach token cap; 1,360 actions | [summary.json](evidence/qualified-dry-all/summary.json), [run log](evidence/qualified-dry-all.log) |
| Mock token accounting | 1,206,720 generated tokens, 145,300 prompt tokens; 2.15460 fixture solves per 100K generated tokens | [summary.json](evidence/qualified-dry-all/summary.json). These are scripted fixture usage values, not model measurements. |
| Repeat completed run | No new receipt files or token charge; all completed starts skipped | [idempotence-all.log](evidence/idempotence-all.log), [final-verification.json](evidence/final-verification.json) |
| Offline behavioral tests | 17 pass | [risk-tests-final.log](evidence/risk-tests-final.log), [test_screen.py](test_screen.py) |

The tests cover partial-start recovery through cached responses, unchanged receipt hashes, exact token limits, final-level wins, unknown usage, a provider exceeding requested output limits, state and tool-observation replay mismatches, 429 backoff, request accounting across retries, daily rollover, fatal HTTP errors, key preservation, and HTTP request serialization. The engine test also checks the 15-action tn36 L1 solution and the final-level lp85 L7 solution. These are test fixtures supplied only to `--dry`. Real endpoints receive no sibling solutions, hints, routes, game source, or prior successful reasoning.

The first-request fidelity checks establish prompt, tool, and observation parity. PNG encodings differ between environments, so equality of decoded RGB pixels is the relevant check. They do not establish equality of every subsequent model decision or hidden competition score.

## Historical A0 reference, with the correct denominators

[audit_inputs.py](audit_inputs.py) reads the existing corrected solve readout, benchmark histories, and every A0 per-attempt response log. It does not rerun the model. The reconstructed reference is in [baseline.json](evidence/baseline.json).

| A0 population | Solves / attempts | Solve rate | Raw response completion tokens | Solves per 100K generated tokens |
|---|---:|---:|---:|---:|
| All 40 starts, two replicates | 17/80 | 0.2125 | 5,791,151 | 0.293551 |
| 26-level hinted-comparison subset, two replicates | 17/52 | 0.326923 | 3,659,542 | 0.464539 |
| 14 never-solved starts, two replicates | 0/28 | 0 | 2,131,609 | 0 |

The brief's 0.33 is reproduced for the 26-level subset. It is not the solve rate of all 40 starts. Mean observed output was 72,389 tokens per attempt across all 40 starts. A0 had a 1,200-second time cap, not an 80K token cap. A future screen at a hard 80K token budget is therefore not a matched reproduction of that reference.

The benchmark's action-associated ledger totals 5,443,843 generated tokens across all 80 A0 attempts. The raw response logs contain 347,308 more. Using the ledger gives 0.312279 solves per 100K, while using received endpoint responses gives 0.293551, which reproduces the approximately 0.29 figure in the prior failure analysis. The new tool uses response usage for its denominator even when generated tokens never lead to an action. All 80 reference logs report completion usage, with zero missing-usage responses. Exact paths appear beside each row in [baseline.json](evidence/baseline.json).

The serving receipt also differs from the brief's precision label. [A0 serve.log](../oracle-hint-20261010/sessions/a0/output/serve.log) reports target path `intel-qwen3.8-flash-next-w4a16-autoround`, target `quant=auto-round, bits=4`, and a separate MTP drafter with `quant=compressed-tensors`. Server arguments report EAGLE, three speculative steps, and four draft tokens. See [a0-serving-identity.json](evidence/a0-serving-identity.json). This is an AutoRound 4-bit A0 reference. These files do not establish an NVFP4 target reference or the precision of every historical hidden submission.

## How to use the tool

Run from the repository root. The parent uses Python's standard library. The worker reuses the existing `solvers/tycho_nooa/.venv/bin/python`, the audited dfranzen source under `analysis/m2-winners-20261001/audit/dfranzen-repo`, and the read-only competition environment files. No dependency installation is needed on this workspace. Override `--env-dir` when the environment files live elsewhere. A copied tool needs those dependencies and paths.

An offline three-start smoke test:

```bash
python3 -B analysis/endpoint-judgment-20261013/screen.py \
  --dry --token-budget 4096 --keys tn36-L1 lp85-L7 bp35-L1 \
  --output analysis/endpoint-judgment-20261013/my-dry-smoke
```

All 40 starts, with the same default token budget as the completed qualification:

```bash
python3 -B analysis/endpoint-judgment-20261013/screen.py \
  --dry --token-budget 80000 \
  --output analysis/endpoint-judgment-20261013/my-dry-all
```

Offline tests and the reference audit:

```bash
python3 -B analysis/endpoint-judgment-20261013/test_screen.py
solvers/tycho_nooa/.venv/bin/python -B analysis/endpoint-judgment-20261013/audit_inputs.py
python3 -B analysis/endpoint-judgment-20261013/audit_fidelity.py
```

The following is a future usage template, not authorization to start screening under this brief. Claude coordinates real screens. The operator must first verify the exact model/route is free and the key cannot spend money:

```bash
python3 -B analysis/endpoint-judgment-20261013/screen.py \
  --base-url "$ENDPOINT_BASE_URL" --model "$ENDPOINT_MODEL" \
  --api-key-env ENDPOINT_API_KEY --provider PROVIDER_NAME \
  --zero-cost-confirmed --token-budget 80000 \
  --max-rpm 5 --daily-cap 50 \
  --output analysis/endpoint-judgment-20261013/provider-model-run
```

Use the same output directory and configuration to resume. Each start has `result.json`, `entry.json`, `checkpoint.json`, native transcripts, and compressed `receipts/NNNNNN.json.gz` containing the full wire request and response. `summary.json` reports selected/completed/interrupted coverage, solves, generated tokens, prompt tokens, actions, and solves per 100K generated tokens. Prefix replay actions are excluded from actor action counts. Executed RESETs and actions that cause deaths are included.

`--daily-cap` counts attempted requests across starts and retries in the run directory, persisted before sending. The local day is UTC. `--max-rpm` uses a persisted rolling minute; 429 responses honor numeric or HTTP-date Retry-After, then exponential backoff when absent. Request/token caps belonging to other runs or other account users are not inferred. Google resets its provider day at midnight Pacific. Subtract external usage and choose conservative local limits. A token-per-minute quota still may reject a long prompt; that rejection is logged, not counted as a failed puzzle solve.

## Adaptations and stopping behavior

The notebook's `setup_env` is parsed without executing notebook cells. The runner keeps its temperature 0.7, top-p 0.95, context settings, game-level retained functions, action guards, image generation, and Python tool. It uses one start per subprocess and bypasses the scheduler's thread pool, while retaining the native game-session play method. No 9-hour competition scheduling or GPU server is started. The real replay overlay stops at the target level completion, including a game's final WIN.

The default `--wire-style standard` removes vLLM-only top-k/template/priority fields and prior assistant reasoning fields from outbound requests. `--wire-style native` retains them for compatible servers. `--modality native` retains images; `--modality text` sends the existing text views only. A text-only screen is a different observation condition and must be labeled that way. `--context-window`, `--max-output-tokens`, `--max-tokens-field max_completion_tokens`, and `--extra-body` allow documented provider adjustments. Configurations and source hashes cannot change silently inside an existing output directory.

Before every new request, the transport bounds output by the remaining per-start generated-token allowance. It records the exact returned model ID, reported provider, fingerprint, precision fields/metadata, and relevant headers. Unknown precision stays unknown. A response without `usage.completion_tokens` is preserved and its tools are withheld; efficiency becomes unknown. Reasoning is counted when the provider includes it in completion usage, and raw usage details are retained so that this semantic assumption can be audited.

Recovery reconstructs the native actor by replaying received responses locally. It checks action history, level, frame hash, and model-visible context before reusing a response or sending the first new request. It normalizes only named clock fields. A replay difference stops the start. A crash or transport timeout with a pending request also stops, because the provider may already have executed it. There is no automatic duplicate call. Inspect `pending.json` and reconcile the provider receipt before clearing it; this tool cannot resolve that ambiguity offline. HTTP failures and caps are distinct from completed unsuccessful starts in the report.

## Remaining unknowns and review

The dated [provider catalogue](ENDPOINTS.md) covers all seven requested providers. It lists published open-model candidates and limits, and marks missing numerical account limits. Cerebras currently has a bounded trial rather than a permanent free tier; Hugging Face free accounts currently have no inference credits. Those services cannot be assumed to offer a new unconditional $0 screen. Current exact OpenRouter free availability was not completely retrieved: the unauthenticated catalogue download failed local DNS, while the official dated guide supplied candidate names. No real provider's tool/vision compatibility or model efficacy was tested.

The existing Python sandbox is reused. It is not a new security isolation boundary. Nondeterministic Python can prevent replay and will require inspection if the context check rejects it. Provider-reported token semantics and undisclosed precision remain measurement limits. The screen measures fresh-start solve efficiency, not the competition's action-based score or the performance of an actor retaining previous-level knowledge.

[REVIEW.md](evidence/REVIEW.md) records the scoped requirements/correctness review and fixes. [risk.json](evidence/risk.json) records the explicit AST/statement-coverage CRAP measurement on 43 new functions, with maximum 34.7121 and no invented quality gate. There were no existing functions to compare for improvement. Failed intermediate runs and the initial plain-receipt qualification are preserved in [development-runs.tar.gz](evidence/development-runs.tar.gz), with [archive hashes](evidence/development-archive.json). All final deliverables are in the assigned directory. No git commit was attempted because `.git` is outside that directory.

The required summary was posted as `codex-endpoint-judgment` on the shared board. See [board-post.txt](evidence/board-post.txt).
