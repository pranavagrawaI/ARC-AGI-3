# Two prepared notebooks; no Kaggle launch

## Matched Actor replay

Package: deployment/kaggle/actor-mtp-20260906. Notebook SHA256:
2a2a081d5f316f9b83024e7d962fd31b6cfc7fb96dac7732040c3edc3fc7907c.
Sealed corpus SHA256: b90fc104d52f4059bc793cb4fa4455c10fd35b15607f35b433e08ab142de2383.

Changed the formerly unsealed four-category contract to accept the actual capture:
all14 requests and their provenance, including three invalid-tool outcomes. Request
and paired response hashes were verified against capture events. Preserved omitted
max_tokens/temperature/top_p rather than injecting assumed defaults. Captured
server sampling_defaults is model; mounted generation_config must match exactly.

Runner now retains invalid tool outcomes and continues both arms. HTTP, input
identity, server and cleanup failures remain fail-closed. Natural input token
counts must match capture; rendered server IDs must match across arms. The fixed
decode diagnostic uses the longest observed case (~21k), not fictitious ~102k
coverage. Same14 historical requests per arm, no generated tools executed.

The host already supplied unknown-tool feedback in the capture, including before
attempt13. No duplicate recovery layer, alias-to-tool execution, or silent argument
coercion was added. This preparation does not claim to cure invented-tool behavior.

## Integrated solver candidate

Package: deployment/kaggle/q6-mtp-integration-20260906. Exact donor notebook:
analysis/deployment-probes/20260906-q6-prediction-feedback/long-run/candidate/probe.ipynb,
SHA256 f40d36d33deed4ac909c515247795a581aa54af2ca092eefae49e6c7af223893.
Solver contract e9b0cae6d1b4e0a1c7224d9e34b3d57a0d0bbe8985764e7bfa9bd443b80bddbf.

Every embedded solver file remains unchanged, including dispatcher, worker,
prediction feedback and integrity guards. Only engine addition is shallow native
MTP (EAGLE1/topk1/drafttokens2) plus metrics and a best-effort pre-teardown server-info
snapshot. The existing lifecycle procedure is preserved. Pilot caps:10 actions,
1800s admission,900s pending worker,3000s launcher,4800s session. Reasoning settings
and model manifest unchanged. Actual official offline ls20 environment/launcher.
The ongoing donor run was not modified or relaunched.

## Local evidence

- Replay tests:6 passed. Includes all14 captured responses through both mocked
  arms;28 natural responses retained,6 invalid-tool outcomes,4 fixed diagnostics.
  This is runner control-flow coverage, not local SGLang serving.
- Integration tests:3 passed. Executed actual notebook asset extraction redirected
  into a temporary directory, verified embedded solver hashes, exercised generated
  command function (original plus exactly selected flags), checked pilot bounds.
- Existing prediction-feedback tests:5 passed, including retained real transition.
- Tokenizer smoke:14/14 inputs render with verified checkpoint tokenizer files and
  Transformers5.12.1; list-of-int serialization passes. Direct HTTP histories first
  failed template rendering because historical arguments are JSON strings; the
  test-only copy now follows SGLang's argument normalization. Template counts are
  consistently22 below captured server usage. No server-equivalence claim is made;
  the runtime uses server-rendered IDs and verifies counts instead.
- Both deterministic notebook rebuild checks passed.
- Installed Kaggle CLI request construction passed with transport mocked: correct
  private slug, RTX PRO6000, offline, competition arc-prize-2026-arc-agi-3, caps7200s
  replay/4800s integration. Zero transport calls, zero pushes.
- Focused Ruff E9/F63/F7/F82 passed over both packages and generated notebooks.
  It caught a missing hashlib import in the integration asset cell; fixed and
  actual extraction rerun successfully before handoff.

## Disposition

Ready for user upload, not hardware-qualified. Run matched replay first for engine
attribution; integration is a separate bounded action/accounting pilot. Neither
demonstrates summarization, rejected hybrid-state correctness, or long-run score.
The experiment cards keep cost and promotion gates separate; success of one does
not automatically qualify the other. No files committed and no raw capture deleted.
