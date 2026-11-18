# Independent Flash-Next evidence review

Both attempts remain operationally invalid under the frozen OOM clause. This review closes several manual evidence obligations; it does not rehabilitate either attempt. Reviewed the frozen card, donor source, raw requests, raw SSE, response files, result rows, provenance, cleanup, and platform output before consulting the authors' result interpretation. No server, tool returned by the model, game, or model request was executed. Used the code-review skill within the assigned evidence axis.

## Closed obligations

| Obligation | v2 (`raw/`) | v3 (`raw-v3/`) |
| --- | --- | --- |
| Recorded model identity versus pins | Matches RadixArk repository/revision, model manifest/config/quantization hashes, 419 files and manifest byte total, Kaggle version-1 model path | Same |
| Recorded runtime/patch/GPU identity | Pinned vLLM version, image index/AMD64/runtime manifest hashes and patched PLE hash agree; RTX PRO 6000 Blackwell Server Edition | Same; a different GPU UUID and boot identity establish a separate session |
| Issued response accounting | Three requests, three response files and terminal SSE streams; two successful result rows; third response preserved before assertion failure | Eight requests, eight response files, eight terminal SSE streams, eight result rows |
| SSE reconciliation | Content, reasoning, assembled calls, finish, usage, request SHA256 agree with saved responses; successful rows agree with responses | Same for all eight |
| Synthetic semantics | Text `4` and offered/schema-valid `submit_number({"value":4})` pass. JSON is syntactically valid `{"key":4}` and fails the frozen exact-output check | All three pass |
| Actor semantics at the card's boundary | No Actor request issued following the stop | All five return offered `execute_python` calls with JSON arguments valid against the actual request schemas; no execution/correctness claim |
| Observed setup cap | 551.686 s < 1800 s | 514.107 s < 1800 s |
| Observed individual request caps | 0.515–0.918 s; completion counts 2, 26, 11, all <=1024 | 0.549–3.551 s; each <=1024 output tokens |
| Cleanup record | Owned identity matched, stopped true, no survivors, port closed | Same |

Reproduce the independent reconciliation with:

```bash
uv run --no-project --with jsonschema python analysis/deployment-probes/20260909-experiment-checks/reviews/flash-evidence/verify_raw.py
```

The script passed on both directories. It reads each issued request rather than using `results.jsonl` as the complete denominator, reassembles raw SSE independently, validates actual offered schemas, and compares provenance fields with constants parsed from the retained donor source.

Evidence roots: [v2](../../../20260909-flash-next-feasibility/raw/) and [v3](../../../20260909-flash-next-feasibility/raw-v3/). Within each root, `vllm-setup-provenance.json:45` records the GPU, `:115–135` model identity and hashing limits, `:259` the mounted model root, and `vllm-server-identity.json` the launched argv. `donor-serving_setup.py:37–58` supplies the model/runtime pins; its `:646–764` checks model manifest and config identity. The served alias `Qwen/Qwen3.8-Flash-Next-NVFP4` is not the checkpoint repository identity and is not a mismatch with `RadixArk/...`.

## Exact-output failure classification

The v2 request asks `Return JSON with the single key value equal to integer 4.` and requests `response_format: {"type":"json_object"}`. Its raw SSE yields `{"key":4}`, terminal `stop`, one usage block, and `[DONE]`; `json.response.json:13–18` agrees. `json.failure.json` identifies the exact equality assertion at `probe.py:98`. The outer `failure.json` records propagation of that same failure, not a second request or a timeout.

This is a **card-authorized synthetic exact-output stop**, with valid JSON transport/format and failed expected content. It is not malformed JSON. The prompt's ambiguous key wording limits interpretation; raw evidence does not establish that ambiguity caused the answer, nor that the server or model is absolved of an instruction-following failure. Because execution already contained an OOM, the card's condition for drawing a compatibility conclusion from structured-output failure (otherwise-valid execution) is unsatisfied anyway.

Five absent Actor responses are explained by the authorized early stop. They are untested cases, not five lost issued responses. Accordingly the current offline audit's `response_count:2` is a successful-results count, not an issued-response count; its generic `Eight unique expected responses are not accounted for` should be refined into `3 issued/accounted; 5 not issued after authorized exact-output stop`. The failed third request's files should be included in its evidence hash inventory, which currently omits `json.request.json`, `json.response.json`, and `json.sse`.

## Limits that remain

- **Payload integrity:** both provenance files explicitly say model and runtime payload SHA256 verification was false. Fast mode checks pinned identity metadata/configs and loads the model; this establishes the recorded identity comparison above, not independent hash verification of every weight shard or runtime layer. Those payloads are absent locally. Do not relabel this limitation a demonstrated wrong identity.
- **Platform/quota and launch policy:** retained platform output reaches final notebook export at 565.751 s for v2 and 540.577 s for v3, supporting execution comfortably within 2500/2700-second bounds. These are log-relative execution observations, not independent platform quota billing or scheduler configuration records. The card says one session and no retry after operational failure. Distinct boot/GPU identities show two sessions used a byte-identical card; v3 therefore cannot be represented as a single compliant attempt under that card. Whether a human separately authorized the subsequent attempt cannot be determined from raw run files. No automatic relaunch is visible inside either retained probe, which starts setup once and exits on failure.
- **Zero actions/submissions:** the retained probe records responses without executing returned tools and contains no game/submission call. Raw artifacts do not independently audit all account activity outside the probe. No reason to invent such activity; scope the claim to this program.
- **Counterfactual caps:** observed durations pass, but that does not prove timeout enforcement under stalled cleanup or every subprocess path. The reconciliation does not test such counterfactual lifecycle behavior.
- **Tokenizer and workload coverage:** actual usage counts validate these issued fixtures only. No candidate-tokenizer evaluation of the live tail was performed in this review.

## Residual report overclaims

The withdrawal banners correctly retain the invalid verdict. The original v2 report's assertions that the abort was caused by the prompt and that its falsifying half was “not Keith's server” go beyond the observed response (`RESULT.md:18–30`, `:57–75`). The original v3 report says prompt disambiguation “did matter” and identifies the prompt as the defect (`RESULT-v3.md:52–54`, `:107–115`). Different sampled runs with multiple client changes do not isolate the cause. These statements should remain explicitly superseded/hypotheses, not become verified facts after the OOM correction. A valid exact-output failure and an ambiguous instruction can coexist.

The old capability claims are already withdrawn; this review adds no further capability verdict. The useful remaining observations are the raw outputs, the authorized stop boundary, matching recorded identity, measured durations, and cleanup records.
