# READY — launch prohibited

Claim: **READY FOR READ-ONLY REVIEW AND A SEPARATELY AUTHORIZED ONE-RUN LAUNCH; NOT LAUNCHED.** This r2 package repairs the logical OpenRouter-to-local SGLang boundary and all six independent-review blockers without mutating the earlier candidate. It made zero Kaggle pushes, provider/model calls, gameplay actions, or competition submissions.

## Repaired contracts

- `request_contract.logical_to_local_request` separately validates the logical OpenRouter request, removes gateway `provider`, binds the exact model returned by the proven v15 `/v1/models` contract (`/kaggle/input/datasets/jakobbrggen/qwen3-8-27b-fp8-hf-snapshot`), preserves messages/tools/tool choice/token cap, and adds the proven local controls `reasoning.enabled=false` plus `chat_template_kwargs.enable_thinking=false`.
- `LocalHTTPTransport` persists the logical attempt before transport, exact POST bytes and parsed wire, raw HTTP response or failure, status, latency, and lifecycle incrementally. It performs no retry and rejects model/config drift, gateway-only fields, absent controls, HTTP failure, malformed JSON/envelope, timeout, and responses over 2,000,000 bytes.
- `remote_entry.py` honors the donor launcher’s exact `--base-url`, `--model`, `--workspace-root`, `--output`, and `--timeout-s` CLI. It writes a truthful `official-launcher-result.json`-compatible status containing construction evidence and an explicit null gameplay result.
- Opaque provider tool-call IDs are preserved and uniqueness-checked without requiring invented `call-N` ordinals. Response-only SGLang/vLLM metadata remains lossless in raw receipts but is projected out before continuation.
- Inspection’s advertised and mounted path is consistently `/evidence/case.json`. Candidate evaluation preserves the baseline 15-second shared deadline, 3-second per-call timeout, 200,000-byte source/argument/result caps, and 1,000,000-byte process-output cap; the post-construction assessment gets a fresh boundary.
- A durable exclusive run-owner marker prevents output reuse. The irreversible suffix-attempt marker is written before assessment, durable totals are audited, and transport/assessment exceptions produce terminal `RESULT.json` evidence before propagating.

## Verification

- Existing constructor suite: **46/46 passed** (`evidence/constructor46.stderr`).
- Existing qualified containment suite: **53/53 passed** under mapped root (`evidence/containment53.stderr`).
- r2 integration/wire suite: **17/17 passed** under mapped root (`evidence/mapped-root-suite.stderr`), including byte-decoding and pinning the actual v15 embedded supervisor, exact donor launcher CLI invocation, and three-request local HTTP continuation (two inspections plus submission).
- Actual generated notebook passed in deterministic normal and conflicting-cache modes and in local HTTP mock-server mode. All three report `INTEGRATION_VALID_CANDIDATE_SUPPORTED`, exactly one suffix attempt, and embedded module origins under `/tmp/construction-policy-integration` (`evidence/notebook-*-fs/working/`).
- Prelaunch passed with closed launch guard (`evidence/PRELAUNCH.json`). Explicit launch invocation exited 1 with `LAUNCH_DISABLED: separate one-run authorization is absent`.
- Consolidated machine-readable checks and hashes: `evidence/FINAL-VERIFICATION.json`; reproducible source/package hashes: `MANIFEST.json` and package `BUILD.json`.

The direct non-root boundary failure is expected and retained at `evidence/direct-nonroot.stderr` (`NON_ROOT_OWNER`); `require_root` was not weakened. Canonical local command:

```bash
unshare --map-users 0:1000:1 --map-users 1:100000:65536 \
  --map-groups 0:1000:1 --map-groups 1:100000:65536 \
  --setgroups allow --setuid 0 --setgid 0 \
  python3 -m unittest discover -s tests -v
```

## Launch control and remaining gap

`deployment/kaggle/construction-policy-integration-20260913T221412Z-r2/launch_once.py` remains a disabled placeholder. `AUTHORIZATION.json` is absent and no Kaggle push command exists. A coordinator must separately authorize any launch and re-review the exact manifest digest.

No target Kaggle execution, actual model load/inference, live SGLang server observation for r2, accelerator observation, quota receipt, or remote integration receipt exists. Local HTTP tests replay captured SGLang/vLLM-shaped envelopes through real sockets and the full continuation/containment path; they are transport qualification, not model or competence evidence.
