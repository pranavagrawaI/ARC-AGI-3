# R10A version provenance recovery — DONE; launch HOLD unchanged

## Finding

The supported exact-file download API accepts **integer `version_number=1`**, separately from `version_label="1"` used by the failed get/status/list APIs. Two fresh, read-only requests to `pranavagrawai/r10a-cd82-constructor-repair`, numeric version 1, returned HTTP 200. No latest fallback, guessed identifier, output listing, zip, retry, or mutation was used.

This resolves bounded output retrieval for the **actual returned target**. It does not repair its mismatch with the approved `pranavagrawai/arc3-r10a-cd82-20260916`. Original launch approval, incident addendum, CONSUMED, and launch HOLD remain untouched; `launch.json` remains absent. This report grants no approval or acceptance, and does not clear the R10b prerequisite.

## Reproducible evidence

`exact-observation-1789558948333208545/requests-responses.json` preserves method, exact request fields, timestamps, status, byte counts, and SHA-256. Raw response bodies are retained at the requested relative paths in that observation directory. Authentication headers and redirect URLs are deliberately excluded.

| Exact-version response | Bytes | SHA-256 |
|---|---:|---|
| `model-runtime/official-launcher-result.json` | 32944 | `fdcfe9457f4d4642559b746278d1539860a7647c1ad8ba8ee95dd8d8d4792976` |
| `actor-critic-construction/SEAL.json` | 137 | `37e20784c9214e87dc7aa343cc85809b4b589fced763c9c9c6e0365a5efb4ac6` |

The selector comes directly from the preserved push receipt, SHA-256 `40ce606c0f820cb8523a739df46b2ddbe5333b8e64126808c33d3e50a8a1ec9c`: returned owner/slug and printed provider version 1. The receipt is not interpreted as authorization for the returned identity.

Installed SDK primary evidence:

- `kagglesdk/kernels/types/kernels_api_service.py:397` defines `ApiDownloadKernelOutputRequest`; lines 455–465 expose/check integer `version_number`; lines 2656–2661 serialize it as `versionNumber`, alongside ownerSlug/kernelSlug/filePath.
- `kagglesdk/kernels/services/kernels_api_service.py:84–96` routes this request to `kernels.KernelsApiService/DownloadKernelOutput`, returning an HTTP redirect response (the installed client follows the download).
- R8 `analysis/actor-critic-construction-kaggle-verification-r8/collect_exact_files.py` and R9 `analysis/actor-critic-construction-kaggle-verification-r9-cd82/collect_exact_paths.py` use precisely this numeric selector for known paths. Their scripts establish precedent; the fresh HTTP 200 downloads establish this task's result independently. Two initially sought predecessor receipt/report paths were absent; no historical download success is inferred from those missing paths.

`INPUT-HASHES.json` binds inspected local receipts, current metadata summary/raw metadata, failed exact-label readbacks, SDK definitions, predecessor collectors, original review controls, and collection planner. `EVIDENCE-HASHES.json` binds this report and new evidence.

## Session mapping and handoff

The downloaded exact-version launcher receipt names:

- Arm A: `model-runtime/solver-workspaces/arm-A/cd82-fb555c5d-rmrz0g5o`
- Arm B: `model-runtime/solver-workspaces/arm-B/cd82-fb555c5d-b7azqbdf`

`session-mapping.json` records the source request/hash and 60 paths derived offline by the existing `r10a-verification/collect_plan.py`. These are launcher-reported session paths, not independently downloaded session contents or provider session IDs. No session artifact was fetched in this investigation. The planner's output is a candidate allowlist for implementer validation, not permission for unrestricted collection.

Exact reproduction command (from repository root; writes only new timestamped evidence in this review directory):

```bash
/home/pranav/.local/share/uv/tools/kaggle/bin/python analysis/r10-prep-20260916/version-provenance-review/retrieve_exact.py
```

For further authorized bounded collection, use that script's `ApiDownloadKernelOutputRequest` pattern with `owner_slug='pranavagrawai'`, `kernel_slug='r10a-cd82-constructor-repair'`, `version_number=1`, and only a validated known receipt/session `file_path`. Preserve each request and raw response/hash. Do not use `version_label`, current output listing, or substitute the expected slug. The supplied reproduction fetches only the same two receipts and caps each at 2 MiB; it never overwrites the prior observation.

## Provenance limits / acceptance consequence

The 404s do not imply numeric version 1 outputs are unavailable. They establish failure of the particular `version_label="1"` request paths; this investigation does not establish what opaque/string labels those endpoints expect or why they return 404. Metadata `id=134588849` is not promoted to a scriptVersionId; currentVersionNumber, COMPLETE, and the anomalously old lastRunTime are not used as exact-output proof.

The evidence binds returned bytes to the provider's explicit numeric-version download contract for the actual slug. It is not a provider-signed immutable attestation, exact-version metadata/status retrieval, or verification of all runtime claims. SEAL's reported `passed` is retained as raw runtime testimony, not independently adjudicated containment. Mechanical/scientific acceptance, cleanup, score trust, and accounting require their own artifact review, and the identity mismatch independently remains HOLD even if those checks pass.

No controls, sources, old reviews, credentials, or remote objects were changed. The original review/control hashes still match the previously recorded report, launch-review, and CONSUMED bindings. Investigation stopped once a supported exact retrieval route succeeded.
