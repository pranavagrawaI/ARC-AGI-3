# Launch accepted; readback blocked

The sole launch gate returned success for target pranavagrawai/actor-critic-construction-kaggle-smoke, version 1, URL https://www.kaggle.com/code/pranavagrawai/actor-critic-construction-kaggle-smoke. Raw admission receipt: push-af6eadd797d64f3b804b8b7836f7a86c.json. Launch binding is in launch.json; authorization remains consumed in CONSUMED.json. No retry occurred.

Local evidence: coordinator and independent package suites passed25 tests; coordinator and independent gate suites passed119 tests. Package source reconciliation passed62 assets55 Python sources422 protected bindings. Final independent launch approval binds manifest0f01cc456dc452a1895dcd30e8594a7c9deee55e0f90c4cc8c227295d84b8aab. These facts are local qualification, not remote success.

Post-launch exact-version status returned404, exact-version and current-source pull returned403, and subsequent CLI list reported authentication required. Raw failed receipts are preserved. Consequently remote existence/state, server metadata/competition/image readback, actual GPU/model/containment/integration/teardown and final quota are NOT VERIFIED. Push acceptance alone is not remote qualification.

The read-only local monitor proc_c52890eaf3fe was stopped after authentication failure. This did NOT cancel the remote job. The sole push requested provider session_timeout_seconds3600; no exact-session cancellation was issued or claimed. Provider enforcement and charged usage still need readback.

Recovery requires restoring Kaggle authentication (CLI: kaggle auth login), then read-only status/source/output/quota collection against the SAME target/version1. Do not invoke push again, remove CONSUMED.json, mutate launched candidate, or label current-source readback exact-version evidence. Preserve failed receipts even if later recovery succeeds.
