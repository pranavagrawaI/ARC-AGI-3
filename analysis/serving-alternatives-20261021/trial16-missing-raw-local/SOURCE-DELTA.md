# Separate collection-only source delta

Baseline: ../trial-10-exact-wire-local/remote-readback/collect_missing_raw.py. That script delegated unbounded recursive paging and bulk response persistence to installed kernels_output with a regex. This packet reuses its current-session listing semantics, missing-only purpose, exclusive files and retained errors; it adds explicit page/name/one-attempt/byte/time/identity/provenance boundaries. No baseline file is edited.

New files:
- collector.py: transport-independent sequential exact-manifest collector and HTTP prerequisite decision.
- remote.py: installed SDK list request omitting version_label; reviewed Decimal-overlay subprocess before/after; bounded streamed GET; no URL or credential persistence.
- run.py: source/input validation and separate exact root-approval guard. Local-only preparation receipts cannot execute it.
- audit.py: separate supplemental required-file audit, explicit independent actual-result origin, retained source/origin/hash checks, unchanged scientific checker and publication verifier.
- build_manifest.py/binding.py: exact retained inputs and the one explicitly approved dependency expansion.
- verify.py/tests: offline public-path request/filter/path/size/budget/failure/provenance checks; no runtime baseline retry.

The original13118 bindings, approved Decimal overlay, original failed primary404, numeric ZIP request/error, partial ZIP/recovery, historical handoff and original INVALID/null reduction remain byte-identical. No new GPU packet, source freeze, model/workload change or launch authority is created. SOURCE-PINS.json is a collection-adapter review pin only.
