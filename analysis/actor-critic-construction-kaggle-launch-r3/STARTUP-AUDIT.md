# R3 shared-premise diagnosis and sibling startup audit

## Before changing the boundary

Shared failed premise: **the donor supervisor still works unchanged under the scrubbed notebook environment.** R1's request-timeout parser rejection and R2's later loader failure are distinct manifestations of an unqualified donor invocation boundary. Merely fixing each observed downstream symptom would preserve that false premise.

Preserved R2 evidence: `analysis/actor-critic-construction-kaggle-20260914T183029Z/terminal-logs-receipt.json`. The real parser accepted 180; offline installation/venv completed. Supervisor failed at `accelerator_preflight`: `/opt/bin/nvidia-smi` exited 12, unable to load libnvidia-ml.so. No model serving or model calls occurred. Initial bootstrap GPU discovery had already succeeded using `accelerator.discover`, which passes its root-owned allowlisted `loader_environment` explicitly. Supervisor `system_snapshot -> run_capture` instead inherits the scrubbed environment. This distinguishes missing *propagation* from absent GPU, wrong identity, failed installation, or parser incompatibility. It does not establish all CUDA dependencies are available.

Bounded intervention: reconstruct only LD_LIBRARY_PATH from the unchanged accelerator allowlist and permission checks when launching the trusted supervisor; remove any preexisting loader value before applying the reconstructed one. Keep the notebook scrub, initial discovery, installer environment and generated-code boundaries unchanged. Preserve the donor supervisor bytes relative to R2. No ambient driver path, preload, Python path or credentials are restored.

## Source-wide startup consumers audited before rerun

| Path | Environment / finding |
|---|---|
| Notebook -> process_guard -> donor_bootstrap | Explicit small notebook environment; isolated `-I -S -B` interpreter; guard inherits only that environment. No restoration at notebook level. |
| Bootstrap -> kaggle_probe.inspect_gpu -> accelerator.discover | Explicit BASE_ENV plus validated default library directories for both ctypes dependency probes and GPU query. Already works; keep checks unchanged. |
| Bootstrap -> require_seal | Real CandidateRPC and inspection before installation; default bounded executor scrubs environment and FDs, sealed root excludes driver libraries. Never give it trusted loader env. |
| Bootstrap -> runtime_install | Offline hashes/venv installs inherit notebook scrub; R2 completed this stage. Do not change install behavior. |
| Bootstrap -> supervisor | Missing loader propagation. Repair exactly this trusted exec environment; preserve /opt/bin PATH and work deadline, not wholesale notebook ambient state. |
| Supervisor system_snapshot/run_capture -> GPU query | Inherits supervisor environment; needs reconstructed driver path. GPU identity parser unchanged. |
| Supervisor lifecycle observer -> compute-app query; before/after snapshots | Same inherited environment through run_capture; repaired by the same exec boundary, not a second patch. |
| Supervisor -> model server Popen | process_environment=os.environ.copy(), plus owner/profile flags; same missing driver env would recur here. Exercise actual generated server-launch statements with compiled dlopen consumer. |
| Supervisor -> kaggle_probe application driver | Trusted host subprocess inherits supervisor environment; its GPU discovery independently reconstructs trusted env. Model relay/constructor host can inherit it; this is not the generated-code environment. |
| construction_service -> construction_job, v11 nooa_worker/shared_local_solver | Trusted host orchestration inherits host environment. Generated candidate and inspection calls still cross SealedBoundary; v11 step_check subprocess has its own explicit environment. No new permission or mount. |
| CandidateRPC / SealedBoundary.inspect -> containment_base.bounded -> qualified_runner | Default executor supplies only PATH/LANG; runner closes inherited descriptors, seals filesystem and scrubs environment. Regressions must show no LD_LIBRARY_PATH or readable/loadable synthetic driver in both candidate and inspection. |

## Fixed bounds and unresolved scope

R3 target/title `pranavagrawai/actor-critic-construction-kaggle-r3` / `actor-critic-construction-kaggle-r3`; coordinator run `20260914T192100Z`. No provider calls or pushes by package owner. Preserve all predecessor files and 422 OpenRouter bindings.

No application/game/policy changes. Preserve one-hour reservation, guardian 3500+8, work 3100, startup 1200, model admission 900, supervisor CLI180, total chat24 including warmup/application23, Actor/summarizer20, actions8, constructor45soft/50hard, application HTTP30. Actual warmup observation uses remaining startup budget, not a changed request timer.

Compiled fixture evidence is **synthetic hardware/loader evidence only**. It must execute the real generated bootstrap invocation, real supervisor parser and GPU query, and real server-launch environment statements. A locally compiled dynamic library is not NVIDIA hardware, CUDA, SGLang, mounted weights, provider availability, or shutdown qualification. Real target-image startup, inference and teardown remain coordinator-owned remote gates. Any failure beyond this trusted bootstrap environment boundary requires escalation, not a broader repair.
