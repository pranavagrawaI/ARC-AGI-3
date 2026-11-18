# Trial8 terminal diagnosis; trial9 local correction only

Trial8 is closed and released by its coordinator. It never started control replay or any candidate phase. Its frozen sources are unchanged. Its unchanged reducer returns `FEASIBILITY_ONLY`, ratio null. This directory is a separately identified, unfrozen CPU successor; it has no notebook, GPU assignment or launch authority.

## Established sequence

The version-bound archive is `../trial-8-wire-capacity-local/remote-readback/output-v1-kernel140368352.zip`, SHA256 `1794285c63dcb9803a14570999f48d19a02efd1f912ed16f546f7154b1bd8372`. The final collection/inventory receipts are that trial's `remote-readback/TERMINAL-HANDOFF.json` and `ARCHIVE-INVENTORY.json`.

1. Actual RTX PRO6000 Server/97887MiB passed; corpus preparation, control preparation and identity passed.
2. The control logged weight load1063.92s, tokenizer startup1247.91s, application startup complete10:04:57 UTC. These are observed startup timings, not request capacity.
3. Probe614 exited75 (connection refused). Probe615 returned HTTP503, Content-Length0, empty body, and exited1. No successful health200 or replay followed.
4. `ready()` only allowed code75, so this response generated the original readiness error. The existing absolute2700s startup allowance had not expired.
5. Arm-finally cleanup sent TERM/KILL, then checked group membership. Seq1867 observed remaining PID566 and raised cleanup-incomplete. This secondary exception replaced the original readiness exception. Seq1868, during final cleanup, observed no remaining group members. Terminal result reports empty surviving groups.
6. Terminal ERROR/typed quota61844.901, reserved0; baseline60473.273, charge1371.628s. Final inter-arm release/GPU-idle checks did not run and remain unqualified.

A source-supported causal representation is `server health503 -> readiness rejection -> arm cleanup -> observed remaining member -> cleanup exception -> original error masked`. The exact state producing503 and PID566's state at the earlier cleanup check were not captured. Application startup does not identify model readiness, and an empty503 does not identify a transient state.

## Exact pinned health semantics

The runtime identity binds the SGLang wheel SHA256 `d0620216732b82b6efd4f8a78cc9b5f55343a4d3837fdfa3bcbb841b7718a225`. Its installed `sglang/srt/entrypoints/http_server.py` SHA256 is `833d5da3572305a96df3607833256ee31f6e7a03d7ea91c260a8bde076846ca6`; the extracted bytes under `evidence/pinned-*` match the wheel and captured installed-source receipt.

`health_generate` lines669–675 returns empty503 for `gracefully_exit` or `ServerStatus.Starting`. With health generation enabled it can also return empty503 after its scheduler/detokenizer heartbeat timeout (lines709–731), marking UnHealthy. The default timeout at line193 is20s; the pinned environment default at `environ.py:336` enables generation. General warmup runs in a separate thread (`http_server.py:415`); its success sets Up (line2319), independently of the ASGI application startup log.

The actual trial8 effective values of `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION` and `SGLANG_HEALTH_CHECK_TIMEOUT` are unavailable. Control preparation preserved inherited environment but recorded only overrides, and neither variable was overridden. The wheel defaults are verified; the inherited remote environment was not recorded. The immediate~0.06s503 is compatible with Starting/shutdown and does not demonstrate a20s heartbeat timeout. SIGTERM was observed later. Starting is a supported hypothesis, not an observed field value.

## Narrow local correction

Only successor `supervisor.py` differs from the baseline. `http_once.py` remains byte-identical.

An explicit per-arm `ready_retry_http_statuses:[503]` opts into waiting on the exact pinned empty503 response shape: helper exit1, status503, exactly one Content-Length0, no transfer encoding, no body, expected capture format. Default remains no HTTP retry. This is a not-ready observation; it never marks success or labels the state transient. A real200 is still required before replay. Connection refusal remains eligible; other/malformed/truncated/body-bearing failures stay fatal. Persistent503 expires the original absolute startup/readiness deadline; retries and individual probes cannot reset or extend it.

Cleanup now waits for actual group disappearance after SIGKILL through one shared cleanup deadline, bounded by the configured cleanup allowance and overall end. Leader exit alone is insufficient. Remaining groups at that deadline stay fatal. The release checker is unchanged, and its failure still blocks the candidate.

A cleanup context captures only an exception raised by its own phase. It retains that original error and records a separate cleanup failure; a cleanup failure with no phase error remains fatal. This avoids using an ambient caller exception as the phase cause.

The cleanup test delays actual SIGKILL delivery to a real descendant whose leader has exited, recreating the timing mechanism deterministically; it is fault injection, not an incidence estimate or proof of PID566's exact kernel state. A separate withheld-kill test proves bounded failure. Native health tests execute the exact pinned function with CPU dependency stubs; they establish branch semantics, not remote state or model function.

## Probe-budget proposal and remaining integration gate

The old10s probe can expire before the pinned default20s generation health result. This adjacent mismatch did not cause the observed immediate503. Do not infer the old effective environment from defaults.

`PROPOSED-DELTA.json` proposes explicit generation-enabled/20s settings for both control identity and server phases, a25s control probe allowance (20s plus5s overhead), and the explicit control503 startup policy. Every probe remains clipped to the original startup/work deadline; overall10200 platform/9900 cooperative/300 cleanup and the2700s control startup allowance are unchanged proposals, not launch grants. Candidate probe/policy remain unchanged. Before a future packet is frozen, the effective two control settings must be demonstrated/bound identically at identity and serve, with an allowlisted runtime receipt if config-to-environment evidence is insufficient. That integration is not silently claimed complete here. No native occupancy, corpus, request, quality or scientific acceptance change is proposed.

## Reproduction

From this directory: `PYTHONPATH=.:tests python3 -m unittest discover -s tests -v`.

Before/after logs are `evidence/readiness-{red,green}.log`, `exception-preservation-{red,green}.log`, `cleanup-race-red.log`, `cleanup-preservation-green.log`, and `cleanup-context-green.log`. A local-review edge and its correction are preserved in `ambient-exception-red.log`; its observed bad result was a later port-in-use error after the cleanup exception had been suppressed, not an accepted successful run. `baseline/` retains the exact trial8 supervisor/helper.
