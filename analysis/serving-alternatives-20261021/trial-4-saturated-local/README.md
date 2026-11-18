# Local saturated capacity design

Start with EXPERIMENT-CARD.md, then QUALIFICATION.json, REVIEW.md and sensitivity.json. Nothing in this directory launches a GPU, requests a slot, installs a serving runtime or loads a model. All apparent speeds are synthetic fixture outputs, not engine measurements.

From this directory, using already cached CPU dependencies:

```bash
uv run --offline --no-project --python 3.13 --with tokenizers==0.23.3 --with jinja2==3.1.6 python -B qualify_local.py
uv run --offline --no-project --python 3.13 --with tokenizers==0.23.3 --with jinja2==3.1.6 python -B sensitivity.py
```

Both commands impose a local 1800s hard alarm. qualify_local.py checks every original corpus manifest file, the immutable catalog, preserved prior trials and all public contract tests. It records a scoped AST/statement-coverage/CRAP diagnostic without a threshold, and archives a raw synthetic control/candidate event pair in cpu-proof/. It is not a launch verifier or adapter qualification.

The proof preserves client/native event JSON, raw synthetic response JSON and request-to-corpus key mapping. Wire and rendered-prefix references resolve through fixtures.Reader against actual local source bytes and the pinned processor template; response artifacts contain synthetic fixture outputs. Reconstruct Reader.requests from request-keys.json and Reader.blobs by canonical-encoding responses.json values, then call capacity.compare on the archived control/candidate events. No simulation is needed for this readback.

Raw contract: each client/native event has finite host-monotonic t, contiguous seq, stable positive pid and boot_id. First records carry session_id/gpu_uuid; source captures start before any traffic, emit heartbeats at most 2s apart and close explicitly. Client replay_start/measure_start/measure_end derive the fixed window. send joins immutable key and actual wire_ref; response joins raw response_ref and status. Native accept joins rid/nid/response_id; prefill carries source image hashes, processor grids, prompt token count and actual render_ref; decode_start must mean first committed target output; finish carries committed output_tokens. Missing, abort/preempt/retract or unknown events invalidate. A real adapter must establish these semantics rather than rename a gauge.

capacity.compare(control, candidate, catalog, artifact_reader) is the public reducer seam. artifact_reader returns actual bytes for each recorded reference. No producer scalar for occupancy, warmup, mix or fidelity is trusted. The catalog is hard-bound by SHA256 and both actual payload and prefix artifacts are checked against it. Source sequence, artifacts and identity remain a trust boundary that future capture/adapters must qualify.

Sensitivity varies prefill rate, recorded cache-hit reuse and hypothetical decode speed separately. Its decode curve and serial-prefill approximation are uncalibrated; this is a falsification check on the design, not a forecast of either stack. CPU_FIXTURE_ONLY is retained on outputs. No GPU cap is proposed and no files are frozen for launch.
