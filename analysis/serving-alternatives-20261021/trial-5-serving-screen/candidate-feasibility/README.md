# First actual candidate feasibility packet

Read EXPERIMENT-CARD.md and review-packet.json. This directory prepares one private GPU feasibility run; it has NOT launched. No speed/adoption/production claim is possible. No paired native overlay is enabled.

The latest packet directory is recorded in QUALIFICATION.json. packet-r1 is retained as superseded preparation evidence; use the final review packet only. Each unassigned notebook raises PREPARATION ONLY before extraction or GPU work. launch.py also refuses a missing, false, stale or mismatched coordinator assignment and any prior launch-attempt marker. No slot-confirmation.json exists.

Reproducible local checks (no GPU):

```bash
python3 -B -m unittest -v test_client test_launch
```

The separate check_long_headers.py tests an actual62-second wait before HTTP headers; it failed at60.125s before the timeout repair and passed at62.075s after it. Avoid repeating it without a new transport concern. Runner tests and independent rechecks are described in REVIEW.md; preserve its unresolved timing anomaly rather than relabeling a passing rerun as the only outcome.

To rebuild preparation after a reviewed edit, retain the old packet and use a new destination:

```bash
python3 build_inputs.py
python3 build_packet.py packet-rNEW
/home/pranav/.local/share/uv/tools/kaggle/bin/python verify_packet.py packet-rNEW
```

verify_packet.py exercises the ACTUAL installed Kaggle SDK save constructor with its endpoint intercepted. It makes zero network/GPU calls. It checks the5400s timeout, tested image/machine/competition envelope, minimal runtime/model sources, case hashes and source/config binding. A later arming step adds only the reviewed coordinator assignment; runner/build.py verifies the same embedded-source digest/config/cap. launch.py independently checks the reviewed preparation freeze, fresh typed quota and terminal account sessions before creating its exclusive one-attempt marker and making one push with timeout5400. It never retries.

Future coordinator slot-confirmation.json must contain launch_authorized:true, coordinator_thread_id d74e17d0-679e-4189-a52b-bfcc9e6895be, approved_freeze_sha256, sources_sha256, config_sha256, confirmed_utc (less than600s old), hard_timeout_s5400, supervisor_timeout_s5100 and serialized_assignment. The read-only earlier quota audit is not clearance. The preparation freeze protects the launcher/builder, actual embedded inputs and review evidence; it does not itself authorize launch.

Candidate setup checks actual RTX/CUDA/host/disk before install, pins compressed image layers/applier/patch identity, validates model version files/sizes/configuration and checks installed overlay/frontend source bytes. It does not read/hash all180GB of mounted model weights a second time. The real server's model load, image handling and native behavior remain unproved until this run. Both source and model-layout failures are useful preserved feasibility outcomes.

All six full request bytes are embedded and revalidated. HTTP success is not sufficient for long-input qualification: raw prompt counts must equal the normalized local expectations; Python call code is parsed but never executed. A mismatch remains FUNCTIONAL_PROMPT_MISMATCH evidence with no rate/fit promotion. Non-200 headers immediately halt further admission while the bounded error read retains bytes. Qualification re-reads actual request/HTTP/response artifacts instead of trusting the client's summary verdict.

After any future single launch, preserve remote metadata/source readback, actual output files and typed quota delta. Read the raw server/request errors before interpreting the verdict. No failure or partial success authorizes a retry, control load, rate run, hidden submission or promotion.
