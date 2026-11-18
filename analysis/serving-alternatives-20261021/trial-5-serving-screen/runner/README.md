# Bounded trial-5 process runner (preparation only)

No installer, model path, GPU cap or launch defaults are supplied. The owner plugs in the exact reviewed commands and cap. `CANDIDATE_FEASIBILITY` is the current purpose: exactly one candidate load and functional screen, no control, replay rate or adoption decision. `PAIRED_CAPACITY` remains an explicit separate interface, never selected implicitly.

Run the CPU fault suite:

```bash
python3 -B -m unittest discover -s analysis/serving-alternatives-20261021/trial-5-serving-screen/runner -p 'test_*.py' -v
```

`supervisor.py CONFIG OUTPUT --assignment ASSIGNMENT` requires GPU_SCREEN mode and a serialized assignment matching the canonical config SHA and exact total cap. The notebook additionally binds the full embedded source map to `assignment.sources_sha256`, computed as SHA256 of `json.dumps({name: SHA256(bytes)}, sort_keys=True).encode()`. CPU fixtures use the separate `CPU_FIXTURE_ONLY` mode and `--cpu-fixture`; that flag rejects GPU mode. An existing supervisor trace prevents an accidental retry into the same output. A notebook extraction uses exclusive file creation as another one-shot boundary. Remote submission anti-retry, quota snapshots and source/readback verification remain the parent launcher's responsibility; this directory makes no platform API calls.

Config keys:

- `run_id`, `mode`, `purpose`, absolute `cwd`, explicit `total_seconds`, explicit `cleanup_seconds` reserve.
- `hardware`: `argv` must equal GPU_ARGV in GPU mode, `seconds`, `minimum_memory_mib` at least 90000. Actual single-device nvidia-smi identity is checked before prepare/identity/server; output and driver/UUID are retained. Host boot, memory and cgroup resource records are retained.
- `candidate` (and `control` only for paired capacity): commands `prepare`, `identity`, `server`, `functional` (or `replay`), `qualify`. Every command is `{argv: [...], seconds: NUMBER, env: OPTIONAL_STRING_MAP}`. No shell is used. Secrets should not appear in argv/config. The trace records argv and config hash, never environment values.
- Arm `ready_url` must be loopback HTTP; `ready_seconds`, `request_seconds`, `drain_seconds` are explicit. `server.seconds` limits startup through readiness, rather than the later declared functional/replay allowance. The server's absolute lifetime is the overall work deadline. No phase extends that overall deadline.
- `arm_order`, when present, must exactly equal `['candidate']` for feasibility or `['control', 'candidate']` for paired capacity.

Every subprocess receives `SCREEN_PHASE`, `SCREEN_DEADLINE_MONOTONIC`, `SCREEN_OVERALL_DEADLINE_MONOTONIC`, `SCREEN_OUTPUT`, `SCREEN_RUN_ID`. Functional/replay also receives `SCREEN_DRAIN_SECONDS`. The client must honor this absolute deadline and bound drain by it; the parent supervisor independently kills a hung client at its phase deadline. `qualify` must exit nonzero for incomplete/failing functional or replay evidence. No config scalar is interpreted as successful qualification.

Health requests execute in their own killable process groups. Both healthy and HTTP-error bodies are bounded by the same absolute request deadline, so trickling cannot reset it. Each helper stdout file starts with one JSON status/headers line, followed by exact raw body bytes up to 65537 bytes (the extra byte detects an over-limit body). Partial trickle bytes are flushed before a timeout kill. Only ECONNREFUSED while an already-started server binds may repeat the health check. GPU polling cadence defaults to two seconds and cannot be shortened; CPU fixtures may explicitly shorten it. There is one server spawn, no model restart, no functional retry. An unrelated existing port listener, HTTP status errors and other transport errors stop the arm. Loopback redirects are rejected.

Each command starts an isolated process group. Successful command parents that leave descendants are also cleaned up. Cleanup sends TERM then KILL within the reserved total budget and records live `/proc` group members. Candidate starts only after control qualification and cleanup; server death during client execution fails the arm. This covers ordinary inherited process groups, not malicious processes that deliberately escape into another session or a kernel that cannot schedule/kill processes. External platform hard timeout remains an independent required boundary. No raw native event merge or PID rewriting is performed here.

`build.py SPEC DESTINATION [--assignment ASSIGNMENT]` builds locally only. SPEC contains `config`, exact `metadata`, and `files` mapping safe relative packet paths to local source paths. Tested image/machine/competition envelope is enforced. The reviewed supervisor/helper are embedded automatically and cannot be overridden. Without assignment, notebook execution stops before file extraction or any GPU command. A prepared packet is structural evidence only: runtime commands/assets, actual successful load, fit, source/readback/quota and serialized launch still need owner review.

Reused evidence: trial-2's successful image, machine shape, competition source and early hardware guard; trial-3 runtime identities/argv references only. The old 3-hour default, fixed-gap traffic, production occupancy bands and historical invalid rates are not imported. Historical installer notebooks show the offline Pennyroyal installation method, but no installation is fabricated or run by this runner work.
