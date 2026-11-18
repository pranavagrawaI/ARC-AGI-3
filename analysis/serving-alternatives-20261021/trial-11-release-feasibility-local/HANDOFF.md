# Release11 review handoff — UNASSIGNED

The control-load/health200/release-only packet is ready for exact review. No slot or
launch attempt exists. One control load only; no replay/candidate/capacity claim.

Identity: `pranavagrawai/serving-release11-20261023`; run `serving-release11-20261023`;
remote work root `/kaggle/working/serving-release11`.

- Final freeze: `8c2d7a4217b5c75c401b1d302f534d996b8bc270ac4e4b9d0325b48cce5139e2` (46 files +52 external references).
- Sources: `4679507b8e37b4b6c0284e5fc3257e7bba608565258ea8040822582a44e4e060`.
- Config: `27898a12cc9d0d8d509f9948b884d9988e4739e408838b2e347d2f09d8f1d27d`.
- Card: `d850740196a42540f2c9c6d005b6805f130135e0fff96411828265dba3cc3bbd`.
- Policy: `ef6d9a55be07418ba06b96c2f432775d63f14cfbc287efa95a179988a25399e5`.
- SDK proof: `4f1574465847d6106842cbff6eb646fa587fe02c08c0434b184013391df56a8e`.
- Full suite proof: `7363af8c5bf9db53a2b32fb917e052d9a9b612fe592fd24680b6fe480ef97166`.
- Documentation-only final rebind: `11be0ef49cd441b5e5b1bf989ed150feef6d13caa6ae9a41e2fffd46fbb8b35f`.
- Exact integration delta: `dc2f4352f922e063875f807289dac99f5744999082125d57ae9c8ac06052b979`.

Proposed mandatory platform3900s /cooperative3600s, including work3300s and cleanup300s.
The clock begins before hardware/preparation. Fixed ceilings: hardware+two exact
filtered compatibility queries30s, preparation480s, identity60s, readiness2400s,
release180s including diagnostic sample, verification60s; work slack90s.
This is a proposal, not a launch grant or a live quota audit.

The unchanged full rerun passed65 baseline +14 packet tests in15.489s, with452
baseline artifact references checked. Installed Kaggle SDK serialization was
intercepted with sockets/requests blocked: network0, all22 decoded payloads/config
exact, mandatory3900s preserved, correct new metadata/mounts/image/machine.
The UNASSIGNED notebook is68817bytes, below1,000,000;
unassigned and local-only fixture both refuse before extraction. Missing/false/
stale/wrongfreeze/oldidentity authority and one-attempt boundaries have regressions.

The actual CPU lifecycle reached loopback HTTP200, bound live sources/process/
cgroup/mount/native-log evidence and ended with no owned native/query/phase groups.
Its real local resource sample failed the original host/disk gates, so the runtime
result remained NOT_QUALIFIED. This proves wiring/fail-closed behavior, not GPU
release. Exact pinned runtime modules/filter/thresholds are unchanged. See
INTEGRATION-DELTA.json and SOURCE-LINEAGE.json for the precise packaging delta.

Preserved qualification anomaly: qualification-final/ failed one unchanged core
test with UnsafeIdentity('live process identity incomplete: 1409936'); passfalse,
remaining[], cleanup_error null, before original end, all fixtures gone. The exact
missing field/transition was not recorded. Coordinator independently reviewed this
as a potential conservative identity abort, not false clean. Only ONE unchanged
rerun followed; its pass does not explain away the failure. The final card now
includes this risk. FINAL-REBIND-PROOF.json proves only card bytes changed after
that passing suite; all46 final input files and52 references rechecked.

Exact reproduction (CPU/loopback only; use a fresh output name):

```bash
cd /home/pranav/projects/arc-agi-3/analysis/serving-alternatives-20261021/trial-11-release-feasibility-local
timeout 600s python3 verify.py --output review-independent-FRESH
```

The original passing full-suite freeze is archived at
drafts/freeze-before-anomaly-disclosure.json. The current freeze differs only in
EXPERIMENT-CARD.md. No extra reruns are needed for this documentation rebind.
The executable launcher is reviewed as an input; DO NOT run it without a fresh
exact serialized assignment. Live preflight outputs/slot/attempt/runtime outputs
are deliberately excluded from immutable launch inputs.

Actual nvidia query/filter compatibility, Kaggle/native shutdown and GPU release
remain unproved. The140GiB rationale/eligibility gap is independent and unresolved;
no threshold/cache changes were made. POST_LOAD_RELEASE_ONLY would not qualify
release after replay or a paired comparison. Original trials/failures remain
unchanged, including trial10 partial archive and unavailable R5 first-return gap.
Root exact review and NEW serialized assignment are required before any GPU work.
