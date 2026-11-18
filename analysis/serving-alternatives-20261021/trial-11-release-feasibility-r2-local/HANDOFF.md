# Release11 superseding integrity correction — UNASSIGNED

Complete exact packet: trial-11-release-feasibility-r2-local. Original submitted
trial-11-release-feasibility-local remains byte-exact, including freeze, card,
payload, proofs and failed identity receipt; PRESERVED-R1.json binds them.

- Freeze: `b777001d4a61e15f8292bf391257aa61ea195657bba0bc8fe0df39aaab595dba` (49 current files +117 external references).
- Sources: `75e663bb3653936b10ae3471f766d6fa1d12d835875d29ace41ad9cf13f83baf`.
- Config: `27898a12cc9d0d8d509f9948b884d9988e4739e408838b2e347d2f09d8f1d27d` (UNCHANGED).
- Policy: `ef6d9a55be07418ba06b96c2f432775d63f14cfbc287efa95a179988a25399e5` (UNCHANGED).
- Card: `ed1d0d9431582341a6e22e62acaad4732ea2c93590a4dbbf9937cb05a3dd454f`.
- Full qualification: `19b578daa15221680db2b62dbe082fd6bc8526f6f1a09745cab1b7d47656d3f5`.
- Correction proof: `d6de9a0fd68b7ea675387697c9f0e19debcae45c0a8db27f4c74b6e0fd5a45c1`.
- SDK proof: `c7e9f07fce5689c34ced1a4eb61d2d35de0a14f32c6c463764076546ff946832`.

Only runtime change: adapters.prepare_controller rejects either command.ok false
or cleanup_verified false before apps/receipt-worker admission. Only payload
files adapters.py and its runtime-pins.json changed. No error-string taxonomy.
Clean pre-execution ControllerProbeDeadline remains distinct. Successful captures
whose resource values are ineligible still poll within the original absolute end.
The proposed cap remains3900 platform /3600 cooperative /3300 work /300 cleanup.
One exact shipping control load, actual200, release only; no replay/candidate.
Same unassigned Kaggle identity: `pranavagrawai/serving-release11-20261023`. No attempt existed on the
original or this superseding packet; no old assignment is reused.

RED: both real-CPU containment-rejection tests dispatched GPU then apps and reached
forbidden receipt-worker admission despite GPU okfalse/containmentfalse and clean
cleanup. Logs red-query.log and red-release.log and full raw receipts preserved.
No full release pass was observed or claimed in those witnesses.
GREEN: both now retain the rejected GPU receipt, dispatch GPU only, start no worker;
release returns false with unclean_or_failed_query and zero samples. The separate
resource witness uses4 successful real guarded query captures/2 samples: GPU5000
ineligible followed by GPU0 eligible. It ends by an explicitly injected CLEAN
pre-execution deadline, retaining the existing deadline branch; no full release
pass is claimed. All451 red/green artifact references reverified; exact member
inventory in EVIDENCE-INVENTORY.json.

Affected/full qualification:65 accepted regressions ran against THIS packet's
actual adapter/capture/controller/dependencies (verify_runtime.py), plus17 packet
tests including these3 regressions. All passed in16.433s; clean-query
phase expiry and original deadline/identity/cleanup tests remain unchanged.
The qualifier explicitly binds current module file paths rather than rerunning
only the prior adapter. Test fixture/source dependencies and original root review
receipts are in SOURCE-LINEAGE.json. Development failures are preserved and
explained in CORRECTION-PROOF.json, not silently discarded.

Actual installed Kaggle SDK serialization intercepted with zero network calls/
attempts.22 decoded payloads and config exact; compressed unassigned notebook
68841B; new source hashes bound in a local-only fixture.
Unassigned/local-only fixture refusal before extraction and missing/false/stale/
wrongfreeze/oldidentity/one-attempt/fresh-check lifecycle tests pass.

Reproduce locally with a fresh output directory, CPU/loopback only:

```bash
cd /home/pranav/projects/arc-agi-3/analysis/serving-alternatives-20261021/trial-11-release-feasibility-r2-local
timeout 600s python3 verify.py --output review-independent-FRESH
```

Thresholds2048MiB/140GiB/100GiB, original accounting/filter/cache behavior, health
1/20+25s and all science/caps unchanged. Actualnvidia/filter/Kaggle/native release
and140GiB eligibility/rationale remain runtime questions. Prior conservative
identity abort remains card-disclosed; exact missing/proc field unknown. CPU
loopback wiring can fail actual local resource thresholds and does not qualify
GPU release. Original trials/partial archive/unavailable R5 first return preserved.
NoGPU, network/API/auth, assignment, launch, staging or commit. NEW exact review
and serialized assignment remain required. Do not execute launch.py now.
