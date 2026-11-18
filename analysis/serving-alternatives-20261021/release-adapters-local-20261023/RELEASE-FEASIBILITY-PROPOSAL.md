# Proposed bounded control-only post-load release screen

Status: **LOCAL PROPOSAL ONLY**, no experiment freeze, assignment, GPU slot request,
launch or retry authority. A new identity and exact reviewed bundle would be needed.
Historical trial10 identifiers in intended-command receipts are evidence references,
not successor identity or authority. No executable Kaggle packet is built here.

## Question and scope

Can frontend-first shutdown, within one absolute release/cleanup deadline, produce
an observable post-load state satisfying all original release predicates for the
shipping unpruned Pennyroyal v2.5.3 control? Keep the exact control runtime, model,
NEXTN3/draft4/FR-Spec/fp8KV, health configuration and prefetch behavior.

One control load, actual health200, then shutdown/release. No replay window,
candidate load, corpus cases, rate ratio, quality or adoption claim. A post-load
pass would not qualify the post1500s replay shutdown boundary or a paired run.
The prior standalone control remains valid only for trial10's own run.

Attribution assumptions: a short resource series can distinguish a transition
within the observed deadline from persistent ineligibility. It cannot, by itself,
attribute high FB memory to driver lag, a leak or an unseen holder. The new native
targeted shutdown intervention changes shutdown ordering; GPU and host outcomes
must still be observed independently. Host cache/reclaim-list state is not adjusted
or credited as a mediator to make the gate pass. No threshold is fitted to trial10.

## Proposed absolute budget

Mandatory platform3900s; cooperative supervisor3600s including300s cleanup reserve;
work ends at3300s. This is a fresh proposed cap, not the former3h comparison cap.

- Early actual RTX/device/driver and initial resource receipts:30s.
- Offline control preparation:480s.
- Interpreter/source/health-configuration identity:60s.
- Native launch, registration and actual health200:2400s. Registration10s is inside
  this phase. Keep generation-health1/timeout20 and25s probes, bounded2s cadence,
  only the previously qualified empty503 startup retry shape; other errors fatal.
- Shutdown plus resource series:180s total, graceful120s/fallback reserve60s.
- Receipt verification/local summary:60s.

These work ceilings total3210s, leaving90s work slack; unused time does not extend
any phase. The platform cap remains an external containment limit, not proof of a
hard cooperative syscall bound. Query stdout/stderr/identity hashing and cleanup
must all fit the absolute end; late or incomplete evidence cannot qualify.

Grounding: trial10 prepare56.175s, identity6.925s, server-spawn to health-ready
1160.329s, whole attempt to ready1270.939s. The coordinator's earlier guard load1997s
is additional variability context, not a calibrated percentile for this control.
The proposed2400s startup is a bounded failure point; it does not guarantee fit.
The evidence reference for trial10 phase clocks is bound in LOCAL-DELTA.

## Registration and release observations

Publish a complete immutable wrapper/frontend registration at spawn, using the
actual prepared receipt, shebang/interpreter, entrypoint and pinned installed
sources. Registration is not readiness. Only after actual health200 does the
controller bind the live process tree and exact registration to its owned Popen.
No signal is permitted for missing/ambiguous/reused/source-drifted identities.

Record pre-shutdown observations diagnostically; a loaded device is expected to
fail release eligibility. Then send TERM only to the bound native frontend.
Wait for native/owned evidence; fallback only verified registered identities under
the same end. Native text logs can show draining/crash markers and process exit;
they do not directly expose watchdog-stop or GPU context destruction.

During the180s release phase, use2s minimum pause between samples and an explicit
sample ceiling of25s (two10s GPU/apps commands plus bounded file/port/native work),
always capped by the remaining graceful/absolute end. GPU and apps commands are
the exact argv in INTENDED-COMMANDS.json. They are sequential, not a simultaneous
snapshot. Each query/file/artifact/port observation retains its own start/end.
If a sample times out, missing fields remain ineligible; preserve raw prefixes and
query intents. Query descendants must not survive a killed probe worker.

Require original GPU memory<=2048MiB, empty apps, empty verified owned processes,
reclaimable host headroom>=140GiB using only the original max-current+inactive_file
formula capped by MemAvailable, free port8001 and /tmp>=100GiB. No active_file
addition, cache clearing/reclaim command or GPU reset. Unknown identities never
authorize signaling unrelated processes. Query subprocess intents carry process
keys; final verification must reject unresolved query cleanup/evidence gaps too.

Large raw evidence is bounded to8MiB per local artifact and referenced by exact
SHA/size rather than IPC payload. A size cap/incomplete capture is a failed gate,
not silent truncation. Primary native logs remain preserved; a failed snapshot
does not qualify them. Artifact verification is part of the final gate, not merely
an upload inventory. Platform terminal status, cleanup and typed quota would be
collected under a future assignment and closed by the coordinator.

## Discriminating outcomes

- All original predicates pass: establishes post-load release for that exact
  control/shutdown policy and observed session only; permits planning the still
  separate replay/paired qualification. It establishes no rate/adoption result.
- GPU/apps/process/port release succeeds but host remains below140GiB: retain that
  GPU release evidence separately; candidate launch remains blocked. Graceful
  shutdown or waiting is not asserted to fix the host gap.
- Device or process state remains failed/unknown at the end: preserve the bounded
  series, primary and secondary cleanup failures; no retry or candidate fallback.

The140GiB load-derived rationale remains missing. Its original contract stays
binding. If that gate cannot be justified/satisfied, the coordinator must make a
separate design judgment with new evidence before any same-session comparison.
This proposal does not choose a replacement.

## Remaining before any GPU packet

Review this local integration/proposal first. A later exact packet would need a
new identity, runtime-relative rebinding of the accepted-core loader, current
allocation/source/transport/registration bindings, reviewed enabling of only the
two GPU command allowlist entries, an experiment card and fresh serialized launch
authority. None is created, inferred or requested by this directory.
