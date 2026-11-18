# Release11 bounded preparation checks

Failure scene: platform consumes a full session yet neither trustworthy release
eligibility nor a complete failed-boundary receipt can be established.

- Identity validation crosses the gate clock/cancellation boundary and starts work
  late: accepted R5 checkpoint/final-time check, deterministic red/green and full
  suite retained. Syscall scheduling remains cooperative; platform3900 cap required.
- Short-lived worker dies after thread exec cleared PDEATHSIG: long controller
  owns query pidfd before execution; workers start after queries reaped. Retained
  cleanup errors stop dispatch. No new filter layer. Actual CLI compatibility is
  unknown: pre-load exact filtered queries must fail closed, not bypass enforcement.
- Old group TERM kills detokenizer before native watchdog shutdown: native wrapper
  is excluded from phase-command cleanup registry; frontend-only TERM and verified
  registered pidfd fallback. Failures before binding cannot justify unknown kills;
  surface unqualified cleanup/platform containment instead.
- Native children or observer namespaces drift: live registration after actual200,
  strict original process identity/ancestry/source checks, actual cgroup/mount paths,
  fresh fd2 binding. Unsupported runtime layout is an explicit abort, not a guessed
  path. Actual remote integration remains unproved.
- A loaded device looks released because apps list empty: retain full FB-used query,
  exact initial UUID, all six original predicates and per-observation clocks. Host
  headroom remains independent; no active_file addition or empirical threshold fit.
- HTTP503/application-startup gets mistaken for ready: unchanged pinned empty503
  retry shape only before actual200, original end/25s probes/2s cadence, explicit
  raw1/20 and effectiveTrue20 identity/serve receipts. Body/status/error preserved.
- Raw evidence disappears during an assertion/cleanup race: accepted R6 test-only
  pre-cleanup/final persistence plus retained verified test pidfds; original failed
  R5 full-suite and missing first receipt stay preserved. Packet CPU lifecycle
  copies output before assertions; early development failure is retained as a log
  with its first output gap, not reclassified.
- Mutable quota/session files invalidate the manifest between preflight and push:
  explicit freeze-input list excludes operational outputs. Full mocked launch
  sequence tests actual fresh_checks writes; input drift still blocks before push.
- Credential/API error or stale assignment triggers retry: launch one-attempt marker
  before the single push, explicit exact source/config/card/policy/new authority,
  bounded error-response capture, no retry/auth repair. Local SDK proof intercepts
  SaveKernel and blocks sockets, so it cannot establish remote acceptance.

Local tests cover construction, actual CPU loopback/registration/cgroup/fd2 plumbing,
resource-failure reporting and empty final owned groups. They do not establish GPU
release or satisfy140GiB on Kaggle. No launch is performed by this card/proof.

Superseding query-integrity failure: a failed containment/command with verified
cleanup must abort before apps/worker/future release samples. QueryIntegrityTests
injects the actual containment validator failure into real guarded CPU capture,
checks okfalse/containmentfalse/cleanuptrue and no later dispatch. Successful
captures with merely ineligible GPU values continue within the original release
end. Runtime correction uses boolean fields, never error-message matching.

Notebook runpy context failure: actual new outer notebook must resolve extracted
package imports from a foreign cwd/clean sys.path before any runtime phase. The
accepted root-path/source-origin guard is exercised through real packaged main
and Runner at a safe CPU boundary, plus wrong preloaded entry/dependency and
foreign shadow controls. Unassigned/local-only refuse before extraction. No
hostile import-hook/live-code attestation or actual GPU release is inferred.
