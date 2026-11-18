# Superset containment review

Status: local stdin defect repaired; Kaggle execution and split transport unqualified. No launch authorized by this document.

Review coordination: workspace e3401039-8f11-4473-821f-c114a2bfd55b; reviewer terminal 2ade96aa-e767-4645-857e-e1e8a6773de1; coordinator terminal 396b418e-500e-492a-b619-2a6fa4bf0373.

## Verified by coordinator

Both execute_inspection and CandidateRPC inherited caller stdin before repair. Real subprocess regression tests failed with synthetic CONTAINMENT_REVIEW_STDIN_CANARY. bounded_run now explicitly uses stdin=subprocess.DEVNULL and close_fds=True. Both regressions then passed: stdin EOF, FD20 closed, synthetic environment credential absent. verify_local.py passed 46 tests and confirmed prior v4 frozen sources unchanged.

## Independent reviewer report

Reviewer independently reproduced both stdin leaks and reran the repaired two-path regression: 2/2 passed. Additional reviewer-reported local probes found an outside sentinel inaccessible, synthetic credential environment absent, FD120 closed, host-loopback listener unreachable, and only namespace PIDs visible. These additional results are reviewer self-reports, not coordinator-reproduced checks or comprehensive sandbox proof. No overlapping edits or paid launches reported.

## Boundary decision

Keep current constructor ABI and outcome criteria. Local bwrap is the currently exercised execution path. Kaggle may perform inference only if a practical authenticated transport is established. Do not upload evaluator_payload, the full case, verifier or suffix. Only validated generation requests built from compact_view/inspection_view and resulting visible tool messages may cross. Kaggle must return inert output, never execute generated calls. Freeze and suffix assessment stay local.

No reusable authenticated relay has been found. Do not build a general tunnel/control plane merely to avoid a bounded platform feasibility check. A root-chroot plus permanently dropped UID/groups/capabilities, no_new_privs, narrow seccomp, closed descriptors, no host proc/devices and exact runtime allowlist is a conditional alternative, NOT an implemented or qualified boundary. Historical CPU capability evidence is not current GPU qualification. Existing hardened_host lacks the required filesystem confinement.

## Split transport acceptance obligations

- Authenticated TLS, expiring run/session identity, exact request ID and canonical body hash bound to response.
- Single in-flight generation; durable remote admission before inference. Repeated IDs cannot regenerate; changed-body duplicates rejected. Duplicate receipts cannot execute tools twice. Uncertain admission stops without retry.
- Fixed server dataset/version, tokenizer/template, reasoning-off, tool parser and token/context/time/call bounds. Qualify actual vLLM fields, not OpenRouter-specific validation.
- Durable local and remote raw payloads, usage and lifecycle. Distinguish local pre-send failure, remote admission, completed receipt and unknown status. Count queue/relay wall time and report GPU usage separately from tokens; no invented API cost.
- Relay credentials absent from model prompts and untrusted child environment, stdin and inherited descriptors.
- Expiry/cancellation rejects later work, stops inference, revokes relay and verifies server/job exit; local timeout is not remote shutdown.
- Offline qualification: two real isolated tool turns and fixture submission/freeze/suffix, plus wrong session/hash, stale/duplicate/reordered receipts, lost response after admission, missing usage, drift, oversized output, missing relay and cleanup-timeout faults. Fail closed before subsequent side effects.

## Remaining availability limitation

Current resource bounds are per-process/per-file, not aggregate descendant PID, memory or tmpfs quotas. Reviewer did not run destructive resource bombs. Evidence confidentiality and hostile-resource availability must be reported separately.

## Scope

Completed action-only checkpoint and observation/action pilot reports are not construction-policy evidence. Neither their results nor this containment review establishes model construction competence. Remote containment/transport qualification and actual vLLM request qualification remain outstanding.
