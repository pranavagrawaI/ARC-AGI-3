# Qualification: local PASS, OpenRouter INCOMPLETE, launch HOLD

Implemented separately from retained q6:

- Compact evidence packet; changed-only prospective experiment records with
  original observation identity and explicit stale-record marking.
- Frozen encoder checked after each action, queue interrupted before another
  action on mismatch/unresolved evaluation. Existing terminal/no-impact rules
  retain precedence. Disposable checker with output/CPU/memory/wall bounds,
  separate result channel, and race-safe parent-death termination.
- Designer-player reasoning context only in mindset arm; loop guidance in both.
  Transport allowlist forwards selection and worker records its actual arm.
- Manifest-derived session validator and exact benign shutdown-log classification;
  all other integrity/lifecycle gates retained.

Local checks: 29 solver/worker tests plus four packaging tests, including actual
adapter/shared-transport/worker subprocess execution from the notebook's extracted
payload. Installed Kaggle request construction is captured without network;
competition attachment, RTX PRO6000, private/offline settings and timeout pass.
One existing set_dynamic deprecation warning remains; no new test failures.

Notebook: probe.ipynb, two isolated arms on public ls20, 30 actions and 1800-second
admission horizon per arm, pending worker allowance900s, launcher2700s, session3h.
Checkpoint/serving baseline and reasoning allowance retained; MTP off. Source
contract df8bf4fa5379a6e069157a96762b6fdcc967599efa59dc8597834d4ec8ee8847.
Model manifest unchanged. No upload, push, submission, or Kaggle launch performed.

OpenRouter results retained under analysis/deployment-probes:

| Run | Observation | Disposition |
| --- | --- | --- |
| 20260907-designer-player-openrouter | two404s; arm flag dropped by transport | Invalid; fixed flag forwarding and regression tested |
| ...-v2 | loop429; mindset7 responses but inherited150s worker window cut decision off | No matched comparison; final response completed but delivery failed |
| ...-v3 | loop2 completions then DNS failure; mindset12 completions, no action/experiment | Live behavior unqualified; no further paid attempts |

Across successful responses, reported cost $0.11033575; conservative reservation
$1.11303052 within $2. Final v3 successful responses reconcile14=14 recorder calls.
Earlierv2 seventh response after consumer termination is explicitly retained as
completed-but-undelivered, not silently assigned to recorder or scheduler failure.

The policy was present in v2/v3 prompts but the requested interfaces were unused
within the diagnostic window. No causal speed or gameplay gain, summary-quality
claim, live interruption qualification, or generalization claim is supported.
LAUNCH.json has launch_ready=false and openrouter_qualified=false. The notebook
is built and locally checked, not recommended for GPU launch yet.
