# Result — stopped before factorial comparison

## Verdict

INVALID as an A/B/C/D comparison. Arm A made eight successful paid Actor calls but committed no action. Before call nine was forwarded, the relay rejected NOOA's separate summarizer request as `missing loop instructions`; the card therefore stopped the run before B, C, or D.

This is a probe-classification defect, not evidence that the candidate lost its stable instructions. Every one of the eight retained Actor request bodies contains `<decision_workflow>`, `<arc_skill>`, and `<prediction_workflow>`. The unretained ninth request was initiated exactly after Actor prompt usage crossed the configured 19,000-token summarization threshold. NOOA's summarizer is a separate no-tools agent and correctly does not render the parent Actor's treatment blocks. The relay had incorrectly required Actor-only factors on every request.

## What was established

- Frozen candidate contract: `fe37d9cc413533a59f50b3fce651a36fe35a1aa884beb1c7fbafbe54b3b1261b`.
- Reasoning remained off on all eight responses: zero reported reasoning tokens and no reasoning text.
- Provider pin and sampling contract were accepted on all eight calls; every response was HTTP-successful and ended with a tool call.
- The new 19k compaction trigger was reached in real traffic. Prompt tokens rose from 12,552 to 22,081 before the background summarizer request was attempted.
- Prefix caching was active: 103,488 cached prompt tokens across 138,810 prompt tokens reported by the eight responses.
- New cost instrumentation wrote seven completed parent-call records and nine Python-cell timing records before abort. Model wall recorded before abort was 26.24 seconds; Python tool time was 0.034 seconds. TTFT remained explicitly unavailable under non-streaming transport.
- The unchanged A arm still spent eight serial CodeAct calls inspecting the initial state and produced zero actions. Its eighth response finally attempted notes and a baseline helper. This is useful negative evidence about A, but it cannot compare the treatments that were never run.

The apparent eight-response/seven-model-record mismatch is attributable to the probe abort racing the worker after the rejected summarizer request. It reinforces the INVALID verdict; it is not reconciled as a successful arm.

## Cost and bounds

Reported provider cost was USD 0.034395. The run stopped after 36.82 seconds of arm wall time and did not consume the remaining arm or spend allowance. No forwarding retry or relaunch occurred.

## Repair

The packaged probe now classifies requests by their tool surface: requests exposing `execute_python` are Actor work and receive A/B/C/D factor checks; no-tools requests are summarizer work and retain reasoning/sampling/budget checks without requiring parent-only context blocks. Purpose is recorded with each physical request. A unit test covers this boundary.

That repair has not been re-run externally under this card. The Kaggle notebook should use the repaired classifier and preserve separate Actor/summarizer accounting.
