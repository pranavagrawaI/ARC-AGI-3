# Critic framework screen v1
Question: Does the framework prompt improve warranted criticism and restraint on development cases?
Posture: Diagnostic evaluation of a prompt repair.
Contrast: Old critic and new framework critic; identical evidence per case, alternating order; six cases, one sample each per arm.
Prediction: New critic identifies scope/objective overcommitments, permits sound exploration and active repair, and avoids certifying the inconsistent reachability result.
Measure: Raw response fields graded against CASES.json. Detection, unsupported objections, unsupported endorsements, execution validity, latency, and cost separate. Empty no_concern is not detection.
Cap: 12 calls; $0.25 conservative total reservation; 480 seconds overall admission window; 30 seconds per HTTP request; 384 total generated tokens per call; zero environment actions. Same existing Qwen endpoint/provider, thinking off.
Stop: Complete 12 calls; abort on transport, provider, reasoning, cost-accounting, or finish-status failure. Preserve malformed semantic responses and continue without retry. No relaunch.
Invalid if: Frozen inputs drift, diagnosis/later outcomes leak into captured input, wrong model/provider, or unaccounted requests. A malformed response is an output failure, never no_concern.
Result map: Better grounded judgments -> candidate for fresh-context/live adoption test; worse -> retain old deployment and inspect mistakes; mixed -> no promotion, identify exact unresolved boundary. These development cases cannot establish generalization or separate-role efficacy.
Outputs: This directory contains exact requests/responses, source hashes, grading criteria, timing and costs. No Actor continuation is generated.
