# tractability-1

Question: Can observed winning frames and a real wall-clock action reserve recover useful commitments from the saved RE86 and LS20 stalls without breaking a short successful RE86 decision?

Posture: Diagnose capability versus missing evidence and decision control; test small generic repairs before another full-game run.

Contrast: Four arms: unchanged saved request; terminal presentation; time-aware decisions; both. Three exact-call packets (RE86 call70, LS20 call75, RE86 successful call36), two replicas, six matched blocks of four arms. Filesystem restored from hash-committed snapshots. All arms share the original model, sampling and real tool implementation. Only declared treatment prompt additions and time allocation differ. This is a request-level continuation, not a reconstruction of the NOOA event graph.

Prediction: Terminal context should reduce confusion between the final pre-winning frame and observed winning state on RE86. The time reserve should produce an action before the external deadline on both stalls. Their combination may help; emitting an action alone does not establish solving.

Measure: Per-case raw requests, responses, tool transcripts, interrupted-provider receipts, completion usage, elapsed time and action commitment. Reconstruct the executed prefix in the matching offline engine and require exact public-observation agreement before evaluating proposed actions. Record legality, interior changes, deaths and level completion separately. Compare the successful branch with its recorded LEFT LEFT completion. Inspect unsupported claims and repeated computations in raw reasoning. Useful progress requires concrete game evidence; an arbitrary changed grid is not a solve.

Cap: One private Kaggle session, exactly one RTX PRO 6000, hard timeout3600 seconds (at most1 GPU-hour), soft stop3300 seconds. At most24 cases,180 seconds percase including tools and final demand; at most25 model calls and25,000 completion tokens percase. Admit calls against requested maximum output and reserve unreturned requests at that maximum; actual usage replaces reservations only with a completed receipt. Time arm reserves min(120 seconds,25% of usable time) for action and5 seconds for host return; final demand maximum2048 tokens, thinking disabled, take_actions only, tool_choice auto as in the frozen lean policy. All arms reserve the last two calls at the common call ceiling for action demands, and may use a final demand if the next full exploratory request would violate token admission. At most4 simultaneous case streams; no competition submission, no paid external provider, no automatic session retry.

Stop: Finish after24 cases; stop spending after any packet integrity mismatch, model/serving identity mismatch, unbounded call admission, missing durable attempted-request record, or an experiment-wide deadline. Individual legal refusals/tool mistakes remain visible and bounded; failed requests retain unknown usage and consume maximum-output reservation. A unavailable/mismatched environment makes that branch's gameplay outcome unevaluable rather than successful. Preserve all incomplete cases on soft stop.

Invalid if: Saved request or workspace hash mismatch; changed undeclared sampling/model; fabricated tool output; engine prefix differs; arm request construction changes more than declared treatment; serving startup or load makes a matched block uninterpretable. Unknown canceled-provider work limits token efficiency comparisons and is not confirmed cancellation. Two replicas cannot establish broad generalization or a full-game score improvement.

Result map: Verified useful progress in both replicas and preserved successful control -> retain repair and test another bounded case. Actions without verified game progress -> operational repair only; investigate missing abstraction or evidence. No improvement despite correct presentation/clock -> stop treating these as sufficient reasoning fixes. Mixed or invalid results -> preserve source and receipts, identify the specific confound, do not retry this run automatically. Full-game promotion remains unproven.

Outputs: This directory's kernel/source payload identities, packet-preparation.json, local qualification logs, launch/quota receipts, output/cases/* and output/summary.json. Raw records precede any verdict.

Authority: Pranav: "I agree with your plan, do it. start with terminal-frame presentation and time-aware decisions" (2026-10-07), within the existing22 GPU-hour campaign. This card fixes the run; it is not the authority itself.
