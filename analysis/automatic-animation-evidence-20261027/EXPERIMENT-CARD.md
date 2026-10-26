# automatic-animation-peak-screen1-20261027

Question: Does automatically showing one real intermediate frame make transient action evidence useful sooner, without harming a one-hour public run?

Posture: Mechanism screen followed by a no-harm check. The CPU renderer check is complete; model benefit remains untested. This card does not qualify hidden execution or diagnose D16's score.

Contrast: Two private, serial public-25 runs of the exact D16 F1x+REAP14 parent. Control is byte-identical to parent SHA256 9d634d6accb024d53b875106c4d77ad12948d895ea485cd4be44b7b4f83f56c0. Treatment adds only ARC3_ANIMATION_IMAGE='peak' to the environment cell. Prepared source hashes are in PREPARATION.json. Same weights, serving flags, game IDs, runtime configuration, sampling, context, retained pieces and 60-minute post-startup stop. No hidden submission or queue change. The same public game set on different workers is not a paired seed-level causal comparison.

Prediction: On a qualifying action, the treatment puts an actual engine frame containing transient information into the next ordinary model prompt. The actor can use it without requesting a timeline or writing an inspection snippet. Level/score effect is uncertain. Extra image input, distraction and early mistaken interpretations are plausible harms.

Measure:
- Actual request JSONL: count automatic-image attachments; verify the attached PNG is exactly the existing renderer's selected engine frame. Count only fresh user messages separately from repeated request exposure.
- Inspect the first qualifying user message in each of the first ten distinct games with exposure (or all games if fewer). Preserve the next two assistant/tool turns. Report explicit accurate descriptions of a transient feature, contradictory descriptions, inspection calls and subsequent actions separately; absence of a description is unknown use, not failure. Ground descriptions in the real frames, without adding labels to the actor prompt. This trace audit nominates a mechanism; it is not a causal uptake rate estimate from unrelated trajectories.
- benchmark.json: all 25 games, levels, mean score, wins, game states and error-ending notes. No-harm decision floor: treatment levels >= control levels - 7; this legacy screen margin is a deployment tolerance, not a confidence interval.
- serve.log and request receipts: startup/readiness, wrong GPU, endpoint errors, CUDA OOM, successful responses, input/completion tokens and gameplay duration. Images may cost context/prefill; do not call the intervention free.

Cap: One control and one treatment kernel, serial, one GPU slot at a time. 6,000 GPU-seconds per arm including startup and export; 12,000 seconds total (3 hours 20 minutes maximum). Gameplay stop 3,600 seconds after benchmark start per arm; retain the parent's 1,300-second screen startup limit. No retry, training, paid API, engine change, hidden submission or automatic queueing. Local preparation/CPU probe uses no GPU or model calls.

Stop: End each arm at its inherited stop and cap. Before the first launch, independently verify exact source bindings, effective GPU class/configuration, available quota and no competing owned GPU run. If either arm cannot become ready in the inherited startup envelope, fails an input/gateway/readiness check, or breaches its cap, stop the pair and preserve it. Do not extend loading patience after observing a failure. If the control fails, do not spend the treatment launch. Source preparation is complete; runtime/upload authority and preflight remain separate.

Invalid if: Parent/source drift; ineffective flag; unexpected game/version mismatch; unhealthy or truncated run; wrong GPU; incomplete raw collection; unverified stop/configuration; comparison arm lacks its intended hour of play. A low score from such a run is not a negative gameplay observation.

Result map:
- Healthy exposure, accurate use of transient evidence in at least two distinct games, and no-harm floor met -> keep as a candidate for a later matched full-horizon test; no efficacy or incumbent-promotion claim from this screen.
- Correct exposure but no interpretable use -> retain as inconclusive; do not launch another identical full screen. Review only the affected turns to decide whether a narrower information format is warranted.
- Inaccurate interpretation or no-harm floor missed -> preserve the negative result and do not adopt; separate distraction, changed trajectories and resource costs rather than declaring all animation evidence useless.
- Too few qualifying turns (fewer than five across two games) -> exposure screen inconclusive; no automatic rerun.
- Infrastructure invalidity -> preserve artifacts, stop under the cap, and return to the failing boundary. No score inference and no automatic retry.

Outputs: analysis/automatic-animation-evidence-20261027/run-screen1/ with source/preflight receipts, launch receipts, terminal status, request JSONL, exact image bindings, benchmark.json, serve.log, quota receipts and collection hashes. This directory has not been created by a live run.

Why this attempt: The CPU check exercises the existing renderer and isolated production method on the archived tn36 action 41: the settled board has no interior change, while the selected real frame includes the recorded rotation. D16's public logs have 47 distinct qualifying user messages across 10 games and zero automatic-image exposures. Half the exposures are in tn36, so breadth is limited. Existing images were already available on request; automatic delivery, not new game knowledge, is the treatment.
