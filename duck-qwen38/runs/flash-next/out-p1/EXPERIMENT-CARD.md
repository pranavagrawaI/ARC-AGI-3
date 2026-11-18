# duck-flash-next-p1
Question: Does the unchanged qualified prune-at-load P configuration improve public-25 full-run levels enough to retain it?
Posture: Execute the existing Arm P precommitment in ../CARD.md, then verify raw outputs.
Contrast: This pass and its paired P replicate versus historical KF=44 and C=42, control mean 43. This is a bundled pruning/serving treatment, not a pure-speed causal contrast.
Prediction: P pair mean >=51.5 levels, pruning guard LIVE, median running requests >=7 while at least 20 requests are active or waiting.
Measure: benchmark.json game_runs[].levels_completed summed per pass; pair mean minus 43 >=8.5. Verify 25 distinct matching games, final states, per-game 7920-second configured horizon and PUBLIC25_AUDIT. vllm-openai-server.log for load, KV, running/waiting, throughput, acceptance; artifacts/*_events.jsonl joined to benchmark histories for time/action/level curves.
Cap: One launch of this unchanged notebook; 25 games, one pass, 7920 seconds per game, no action cap (existing configuration). Approximately 3 GPU hours for this pass, 6 for pair; 3 GPU-hour operational wall limit from execution start per pass, with platform maximum 32400 seconds unchanged. Paid API spend not applicable. No automatic retries or additional treatment runs.
Stop: Normal completion of all games; stop this run on integrity/init failure; if still executing at 3 GPU hours, terminate using available Kaggle controls. Do not early-stop on intermediate gameplay results. Queuing consumes no inferred GPU time.
Invalid if: Missing/truncated output, missing games, failed pruning guard, wrong source/configuration, unexplained host failure or runtime truncation. An incomplete run is not a zero score. Mechanism failure prevents promotion even if total levels look favorable.
Result map: Both passes valid and pair mean >=51.5 with mechanism confirmed -> retain P as local submission candidate, preserve KF fallback; valid pair below threshold -> KF remains; invalid or ambiguous -> KF remains and report the gap, no retry. This does not authorize a competition submission.
Outputs: /home/pranav/projects/arc-agi-3-reference/duck-qwen38/runs/flash-next/out-p1
Source identity: ../prune/execution-20260930/manifest.json; original built notebook unchanged. Execution evidence is separate from this immutable card.
