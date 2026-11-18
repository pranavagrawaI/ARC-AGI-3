# Independent q6 speed-probe investigation

Reviewed 2026-09-05. Scope: downloaded speed-probe artifacts, source extracted from deployment/kaggle/q6-speed-probe/probe.ipynb, current accounting/host/scorer callers, and isolated local findings made since packaging. This is a review, not a launch or an edit of the fix underway. Reproduce with `python3 analysis/deployment-probes/20260905-q6-independent-review/reproduce.py`.

## P1: Schema repair alone reaches a second accounting failure

Frozen expected_launcher_solver_requests (frozen-runtime.py:1370) rejects nooa-join-sessions-v3 exactly as reported. There is a second independently reproduced mismatch: server.log contains 108 successful chat requests, warmup-facts.json accounts for one, and ipc/model_calls.jsonl contains 106 rows. The following reconciliation expects 106 solver requests and observes 107. The replay uses a diagnostic Tycho-shaped projection of the 106 records to exercise this next check; that projection is not a production fix.

Frozen solver_agent.py:697 installs a separate summarizer but only the actor event manager receives the llm_call guard (:733). This supplies a concrete unrecorded request path. Actor prompt tokens fall from 102149 to 18323 at call 87, consistent with compression. Missing request ownership is certain; identifying that exact extra request as the summarizer is an inference without a request-ID join. Current working source already adds summarizer guards, apparently as part of the ongoing fix; this review does not claim that change qualified.

Required qualification: successful requests reconcile by source across all episodes, including real summarization and terminal cleanup. Do not manufacture an extra count, substitute action/batch counts, or silently drop unrecorded requests. Reasoning/integrity checks must cover summarization too. A clean old artifact cannot retroactively supply missing usage evidence.

## P1: The HUD heuristic converts a real move into a blocked action

Frozen chassis/host_contract.py:34-70 learns HUD rows from change frequency after five transitions. In this run the first five actions are horizontal. At step 5 the learned HUD rows are [45,46,47,48,49,61,62], containing the moving 12/9 pattern as well as the bottom band. LEFT moves color12's bounding box from [45,46,24,28] to [45,46,19,23], with 52 changed pixels. Nevertheless classify returns hud_only, outcome becomes no_impact, and HostEvidence.observe skips three pending UPs. Replaying the unchanged class over raw frames reproduces it.

The actor's saved step_log then falsely states that the third LEFT hit a wall, while reporting a final position consistent with all three LEFTs. This is demonstrable contamination by host evidence, not merely a model hallucination. A correct spatial prediction gets a failed coarse outcome score.

Candidate direction: do not let row-frequency-only HUD inference assert no_impact or inert-action knowledge. Exact grid equality can establish no change; row frequency alone establishes neither HUD identity nor lack of movement. A conservative changed/unchanged fallback would preserve actual HUD-only changes as uncertain rather than falsely declaring real motion blocked. Changing this host policy requires a new candidate identity and local coverage; no such fix was made here.

## P2: Zero stall detections means zero eligible exposure

Frozen stall_contract.py:11-28 requires six prior submitted batches and state.step >=25. Every attempt here is below both eligibility conditions (latest is turn5/step9). There were zero eligible attempts, not six negative stall trials. Detection runs at before_submit and considers repeated rationale token sets, no level progress, and an untried available action. It does not inspect time or model calls inside a decision. The 56-call turn therefore cannot be a q6 detection under this contract, even if it is an operational deliberation stall.

The scorer agrees with the online contract. Changing the detector to catch long inner loops would be a different intervention, not correcting this result. Report eligible attempts explicitly. Earliest detection is at step25; observing a full 20-action escape horizon requires reaching at least step45. Later detections need correspondingly later follow-up. The frozen >=15 actor-attempt validity bar does not replace this exposure requirement.

## P2: The sizing calculation mixes batches and individual actions

The six timed units are actor turns/submitted batches: call counts 11,19,11,4,56,5 and model-time sums 79.737,203.810,201.471,48.540,720.173,71.228 seconds. Their median is 140.604 and mean 220.827 seconds. The 200 cap is TYCHO_KAGGLE_MAX_ACTIONS: my_agent.py:219 increments it for each individual ActionDecision returned to the environment, including actions drained from a submitted batch. It is not a cap of 200 actor invocations.

This probe submitted 16 actions across six batches and executed 10. Three were skipped by the false no-impact stop, and the last three remained unexecuted at the clean outer cap. Therefore 200 * 220.8 seconds is not the runtime of the configured cap. Literal realized throughput (1332.14 seconds /10 executed actions) would imply about 7.4 hours for 200 actions, or 54 actions in two hours. Neither is a reliable forecast from this short, capped, tail-heavy sample. The 32-51 actor-turn projection may be a rough planning scenario; it is not a guaranteed >=15-turn result or a correct replacement action cap.

Recorded model-call time totals 1324.960 seconds, consistent with the quoted model-bound finding. One provider request is unrecorded, so do not claim a complete decomposition of all inference work. This does not support importing upstream actor timing.

## P2: Zero truncated_batches hides the actual host stop

score_q6.py's truncated_batches counts the key truncated_from in submitted batches. It does not count HostEvidence's termination of a batch after a no-impact result. The raw published step5 state explicitly says three queued actions were skipped, while SCORE.json reports zero truncated_batches. The final cap also leaves three unexecuted actions, for a different reason. These counters must be distinguished in reporting: actor-side proposal truncation, host-side batch stop with reason, and unexecuted tail at outer budget. The zero currently reported is not proof that no batch was shortened.

## P2: History access is broken at a decision-changing moment

Call48 asks trajectory(per_step=True) to compare steps7 and8 and understand the unexpected absence of a level-up. Frozen solver_agent.py:970 reads native agent_logs/nemo/team_leader/events.jsonl, absent on this host. Executing the frozen method against the real run returns [], despite 10 preserved raw transitions. The isolated experiment-loop-v2 implementation already supplies before/after frames from ipc/transitions.jsonl and is directly relevant to the next quick probe. Extract that narrow host compatibility fix independently of the experiment protocol.

Call46 also explicitly complains that the notification says step5 while the latest host state is step8. This is model-reported notification staleness, not a captured request proving exactly which queue item was delivered. It agrees with the separately reproduced current-state dispatch race. Existing isolated StateDelivery repair is a candidate; preserve raw notification/current-state identities in the next probe to verify it.

## Turn4: A concrete experiment the actor eventually ran but failed to interpret

At step7, five palette0/1 glyph pixels exist. At step8, the moving pattern overlaps their coordinates and no 0/1 pixels are visible. After one more UP at step9, all five original coordinates and values reappear exactly. This supports temporary occlusion over permanent consumption. The model repeatedly called this absorption during 56 calls/720.17s. Nine cells are entirely conditional image requests under visual=off, totaling 45.53 seconds of recorded model time and doing no work. Other calls include repeated inspections and actual BFS/model exploration; do not call all 720s idle.

At turn5 the submitted rationale still says 'after glyph absorption'. Thus the executed move-away supplied useful discriminating evidence, but the retained explanation failed to incorporate it. This connects to the recent experimentation probes more directly than a new restoration predicate. A useful evaluation is whether the actor predicts and compares alternatives such as persistence versus occlusion, then revises from the actual next frame. Do not inject ls20-specific rules into a general solver or award a causal-reasoning pass just for issuing DOWN/UP or closing a binary API.

## What belongs in the next quick probe

Keep the next quick run a repaired-host qualification, separate from stall-incidence or treatment-effect measurement. A new identity may include discoveries since the speed probe started; archive a change manifest and do not treat it as a one-factor speed comparison.

Before paid execution, qualify locally on the exact rebuilt/extracted bytes:
1. Native NOOA session aggregation and request reconciliation, including multiple episodes, summarizer requests, malformed/missing evidence, and terminal cleanup.
2. Replay the first five real transitions: step5 must not be classified as an inert move or skip UPs merely from horizontal motion. Include literal no-change and actual terminal/level-boundary cases.
3. Per-step history returns every executed transition with correct before/after grids; stale notifications deliver the latest committed state without duplicate stale-only wakeups.
4. Counters separately report model calls by source, actor batches/attempts, submitted/executed actions, host skip reasons, cap tails and detector eligibility. The accounting-only fix can be tested without another GPU run.

A repeat-sized 10-action/1800-second-gameplay/900-second-worker/3600-second-session quick probe is enough to check basic execution and final accounting; these are the previous speed-probe bounds, not a claim of completion probability. It cannot measure q6 stall incidence and should not carry the >=3-evaluable-stalls acceptance bar. Summarizer execution is not guaranteed in ten actions, so a deterministic local compression test must precede it. No launch is authorized or performed by this review.

After that, size a separate incidence run in executed actions AND actor attempts, with eligible-exposure counts and the 20-action follow-up requirement. Do not pick its cap by multiplying the batch mean by an action target. Do not add the mandatory experiment API, a restoration predicate, reasoning-on, or speculative server-abort changes to this qualification bundle: they lack evidence of repairing this run and confound its interpretation.

## Evidence and limits

EVIDENCE.json contains independent pixel replay, request counts, detector eligibility, timing, and glyph coordinates. TURN4-CODE.txt expands the actual recorded model tool calls. frozen/ and frozen-runtime.py are extracted source, not production edits. REPLAY-OUTPUT.json records the passing reproduction. Shared-board thread292 reply296 informs the fix underway of the principal findings. No paid requests, launches, production edits or quota queries were made. Current quota and other running kernels were not independently refreshed.
