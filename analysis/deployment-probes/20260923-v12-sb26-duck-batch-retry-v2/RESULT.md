# 20260923 v12 sb26 Duck-batch retry v2 result

Kernel `pranavagrawai/tycho-nooa-v12-sb26-duck-batch` v2: Kaggle COMPLETE. The launcher `result.json` says `passed`; it ran 2026-09-23 07:19:35–08:34:15 UTC. `system-before.json` records a working NVIDIA RTX PRO 6000 Blackwell Server Edition (97,887 MiB, `nvidia-smi` exit 0). This is the first gameplay run of the frozen Duck-batch package. Raw output is in `kernel-output/` beside this file.

## Card reading

The two sb26 episodes ran sequentially with 1800 s admission each. Both stopped at `wall_budget`, with 45 and 49 executed environment actions. The first reached 0 levels. The second reached L1 at action 16 (a `SPACEBAR` transition), at 08:13:03.677 UTC. Its `launch.json` start was 08:03:47.357 UTC, so episode-clock time-to-L1 was **556.32 s (9 min 16 s)**. There was no RESET before L1 in this episode. Final state was `NOT_FINISHED`, with one level complete; this was not a game solve or competition score.

The second pass submitted 14 batches longer than one action (13 of length 2, one of length 5). The host executed both actions in 12 of the length-2 batches and all five actions in the length-5 batch. One length-2 batch stopped after its first action. This establishes that open batching was used, not merely offered. Pass 0 submitted only single actions.

Both `launch.json` files record `duck_batch=1`, `epistemic_critic=0`, `contingent_arm=off`, and `transcript_keep_turns=4`. Model-call ledgers contain 157 and 110 actor calls and zero critic calls. Eviction occurred (41 and 29 `EVICTED` rows) with no `EVICTION_ERROR`; neither worker recorded `ABORT`. The two start times are 1914.46 s apart, versus pass-0 gameplay wall of 1911.64 s, so episode walls do not overlap. The GPU is the required RTX PRO 6000, not P100.

The notebook's `ASSESSMENT.json` says `INVALID` for two reasons that do not negate the above raw observations:

1. The assessor looks for `game-stops.jsonl` within each session. The host wrote the two `wall_budget` records to the parent `solver-workspaces/game-stops.jsonl`.
2. The assessor flags `action executed after interrupted prefix` when an unresolved step check reports a queued suffix. In this Duck-batch mode, `nooa_worker.py` deliberately retains the queue for `unresolved` checks. All recorded checks are `unresolved`, with no real mismatch; the suffixes listed above were intentionally played. The assessor's `skipped_actions` field counts the queue at check time even when those actions are retained, so its 22 interrupted actions for pass 1 are not 22 skipped executions.

## Interpretation

One pass improved on the critic-off episode-clock L1 comparison of 17.5 min; the other failed to reach L1, so the card's prediction that both passes reach L1 failed. The successful pass remained above Duck's 322 s upper target by 234 s. Because four mechanisms changed together and the two passes varied sharply, this run supports a narrower statement: the Duck package can play multi-action batches and produced one faster L1 in this setting. It does not identify which mechanism caused the faster pass or establish reliable sub-322 s timing.

### Follow-up comparison of the successful lives (2026-09-24)

The critic-off pass-1 episode spent 65 actions before a RESET/GAME_OVER split, then reached L1 at action 15 of the next session (`sb26-ef2qddnf`), about 12.5 min after that life began. Duck-batch pass 1 reached L1 at action 16 of its session, 9.27 min after start. Comparing those successful lives gives **15 versus 16 actions**, and roughly **12.5 versus 9.27 min**, a smaller and better-matched time gap than the 17.5 versus 9.27 min episode clocks. Before L1, the critic-off life used 67 model calls, 2,270,619 prompt tokens, and 743.4 s summed model-call wall; Duck-batch used 46 calls, 1,264,479 prompt tokens, and 548.6 s. The 194.8 s model-call wall gap accounts for nearly all of the approximately 199 s life-clock gap. Batching eliminated only three action-turn boundaries before L1; the remaining call difference is not causally attributable from these two different trajectories. This argues for inspecting decision-call count and repeatability before spending another kernel on batching alone.

No further push or submission is authorized by this card.
