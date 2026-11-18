# Verdict — 20260905T-nooa-join-public3-reasoning-v2

Kernel `pranavagrawai/prototype-nooa-join-public3-reasoning` version 2. Status ERROR. Probe `elapsed_s` 14839.57 (~4.12 h). No retry.

## Probe

**Held.** `public_progress_error` null, `solver_sessions_error` null. Games ls20, s5i5, cd82 present. Schema `nooa-join-public-progress-v2` / `nooa-join-sessions-v3`. Chassis `q3c-mdfiles-after`. Kernel ERROR is `check=True` on rc=71 after a worker thread died.

`--request-timeout-s 180` shipped. Error does not contain `requires --request-timeout-s 180` or `turn does not match the last public observation`. GAME_OVER resync was not exercised: no game reached GAME_OVER.

Thinking-on held: notebook `ARC_LLM_EXTRA_BODY={"chat_template_kwargs":{"enable_thinking":true}}`; SGLang `reasoning-parser qwen3`; server log `enable_thinking` default_enabled=True.

## Crash

`TimeoutError: worker timed out waiting for decision` in `LocalSolver._receive` (`Thread-3`/`Thread-4`). Worker cap `TYCHO_NOOA_WORKER_TIMEOUT_SECONDS=7200`. Last recorded play `cd82` 21:42:01; timeout frame 23:42:01 — gap **7200.00 s**.

That gap is not GPU idle. `server.log` Decode batch lines run 19:40:00–23:11:38 with ~450–580 lines per 10 min (almost every second). Decode stops 23:11:38; host times out 31 min later. `#running-req` is 1 for 7043 batches and **2 for 3687** — host C=1 does not keep a second worker off the GPU. `#queue-req` stays 0 (the queue is Python, not SGLang).

## Play (truncated: 6 env actions of 80)

| game | recording frames | grid-hash changes | IPC batches | env actions | max_levels | first action |
|---|---:|---:|---:|---:|---:|---|
| ls20 | 1 | 0 | 1 | 1 | 0 | LEFT |
| s5i5 | 1 | 0 | 1 | 1 | 0 | CLICK 40 40 |
| cd82 | 3 | 2 | 2 | 4 | 0 | UP; then LEFT, RIGHT, SPACEBAR |

Totals: 5 frames, 2 grid-hash changes, 4 batches, 6 env actions, **0 levels**. Episode count 1 each.

Clocks from recordings + launcher stdout:

| t (UTC 2026-09-04) | event | Δ |
|---|---|---|
| 19:34:41 | probe start | |
| 19:39:55 | three games created | readiness ~5 min |
| 20:29:32 | cd82 first frame | 2977 s after create |
| 20:53:08 | ls20 first frame | 1416 s |
| 21:16:26 | s5i5 first frame | 1398 s |
| 21:42:01 | cd82 second frame | 1534 s |
| 23:11:38 | last Decode batch | GPU still generating through the “hang” |
| 23:42:01 | timeout | 7200 s after last play |

ls20 `worker-stderr`: `GenerationError` after 3 LiteLLM retries, `timeout value=180.0, time taken=541.34 seconds`. s5i5 `step_log.md` records ~8 WAIT re-wakes, the same 541 s API timeout, then stall recovery that truncated `actions.jsonl` because the first CLICK was never consumed. Decode median ~38.5 token/s (p90 65). At that rate 8192 max tokens wants ~200 s, which is over the frozen 180 s request cap.

Depth (thinking-off, same three games, 80 cap) recorded 131 frames / 125 grid changes in ~84 min of play before a GAME_OVER continuity crash. This run spent ~4 h to emit 6 env actions.

## Card

Stop: success if any `max_levels_completed >= 1`. Futility if play completes and all three stay at 0 with thinking-on confirmed. Timeout with recordings is a result; do not retry.

- Levels: 0. Thinking-on: held. Continuity crash: did not recur. Argparse: fixed vs v1.
- Play did not complete 80 actions.

Result map: **timeout / truncated.** Score completed games; do not retry. Thinking-on did not rescue this chassis at 80 because 80 was not measured. The next spend on this join, if any, is not another thinking-on 80-action retry and not 160 actions. Representation, first-turn cost, or the WAIT/_receive deadlock is the live fork — not a third push of this card.

## Claim ceiling

Thinking-on Qwen on the frozen q3c join, with GAME_OVER resync present and argparse 180, can emit a first CodeAct batch on ls20/s5i5/cd82. It does not establish a level, an 80-action budget, or that thinking-on is faster or slower as a quality factor independent of the 180 s request cap. A hung `_receive` still burns the 7200 s worker cap while the GPU keeps decoding.

Outputs: `analysis/deployment-probes/20260905T-nooa-join-public3-reasoning-v2/kaggle-output/`.
