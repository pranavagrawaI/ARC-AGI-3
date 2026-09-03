# V15 terminal result

Kernel version **13** of `pranavagrawai/prototype-tycho-nooa-public-3-full-timing`, launched
2026-09-03, elapsed **1920.96 s** (launcher 1596.1 s), terminal Kaggle status **COMPLETE**,
outer probe `status=passed`, `request_path_status=passed`.

## Result — the transport question is answered YES; qualify the bounded action-demand transport

The exact Kaggle SGLang route honors a forced `take_actions` request with
`extra_body.chat_template_kwargs.enable_thinking=false` and `max_tokens=2048`. Every forced
action-demand call returned a `take_actions` tool call in 81–161 completion tokens with **zero
reasoning tokens**. The Solver reached **eight reconciled public `ls20` actions**, the full
permitted horizon, with the request contract intact.

Per the V15 card result map: *"Eight reconciled actions … with the request contract intact and the
worker envelope showing 2,048 / no-thinking → qualify the bounded action-demand transport and
inspect latency and gameplay before expanding to three public games."*

## The V14 fix landed

`worker-metadata.json` decision envelope (session `ls20-794xo7xe`):

```json
"action_demand_disable_thinking": true,
"action_demand_max_output_tokens": 2048
```

Both were `false` / `null` in V14. `solver_sessions_error` is `null` (V14's entire failure was this
raising "Solver decision envelope differs").

## Per-call reconciliation (11 Actor calls + 1 warmup)

| call | sequence | tools | tool_choice | max_tokens | extra_body | finish | completion / reasoning tok | tool call |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 8 | auto | 16000 | — | length | 16000 / 16000 | none |
| 2 | 0 | 8 | auto | 16000 | — | length | 16000 / 16000 | none |
| 3 | 0 | 8 | auto | 16000 | — | tool_calls | 11292 / 10748 | ls, run_python |
| 4 | 0 | 8 | auto | 16000 | — | tool_calls | 132 / 25 | run_python |
| 5 | 0 | 8 | auto | 16000 | — | length | 16000 / 16000 | none |
| 6 | 0 | 1 | required | **2048** | **enable_thinking=false** | tool_calls | 99 / 0 | **take_actions** |
| 7 | 1 | 1 | required | **2048** | **enable_thinking=false** | tool_calls | 161 / 0 | **take_actions** |
| 8 | 4 | 1 | required | **2048** | **enable_thinking=false** | tool_calls | 81 / 0 | **take_actions** |
| 9 | 5 | 1 | required | **2048** | **enable_thinking=false** | tool_calls | 105 / 0 | **take_actions** |
| 10 | 6 | 1 | required | **2048** | **enable_thinking=false** | tool_calls | 89 / 0 | **take_actions** |
| 11 | 7 | 1 | required | **2048** | **enable_thinking=false** | tool_calls | 84 / 0 | **take_actions** |

- Every exploratory checkpoint: `max_tokens=16000`, no thinking override — as predicted.
- Every forced checkpoint: only `take_actions`, `tool_choice=required`, `max_tokens=2048`, exact
  `{"chat_template_kwargs": {"enable_thinking": false}}` — as predicted.
- Every required response emitted `take_actions` — as predicted.

## Action reconciliation (8 for 8)

| seq | batch | released | official recording | count |
|---|---|---|---|---|
| 0 | submit len 1 | RIGHT | ACTION4 | 0 |
| 1 | submit len 3 | RIGHT | ACTION4 | 1 |
| 2 | (same batch) | RIGHT | ACTION4 | 2 |
| 3 | (same batch) | RIGHT | ACTION4 | 3 |
| 4 | submit len 1 | DOWN | ACTION2 | 4 |
| 5 | submit len 1 | RIGHT | ACTION4 | 5 |
| 6 | submit len 1 | RIGHT | ACTION4 | 6 |
| 7 | submit len 1 | RIGHT | ACTION4 | 7 |

`solver_sessions.games.ls20.action_batches`: `submission_count=6`, `release_count=8`,
`submitted_action_count=8`, `invalidation_count=0`, `model_calls_avoided=2` (the sequence-1
three-action batch covered sequences 1–3). Launcher stdout: "agent took 8 actions". Recording
SHA-256 `62a02d37fe3708ecaebfc9bb0ad8dafc62b48b809f39c461299a53e1db076815`
(102,632 bytes), `recorded_action_results=8`.

## Caps — all within limit

| cap | limit | observed |
|---|---|---|
| Actor calls | ≤ 13 | 11 |
| chat-completion accesses incl. warmup | ≤ 14 | 12 (`whole_model_request_count_verified=true`, expected==observed) |
| Actor completion tokens | ≤ 96,384 | 60,043 |
| startup | ≤ 1,200 s | within (warmup 8.1 s, run start 20:07:19, launcher start ~20:12) |
| launcher time | ≤ 7,200 s | 1,596.1 s |
| notebook | one RTX PRO 6000 | one RTX PRO 6000 |

## Lifecycle / containment

`r8_l3b_gate`: `lifecycle_passed=true`, `request_path_passed=true`, `teardown_log_accepted=true`.
`lifecycle.containment_passed=true`, `status=passed`. Teardown classification: `forbidden_count=0`,
68 entries all allowed. `model_verification=matched`. `runtime_verification` hashes recorded.

## Zero competition submissions

`operation_mode=offline`; `KAGGLE_IS_COMPETITION_RERUN` guard present in cell 4; no
`kaggle competitions submit` in the kernel log. The official scorecard is empty
(`score=0.0`, `total_actions=0`, `scorecard_status=unavailable_due_local_initial_reset_contract`) —
evidence is the official agent recording, not a scored submission, exactly as the card states.

## Observations for the pre-expansion review (not invalidating)

1. **Exploratory over-reasoning is real and near the cliff.** Calls 1, 2, and 5 hit
   `finish_reason=length` at the full 16,000-token exploratory budget with no tool call. The
   Solver's six-call sequence-0 budget absorbed all three; call 6 (forced) committed. Only calls 3
   and 4 produced useful exploratory tool work. Before scaling to three or 25 games, inspect
   whether the exploratory budget or the exploratory prompt should change — the forced-commitment
   mechanism now works, but exploratory calls burn ~48,000 tokens per sequence-0 with little
   return here.
2. **No gameplay progress.** Final state `NOT_FINISHED`, 0 of 7 win-levels, actions were
   7×RIGHT + 1×DOWN. ls20 was not solved. This is a gameplay-quality question, separate from the
   transport question the card asked.
3. Per-forced-call latency was low (81–161 completion tokens); the sequence-1 batch avoided two
   model calls. Full per-sequence timing is in the machine-readable reconciliation.

## Claim ceiling

- The exact route (pinned official ARC-AGI-3-Agents launcher → frozen Solver worker → Tycho-NOOA
  actor → LiteLLM → local Qwen/SGLang) honors a 2,048-token no-thinking forced `take_actions`
  request and reliably commits an action; it did so six times in one run.
- The guarded batch machinery released eight actions, all reconciled one-for-one to the official
  recording, with zero invalidations and two model calls avoided.
- V15 does **not** establish gameplay quality, repeated-game reliability, public-25 performance, or
  any competition result. ls20 was not solved.

## Result-map disposition

**Qualify the bounded action-demand transport.** Next permitted step is to inspect latency and the
exploratory-reasoning behaviour, then expand to three public games under a new card — not directly
to 25.

## Machine-readable evidence

`TERMINAL-RECONCILIATION.json` in this directory. Raw Kaggle output and terminal source were
downloaded before interpretation; terminal source is byte-identical to the frozen candidate cells.
