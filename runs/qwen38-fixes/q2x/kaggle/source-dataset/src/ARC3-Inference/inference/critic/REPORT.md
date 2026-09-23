# Duck critic adapter, first half (model-free)

Branch `step2-duck-critic-adapter` on top of upstream `7652836`. Everything here is
mechanical: packet, investigation-only sandbox, trigger policy, report contract, the
prompt-injection hook, and a replay harness. No model is called anywhere.

## Packet schema (`packet.py`, `schema_version: 1`)

| field | type | source |
|---|---|---|
| `packet_id` | 16 hex chars | sha256 of the canonical JSON of every other field |
| `level`, `action_num` | int | `current_frame.level`, `current_frame.step` |
| `trigger_cause`, `trigger_reason` | str | `TriggerDecision` |
| `current_frame`, `previous_frame` | `{grid:int[][], step, level}` or null | `tool_runtime_state.json` (Duck's `frame_to_payload`); previous = `history[-2].frame` |
| `history` | last H=12 `{action, frame}` | runtime-state history, bounded; `history_total` keeps the full length |
| `transitions` | last M=25 `{action, action_num, level, board_changed, level_completed}` | derived from consecutive runtime-state frames (`board_changed` = grid inequality, `level_completed` = level strictly increased, both exactly the solver's definitions); `transitions_total` keeps the full count |
| `valid_actions` | tuple[str] | Duck's `_current_valid_actions` (model names) |
| `last_action_result` | dict | Duck's compact action payload |
| `actor_beliefs` | dict[str,str] | `ToolAgent._summarized_knowledge` verbatim (seven keys) |
| `step_summary` | dict or null | `ToolAgent._last_step_summary` |

`build_packet(state_path, *, summarized_knowledge, last_step_summary, last_action_result,
valid_actions, trigger)` reads the state file via Duck's `load_runtime_state`;
`build_packet_from_state(...)` is the pure core. `to_json()`/`from_json()` round-trip and
`from_json` rejects a tampered `packet_id`. The packet is frozen: nothing built later can
reach the critic through it.

Size: 64x64 int grids cost ~12 KB each, so a steady-state packet with 12 history frames is
~150-180 KB (max seen 230 KB); the first packet of a game is ~25 KB. That is fine for disk
and for feeding the sandbox; it is not something to paste into an LLM prompt.

## Sandbox (`sandbox.py`)

`run_investigation_python(packet, code, timeout_seconds=30)` calls Duck's
`run_sandboxed_python` with `initial_state = packet_to_initial_state(packet)` (built with
Duck's own `_ascii_frame_view_payload` / `_ascii_history_view_payload`, so the critic sees
the same `current_frame`, `previous_frame`, `history`, `transitions`, `last_transition`,
`valid_actions`, `last_action_result` objects the actor sees) and an `action_handler` that
raises. Inside the sandbox `action(...)` raises `RuntimeError: action failed in sandbox
host.`; the result dict is Duck's plus `refused_action_calls`.

## Trigger policy (`trigger.py`)

`DuckTriggerPolicy(stall_no_change=3, stall_no_progress=25)`; feed
`observe(transitions_from_history(history))` (idempotent by `action_num`), then at each
turn start `decide(level=, action_num=, critic_request=)` -> `TriggerDecision(fired, cause,
reason)`. Evaluation order and bookkeeping:

1. `level_start`: level not yet fired -> fire. Any firing marks the current level as fired,
   so a request or stall on a level's first turn does not cause a second level-start firing.
2. `actor_request`: non-blank `Critic:` text -> always fires.
3. `stall_no_change`: a run of >= N executed actions without a board change since the last
   re-arm. **Latched, not a live streak**: Duck batches (ls20 p0 has a 32-action batch
   containing a 12-action no-change run that ends with changes); a plain streak counter
   never fired on it. Deviation from the plan, made on evidence.
4. `stall_no_progress`: >= K executed actions since the last level completion / re-arm.

Re-arm (all stall counters to zero) happens on any firing and on any level completion.
No `wm_signal` port: Duck has no executable world model.

## Hook in `tool_agent.py` (+24 lines, 0 deletions)

- `ToolAgent._pending_critic_report`, `set_critic_report(report)`, `_critic_request`,
  `take_critic_request()`.
- `_build_user_prompt`: when a report is pending, `render_report_block(report)` is
  appended immediately after `"end of world model. "` and the report is cleared.
- `analyze()`: if a report was injected, a `[CRITIC REPORT]` transcript section (the
  report JSON) is appended right after `[USER PROMPT]`, so uptake can be measured from
  transcripts.
- `_extract_scientist_note` parses a `Critic:` label into `critic_request`;
  `_update_summarized_knowledge_from_assistant` routes it to `_critic_request` and never
  into the seven world-model keys.
- Byte identity: `tests/critic/fixtures/prompt_*.txt` were generated from the unmodified
  upstream function (commit `9cc865f`, before the hook existed) for three input sets; the
  patched function reproduces them byte for byte when no report is pending.

## Report block (`report.py`)

```
Critic report (advisory, from a separate investigator; not part of your world model):
- Packet: <packet_id>
- Summary: ...
- Confidence: ...
- Evidence:
  - ...
- Recommended actions (in `action(...)` format; verify before use): [{"action": "UP"}, {"action": "MOUSE", "col": 4, "row": 3}]
- Open questions:
  - ...
end of critic report.
```

## Replay (`replay.py`)

`python -m inference.critic.replay <events.jsonl>... [--stall-n 3 --stall-k 25]
[--dump-dir D] [--no-sandbox] [--no-log]`. Facts about the trace format that shaped it:

- `*_events.jsonl` is the viewer log: `initial`, then `action` events (one per executed
  action, with `board` already as an int grid and `board_ascii`), and one `analysis`
  event per `analyze()` call, emitted *after* that call's actions, carrying the whole
  transcript of that call (not a delta relative to earlier calls). Retried / yielded
  calls produce their own `analysis` event with the same `analysis_step`.
- Auto-`RESET` after GAME_OVER is issued by the solver before the next `analyze()` and
  carries the previous `analysis_step`; the replay groups it as a pre-turn action.
- `board` and `board_ascii` agree on every event checked (0 mismatches across 500 files,
  via `ARC_COLOR_CHARS.index`). Derived `board_changed` matched the event flag on all
  68,680 actions.
- Valid actions are not in action events; they are read from the turn's `USER PROMPT`
  line `Valid actions right now: ...`.
- Beliefs: for each turn, ASSISTANT sections are fed through `_extract_scientist_note`
  in order (THINKING sections are ignored, as Duck ignores them), then the six per-level
  keys are wiped if that turn's actions include a level transition / game over / run
  complete, matching `_update_summarized_knowledge_from_step_summary`.

## Results on real traces (defaults N=3, K=25; sandbox snippet = segmentation node count + previous->current cell diff)

| file | turns | actions | levels done | level_start | actor_request | stall_no_change | stall_no_progress | packets | sandbox ok/err | mean KB | max KB | mean sandbox s | ASSISTANT sections | updating beliefs | turns with beliefs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lp85-305b61c3_p0 | 33 | 58 | 1 | 2 | 0 | 0 | 2 | 4 | 4/0 | 126 | 177 | 0.082 | 0 | 0 | 0 |
| lf52-271a04aa_p0 | 77 | 131 | 1 | 2 | 0 | 0 | 4 | 6 | 6/0 | 180 | 211 | 0.079 | 0 | 0 | 0 |
| ls20-9607627b_p0 | 53 | 132 | 0 | 1 | 0 | 1 | 3 | 5 | 5/0 | 145 | 175 | 0.049 | 0 | 0 | 0 |
| sb26-7fbdac44_p0 | 53 | 171 | 1 | 2 | 0 | 0 | 5 | 7 | 7/0 | 156 | 178 | 0.066 | 0 | 0 | 0 |
| lp85-305b61c3_p1 | 31 | 79 | 1 | 2 | 0 | 2 | 1 | 5 | 5/0 | 139 | 177 | 0.084 | 20 | 20 | 28 |
| wa30-ee6fef47_p11 | 75 | 170 | 0 | 1 | 0 | 0 | 6 | 7 | 7/0 | 153 | 175 | 0.042 | 75 | 73 | 74 |
| ft09-0d8bbf25_p0 | 39 | 85 | 3 | 4 | 0 | 0 | 1 | 5 | 5/0 | 147 | 180 | 0.070 | 41 | 22 | 18 |

Firing positions (turn, action_num, level, cause), e.g. ls20 p0:
`(1,0,1,level_start) (3,32,1,stall_no_change) (13,58,1,stall_no_progress) (24,84,1,stall_no_progress) (40,110,1,stall_no_progress)`.

All 500 example-run files (25 games x 20 passes), same defaults:

| files | turns | actions | levels | level_start | actor_request | stall_no_change | stall_no_progress | firings / 100 actions | sandbox ok / err | mean packet KB | mean sandbox s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 500 | 23,073 | 68,680 | 309 | 808 | 0 | 259 | 1,851 | 4.2 | 2918 / 0 | 151 | 0.066 |

### Surprises

- **The four requested traces have zero ASSISTANT sections.** The model answered every
  turn with THINKING + a tool call and no assistant text, so Duck's carried world model
  was empty for the whole game and the `Working world model carried from earlier turns:`
  block never appeared in a prompt. Across all 500 files: 282 contain ASSISTANT sections;
  6,031 ASSISTANT sections total, 4,655 of which update at least one belief key; only
  6,956 of 23,073 turns (30%) started with any non-empty belief. `actor_beliefs` will
  often be empty in production packets.
- ASSISTANT sections are fully recoverable from the `transcript` field (each analysis
  event carries its own call's full transcript).
- Labeled lines also appear inside THINKING sections; Duck ignores those, and so does
  the replay (tested).
- No real transcript contains `Critic:` (0 of 500 files), so the new label has no
  false-positive history.
- `stall_no_progress` dominates (63% of firings) and re-fires every 25 actions on a
  stuck level; `stall_no_change` is rare because Duck's actions almost always change
  something (often HUD only).

## Verification

```
$ uv run --no-sync ruff check inference/critic tests/critic
All checks passed!
$ uv run --no-sync pytest tests/critic -q --basetemp=/tmp/duck-critic-pytest -p no:cacheprovider
..............................                                           [100%]
30 passed in 0.17s
```

(`--basetemp` only sidesteps stale pytest temp dirs from other projects in `/tmp`.)

## Open questions for the second half

1. Packet cost vs. prompt: frames should stay in the sandbox; the critic prompt should
   carry `actor_beliefs`, `transitions`, `last_action_result`, and sandbox output, not
   the packet JSON. Should frames be stored as letter ASCII (3x smaller) instead of
   Duck's int-grid payload?
2. With beliefs empty in 70% of turns, what does the critic critique? Probably the
   transition record and the actor's tool code (also recoverable from `TOOL CALL: python`
   sections) rather than prose beliefs.
3. `stall_no_progress` re-firing every K actions is a budget decision: one critic call per
   25 actions on a stuck level, ~4 per 100 actions overall. Wall clock binds in the
   competition, so K may need to scale with remaining time.
4. Wiring: the trigger must be evaluated in the solver loop (or a `ToolAgent` subclass)
   before `analyze()`, after the auto-reset, with `take_critic_request()` from the
   previous turn; `set_critic_report()` before `analyze()`. Not wired here on purpose.
5. Uptake metric: the `[CRITIC REPORT]` section plus the next `[TOOL CALL: python]`
   gives a direct measure of whether recommended actions were executed.

---

# Second half: critic policy, gate, and solver wiring

Commits `8b2d86f` onward. Model-facing pieces only; the first half's packet, sandbox,
trigger, and report contract are unchanged.

## What was ported from the Tycho trigger builder, and what was not

| Tycho `builder.py` / `builder.system.j2` | Duck critic (`policy.py`) |
|---|---|
| investigator, never acts | same; `action(...)` refused in the sandbox |
| `notes/world_model.md` scratchpad across invocations | `CriticNotes`: the report's `scratchpad:` line, kept per game (last 12) |
| `_builder_budget_notice`: investigate → commit by calls/seconds left | `budget_notice()`: same two modes, host-authoritative, appended after every tool result |
| `FINAL_REPORT_REQUEST` when the budget hits | same; report also parsed from the reasoning channel when `content` is empty |
| `FALLBACK_REPORT` | same shape (`confidence: none`, no recommendation) |
| compact labeled report read by the actor every turn | `confidence/model/objective/plan/recommended_action/subgoal/note/scratchpad` |
| `world_model.py`, verifier, `write_file`/`edit_file`, planner | **not ported** -- Duck has no executable world model; investigation uses Duck's own sandbox |

## Budget semantics

`CriticPolicyConfig(max_tool_steps=2, max_seconds=90)`: investigation calls per firing =
`max_tool_steps - commit_calls_remaining` (= 1), then one report call. Every request's
timeout is bounded by `seconds_left + request_grace_seconds` (20 s), so a runaway
reasoning call cannot push a firing more than ~20 s past its budget; a timeout yields the
fallback report, never an exception into the game loop.

## Gate runs (OpenRouter `qwen/qwen3.6-27b`, 12 real packets from example-run: lf52 p0, lp85 p1, ls20 p0, sb26 p0 x3 each; 7 level_start, 2 stall_no_change, 3 stall_no_progress)

| run | budget | parsed | fallback | sandbox errors | legal rec. | median s | median tokens in+out |
|---|---|---|---|---|---|---|---|
| run1 | 4 steps, 120 s, first prompt | 11/12 | 1 | 14/36 | 11/11 | 60 | 9,059 + 1,739 |
| run3 | 2 steps, 90 s, field-accurate prompt | **12/12** | 0 | **0/12** | 11/11 | **33** | 3,807 + 1,113 |
| run2 | 4 steps, 120 s, no request bound | firing #2 ran 667 s / 16k completion tokens with no report; killed | | | | | |
| run4 | 2 steps, 90 s, bounded requests | 10/12 | 2 | 0/12 | 10/10 | 40 | 3,903 + 1,059 |
| run5 | **final defaults** (run4 + one report retry) | **12/12** | **0** | **0/12** | **12/12** | **31** | 3,977 + 992 |

run4's two fallbacks were the model ignoring "tools are closed" on the report call (raw
`<tool_call>` markup as content, or empty content mid-reasoning) -- the same failure Duck's
actor loop retries once. run5 adds that single retry (`report_retries=1`); it was not
needed on any of run5's 12 firings, so it is insurance, not a cost.

Per-firing transcripts for every run are under `records/step2-gate/` at the worktree root.

What run1 → run3 changed: (a) the system prompt now states the segmentation node fields
exactly as Duck's actor prompt does (`pixels` is an integer count) -- 13 of the 14 sandbox
errors were `len(node['pixels'])`; (b) the report call gets 2x output headroom and the
reasoning channel is parsed as a fallback; (c) per-request timeouts are bounded by the
remaining budget (run2 is the failure this prevents).

Cost on Kaggle, extrapolated: Duck generated ~1.1k tokens per turn at ~46 s per turn
(10-way, RTX PRO 6000), so a lean firing (~1.1k completion tokens, 2 prefills of 3-5k)
is **about one Duck turn**. At the replay rate of ~2-3 firings per 40-action game that is
roughly 5-12 % of a 1200 s game.

## Wiring (`runner.py`, `solver.py`, `tool_agent.py`)

- `ToolAgent.critic_context()` / `model_config()`: read-only accessors (+12 lines).
- `_HarnessGameSession._run_critic_if_triggered()` runs before every `analyze()`:
  `take_critic_request()` → `CriticRunner.decide()` → on fire `build_packet_from_state` →
  `CriticPolicy.run` → `set_critic_report()` → a `type: "critic"` viewer event and a
  `<run_stem>_critic.jsonl` trace next to the transcript. Any exception is printed and
  swallowed: the critic is advisory and never ends a game.
- Synchronous in the game thread, so the per-game cap charges it automatically.
- `HarnessSolver._make_critic()` returns `None` unless `DUCK_CRITIC_ENABLED=1`
  (test: `test_critic_is_off_by_default`); the incumbent arm is byte-identical upstream.
- Kaggle: `duck_kaggle_setup_command()` now copies every `DUCK_CRITIC_*` from the launcher
  env into the runner's setup env, so the arm is chosen at launch by environment alone.
- End-to-end offline test: `test_session_hook_injects_report_into_next_prompt` drives the
  hook with a stub game and checks the next user prompt carries the block exactly once.

## Not done / decisions for Step 3

- Uptake is measured post hoc from `_critic.jsonl` (recommended actions) joined to the
  following action events; no live metric.
- The critic shares the actor's vLLM endpoint and concurrency slot; its latency on Kaggle
  is inferred from token counts, not measured. Step 3's first wave measures it.
- `stall_no_progress` re-fires every 25 actions on a stuck level; not scaled to remaining
  wall clock.

# Question mode (default from this commit)

Report mode's tool loop costs ~one Duck turn per firing and hands the actor a plan it did
not derive. Threads 423/426 on the board showed the same shape: grounding is fine, the
pivot arrives late, zero levels. Question mode keeps the trigger and the packet, drops the
tool loop, and makes the host do the grounding the model was spending calls on
(`question.py`: segmentation summary, previous->current cell diff, the actor's last
`python` code from the transcript). One call, reasoning off, 160 output tokens, 30 s bound.
The model answers three lines -- `assumption`, `question`, `probe` -- and the actor sees
the assumption as the summary, the question under "Open questions", and the probe as a
recommended action only if it is legal on this game (the host owns legality; an off-list
probe is dropped, the question stands).

`DUCK_CRITIC_MODE=report` restores the tool loop; `question` is the default in
`CriticPolicyConfig` and the Kaggle setup env carries the choice. Packet schema is now 2
(`actor_last_code`).

## Gate (same 12 packets as above, OpenRouter `qwen/qwen3.6-27b`, `reasoning: {enabled: false}`)

| replay | parsed | fallback | probe present / legal | median s | median tokens in+out |
|---|---|---|---|---|---|
| question-mode-v1 | 12/12 | 0 | 11 / 9 | 2.6 | 1,154 + 57 |
| question-mode-v2 | 12/12 | 0 | 12 / 12 | 3.0 | 1,156 + 54 |
| question-mode-q38 (final tree) | 12/12 | 0 | 10 / 10 | 3.2 | 1,158 + 70 |

Three samples of one tree: 36/36 parsed, no fallback, no sandbox (there is none). Per
firing that is roughly 1/15 of a Duck turn at Kaggle rates, versus ~1 turn for report
mode. Transcripts under `records/step2-gate/question-mode-*`.

## Live, local (LS20, 15-action cap, `records/step2-live/`)

- `ls20-q1-auth401`: the first firing failed with `401 No cookie auth credentials`. The
  actor authenticates through `ToolAgent._headers` env fallbacks but `model_config()`
  returned its stored key, which was empty. `model_config_from_env` now falls through
  `DUCK_CRITIC_API_KEY` → actor key → `LOCAL_ANALYZER_API_KEY` → `OPENROUTER_API_KEY` →
  `OPENAI_API_KEY` (test: `test_model_config_falls_back_to_env_key`). The fallback report
  reached the actor and the game continued: the advisory contract held under a real error.
- `ls20-q2`: `level_start` fired at action 0, parsed in 2.9 s (824 + 67 tokens); the probe
  `MOUSE(row=32, col=32)` was dropped as illegal on LS20, the question reached the prompt.
  The stall trigger never armed in 15 actions (longest no-change run 1 < 3; 15 < 25), so
  its silence is correct behavior and **no live game has yet fired on a stall**. Only gate
  replay has exercised `stall_no_change`/`stall_no_progress` end to end. Step 3's first
  Kaggle wave is the first place that can happen.

## Not done

- `test_question_mode_falls_back_without_raising` originally accepted a raise; it now
  asserts the fallback on `requests.exceptions.Timeout`. Non-`requests` exceptions from a
  custom `post` still propagate; the solver hook swallows them, so the game is safe either way.
- Kaggle has not executed this tree. Step 1's kernel ran upstream `7652836`; Step 2 only
  rendered the setup script with and without `DUCK_CRITIC_*` and compiled both.
