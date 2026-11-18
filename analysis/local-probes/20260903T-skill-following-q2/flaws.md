# Flaws — chassis, prompt, and q2d

Evidence from the live q2d prompt (`results/nemo_solver/20260903_160733_ls20_mdfiles_q2d/agent_logs/nemo/team_leader/messages/step_000_round_00_user.md` and later turns) and from that run’s cells, notes, helper, and frames.

The ARC skill is not the main problem. The chassis around it is. The skill already asks for `encode`/`predict`, retrodiction, search, then `submit_actions`. The layout makes notes-then-batch the fluent path.

Companion: `perfect-approach.md`.

---

## A. Prompt and chassis (we are in the way)

### 1. System prompt names the wrong memory first

Every turn, `MdArcSolverAgent` docstring:

> Follow the `<arc_skill>` context block. Your accumulated knowledge lives in markdown files (`read_notes`/`append_notes`/`write_notes`) — read them before deciding, append after observing, curate them periodically.

The plan is supposed to be `self.h.*.predict`. The first sentence they see is “knowledge = markdown.” q2d: 15 `append_notes`, 3 `predict(` calls.

### 2. Per-turn workflow puts notes before the grid, and weakens `predict:`

`handle` (what they execute):

1. parse state, check last prediction
2. **consult the knowledge store BEFORE deciding**
3. `grid_array` / `load_helpers` / `trajectory`
4. record notes
5. `submit_actions([...], rationale="prediction...")`

The skill says `rationale="predict: ..."`. The workflow says `"prediction..."`. Retrodiction rate was scored on the skill’s string. The host never required it.

### 3. CodeAct `strategy_prompt` is a different product

Injected above the skill, every turn:

- Finish with `return_result`. The game finishes with `submit_actions`.
- “For language tasks (classification, extraction, interpretation), use LLM reasoning — answer directly via `return_result`.” Reasoning was **off**. This line tells them to squint and talk.
- “Helpers persist as REPL locals across cells.” A second meaning of “helper,” next to `write_helper`.
- Fan-out: `@strategy(PredictStrategy())` — more LLM calls, not numpy search over `z`.
- `await self.method(...)` — `grid_array` and `submit_actions` are sync.

### 4. “Keep cells short” vs search

Sandbox block, every turn: *Keep cells short and return partial results.* The 60 s kill is a real cap. The English is an extra brake on an in-cell BFS, which is the compile step once `predict` holds.

### 5. Batching is the easy move from turn 0

`submit_actions` docstring: “one or more,” example `["UP", "UP", "RIGHT", "USE"]`, cap **20**. The skill says while `predict` is weak, explore. It never says batch size 1. The example is a four-action spray. q2d followed the example.

### 6. Prefill dumps the hex grid into the prompt

Round 0 is two framework cells: pprint the notification (truncated `grid_rows` as a 1000-character hex string), then `show(grid_image)` which is a no-op because visual is off. First pixels shown: ASCII hex. Skill’s first sentence: parse with `grid_array`.

### 7. `self.h` is hidden from `<state>`

Deliberate (run-dir path must not leak the game name). After `load_helpers()` the namespace is invisible unless they remember the printout. Notes stay on the dashboard: `notes: hypotheses.md, step_log.md`. Markdown is signposted; the world model is not.

### 8. Distractor skills

`<skills>` lists `nemo.shell`, `nemo.web`, `nemo.memory`, `nemo.todo`, … and a tip to `queue_manager.cancel('game_states')`. Shell/network are jailed. It still occupies the top of the context.

### Signifiers to change (not a new skill)

| Live text | Affords | Want |
|---|---|---|
| “knowledge lives in markdown” | notes as the plan | “the plan is `self.h.*.predict`; notes are the lab book” |
| `rationale="prediction..."` | any sentence | `predict:` or reject |
| submit example 4-pack, cap 20 | spray | cap 1 until last retrodiction matched; then search may batch |
| “Keep cells short” | don’t write a BFS | “a search over z is a valid cell; 60 s is the kill, not a style rule” |
| `return_result` / “use LLM reasoning” | talk or end the session | this agent ends only via `submit_actions` |
| hidden `self.h` | world model disappears | status line: `helpers: world_model.encode,predict` |

---

## B. Actor on q2d (compilation and recovery)

Live dir: `results/nemo_solver/20260903_160733_ls20_mdfiles_q2d/`. VOID: `20260903_160506_*` (PDEATHSIG). Details: `RESULT.md`, `ECONOMY.md`.

### Knowledge stayed in the air

- `hypotheses.md` H29: wide bar cols 14–18 is a death trap, confirmed. Turn 21 sent LEFT×8 “to col 14.” Step 103 GAME_OVER, stack at cols 9–13.
- Helper written turn 6. 4 `predict(` / 6 `encode(` / 1 `retrodict` in 126 cells. 0 of 24 rationales start `predict:`. No search over `z`.
- `predict` on RESET still returns `(45, 34)` / `bar_cells=84` (first-room spawn). After level-up the spawn is `(40, 29)` / 100 bar cells (step 24 and again at 104). Helper not rewritten. `read_helper` never called.

### Signals not used

Turn 9 UP×3 into the shaft: 2 pixels each, rows 61–62, colour 11→3. Block did not move. Batch continued. Same pattern for 31 HUD-only steps in the second room. 11-bar as a 42-move budget was written (H9, H34), then a 10-action batch ran it empty.

Trajectory cells *saw* `diff=4` = blocked (including after death: “6th LEFT blocked, 7th LEFT died”). The next submit was still a 10-action batch.

### No recovery; second room is not an accidental win

First room: 24 actions vs baseline 22, RHAE 0.84. Second room: 99 steps, 0 completions, 32% HUD-only/blocked. After GAME_OVER the harness RESET to the second-room spawn. Turn 22 rationale: still “UP to the wide bar, DOWN to the middle.”

### Tool use: hands good, control weak

177/177 cells succeeded, 0 `REJECTED`. `grid_array` 88/126. Notes files actually written. Perception works. Control (`predict` constraining the batch, helper refresh on level-up, diffs stopping a spray) does not.

Unused: `grid_image` (visual off, fair), `game_status`, `UNDO`, submitted `RESET`. ~25 cells are `await self.user_messages.get(timeout=5.0)` instead of `latest_state()`.

### Economy (thinking off)

Wall 1492 s ≈ LLM latency. Env 81 ms. Room 0: 10 min, 12% waste, 25 s/env. Room 1: 15 min, 32% waste, 8.8 s/env. After turn 6 they batch and depth falls to 1. The death is the cheapest turn (8.4 s, one LLM call, LEFT×8). Output 787 tok/env. 80% of 7.0M input was cache_read. Still above Son Pham’s 644 tok/action death line.

q2c (thinking on) is not a pair: EmptyContentError, never a finished turn, ~26,700 output tok/env, 91% reasoning.

---

## C. Measurement and ops

- `score_q2.py` prints FAIL (compute 0.202) by dividing by 124 env steps. The precommit named **turns**. 24/24 compute, helper present, 24/24 one submit. Leave the script; do not retune it to this result.
- `run_solver.py` `PR_SET_PDEATHSIG`: `setsid nohup` is not enough. Use `daemonize_q2d.py`.
- Only `ARC_LLM_EXTRA_BODY={"reasoning": {"enabled": false}}` turns thinking off on this OpenRouter route. Default `--effort-ladder 600:medium` would re-open it. `chat_template_kwargs` is ignored.
- +8 pipe-column offset does **not** bind here (numpy `(45,34)` = notes). Do not spend a run on it.
- HUD no-impact is detectable without naming colour 11 (rows 61–62 by change frequency). Step 103 is HUD-only **and** GAME_OVER. See `no-impact-offline/`.

---

## D. What not to treat as the next fix

WikiSkill / evolved `SKILL.md`: the science is already in the skill. The failures are notes that do not constrain batches, a helper that does not refresh, HUD-only steps that do not stop the batch, and a chassis that talks over the skill.

Latent bridge: not reachable on one Qwen-3.8-27B by 30 Sep.

V16 vision re-push: local readback already answered; image cells at chance.

Thinking-on: live hypothesis for compilation/recovery; unmeasured; its own card, not a silent hatch change inside a host change.
