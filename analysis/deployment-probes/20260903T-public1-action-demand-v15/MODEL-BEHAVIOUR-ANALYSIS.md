# V15 model-behaviour analysis — what the model does, the rate-determining step, why it over-thinks

Evidence: V15 kernel-13 output (`ls20-794xo7xe`), all 11 decision checkpoints, `llm-trace.jsonl`,
`action-batch-trace.jsonl`, `server.log` (SGLang), `official-launcher-result.json`, per-turn diffs.
Every number below is from those artifacts.

---

## 1. Timeline — where the 1,596 s went

| clock | event |
|---|---|
| 20:07:19 | probe start |
| 20:12:43 | scorecard created, first Actor request issued |
| **20:38:44** | **first action committed (ACTION4 / RIGHT) — +1,561 s (26 min)** |
| 20:38:50 | actions 1–3 (batch release, no model call) |
| 20:38:57 → 20:39:18 | actions 4–7, one forced model call each, ~7 s apart |
| 20:39:19 | launcher finishes: "agent took 8 actions, took 1596.1 seconds" |

**98% of the run (1,561 of 1,596 s) was spent before the first action.** The other 7 actions
took 34 s combined.

---

## 2. What the model is actually doing

### Input it receives each Actor call
- **system** (9,842 chars): abstract doctrine — *"An option is asymmetric exposure plus the right
  to choose and keep… state doubt is paid by rebuild, goal doubt only by contact… Use optionality
  intelligently."* Plus tool docs for `run_python` / `symmetry` / `wmlib`.
- **user #1** (10,334 chars): the turn dict, then the **full 64×64 grid rendered as pipe-delimited
  integer text** (`y31: 3 3 3 3 3 0 3 3 | …`), 8 column-groups per row, 64 rows.
- **user #2** (298 chars): "DECISION CLOCK — Model calls used: N of 6… commit one ARC action…
  Investigate only for a result that could change that commitment."

### What it produces (sequence 0, the 26-minute phase)

| call | completion / reasoning tok | finish | tool call | outcome |
|---|---|---|---|---|
| 1 | 16,000 / 16,000 | length | **none** | parsed grid in prose, ran out of budget mid-sentence ("So the object") |
| 2 | 16,000 / 16,000 | length | **none** | same; drafted a `run_python` it never emitted |
| 3 | 11,292 / 10,748 | tool_calls | `ls`, `run_python` | **the only productive exploratory call** |
| 4 | 132 / 25 | tool_calls | `run_python` | forced retry: call 3's output was "too large… externalized" to a file |
| 5 | 16,000 / 16,000 | length | **none** | re-reasoned the grid from scratch again; drafted code it never emitted |
| 6 | 99 / **0** | tool_calls | `take_actions` | forced call (thinking off): committed RIGHT in 99 tokens |

**Three of the five exploratory calls (1, 2, 5 = 48,000 tokens) emitted nothing.** They each
manually re-parse the ASCII grid — deriving bounding boxes and column offsets by hand — and
repeatedly catch themselves miscounting: *"wait, I misread earlier"*, *"I mislabeled the groups
earlier"*, *"Oh! I misread"*. Call 2 even writes *"I shouldn't keep reading from the text display —
better to use numpy"* and then keeps reading from the text display.

### It never identified what it controls

Call 3's `run_python` got the exact answer in ~200 tokens: palette `{0:3, 1:2, 3:892, 4:2609,
5:439, 8:12, 9:45, 11:84, 12:10}`, sprite `0/1` at rows 31–33 cols 20–22, `c` block at rows 45–46
cols 34–38, `9` block at rows 47–49 cols 34–38.

The per-turn diffs show the real mechanic:
- **RIGHT translates the `c`+`9` compound block +5 columns per press** (cols 34–38 → 39–43 → 44–48),
  then it stops.
- The `11` band on rows 61–62 **depletes one column per action** — a move counter / fuel bar.
- The `0/1` sprite never moves.

The model's committed rationales show it never worked this out:
- c6: *"grid cell to the right appears 4 = walkable"* — guessing.
- c7: *"RIGHT clearly translated the visible 5x3 marker/carry cluster… Keep moving RIGHT toward the
  9-block (which appears to be the objective/door)"* — it is **moving the 9-block and calling it the
  goal**.
- c9: *"RIGHT advances the path segment even though I issued DOWN"* — it is reading the fuel-bar
  depletion as a path and is now fully lost.
- c8–c11: walked RIGHT into a wall for the last four actions.

`notes/actor_beliefs.md` after the run: six entries, all *"model committed \<action\> with no
model-authored beliefs update"*. It never wrote `world_model.md` or edited `world_model.py`. Every
turn started cold.

**Result: 8 actions, block nudged ~10 columns, 0 levels, score 0.0.**

---

## 3. The rate-determining step

### Primary: reasoning-token **decode** at ~41 tok/s

- SGLang decode throughput: median 38.3, max 41.2 tok/s (1,501 samples), single stream
  (`#running-req: 1` — one game, nothing to batch).
- Sequence-0 completion tokens ≈ **59,400** (calls 1–5). At ~41 tok/s that is **~1,450 s ≈ 24 min**,
  which accounts for essentially the entire 26-minute pre-first-action gap.
- Of those 59,400 tokens, **~48,000 (calls 1, 2, 5) produced no tool call** — pure waste.

Why 41 tok/s is low for this GPU: 27B FP8 model, `--attention-backend triton` +
`--sampling-backend pytorch` (both unoptimised), `--kv-cache-dtype fp8_e4m3`. A tuned backend
(flashinfer / fa3) would likely be 2–3× faster, but that only rescales the problem — the model
still should not be generating 48,000 useless reasoning tokens.

### Secondary (only visible once thinking is off): uncached **prefill**

- `#cached-token: 0` on almost every prefill. The NOOA actor rebuilds the conversation each turn,
  so there is no stable prefix for SGLang's radix cache. (Two consecutive forced calls got partial
  hits: `#cached-token: 10240` and `2048`.)
- Prefill runs ~6,900 tok/s (median, chunked at 2,048). Each forced call re-prefills a growing
  prompt: 13.6k tokens (call 6) → 23.2k (call 7) → 31.2k (calls 8–11).
- The 7 post-first actions took ~34 s total — ~5–7 s each, **dominated by re-prefilling 23–31k
  uncached tokens**; the actual decode is 81–161 tokens (<4 s).

### Not the bottleneck
Tool execution, host reconciliation, containment, and network are all sub-second. The launcher's
own overhead is negligible against 26 minutes of model decode.

---

## 4. Why it thinks so much

1. **The prompt rewards unbounded analysis.** "Actions are precious" (8-action budget) +
   "investigate only for a result that could change the commitment" → the model tries to fully
   understand a 64×64 world before spending one action. That is unbounded work, and it has 16,000
   thinking tokens of rope. The forced calls prove the capability is there: with `enable_thinking=
   false` and 2,048 tokens it commits in 81–161 tokens, every time.

2. **Grid-as-text is the worst possible input format for an LLM.** The model cannot reliably index
   `grid[r,c]` from pipe-delimited text. It burns thousands of tokens re-deriving column offsets
   and still gets them wrong. Meanwhile `run_python` returns the exact same information in ~200
   tokens (call 3) — but calls 1, 2, 5 front-load the manual parse and never reach the tool call.

3. **A compounding failure loop.** Call 3's `run_python` bundled too much (ls + three file dumps +
   region slices); its output was "externalized" (written to a file, not returned inline), so call
   4 had to re-run a trimmed version, and call 5 then re-reasoned the grid from scratch — a third
   full 16,000-token parse.

4. **No thinking cap on exploratory calls**, and no "act within N tokens or call `run_python`"
   forcing function. The model is free to spiral, and does.

5. **No durable learning.** Because it never writes `world_model.md` / `actor_beliefs.md` content,
   each turn re-derives everything. (In this run it barely mattered — it only got 6 turns — but at
   3 or 25 games this multiplies.)

---

## 5. Levers (for the pre-expansion review — not acted on)

Ranked by expected impact:

1. **Feed structured state, not raw ASCII.** Put the `wmlib` output in the prompt — palette,
   per-colour bounding boxes, the already-parsed diff — instead of (or alongside) the pipe grid.
   This removes the single largest reasoning sink.
2. **Bound exploratory thinking.** Cap exploratory `max_tokens`/thinking at ~4–6k, or disable
   thinking on exploratory calls too, with a "call `run_python` or commit" forcing rule. Calls 1,
   2, 5 would have been forced to produce *something*.
3. **Reframe the exploratory doctrine.** "Actions are precious / investigate only if it flips the
   decision" is inducing analysis paralysis. "Act, observe, revise" fits an 8-action budget better
   — the cheapest way to learn the mechanic here was to press a key.
4. **Stabilise the prompt prefix** so SGLang's radix cache hits — would cut forced-call latency
   ~2–4×. Smaller effect than 1–3.
5. **Faster serving backend** (flashinfer / fa3 attention, flashinfer sampling) — ~41 tok/s is low
   for an RTX PRO 6000. Rescales the problem; does not fix the token waste.

---

## 6. Three follow-up questions

### "Weren't we feeding structured state?" — No.

The V15 turn-0 Actor prompt is: the turn dict + the **raw 64×64 grid as pipe-delimited integer
text**, and nothing else. There is no pre-computed palette, segmentation, entity list, or bounding
boxes. Later turns add a text diff, still no structure.

The structuring tools (`wmlib`, `symmetry`, `dynamics`) are *preloaded into every `run_python`
call* and the system prompt describes them at length — but the model has to choose to invoke them.
The prompt hands over raw ASCII and says "the full grid and exact transition remain available
through run_python". So the machinery to produce structured state exists; it is not run *for* the
model, and in this run the model barely ran it itself (see below).

### "Didn't we solve the model treating a depleting resource as the goal?" — Solved once, in a different context, not carried into deployment.

A prior replay workspace for this exact game
(`analysis/production-actor-v1-replay/.../plan-assumption-repair/runtime/ls20/notes/world_model.md`)
had already worked out, correctly:
- the movable 5×5 block (colour 12 over colour 9), moving **by its own height/width per action**;
- *"Step counter / yellow bar: color 11 bar on rows 61–62… Each action shrinks bar from left by one
  column… This suggests a move budget / life counter"* — explicitly identified as a resource, not a
  goal;
- open hypothesis: *"move block to target location before bar depletes… maybe bar depletion causes
  GAME_OVER"*.

None of that reached the V15 run. The deployment seeds a **fresh** workspace: `notes/` was empty
(`notes -> []`), `world_model.py` was the generic 5,075-byte stub. There is no mechanism that
carries a solved `world_model.py` / `world_model.md` for a known game into a new deployment — the
wheelhouse ships only the generic stub. The system prompt *warns* about exactly this failure
(*"Counters, HUD/interface cells… lives, timers… can matter. Store such state explicitly"*), but a
warning is not a mechanism, and the model — which never wrote a model — ignored it and read the
bar depletion as *"the path advancing"*.

### "Did it use wmlib / symmetry / dynamics?" — No. None of it.

Complete tool usage for the entire run: `ls` ×1, `run_python` ×2, `take_actions` ×6. The two
`run_python` calls used only `os.listdir`, `open().read()`, `np.unique`, `np.nonzero`, and raw
`grid[...]` slicing.

Never called: `wmlib.segment` / `.frames` / `.transitions` / `.attempts` / `.diff`;
`symmetry.canon`; any `dynamics.*`. Never edited `world_model.py`. Never ran `plan.py`. Never wrote
`notes/world_model.md`. `verify.py`, `theory_probe.py`, `host_planner_contract.py` untouched.

The entire Tycho stack — segmentation, symmetry testing, the Moore-machine world model, the
planner, dynamics verification, the per-edit verification feedback loop — **sat idle**. The model
never reached the verification loop because it never authored a model. It did hand-numpy on the raw
grid and then guessed.

---

## 6b. Why local runs used wmlib/symmetry heavily and this one did not — different agent

The V15 deployment runs a **different agent configuration** from the default local one.

| | local default (`TYCHO_MODE` unset) | Kaggle V13–V15 |
|---|---|---|
| mode | **`trigger`** (`DEFAULT_MODE`) | **`single`** |
| Builder | `harness_fires_builder=True` — **a dedicated Builder agent runs automatically** on every dynamics divergence | `include_builder=False` — **no Builder exists** |
| who writes `world_model.py` | the Builder (`actor_writes_wm=False`) | the Actor (`actor_writes_wm=True`) |
| Builder budget | `max_tool_steps=40` per pass, `max_tokens=16000`/step, multiple passes | n/a |
| treatment | `control` → identity *"You are one persistent investigator playing an unfamiliar grid game"* | `single_decision_discipline` → identity swapped to *"You are a bounded decision-maker"* + the "act, don't investigate" strategy prompt |
| Actor call budget | `DEFAULT_MAX_ACTOR_CALLS = 26` | `ARC_LLM_MAX_ACTOR_CALLS = 6`, `…_AFTER_INITIAL = 1` |

**The Builder is the component that uses `wmlib` / `symmetry` / `dynamics` heavily** — its entire
job is to construct and verify `world_model.py` (dynamics(), outcome(), plan), it gets 40 tool
steps per pass, and its prompt is the one that enumerates the `wmlib` / `frames` / `transitions` /
`symmetry` preloads. In `trigger` mode it fires on its own; the Actor just consumes its verified
model and acts. That is almost certainly the frequent tool usage seen locally.

In `single` mode the Builder is **gone**. The Actor is supposed to do that work itself
(`actor_writes_wm=True`), but the same deployment simultaneously:
- swaps its identity from "investigator" to "bounded decision-maker",
- adds the discipline prompt telling it to act rather than investigate, and
- cuts its call budget from 26 to **6 for the first decision, 1 thereafter**.

So the machinery is not broken — it is **switched off by the deployment config**, and the fallback
path (Actor builds its own model) is starved of both the mandate and the budget to use it. In V15
the Actor spent 5 of its 6 first-decision calls flailing at the raw grid and never reached
`wmlib.segment`, `world_model.py`, `verify.py`, or `plan.py`.

Why the deployment is configured this way: the V13→V15 line deliberately minimises model calls for
Kaggle (27B model at ~41 tok/s, 7,200 s launcher cap, eventually 25 games). A `trigger`-mode
Builder doing dozens of tool steps per game would not fit that budget. `single` mode is the
deployment's simplification — and this run quantifies its cost: no world model is built, the Actor
guesses.

**Consequence for the roadmap:** V15 qualified the *transport* for the deployment agent, but the
deployment agent is a stripped-down `single`-mode Actor, not the `trigger`-mode Builder+Actor
validated locally. Any "expand to 3 / 25 games" decision should first settle which agent the
public-25 run is actually meant to be, and whether a bounded Builder (e.g. `orchestrator` mode, or
`trigger` with a small `builder_max_tool_steps`) fits the time budget.

---

## 7. Claim ceiling for this analysis

- The bottleneck is **exploratory reasoning-token decode**, quantified: ~48,000 wasted tokens ≈
  ~20 min of the 26-min pre-first-action phase.
- The model **did not solve or make progress on ls20** and **never identified the controllable
  entity**; V15 qualified the *transport*, not the *agent*.
- The levers above are hypotheses from one game on one public environment. Each needs its own
  bounded run before it is a claim.
