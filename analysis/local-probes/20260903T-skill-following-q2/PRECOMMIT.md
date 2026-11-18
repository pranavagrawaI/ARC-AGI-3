# Q2 precommit — does this executor follow a short procedural skill?

Written before the run. Cheap local probe, not a Kaggle burn, but the measure is fixed in
advance so compliance cannot be scored after seeing the trace.

## Question

Given the reference 51-line `interactive-game-solver` skill, CodeAct, and real public `ls20`
turns, does `qwen/qwen3.8-27b` execute the skill's procedure, or substitute its own?

This is the gate the WikiSkill/B2 work is blocked on. If the executor does not run a hand-written
procedure, no result about an *evolved* procedure is interpretable — a win and a loss would look
the same.

## Incumbent, from V15 (kernel 13, our Tycho actor, same model family)

The V15 prompt already told the actor `wmlib`/numpy were available and preloaded. Observed:

| | V15 |
|---|---|
| turns with a computation call before committing | 2 of 11 Actor calls ≈ 0.18 |
| persisted world model (`world_model.py` edited) | never |
| beliefs written (`notes/actor_beliefs.md` content) | 0 of 6 entries; all "no model-authored beliefs update" |
| an explicit self-correction that was then ignored | call 2: *"I shouldn't keep reading from the text display — better to use numpy"*, then kept reading it |

## The skill's checkable obligations

From `nooa/examples/arc_agi_3/skills/interactive-game-solver/SKILL.md`:

1. compute, don't squint — parse with `self.grid_array(...)`, work in numpy
2. persist the model as helpers — `encode(grid) -> z` under a declared `Z_SCHEMA`, `predict(z, a) -> z'`
3. retrodict each turn — compare `predict(z, a)` against the real next state
4. record hypotheses, knowledge and plans before deciding
5. turn contract — end with exactly one `submit_actions([...], rationale="predict: ...")`

## Measure (from the run's own artifacts, not from my reading of the transcript)

| metric | source | definition |
|---|---|---|
| compute rate | `agent_logs/**/events.jsonl` | turns with ≥1 `repl_execute` before `submit_actions` ÷ turns |
| helper persistence | `team_nemo/shared/helpers/` | any module defining `encode` and `predict`; first turn it appears |
| retrodiction rate | `ipc/actions.jsonl` | submissions whose `rationale` begins `predict:` ÷ submissions |
| turn contract | events + ipc | turns ending in exactly one `submit_actions` ÷ turns |
| knowledge writes | `team_nemo/shared/knowledge/` | files and bytes written |
| batch size | `ipc/actions.jsonl` | actions per submission |

Gameplay (levels, actions, RHAE bounds) is recorded but is **not** the gate.

## Precommitted gate

**Passes** if all three hold: compute rate ≥ 0.50, at least one persisted helper defining both
`encode` and `predict`, and turn-contract compliance ≥ 0.90.

**Fails** if compute rate < 0.50 or no `encode`/`predict` helper is ever written — i.e. the
executor is told the procedure and does not run it.

**Inconclusive** if the run aborts on harness, transport, or budget grounds before 15 turns.

## Result map

- Passes → skills are a live lever for this executor. B2/WikiSkill becomes writable as a card,
  still under its own holdout and contestant-prior constraints.
- Fails on compute rate → the interface, not the skill text, is the binding constraint. Next
  question is host-side forcing (a required computation call before a commit is accepted),
  not better skill prose.
- Fails only on helper persistence → it computes but will not externalise a model. That is the
  "good enough model" question, and it argues for the actor keeping notes rather than code.
- Passes procedurally but plays badly → expected, and not a reason to reject. This gate is about
  instruction-following, not RHAE.

## Caps

One public game (`ls20`), offline, `mdfiles` variant (no embedding dependency), turn cap 40,
spend cap $6 of the $33.50 remaining, checked against the OpenRouter credits endpoint before and
after. Abort on transport failure or if spend passes the cap mid-run.

## Claim ceiling

One game, one model, one skill, one run. Establishes whether this executor runs this procedure on
this game — not whether it would on 25 games, not whether an evolved skill would do better, and
nothing about the hidden half. Hosted `qwen/qwen3.8-27b` is the deployment's family and scale, not
its FP8-on-SGLang serving.

---

## Addendum — deviations forced by the local environment (recorded before results)

Three changes from the reference invocation, none of which alter the skill, the model, or the
measure:

1. **`--sandbox inproc` instead of the default `drop`.** The uid-drop layer needs root
   (`must start as root to drop privileges`), which this session does not have. The run keeps the
   in-process AST cell guard, module denylist and `open()` jail, but loses the kernel-enforced
   backstop that makes the game source unreadable.
   *Consequence:* this run is **not containment-qualified**. Any gameplay number from it is not
   admissible evidence. The compliance metrics are still usable, but I must verify no cell read
   the game source rather than rely on the sandbox — post-hoc check: grep every `repl_execute`
   cell for `environment_files`, `ls20.py`, and absolute paths outside the neutral run dir. If any
   hit, the run is void.
2. **`ARC_DATA_DIR` set to the repo root.** The example derives its data dir from the nooa
   checkout, which has no `environment_files/`; ours does. This only tells the SDK where the
   offline game lives.
3. **Model id passed bare as `qwen/qwen3.8-27b`.** The example documents an `openai/` provider
   prefix, but LiteLLM forwards the string unchanged and OpenRouter rejects
   `openai/qwen/qwen3.8-27b` as "not a valid model ID".

Arm: `--visual off`. The skill's first instruction is "compute, don't squint", so the cleanest
test of whether it computes is the one where squinting is not on offer. The vision probe already
established that images do not deliver exact cells to this executor.

## Addendum 2 — which skill, recorded before results

The example ships **two** skills and the CLI default is not the one this precommit was written
against:

| skill | lines | contract |
|---|---:|---|
| `grid-game-solver` **(CLI default)** | 93 | hypothesis-driven: observe, hypothesise, experiment, record; "build a world model in code" as a loose `parse(grid) -> dict` |
| `interactive-game-solver` | 51 | the paper's world-model skill: `encode(grid) -> z` under a declared `Z_SCHEMA`, `predict(z, a) -> z'`, retrodict every turn, search once trusted |

These are the paper's two arms: `interactive-game-solver` is the "world-model skill" that reached
50.2 / 85.1 RHAE, and `grid-game-solver` is the "hypothesis-driven baseline skill" at 41.7.

This probe runs **`--skill interactive-game-solver`**, chosen because the precommitted metrics
(`encode`/`predict` persistence, retrodiction rate) are that skill's obligations. The first two
aborted attempts (`q2a`, `q2b`) silently used the `grid-game-solver` default; their metrics would
not have matched the gate, and they produced no completed turn in any case.

A run of the 93-line baseline is a reasonable follow-on — comparing compliance across two skill
forms would say whether *specificity* helps this executor — but it is not this gate.

## Addendum 3 — the hang, and what it says already

Attempt `q2a` stalled for 9.5 minutes inside a single CodeAct cell (655 characters, no output,
no completion) under `--sandbox inproc`, which has no per-cell timeout. This reproduces the
failure the NOOA paper reports for two of its 25 games: *"ad-hoc, in-cell searches that lacked the
bounds (max_depth, visited sets, node budgets) their own persisted planners carried."*

The fix is `--sandbox-cells --sandbox-require --sandbox-cpu-s 60`, which adds a guarded worker per
cell with a hard timeout, memory/CPU caps, **Landlock filesystem confinement and a seccomp network
block**. That also restores most of the containment lost by dropping from `--sandbox drop`: the
game source is confined by Landlock even without the uid drop. The post-hoc game-source grep in
Addendum 1 still applies.

Two observations that are already evidence, before the gate is scored:
1. This executor **does emit CodeAct cells** — `repl_execute` appears in the first turn of every
   attempt, unlike V15's actor, which reached `run_python` twice in eleven calls.
2. It writes cells that can run unbounded. If the target design ships, per-cell timeouts are not
   optional hardening; they are load-bearing.

## Addendum 4 — reasoning off is a treatment change, not a transport repair

`q2c` was killed on `EmptyContentError`: OpenRouter's default-on thinking put every token in
`reasoning` and left `content` null; NOOA retries that three times and fails the turn. The turns
never returned. A conclusion from that session — that this executor computes fluently and cannot
terminate a turn — is retracted. It is not supported.

The hatch that stops the error is `ARC_LLM_EXTRA_BODY={"reasoning": {"enabled": false}}` in
`~/projects/arc-agi-3-reference/nooa/examples/arc_agi_3/arc_llm.py` (uncommitted). Verified live
through `CompletionClient.call()`: with the hatch, a short "OK" costs 2 completion tokens and 0
reasoning tokens; without it, 164 completion / 145 reasoning and a leading blank `content`.

That is a different actor. The original gate asked whether this executor follows a short
procedural skill. With thinking disabled it is a faster actor, and given the tempo evidence it is
the actor the rest of the month needs to know about. It is **not** the same treatment as `q2c`.
Log it here rather than let it pass as a repair of `q2c`.

`q2c` numbers that survive because they do not depend on a completed turn contract (measured from
emitted cell source and OTel usage, 2026-09-03 scorer): compute_rate 0.727; 98 of 109 emitted
cells (90%) touch `grid_array` / numpy; no `encode`/`predict` helper; 1 agent submission against
11 env steps (the rest are harness force-advances); 267,057 output tokens of which 243,114 are
reasoning, for 10 gameplay steps ≈ 26,700 output tokens per env action. Son Pham & Mark Barney's
state-graph variant died at 644 tokens/action. That ratio is why this actor, not the thinking-on
one, is the next measurement.

Arm `q2d`, same skill / game / sandbox / visual-off / caps as above, plus:

4. **`ARC_LLM_EXTRA_BODY={"reasoning": {"enabled": false}}`.** Thinking off for every call.
5. **`--effort-ladder ''`.** `run_solver.py` defaults to `600:medium`, which injects
   `reasoning_effort` on later calls of a long turn. Leaving that default on would re-open the
   treatment after ten minutes. Empty string keeps a single fixed (off) effort.

Same gate, same 15-turn inconclusive floor, same $6 spend cap. Invalid if `EmptyContentError`
still dominates, or if usage shows reasoning tokens returning. Tokens per committed environment
action are reported next to the gate; they are not the gate.

`--sandbox drop` still needs root, hence `inproc`. Unbounded CodeAct cells still hang, hence
`--sandbox-cells --sandbox-require --sandbox-cpu-s 60`.

`20260903_160506_ls20_mdfiles_q2d` is **VOID**. `run_solver.py` sets `PR_SET_PDEATHSIG` to
SIGTERM, so when the launch shell exited the kernel killed the orchestrator. Two CodeAct cells
ran; no gate-length trace. Not a model result. Relaunch is daemonized (double-fork, parent PID 1)
so that signal cannot fire. Same tag `q2d`; timestamp distinguishes the dirs.
