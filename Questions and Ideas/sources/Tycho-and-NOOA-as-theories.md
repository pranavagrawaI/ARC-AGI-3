# Tycho and NOOA as theories

Read 2026-09-03 from the papers and implementations in
`~/projects/arc-agi-3-reference/`. This is a reconstruction of what you must
hold in mind to change the solver without destroying it. It is not a substitute
for the papers.

Sources:

- Tycho paper: `papers/tycho-2607.28287.pdf` (Lehmann, Aioanei, Vahdati)
- Tycho code: `tycho/` at `f68912a` (exact solver pin)
- NOOA paper: `papers/nooa-2607.20709.pdf` (Furgale et al.)
- NOOA official: `nooa/` at `2d3f6eb`
- NOOA pin the hybrid imports: `nooa-pin/` at `55a4b57`
- Hybrid: `solvers/tycho_nooa/`
- Live Kaggle contact: `analysis/deployment-probes/20260903T-public1-action-demand-v15/MODEL-BEHAVIOUR-ANALYSIS.md`

---

## 1. Tycho's theory

ARC-AGI-3 is active identification of a **rendered deterministic Moore machine**.
The grid is an output of hidden state, not the state. Two identical grids can
have different successors (scrolling viewport). Animation frames are evidence
about a transition; they are not frames you act from.

A useful model is a Python program with `init_state`, `transition`, `render`,
`outcome`. `render` may abstain (`-1` / `⊥`) on unknown cells. Outcomes are
checked separately from transitions: a model can replay motion and still have
the wrong goal.

The actor alone commits environment actions. The model is advisory. A validated
plan is guarded continuation: it is shown only while the model hash, observed
frame, and availability still agree.

**Active abstraction** is the joint problem: build a testable model from costly
interaction, *and* decide when constructing, repairing, using, or bypassing it
is worth the cost.

### The matched result that is the theory

Opus 4.8, one trajectory per game, all 25 public games:

| Policy | Who edits the model | When | RHAE | Transition match | Builder calls |
|---|---|---|---|---|---|
| No world model | — | never | **79.07** | — | 0 |
| Single | actor | at its discretion | 85.36 | 13.9% | 0 |
| Orchestrator | builder | actor requests | **88.49** | 16.2% | 147 |
| Trigger | builder | harness after falsification | 83.07 | **88.1%** | **1,192** |

Trigger produces the most exact simulators and the worst of the three
model-using scores. Transition match is not play. sk48 is the case: exact
gripper-and-block mechanics, unresolved objective, 13 builder calls, 314
actions, 0 levels. Trigger happened to solve that game; orchestrator did not.
No policy dominates every game.

Selected policy: **orchestrator**. GPT-5.6 Sol and Opus 5 both reach 100.00
RHAE under it. Opus 5 uses 6,641 actions; GPT uses 7,766.

At a $100/game inference prefix, **no-world-model leads**. Model construction
has not paid off yet. Orchestrator has mostly paid off by $250.

Tycho also reports that a frontier model with only the durable typed evidence
workspace is already a strong agent (79 RHAE). The workbench is optional
machinery around that.

### What Tycho's code actually is

Four modes in `tycho/agent/modes.py`, one `ModeSpec` each. The actor prompt
paragraphs are `wm_none.j2`, `wm_single.j2`, `wm_orch.j2`, `wm_trigger.j2`.
Orchestrator's instruction: call `invoke_builder` a few times per level, not
every turn; put beliefs in `reason`; decide the action yourself.

Evidence lives on disk. `wmlib` queries decision frames, archived resets,
terminals, and animation separately. Conversation clears at level boundaries;
a scribe writes summaries first. Workspace persists.

---

## 2. NOOA's theory

An agent is a Python object. Fields are state. Methods are capabilities.
Docstrings are prompts. Type annotations are contracts. `...` means an
LLM loop; a real body stays deterministic Python.

Six model-facing capabilities on one surface:

1. typed input/output
2. pass-by-reference over live objects
3. code as action (CodeAct REPL)
4. programmable loop engineering (ordinary Python)
5. explicit object state
6. model-callable harness APIs (context blocks, event query)

The model does not receive a JSON tool menu as its principal interface. It
writes Python against `self`, sees bounded previews of live values, keeps
intermediates as locals, and finishes with `return_result` validated against
the method's return type. Tool outputs do not have to re-enter the prompt as
text.

The interface matters most when the model supplies less planning discipline of
its own. On SWE-bench, NOOA's lead over OpenCode/PI is largest with reasoning
off.

### The ARC result that is the theory

NVIDIA compressed DreamTeam (six specialized agents, 1,821 lines of role
prompts, 4,690-line retrodiction engine) into **one agent and a ~50-line
skill**. The CodeAct REPL is the simulator. Context blocks are shared state.
Memory is the team's ledger.

World-model skill: persist `encode(grid) -> z` and `predict(z, action) -> z'`,
retrodict every turn, search once trusted, end every turn with
`submit_actions([...], rationale="predict: ...")`.

Under the two-hour fleet cap:

| Condition | Model | RHAE |
|---|---|---|
| Raw model (ARC Prize eval) | GPT-5.6 Sol | 13.3 |
| Hypothesis skill + memory | GPT-5.5 | 41.7 |
| World-model skill + markdown files | GPT-5.5 | 38.4 |
| World-model skill + MemoryManager | GPT-5.5 | **50.2** |
| World-model skill + MemoryManager | GPT-5.6 Sol | **85.1** |

Memory vs the same skill with markdown files: **+11.8 RHAE**. That is one of
the few measured deltas in this literature. Tycho with the same GPT-5.6 Sol
reaches 100; NOOA reaches 85 at ~$13–20/game versus Tycho's ~$179 mean.

NOOA's official solver (`nooa/examples/arc_agi_3/solver_agent.py`) is CodeAct.
The skill says: compute, don't squint; `self.grid_array`; numpy; persist
helpers; `submit_actions` a sequence; consult memory before deciding.

Their world-model contract is **encode/predict**, not Tycho's
State/transition/render/outcome. They batch actions. They do not run Tycho's
four-policy experiment.

---

## 3. The hybrid's actual theory

`solvers/tycho_nooa/` is Tycho's workspace, tools, modes, verifier, and
planner, hosted by a NOOA `Agent` whose live strategy is
`NativeToolStrategy`, not `CodeActStrategy`. Both actor and builder set
`context={"self": None, "state": None}`. The model sees a Tycho tool menu
and a filesystem.

This is documented in `Tycho-NOOA Architecture - What Remains to Steal.md`
(2026-08-21). It remains true.

Local defaults that contradict the papers:

- Hybrid `DEFAULT_MODE = "trigger"` (`solvers/tycho_nooa/src/tycho_nooa/modes.py`).
  Tycho selected orchestrator. Trigger is the policy that buys transition
  match and spends 1,192 builder calls for a worse score.
- Kaggle V13–V15 set `TYCHO_MODE = 'single'`
  (`deployment/kaggle/s0-public1-action-demand-v15-candidate/build_notebook.py`).
  Single is the expensive integrated policy. At low inference budgets,
  Tycho's no-world-model leads.
- The hybrid added forcing Tycho does not have: world-model checkpoint after
  every builder pass, actor-beliefs breadcrumb after every action. Those
  make the system more trigger-like.
- CodeAct exists only as `sterile_execution`, offline, not on the live path.
- MemoryManager is unused. Notes are files. V15's `actor_beliefs.md` after
  eight actions: six breadcrumbs, all "no model-authored beliefs update".

The 2026-08-21 note already said the largest unstolen NOOA strength is the
agent as a typed live object. The Kaggle path has since spent its days on
transport: SGLang, thinking-off action-demand, eight reconciled `ls20`
actions. That work answered a different question.

---

## 4. What V15 showed about the missing theory

Kernel 13, Qwen 3.8 27B, `ls20`, 8 actions, 0 levels, score 0.0.

98% of 1,596 s was spent before the first action. Three of five exploratory
calls emitted no tool: 48,000 tokens of hand-parsing the 64×64 ASCII grid
inlined in the prompt. Call 2 wrote "I shouldn't keep reading from the text
display — better to use numpy" and then kept reading the text display. One
`run_python` got the palette and sprite in ~200 tokens. The model never
identified the controllable object. Forced `take_actions` with thinking off
then walked RIGHT into a wall.

Tycho's theory says: query evidence through `wmlib`, do not reparse the grid
as if it were Markov, spend an action only when it distinguishes hypotheses
or advances a working theory.

NOOA's theory says: the grid is a live array; the model writes numpy against
it; intermediates stay objects; a turn ends with a typed submission.

V15 did neither. It did transduction over ASCII, then a host-forced action.

---

## 5. What this means for a score above 10 by 30 September

The 100 RHAE numbers are Opus 5 / GPT-5.6 Sol. Raw GPT-5.6 Sol is 13.3.
NOOA's harness multiplies that model by ~6×. Tycho's selected policy takes
the same model to 100. We are running Qwen 3.8 27B. The relevant Tycho
number is not 100. It is that **no-world-model already does most of the
work for a strong model**, and that **at cheap inference, modeling has not
paid off**. The relevant NOOA number is that **the interface substitutes
for missing model discipline**, and that **memory vs files is +11.8**.

A score above 10 on the hidden half with Qwen is closer to "does the model
take discriminating actions at all" than to "does the Moore-machine
workbench match Tycho's orchestrator". V15 did not take discriminating
actions.

Ordered consequences, not a build list:

1. The model must not be asked to parse 64×64 integer text as its primary
   perception. That is the V15 failure. NOOA's visual grid + numpy path
   and Tycho's `wmlib`/preloaded `grid` both exist to prevent it.
2. Do not default to trigger, and do not ship single on Kaggle because it
   is "simpler". Tycho measured this. Orchestrator is the selected policy.
   No-world-model is the cheap baseline. Trigger is the expensive wrong
   bet unless a specific game needs automatic repair (sk48).
3. A more exact `world_model.py` is not a reason to keep a builder running.
   Check outcome identification and whether the actor uses the advice.
4. CodeAct is not a taste. For a weaker model it is the interface that
   keeps computation out of the transcript. The native loop paid five
   inferences to fail to emit a tool.
5. Memory vs files is a measured +11.8 on GPT-5.5. It is untested on Qwen.
   It is a cheaper experiment than another transport probe once perception
   and action commitment work.

---

## 6. Reading order

Do not read linearly. Read until the next change would be different.

1. Tycho paper: abstract; §3 Moore machines and typed history; Table 1;
   Tables 7–8; §5.2 case studies (cd82, tu93, sk48).
2. `tycho/docs/ARCHITECTURE.md`, `tycho/agent/modes.py`,
   `tycho/prompts/partials/wm_orch.j2`.
3. NOOA paper: §2 principles; §3 loop; §4.4 ARC (the +11.8 and the 13.3→85.1).
4. `nooa/examples/arc_agi_3/skills/interactive-game-solver/SKILL.md` and
   `solver_agent.py` through `submit_actions` / `handle`.
5. Hybrid `solvers/tycho_nooa/README.md` target diagram, then V15
   `MODEL-BEHAVIOUR-ANALYSIS.md`.

The 2026-08-21 steal-list is still the right map of the hybrid. It is not
the theory of either paper. The papers' theories are the two tables and
the four Tycho cases.
