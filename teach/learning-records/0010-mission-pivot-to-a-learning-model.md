# Mission pivot: from hardcoded planner to a true learning model

Session 5 (2026-06-22). Pranav reframed the mission (confirmed; MISSION.md
rewritten). The trigger was LR 0009 (the real cheat is hardcoded rules).

**The new framing, in his words:**
- Competing in ARC-AGI-3; current SOTA is poor — "barely anything scores above 1,
  even the largest LLMs can't get through." So brute scale is a dead end for us.
- "We clearly cannot compete on scale, data, compute and architecture" — no
  frontier LLM of our own. The bet is **targeted heuristics + priors baked into
  the architecture** so a small model **generalizes new, unseen rules quickly**
  and sample-efficiently.
- Transfer is explicit: ls20 controls a single entity, but other games control
  multiple. The architecture's priors must make that a non-event, not a retrain.
- The 0001–0007 deterministic work was **scaffolding** to build game intuition
  (and he's confident it can be pushed to a full level-7 clear). That phase is
  done. "Now we start working on the real problem: a true learning model which
  can experiment, adapt, generalize and abstract."
- Roadmap: beat ls20 with a learning model → add a few games → beat all with ONE
  model → add more.
- Format: keep the lesson loop (teach → he questions → iterate) so he stays in
  the loop, but **expand scope per lesson** to get more done.

**What this changes for teaching:**
- The new arc's spine (draft, revise as ZPD develops):
  1. Research/decision lesson — the landscape of sample-efficient, prior-rich
     rule-learning architectures under our constraints; ground in real published
     work (RESOURCES is thin — must populate). Ends in the paradigm fork.
  2. Object-centric state — parse the 64x64 grid into typed objects + relations.
     The representation that makes everything downstream sample-efficient and is
     common to ~all viable approaches. (Prior: objectness.)
  3. Controllability discovery — induce which object(s) the actions move, from
     interaction. First honest scientist-loop win; his single-vs-many-controlled
     example IS the generalization test. (Prior: agentness.)
  4. Dynamics/interaction-rule induction — cycler/wall/door mechanics learned,
     not read; the matching gate as a learned hypothesis.
  5. Goal/reward induction; then planning with the *induced* model — beat ls20
     honestly, replacing 0007's oracle.
  6. Cross-game generalization / meta-learning toward one-model-many-games.
- Paradigm fork still open (symbolic/program-induction vs neural object-centric
  world model vs neuro-symbolic hybrid) — to be decided in lesson 1 of this arc,
  evidence-based, not from parametric guesses.

**Lessons now bigger:** combine knowledge + a build artifact per lesson.
