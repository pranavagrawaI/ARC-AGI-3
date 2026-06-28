# Mission: ARC-AGI-3 — Build a *Learning* Agent That Earns the Rules

## Why
Pranav is building an AI agent to compete in **ARC-AGI-3 (ARC Prize 2026, Kaggle)**.
The honest goal is not an agent that *executes* known rules but one that **learns
them**: experiments, adapts, generalizes, and abstracts the rules of a novel game
from interaction alone — the skill ARC actually measures.

The binding reality: we **cannot compete on scale, data, compute, or model size**
— no building a frontier LLM, and even the largest LLMs barely clear the games.
The bet is the opposite: **targeted heuristics + Core-Knowledge priors baked into
the architecture**, so a small model learns new rules sample-efficiently and
*transfers* to games it has never seen (e.g. one controlled entity → many
controlled entities is the *same* mechanism, not a retraining event).

The earlier deterministic-algorithm work (lessons 0001–0007) is now understood as
**scaffolding**: it built intuition for the games and proved a hand-coded planner
can approach a full clear. That phase is essentially done. The real work starts
here — a true learning model.

## Success looks like
- **Beat ls20 with a learning model**, not a hardcoded planner: the agent induces
  the game's rules (controllability, dynamics, win condition) from interaction,
  with `ls20.py` unread. Priors that are GIVEN must be Core-Knowledge only;
  task-specific rules must be LEARNED. (See LR 0009.)
- **Sample-efficiency is the scored north star (Target A, LR 0012):** the
  quantity we optimize is how *fast* the learner reaches competence on a novel
  game from zero, at test time — the slope of the acquisition curve, not its
  asymptote. "Beat all games" means *acquisition cost on game N+1 falls as games
  1…N are seen*, NOT a policy that covers all games after per-game tuning (that
  scores ~0). We scale the **learner/adapter** (transferable: priors +
  exploration-and-hypothesis procedure), never the **player** (weights/policy,
  which overfit per-game by the benchmark's design).
- Architectural priors are explicit and justified: we can say which inductive
  biases (objectness, agentness, relations, …) buy which generalizations.
- Repeatable benchmark + reasoning about sample efficiency and transfer deltas.

## Constraints
- **Resource-constrained**: local consumer GPU; no large-model pretraining. Favour
  object-centric / model-based / program-induction designs that are sample-
  efficient by construction over scale-hungry ones.
- **Lesson-loop with Pranav in the loop**: teach → he questions → iterate on ideas
  until something works. As of session 5, **expand scope per lesson** to move
  faster; each lesson still ties to the mission and ends in a tangible artifact.
- Methodology: Karpathy's recipe (dumb baselines, overfit-one-batch, one-change-
  at-a-time) still governs experiments. Background: ML + RL basics.
- Target: ARC Prize 2026 (Kaggle) — pace against the competition timeline.
- Every lesson states plainly which priors are GIVEN vs LEARNED (LR 0009).

### Locked architectural bets (session 6 — LR 0012/0013)
- **Hand-build the Core-Knowledge priors as a small hypothesis *family*, not
  fixed constants** (4a). Objectness isn't one function — connectivity by colour
  vs adjacency vs motion-coherence are different segmenters for different games.
  The prior is a parameterized space with a few test-time-selectable knobs, or it
  becomes the brittle hardcoding we're escaping. (Lesson 0008's same-colour
  segmenter is recognised as *one member* of that family, not the law.)
- **Amortize little, infer much — but the inference is *budgeted abduction*** (4b
  amended). Keep learned weights small; spend at test time. Inference is scored
  by **hypotheses-killed-per-action**, not unbounded search — every probe is an
  experiment chosen for information gain, because on this metric "slow" = "fail."
- **The world model's first job is FALSIFIER, not planner.** Early, it exists to
  refute candidate rules cheaply (in imagination) before a real action is spent
  testing them. Planning-through-the-model comes after the rules are known.
- **GIVEN = a generic causal-interaction vocabulary; LEARNED = which
  instantiations hold in this game.** The hypothesis grammar is game-agnostic
  (any object-class as trigger × an operator on an avatar attribute); we do NOT
  enumerate ls20's specific rules — that would be the LR-0009 cheat again.

## Out of scope
- Hardcoded per-game rule algorithms as the *end goal* — demoted to intuition-
  building scaffolding (done). The oracle planner (0007) is kept only as a
  skyline / upper bound.
- Training a frontier-scale LLM from scratch; competing on raw compute/data.
- ARC-AGI-1/2 except as context; authoring new platform games except where
  procedurally generated environments serve *training* the learner.
