# ARC-AGI-3 Resources

## Knowledge

- [ARC-AGI-3 Official Docs](https://docs.arcprize.org)
  The canonical source. Quickstart, agents, actions, scorecards, API reference. Use for: any factual claim about the benchmark. Index of all pages: https://docs.arcprize.org/llms.txt
- [Scoring Methodology](https://docs.arcprize.org/methodology.md)
  Per-level score = `(human_baseline_actions / ai_actions)^2`, capped at 1.15; game score = level-number-weighted average; total = mean over games. Use for: anything about what "performing well" means.
- [Agents Quickstart](https://docs.arcprize.org/agents-quickstart.md) and [Create Your Own Agent](https://docs.arcprize.org/create-agent.md)
  Official path for building agents on the toolkit. Use for: agent scaffolding patterns.
- [LLM Agents](https://docs.arcprize.org/llm_agents.md) and partner templates ([Anthropic](https://docs.arcprize.org/partner_templates/anthropic.md), [HuggingFace](https://docs.arcprize.org/partner_templates/huggingface.md), [LangChain](https://docs.arcprize.org/partner_templates/langchain.md))
  Reference implementations of LLM-driven agents. Use for: lesson stages where we study existing agent designs.
- [Benchmarking Tooling](https://docs.arcprize.org/benchmarking-agent.md) and [Swarms](https://docs.arcprize.org/swarms.md)
  Repeatable evaluation across games. Use for: the "measure everything" stage of the mission.
- [ARC Prize 2026 (Kaggle starter kit)](https://docs.arcprize.org/arc-prize-2026.md) and [Competition Mode](https://docs.arcprize.org/toolkit/competition_mode.md)
  Use for: if/when the mission targets the live competition.
- Local: installed packages `arc_agi` (toolkit: Arcade, wrappers, scorecards) and `arcengine` (game engine: sprites, levels, GameAction) in `.venv/lib/python3.14/site-packages/`
  The ground truth for API behavior — readable source. Use for: any question the docs don't answer; verified facts beat docs.
- Local: downloaded game source `environment_files/ls20/9607627b/ls20.py` (~2000 lines, obfuscated names)
  A real game's full implementation: sprites, levels, collision. Use for: understanding what games are made of. NOTE: reading game source to hand-craft a solver is a form of test-set leakage for the mission — use it to understand the *engine*, not to cheat individual games.
- [Blog: "A Recipe for Training Neural Networks" — Andrej Karpathy, 2019](https://karpathy.github.io/2019/04/25/recipe/)
  Our adopted ML methodology (user-requested): fail-silently paranoia, dumb baselines first, overfit-then-regularize, one change at a time. Translated for this project in `teach/reference/karpathy-recipe-for-arc.html`. Use for: checking every experiment plan before running it.
- [Paper: "On the Measure of Intelligence" — François Chollet, 2019](https://arxiv.org/abs/1911.01547)
  The intellectual foundation of ARC: intelligence = skill-acquisition efficiency. Use for: understanding *why* the benchmark scores efficiency, not just success.
- [ARC Prize GitHub org](https://github.com/arcprize)
  Official repos including ARC-AGI-3 agents/toolkit code. Use for: studying open-source agent implementations.

## Learning-model architectures (the new arc)

- [Paper: "ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence" (technical report, 2026)](https://arxiv.org/abs/2603.24621) · [PDF](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)
  The benchmark's own statement of the abilities it isolates: explore novel environments, acquire goals on the fly, build adaptable world models, learn continuously — no NL instructions. As of Mar 2026, frontier models score **sub-1%** while humans hit 100%. Use for: the authoritative "what is actually being tested" and why scale alone fails.
- [Paper: "Graph-Based Exploration for ARC-AGI-3 Interactive Reasoning Tasks" (2026)](https://arxiv.org/abs/2512.24156)
  **Closest prior art to our plan.** Object-centric *graph* state (entities + relations, not pixels); discovers controllable elements by experimentation (not a predefined action set); learns dynamics incrementally as approximate causal transitions; plans by graph-based search working backward from goals. Hybrid learned-dynamics + structured-search, not hardcoded. Open limits: sample efficiency and scaling to many interdependent objects. Use for: validating the spine and seeing where it breaks.
- [Paper: "Object-Centric World Models Meet Monte Carlo Tree Search" (2026)](https://arxiv.org/pdf/2601.06604)
  Object-centric world model + MCTS planning. Built-in priors: object permanence, explicit relational structure, compositional generalization. Demonstrates generalization across **different object counts and unseen object types** — directly the "one controlled entity → many" transfer we need. Use for: the planning half once we have a learned model.
- [Paper: "Object-Centric World Models from Few-Shot Annotations for Sample-Efficient RL" (2025)](https://arxiv.org/abs/2501.16443) and [OC-STORM / FIOC-WM line](https://www.emergentmind.com/topics/object-centric-world-models)
  Why object-centric beats pixel-reconstruction MBRL: capacity goes to semantically meaningful entities, dynamics are modular/disentangled → better sample efficiency and generalization. Use for: justifying the object-centric inductive bias as our core architectural prior.
- [Paper: "Schema Networks" (Vicarious, 2017)](https://arxiv.org/abs/1706.04317)
  The classic argument for object-relational dynamics enabling zero-shot transfer across game variants (Breakout). Use for: the historical case that the priors-in-architecture bet is sound.
- [Tutorial: "A Tutorial Introduction to the Minimum Description Length Principle" (Grünwald, 2004)](https://arxiv.org/abs/math/0406077)
  Primary grounding for lesson 0012's model-selection prior: regularities compress data, so prefer the shortest hypothesis plus residual encoding rather than an elaborate rule that merely memorizes observations. Our lesson uses a deliberately small two-part-code proxy, not full normalized maximum likelihood.
- [Paper: "Generalization as Search" (Mitchell, 1982)](https://doi.org/10.1016/0004-3702(82)90040-6)
  Primary grounding for lesson 0012's view of concept learning as search through hypotheses expressible in a fixed language, retaining descriptions consistent with positive and negative examples. Use for: version spaces, candidate elimination, and understanding why a bounded grammar can fail even when more data is available.
- [Paper: "Inductive Logic Programming" (Muggleton, 1991)](https://www.doc.ic.ac.uk/~shm/Papers/ilp.pdf)
  Seminal primary source for learning logical clause theories from examples plus background knowledge. Lesson 0012 is a deliberately tiny propositional/conjunctive precursor—not full first-order ILP—but inherits the separation between a GIVEN representation language and LEARNED clauses.

## Runway: fork these (agents, baselines, world models)

- [arcprize/ARC-AGI-3-Agents](https://github.com/arcprize/ARC-AGI-3-Agents) — the **official agent framework** with a random-agent baseline runnable against a single game (e.g. ls20). Use for: our harness and our floor. Fork this rather than re-scaffolding.
- [arcprize/arc-agi](https://github.com/arcprize) — the toolkit / Arcade API for running environments locally (we already use the installed `arc_agi`). Use for: the unlimited local simulator.
- [DriesSmit/ARC3-solution (StochasticGoose, Tufa Labs)](https://github.com/DriesSmit/ARC3-solution) — **lightest real RL starting point.** A CNN predicts which of ACTION1–6 will cause a frame change, biasing exploration toward *state-changing* actions; hash-deduplicated experience buffer; runnable per-game (`--game=vc33`). Note: this is a cruder cousin of our lesson-0008 controllability discovery — they bias toward "*something* changed," we already know *which object* changed. Use for: a concrete baseline to beat and a buffer/dedup pattern.
- [dolphin-in-a-coma/arc-agi-3-just-explore](https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore) — **3rd-place preview solution, unusually honest README.** Findings: in-context LLM interaction was *less* promising; brute-force exploration won. **The trap (read twice):** a world model that predicts next-frame from (frame, action) could improve sample efficiency, but it's unclear it helps *prioritize exploration* — "why should the correct pattern in ft09 be more surprising than an incorrect one?" This is the noisy-TV problem; see LR 0011 for why our agency-grounded design dodges it. Use for: the single most important design constraint on our 0009.
- [Paper: ARCLE — a DreamerV3 world-model agent applied to ARC](https://arxiv.org/abs/2407.20806) — argues the world model's vision+memory structure is crucial for ARC. Its stated limitation (world-model work focuses on single-agent-single-task) does **not** bite us — single-game-at-a-time is exactly our scope. Caveats for us: Dreamer is pixel-reconstruction + compute-heavy, and uses the model for *planning/imagination* (trap-free) not *curiosity* (trap-prone) — keep those two uses separate. Use for: the "if we had compute" reference point and the planning-via-imagination pattern.

## Wisdom (Communities)

- [ARC Prize Discord](https://discord.gg/9b77dPAmcA)
  The main community for ARC-AGI-3 builders; toolkit devs are active there. Use for: questions the docs can't answer, agent design discussion.
- [Kaggle ARC Prize forums](https://www.kaggle.com/competitions)
  Competition discussion threads when ARC Prize 2026 is live. Use for: seeing what approaches others try, post-competition write-ups.

## Gaps

- Human baseline data per game (which games are hardest for humans) — check if exposed anywhere beyond `baseline_actions` in metadata.
