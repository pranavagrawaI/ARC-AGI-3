# The curiosity trap, and why we drive exploration by agency + goal, not surprise

Pranav surfaced five runway sources (session 5, 2026-06-22; now in RESOURCES).
The load-bearing one is the dolphin-in-a-coma 3rd-place README and its warning,
which aims directly at the world-model direction we committed to in LR 0010.

**The trap (verbatim worry):** a world model that predicts next-frame from
(frame, action) may improve sample efficiency, but it's unclear it helps
*prioritize exploration* — "why should the correct pattern in ft09 be more
surprising than an incorrect one?" This is the classic **noisy-TV problem** of
intrinsic motivation: prediction error (surprise) is a bad proxy for
task-relevant novelty. A world model is most surprised by stochastic/visually
complex transitions, which need not be the rule that wins; the correct pattern,
once seen, is *low* surprise and gets de-prioritized.

**Two uses of a world model — very different risk:**
1. **Planning / imagination** (Dreamer, MuZero, OC-WM+MCTS): roll the model
   forward to *evaluate* candidate action sequences toward a known goal.
   Sample-efficiency win. **Trap does not apply.**
2. **Exploration / curiosity**: use prediction *error* to decide where to go.
   **Trap applies** — surprise ≠ value. The dolphin author's doubt is precisely
   about this use.
   Keep the two jobs separate. We will use a world model for (1), never (2).

**Our resolution (the design constraint on lesson 0009):** decompose what naive
curiosity conflates.
- **Map the rules with bounded, agency-grounded interaction-novelty.** Count a
  novelty only when it is *causally downstream of MY action* — an effect on the
  object I control, or on an object my avatar touched. This is the real noisy-TV
  defense: a stochastic background isn't caused by me and doesn't involve my
  avatar, so it earns no novelty. The target set is *finite* (try each
  object-interaction type once), unlike pixel-surprise which is unbounded. Our
  lesson-0008 controllability anchor is exactly the hook this needs;
  StochasticGoose's "bias toward state-changing actions" is a cruder version of
  the same instinct.
- **Choose among rules with the explicit GOAL signal**, never with surprise.
  Interaction-novelty tells you to *try* the right cycler AND the wrong cycler
  once each — it cannot tell correct from incorrect (the author's point exactly).
  Only the score / `levels_completed` signal can do credit assignment. So:
  novelty *maps* the dynamics; the goal signal *selects* and *sequences* them
  (planning over the learned model).

In one line: **surprise is a fine driver for mapping rules, a terrible proxy for
the goal.** Use agency-grounded interaction-novelty to learn the dynamics (a
factored, relational object-space world model — NOT pixel reconstruction), and
the score/level signal to plan. That sidesteps the trap by construction.

**Curriculum impact (refines LR 0010 spine):**
- 0009 = dynamics induction in *object/interaction* space (avatar touches object
  of type X → observed effect), explored by agency-grounded novelty, source
  unread. Explicitly NOT a pixel-prediction curiosity agent.
- 0010(new) = goal induction: bind effects to the score/level signal; recover the
  win condition.
- 0011(new) = planning: roll the induced relational model forward (BFS/MCTS) to
  hit the goal — world model used for imagination (trap-free).
- Baseline/floor to beat: StochasticGoose (DriesSmit) and the official
  ARC-AGI-3-Agents random agent.

**Habit reinforced:** he brings the field's hardest-won negative result and asks
whether our plan walks into it. It would have, if 0009 were a predict-the-frame
curiosity agent. It isn't, now — because he flagged the sentence.
