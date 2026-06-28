# The real cheat in 0007 is hardcoded RULES, not read-from-engine STATE

Pranav's critique of lesson 0007 (2026-06-22), and it cuts deeper than the
lesson's own "the cheat" disclaimer:

> "We haven't solved the problem of getting state, or inferring state. The
> issue is not whether we can build an algorithm that reaches the final state
> *if we give it priors* — it is how to get the algorithm to **learn the
> priors in the first place**."

**Why this is correct and important.** Lesson 0007 frames its cheat narrowly:
`read_oracle` reads live STATE (positions, attrs) from the engine instead of
from pixels, and promises lesson 0008 will "perceive it from pixels." But that
understates it. The truly hardcoded thing is the **rules**, all obtained by
reading the obfuscated `ls20.py` in session 4 (LR 0008):

- the win condition (match shape+color+rotation AND stand on the door),
- that three specific tile-tags are cyclers, one attribute each,
- that cyclers advance +1 (mod N), one-way,
- that one tag is a wall.

Perceiving those same facts from pixels (the planned 0008) STILL hands the agent
the rules — you only know to *look for* cyclers/doors/walls and to *apply the
matching gate* because you read the source. Swapping the oracle's data channel
from struct-fields to pixels changes nothing about the cheat.

**The distinction that resolves it (ARC's own framing, Chollet "On the Measure
of Intelligence"):** intelligence = skill-acquisition efficiency given priors +
experience. ARC deliberately fixes the priors to **Core Knowledge** — objectness,
agentness/goal-directedness, counting, basic geometry/topology — which humans
and agents may both assume. Everything else — the **task-specific rules of this
particular game** — must be *learned from interaction*, with no instructions.
ARC-AGI-3 is exactly this in interactive form. Our cheat = we smuggled
task-specific rules in as if they were priors. The benchmark IS the rule
induction we skipped.

**Curriculum correction this forces.**
- Relabel 0007 honestly: it is an **oracle skyline** — an upper bound showing
  what competent play looks like *once the rules are known*. Valuable as a target
  and a planner-skills lesson (taps_needed, BFS-with-avoid, reach-a-blocked-cell),
  NOT as progress on the real problem. Keep it; rename its role.
- The real next thread is the **scientist loop** we've gestured at since session
  2 (hypothesise → intervene → observe → update belief), now made literal:
  induce ONE game rule from interaction alone, source unread. Smallest honest
  win: discover "contacting tile-type-X cycles one of my state attributes" from
  frame-diffs + the observable attribute vector, with zero knowledge from
  `ls20.py`.
- "Perceive from pixels" (old 0008) and "induce the rules" are different axes.
  Real difficulty ladder: oracle+rules (0007) → perceive state, rules given →
  **induce rules from interaction, core-knowledge priors only** (the actual ARC
  task).

**Open strategic fork (do not resolve yet — paradigm choice is deferred to the
research lesson per NOTES):** a from-scratch symbolic rule-inducer vs. an LLM
agent that brings broad priors and does in-context induction. The competition
(ARC Prize 2026) rewards the second more, but the "learn the priors" purity
Pranav is after lives in the first. Surface, don't decide.

**Habit confirmed:** he audits the *integrity of the exercise*, not just its
output — caught that a passing, level-clearing agent had defined away the actual
problem. Teach forward: every future lesson must state plainly which priors are
GIVEN vs LEARNED.
