# 0015 — The closed-alphabet doctrine, and the build it commits us to

Session 7 (2026-06-23). Two of Pranav's questions in a row broke open the
falsifier from 0009: (1) the `changed?` bit is avatar-egocentric, so a non-local
effect (switch opens a distant door) is *silently* absorbed into WALKABLE/BLOCKER
— worse than the cyclers, which at least stay undecided; (2) the role taxonomy
itself is a dead end — to reach goals you'd need arbitrarily fine, game-specific
categories, and a human cannot keep authoring them.

The resolution is **not** open-ended predicate invention. Pranav made the call:
build the **closed alphabet path, eagerly**.

> Hand-build the alphabet (slots, comparison lenses, identity tracking). The
> agent writes clauses over that alphabet. Only a failed clause-space earns a
> new letter.

Fineness comes from **composition over a small fixed primitive vocabulary**, not
from a longer list of categories. WALKABLE/BLOCKER/CHANGER were scaffolding for
the falsification *mechanism*, never the ontology. The mechanism carries over to
*clauses*; the fixed list does not. The human prior line moves from *categories*
down to *primitives + the search that combines them* — frozen in
`docs/learning_rules/LR_0016_closed_alphabet.md`.

**Why this and not the alternatives (grounded in the locked north star,
LR 0012 — sample-efficiency / fast test-time adaptation):**
- Learned latent world model: rejected as the core loop — learning a transition
  function per game from scratch fights sample-efficiency and kills fail-loud.
- LLM-as-proposer: deferred (not deleted) — reserved for the *rare* primitive-
  level gap, never touched in this sprint.
- Bounded symbolic clause search: chosen. Data-thin, interpretable, fail-loud.

**The corrected lesson ladder (sequential lesson files; replaces the 0010+ sketch
in NOTES):**
```
0010  Eager invariance-lattice perception   -> teach/lessons/0010-invariance-lattice-perception.{html,py}
0011  Persistent object identity (occlusion)-> teach/lessons/0011-object-identity-occlusion.{html,py}
0012  Depth-gated clause search (MDL-gated)
0013  Experiment selection by expected info gain
0014  Slot-gap detection
0015  Slot discovery — DEFERRED until 0014 fires a high-confidence gap
```
(NB: depth control lives *inside* 0012 — gate and search are one mechanism;
ungated search overfits immediately, MDL is what makes it sane. The old plan's
standalone "0013 depth control" is folded in.)

**FORMAT CORRECTION (Pranav, same session):** an earlier draft of this record
said "lessons stop being HTML and become real `src/` modules." Reversed — the
HTML lesson + companion `.py`, both in `teach/lessons/`, is the canonical unit and
was the point all along. No parallel `src/` tree. The companion runs from the repo
root: `uv run python teach/lessons/<NNNN-name>.py` (house style: a `__main__`
block of synthetic checks); a lesson reusing an earlier one loads it by path via
`importlib` (and must register it in `sys.modules` so 3.14 dataclasses resolve
annotations). The doctrine doc stays at `docs/learning_rules/LR_0016_closed_alphabet.md`.

**Milestone (NOT "win ls20"):** given interaction traces, the agent can represent
objects under multiple comparison lenses, track them under occlusion (occluded ≠
disappeared), generate minimal clauses over fixed primitives, choose probes by
expected information gain, and report whether a failure is a *logic* gap or a
*slot* gap.

Build order locked (as lessons): 0010 invariance-lattice -> 0011 object-identity
-> 0012 depth-gated clause search -> 0013 info-gain experiment policy -> 0014
slot-gap detector. No LLM proposer, no latent WM, no reading ls20 source except as
oracle skyline, no hardcoded roles. 0010 + 0011 BUILT & VERIFIED this session.
[[0009-the-real-cheat-is-hardcoded-rules]] [[0012-scale-the-learner-not-the-player]]
[[0013-0009-is-explicit-hypothesis-discrimination]] [[0014-falsifier-built-occlusion-and-noise-findings]]
