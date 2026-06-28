# Lesson 0009 built + verified: the falsifier, and two findings real pixels forced

Session 6 (2026-06-23). Built `teach/lessons/0009-dynamics-falsifier.{py,html}` —
the explicit dynamics falsifier from LR 0013. Probed ls20 FIRST (Karpathy: know
your data) before designing: confirmed the avatar's own (9,15)+(12,10) signature
stays constant across 600 random steps — random rarely lands on a colour-cycler,
and cycling shows up as one-frame occlusion artifacts (size-10 blobs of colour
3/11), not a clean avatar change. That probe reshaped the lesson honestly.

**What 0009 is:** every object-class starts with all roles alive {WALKABLE,
BLOCKER, CHANGER}; each role predicts a 2-bit outcome (moved?, changed?); a
contact refutes the inconsistent roles. (moved=False, changed=True) is predicted
by NO role → empty set = grammar-gap alarm (fail-loud, the LR-0013 reason for
explicit-over-implicit). Metric: hypotheses-killed-per-action. Avatar tracked by
controllable SIZE (survives recolouring); target class read one cell ahead via
`component_at` flood. Offline synthetic single-layer world (known roles + a
planted out-of-schema tile) proves all three roles, the alarm, and the metric —
all 4 checks pass.

**Two findings the LIVE run forced (the valuable part):**
1. **Class-by-(colour,size) shatters on real pixels.** The floor is one big
   region whose connected-component *size* flickers as objects occlude it →
   one material splinters into many ghost classes (size 892/820/72/…), and the
   dominant ghost got a FALSE grammar-gap. Fix: key a material's role by
   **colour**; skip any target whose size ∈ avatar sizes (also kills the
   one-frame occlusion artifacts).
2. **Pure falsification is noise-intolerant.** A single mis-read frame emptied
   the hypothesis set (false alarm). One bad datum shouldn't refute a law. Fix:
   **evidence-weighted falsification** — a role dies only when the *majority*
   outcome (≥CONSISTENCY over ≥MIN_CONTACTS) contradicts it. Noise-free → reduces
   to one-shot falsification; noisy → survives. NB this is a refinement of, not a
   retreat from, fail-loud: the alarm now fires on a class whose *dominant*
   behaviour no role explains (genuine), not on noise.

**Verified live result (ls20, 1500 steps, post-fix):** colour 3 (floor) →
WALKABLE (995 contacts); colour 5 (boundary) → BLOCKER (7); colour 0 → WALKABLE;
6 hypotheses falsified; **zero false grammar-gaps**. The wall/boundary and floor
were discovered by refutation with NO colour named — the 0009 analogue of 0008
discovering the avatar.

**The honest seam to 0010 (built into the lesson):** cyclers correctly stayed
UNDECIDED, not forced to CHANGER. ls20 single-layer rendering occludes the avatar
on a tile, so a colour-cycle isn't a clean change to its signature — the coarse
`changed?` bit is blind to it. 0010 = **attribute-level perception of the
avatar's own body** (colour/shape/orientation), turning CHANGER detection real
and the cyclers into a learned transition table. Then goal-induction + planning.

**Pedagogy shipped:** the lesson HTML shows the failure-then-fix openly ("it
broke the first time") as a Karpathy beat — synthetic proved the logic, contact
with the environment proved the perception.
[[0013-0009-is-explicit-hypothesis-discrimination]] [[0008-... controllability]]
