# symmetry-library-20260820T085718Z

Status: BUILT, VERIFIED against mathematics, and WIRED into the solver.

Code: `experiments/cards/symmetry-library-20260820T085718Z/` — `symmetry.py` (136 sloc),
`verify_math.py` (46 checks, GREEN), `overdelete_math.py` (11 passes).

## What changed, and why

Game-derived helpers stopped paying. Three games produced six helpers, and **every one of
them was required by exactly one game and invisible to the other two**; the third game also
inverted a policy the first two had agreed on. Extrapolating from three recordings to
twenty-five games was not going to work.

So the objective moved from the corpus to the mathematics. Klein's answer (1872) is that a
geometry *is* a group acting on a space, and its statements are exactly that group's
invariants. There is therefore nothing to invent separately called `congruent`, `similar`,
`chiral`: there is **one canonical form parameterised by which group you are working in**,
and the named relations are equalities of canonical forms.

Four quotients can be taken independently, and they are the whole design surface:

| quotient | question it answers |
|---|---|
| `point` | which subgroup of D4 (the square's eight symmetries) is allowed |
| `translate` | is position absolute, or free |
| `scale` | does a k-fold magnification count as the same thing |
| `colours` | are labels fixed, or permutable |

Every game-specific comparison found in the earlier work is a choice of those four. That is
the point: **the choosing is the model's job at play time, not the library's.**

D4 is not a modelling preference. Restricting integer 2x2 matrices to det = ±1 is exactly the
condition of being a bijection of the lattice, and the eight signed permutation matrices are
all of them.

## The verifier is mathematics, not recordings

A recording can only ever say "this fired on the frames I happened to have". These 46 checks
say whether the construction is right at all, and they fail loudly when it is not.

- group axioms — closure, identity, existence of inverses, associativity, Lagrange;
- **the subgroup lattice of D4 has exactly 10 members** (closure computed from arbitrary
  generators, never assumed);
- action axioms — `act(gh) == act(g)act(h)`;
- `canon` is invariant under D4, C4, C2, and under the point group about a fixed origin;
- `canon` is **sound and complete**: equal canonical forms iff a group element carries one
  figure to the other;
- **orbit–stabiliser** `|orbit| x |stab| == |G|`; the transporter is empty or a coset;
- **Burnside** on all 97 translation-normalised tetromino-sized figures in a 3x3 box;
- the refinement lattice: rigid ⟹ directly congruent ⟹ congruent ⟹ similar;
- **chirality computed two independent ways** — `chiral(F)`, and "the stabiliser contains no
  reflection" — which must agree on every figure;
- anchors: S-tetromino chiral, T and square achiral, S and Z enantiomorphic, |stab(S)| = 2,
  |stab(square)| = 8.

## Overdeletion: what is mechanism and what is vocabulary

Eleven passes. Seven crossed the boundary of failure, across five distinct axes:

| pass | verdict | what broke |
|---|---|---|
| `no-min-over-group` | RED (9) | `canon` stops being invariant; Burnside reports 97 orbits where there are 20 |
| `no-parameters` (one fixed geometry) | RED (10) | chirality, enantiomorphism, absolute position all die at once |
| `no-scale` | RED (3) | similarity is gone; magnification is not invertible on a lattice, so `primitive` is the only normal form available |
| `no-labels` | RED (3) | colour multiset, recolouring, parts-vs-whole |
| `no-colour-perm` | RED (1) | figures identical up to recolouring stop being one class |
| `no-translate-param` | RED (1) | own-frame and shared-frame comparison collapse into one |
| `no-structure` (first attempt) | crash | verifier named the deleted functions — **not a valid red**, retried fairly below |

Three passes survived on merit, and they say something worth keeping:

| pass | verdict | reading |
|---|---|---|
| `no-inverse` | GREEN | nothing needed it. **Deleted permanently.** |
| `no-named-relations` | GREEN | `congruent` / `directly_congruent` / `similar` / `chiral` / `enantiomorphic` are one line each over `canon`, reproducible in six lambdas |
| `no-structure` (fair retry) | GREEN | `stabiliser` / `transporter` / `orbit` are three lines each over `canon` |

**The core is three functions — `generate`, `act`, `canon`.** Everything else survived
deletion, so it is kept as vocabulary rather than as machinery, and the module says so.

## Two honest corrections to my own method

1. **Two greens in the first sweep were worthless.** My harness deleted the checks along with
   the feature they tested, so `no-scale` and `no-colour-perm` were acquitted by removal of
   the jury. Fixed by restating every requirement through `canon` alone; both then went RED.
2. **Two checks leaned on library conveniences.** "inverses" called `inverse()`, and
   "determinants" called `det()`, which would have made those helpers look load-bearing when
   only the verifier was leaning on them. Restated as existence claims over the group.

The passes also exposed two holes in the verifier, both now closed: nothing tested arbitrary
generated subgroups (the model must be able to name the group *its* game's actions generate,
not pick from a menu of four), and nothing tested absolute position.

## The old finding, restated as a theorem

The cd82 failure — 141 false positives at 22% precision — took three games to find. It is one
line of mathematics: canonicalising each colour class on its own discards the relative
position of the classes, so it is a strictly coarser equivalence than canonicalising the
labelled figure. The verifier exhibits a witness pair whose colour classes are separately
congruent while the labelled wholes are not rigidly equal.

## What this does not do

- **Extraction is untouched.** Getting figures out of a 64x64 grid — segmentation, hole
  filling, background choice — is not comparison and was not part of this pivot. The two
  segmentation policies remain empirically mutually load-bearing.
- **No game was consulted, and none should be** to justify the library. Games are now
  downstream users, not the objective.
- **No run has used it.** Wiring is verified by tests, not by an episode.

## Wiring: a separate file, not a merge into wmlib

Two ways to give the model both toolboxes. The code decides it, not taste:

- **Merge into `wmlib.py`** — `GameWorkspace._seed_worldmodel` rewrites wmlib from
  `tycho/workspace/wmlib_template.py` on *every* fresh start, commented "wmlib + helper
  scripts are OURS — always refresh to current version". Merging therefore means either
  editing a vendored file that any reinstall discards, or forking **1,627 lines** we do not
  own and would have to re-merge forever.
- **Separate file** — costs one `write_text` in our own `__init__`.

The only real argument for merging was discoverability, and it is measured, not theoretical:
this project has already recorded a null result caused by a capability the models could not
see. But both halves of discoverability turn out to sit on extension points **we already
own**, so the split costs nothing:

| what a merge would buy | how the split gets it |
|---|---|
| preloaded, no import to remember | `GatedToolExecutor._PREAMBLE = ToolExecutor._PREAMBLE + "…import symmetry…"` — upstream reads it as `self._PREAMBLE`, so a subclass override just works |
| named in `run_python`'s tool text | `WorkspaceToolSurface.native_tools` appends `_SYMMETRY_NOTE`, leaving upstream's description the source of truth for what it already covers |
| one place to look | the file's own docstring is a toolbox listing in wmlib's style, and names the four quotients and the parts-vs-whole trap |

An audit of what the model can actually perceive found the tool description was not enough,
and that one channel was worse than silent. `builder.system.j2` enumerates run_python's
preloads as a **closed list** — `grid`, `np`, `wmlib`, `frames`/`transitions` — written before
`symmetry` existed. An authoritative list that excludes a control does not omit it; it denies
it. Three channels now carry the module, all sourced from `symmetry_notice.py` so they cannot
drift: `TOOL_NOTE` on `run_python`, and `PROMPT_BRIEF` appended to the actor's system prompt
and to upstream's builder prompt (appended, never replacing — a test asserts upstream's own
handles survive). The distinction that made this worth doing: a tool description is read when
*choosing a tool*, a system prompt when *forming a hypothesis* — and reaching for a comparison
library is a hypothesis-time act.

The split also buys something a merge could not: `symmetry_template.py` is pure stdlib and is
imported directly by `tests/test_symmetry.py`, so **the theorems are proved about the exact
bytes the model receives** — a test asserts the seeded file is byte-identical. Folded into
1,627 lines of numpy-dependent observation code, the mathematical objective would have become
untestable in isolation.

`solvers/tycho_nooa/tests/test_symmetry.py` — 26 tests, all passing; full suite 91 passing.

## Next consequential action

Run an episode. The standing defect is a 1.00-accuracy physics model that never becomes a
goal test; the model now has, for the first time, a vocabulary in which a goal hypothesis can
be *stated*. Whether it reaches for it is an open empirical question, and the honest prior is
this project's own record: seeding a capability is not the same as it being used.
