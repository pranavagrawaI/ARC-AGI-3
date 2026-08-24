"""What the model is told about workspace mathematics, in the one place that says it.

A seeded file the model never hears about is a file it never opens, and this project has
already paid for that once: the durable-surface nulls were void because the models could not
see the affordance. Discoverability here is not one channel but three, and they disagree by
default — upstream's builder prompt enumerates the run_python preloads as a CLOSED list
(`grid`, `np`, `wmlib`, `frames`/`transitions`) that cannot mention ours. An authoritative
list that excludes a control is worse than no list at all, so `PROMPT_BRIEF` corrects it
explicitly rather than merely adding to it.

The three channels, and who reads each:

  TOOL_NOTE     appended to run_python's description   actor and builder (tool surface)
  PROMPT_BRIEF  appended to both system prompts        actor (agent docstring),
                                                       builder (upstream builder.system)
  the file      `ls` lists it, its docstring teaches   whoever goes looking

Kept here rather than beside each use so the three cannot drift apart.
"""

from pathlib import Path

from tycho_nooa.trusted_runtime import RuntimeState, inspect_runtime

# Deliberately terse: this rides on every run_python description, on every call, forever.
_SYMMETRY_TOOL_NOTE = (
    " `symmetry` is also preloaded, for comparing two spatial arrangements: "
    "`symmetry.canon(fig, point='full'|'rotations'|'halfturn'|'none', translate=True|False, "
    "scale=True|False, colours='fixed'|'any')` returns a canonical form, and two figures "
    "are the same — under the quotients you chose — iff their canonical forms are ==. "
    "A figure is a dict {(y, x): colour}. "
    "`congruent`/`similar`/`chiral`/`enantiomorphic` are one-line presets over it, and "
    "`stabiliser`/`transporter`/`orbit` say which of the square's eight symmetries fix a "
    "figure, carry one onto another, or are reachable. There is no helper for this game's "
    "win condition: choose the four quotients yourself and read symmetry.py's docstring "
    "for what each one decides."
)
_DYNAMICS_TOOL_NOTE = (
    " `dynamics` is preloaded too. Use `dynamics.key_counterexample(model, "
    "[left, right], actions=..., max_depth=...)` for the shortest bounded suffix "
    "that refutes a planner-key merge, and `dynamics.action_laws(model, states, "
    "actions=..., projection=..., representation='name')` for bounded idempotence, inverse, commutation, "
    "and many-to-one certificates. No separator within a bound is inconclusive, "
    "never proof of equivalence."
)
TOOL_NOTE = {"run_python": _SYMMETRY_TOOL_NOTE + _DYNAMICS_TOOL_NOTE}

# The four quotients are named here, not just pointed at, because the choice IS the work:
# every named relation in the library is one line over canon with these four fixed.
#
# Every admissible VALUE is spelled out too, and that is not padding. The first live run wrote
# `canon(fig, point='full')` and `point='free'`; the earlier draft of this brief printed
# `point=...` while printing `colours="fixed"` literally, so a reader had no way to see that
# one takes an object and the other a string. The library now accepts the names as well --
# both halves of that fix matter, because a brief that hides the vocabulary costs a turn
# even when the call would have worked.
_SYMMETRY_PROMPT_BRIEF = """
The workspace also holds `symmetry.py`, which is preloaded into every run_python call as
`symmetry` alongside `grid`, `np` and `wmlib` — any preload list you were given elsewhere
predates it. It answers "are these two spatial arrangements the same?", which `wmlib` does
not: wmlib finds and measures things, symmetry compares them.

One function does the work:

  symmetry.canon(fig, point='full', translate=True, scale=False, colours='fixed')

It returns a canonical form, and two figures are the same — under exactly the quotients you
chose — iff their canonical forms are `==`. A figure is a dict {(y, x): colour}; build one
with `{(y, x): int(grid[y, x]) for y, x in zip(*np.nonzero(grid))}` or from a `wmlib.segment`
component's cells. Those four arguments are the whole decision, and every value they take is
listed here:

  point      'full' (all eight symmetries of the square), 'rotations' (the four quarter
             turns), 'halfturn', 'none' (no pose freedom at all), or generate([...]).
             Pick the one this game's actions actually GENERATE, or you will accept a
             mirror image as a match. This is POSE, not position — position is `translate`.
  translate  True = position free (comparing two OBJECTS), False = absolute (two SCENES)
  scale      True = a k-fold magnification is the same thing, False = it is not
  colours    'fixed' if the palette means something, 'any' to compare up to recolouring

Compare two canonical forms only when both sides were computed with the SAME four values;
canon under mismatched quotients answers a question nobody asked, and answers it silently.

There is deliberately no ready-made helper for this game's win condition. Choosing the four
quotients is the modelling act, and it is yours. If a goal hypothesis in your notes is of the
form "X ends up matching Y", `symmetry` is how you turn it into a test you can run instead of
a sentence you leave open. Read symmetry.py's docstring for the traps.
""".strip()

_DYNAMICS_PROMPT_BRIEF = """
The workspace also preloads `dynamics`. It tests a candidate transition system without
choosing the model, states, actions, or representation for you:

  dynamics.key_counterexample(model, [left, right], actions=..., max_depth=4)
  dynamics.action_laws(model, states, actions=..., projection=..., representation='position')

The first returns the shortest tested action suffix that distinguishes two states merged by
the chosen planner key. The second returns bounded certificates for idempotence, directional
inverse laws, commutation, and many-to-one behaviour. A certificate marked `supported` is
supported only on its recorded finite domain and named projection. Name every custom
`projection` or `observe` with `representation=...`. No separator within a bound is
`inconclusive`, never proof of equivalence. Read dynamics.py's docstring before relying on a
certificate.
""".strip()

PROMPT_BRIEF = f"{_SYMMETRY_PROMPT_BRIEF}\n\n{_DYNAMICS_PROMPT_BRIEF}"
HISTORICAL_TOOL_NOTE = {
    "run_python": (
        _SYMMETRY_TOOL_NOTE
        + " This declared exact runtime predates `dynamics.py`; the `dynamics` preload is None."
    )
}
HISTORICAL_PROMPT_BRIEF = f"""{_SYMMETRY_PROMPT_BRIEF}

TRUSTED RUNTIME STATUS: this declared exact version predates `dynamics.py`; its `dynamics`
preload is None. Resume remains on that version rather than silently changing its mathematics.
"""

LEGACY_TOOL_NOTE = {
    "run_python": (
        " This resumed checkpoint has no trusted-runtime manifest. Any `wmlib`, "
        "`symmetry`, or `dynamics` file present has unverified provenance; either "
        "mathematics preload may therefore be None. Inspect before relying on it."
    )
}

LEGACY_PROMPT_BRIEF = """
TRUSTED RUNTIME STATUS: legacy/unknown. This resumed checkpoint has no runtime manifest,
so the host cannot claim byte-exact `wmlib`, `symmetry`, or `dynamics` availability. Files
with those names, if present, are historical and unverified. Inspect them before use; either
mathematics preload may be None.
""".strip()


def tool_note(workspace: Path) -> dict[str, str]:
    inspection = inspect_runtime(workspace)
    if inspection.state is not RuntimeState.DECLARED:
        return LEGACY_TOOL_NOTE
    if inspection.runtime is not None and any(
        module.path == "dynamics.py" for module in inspection.runtime.modules
    ):
        return TOOL_NOTE
    return HISTORICAL_TOOL_NOTE


def prompt_brief(workspace: Path) -> str:
    inspection = inspect_runtime(workspace)
    if inspection.state is not RuntimeState.DECLARED:
        return LEGACY_PROMPT_BRIEF
    if inspection.runtime is not None and any(
        module.path == "dynamics.py" for module in inspection.runtime.modules
    ):
        return PROMPT_BRIEF
    return HISTORICAL_PROMPT_BRIEF
