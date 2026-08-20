"""Overdeletion against a mathematical objective.

Each pass removes something that currently looks like a pillar of `symmetry.py` and reruns
`verify_math.py`. A pass that stays green either found dead weight, or found a hole in the
verifier -- the two are told apart by asking whether the OBJECTIVE needs what vanished.

Where a pass removes vocabulary the verifier merely calls by name, the verifier is rewritten
to call the underlying machinery instead, so the deletion gets a fair trial rather than an
AttributeError.
"""
import re, subprocess, sys

PY_BIN = "/home/hipra/projects/arc-agi-3-tycho-nooa-refactor/arc-host/.venv/bin/python"
SRC, VER = "symmetry.py", "verify_math.py"
GREEN_SRC, GREEN_VER = open("symmetry.green.py").read(), open("verify_math.green.py").read()

def cut(text, start, end=None):
    """Delete from the line beginning `start` up to (not including) the line beginning `end`."""
    i = text.index(start)
    j = text.index(end, i) if end else len(text)
    return text[:i] + text[j:]

# --- the passes ---------------------------------------------------------------------

def no_generate(s, v):
    """Hardcode the named groups; drop closure-from-generators."""
    s = cut(s, "def generate(gens):", "FULL      =")
    s = s.replace("FULL      = generate([R, M])", "FULL      = frozenset({I, R, compose(R,R), compose(compose(R,R),R), M, compose(R,M), compose(compose(R,R),M), compose(compose(compose(R,R),R),M)})")
    s = s.replace("ROTATIONS = generate([R])", "ROTATIONS = frozenset({I, R, compose(R,R), compose(compose(R,R),R)})")
    s = s.replace("HALFTURN  = generate([compose(R, R)])", "HALFTURN  = frozenset({I, compose(R, R)})")
    s = s.replace("NONE      = generate([])", "NONE      = frozenset({I})")
    return s, v

def no_inverse(s, v):
    return cut(s, "def inverse(g):", "def det(g):"), v

def no_translate_param(s, v):
    """Position is always quotiented away; there is no absolute-position comparison."""
    s = s.replace("def canon(fig, point=FULL, translate=True,", "def canon(fig, point=FULL, translate=None,")
    s = s.replace("    if translate:\n        fig = _to_origin(fig)\n", "    fig = _to_origin(fig)\n")
    return s, v

def no_scale(s, v):
    s = cut(s, "def primitive(fig):", "# --- the structure")
    s = s.replace("    if scale:\n        fig = _to_origin(primitive(fig)[0])\n", "")
    return cut(s, "def similar(a, b, **kw):", "def mirror(fig):"), v

def no_colour_perm(s, v):
    s = cut(s, "def _label_canon(fig, colours):", "def primitive(fig):")
    s = s.replace("    return _label_canon(fig, colours)",
                  "    return tuple(sorted((y, x, l) for (y, x), l in fig.items()))")
    return s, v

def no_named_relations(s, v):
    """Delete the one-line names; keep only `canon`. Vocabulary or mechanism?"""
    s = cut(s, "def congruent(a, b, **kw):")
    v = v.replace("S.congruent(", "_congruent(").replace("S.directly_congruent(", "_direct(")
    v = v.replace("S.similar(", "_similar(").replace("S.mirror(", "_mirror(")
    v = v.replace("S.chiral(", "_chiral(").replace("S.enantiomorphic(", "_enant(")
    shim = """
_congruent = lambda a, b, **k: S.canon(a, S.FULL, **k) == S.canon(b, S.FULL, **k)
_direct    = lambda a, b, **k: S.canon(a, S.ROTATIONS, **k) == S.canon(b, S.ROTATIONS, **k)
_similar   = lambda a, b, **k: S.canon(a, S.FULL, scale=True, **k) == S.canon(b, S.FULL, scale=True, **k)
_mirror    = lambda f: S.act(S.M, f)
_chiral    = lambda f, **k: not _direct(f, _mirror(f), **k)
_enant     = lambda a, b, **k: _congruent(a, b, **k) and not _direct(a, b, **k)
"""
    return s, v.replace("FAIL = []", shim + "\nFAIL = []")

def no_structure(s, v):
    """Delete stabiliser / transporter / orbit -- the group-structure queries."""
    s = cut(s, "def stabiliser(fig, point=FULL, **kw):", "def _norm(fig, kw):")
    shim = """
_stab = lambda f, p=S.FULL, **k: frozenset(g for g in p
                                           if S.canon(S.act(g, f), S.NONE, **k) == S.canon(f, S.NONE, **k))
_trans = lambda a, b, p=S.FULL, **k: frozenset(g for g in p
                                               if S.canon(S.act(g, a), S.NONE, **k) == S.canon(b, S.NONE, **k))
_orbit = lambda f, p=S.FULL, **k: frozenset(S.canon(S.act(g, f), S.NONE, **k) for g in p)
"""
    v = v.replace("S.stabiliser(", "_stab(").replace("S.transporter(", "_trans(").replace("S.orbit(", "_orbit(")
    return s, v.replace("FAIL = []", shim + "\nFAIL = []")

def no_min_over_group(s, v):
    """Normalise by one arbitrary element instead of minimising over the whole group."""
    return s.replace("    return min(_normalise(act(g, fig), translate, scale, colours) for g in point)",
                     "    return _normalise(fig, translate, scale, colours)"), v

def no_parameters(s, v):
    """One fixed geometry. This is the design claim under test: is choosing the quotient
    load-bearing, or would congruence-of-shapes alone have done?"""
    s = s.replace("def canon(fig, point=FULL, translate=True, scale=False, colours=\"fixed\"):",
                  "def canon(fig, point=None, translate=None, scale=None, colours=None):")
    s = s.replace("    return min(_normalise(act(g, fig), translate, scale, colours) for g in point)",
                  "    return min(_normalise(act(g, fig), True, False, \"fixed\") for g in FULL)")
    return s, v

def no_labels(s, v):
    """Figures as bare cell sets: colour is discarded before comparison."""
    return s.replace("    cells = tuple(sorted((y, x, l) for (y, x), l in fig.items()))",
                     "    cells = tuple(sorted((y, x, 1) for (y, x) in fig))"), v

PASSES = [("baseline", lambda s, v: (s, v)), ("no-generate", no_generate),
          ("no-inverse", no_inverse), ("no-translate-param", no_translate_param),
          ("no-scale", no_scale), ("no-colour-perm", no_colour_perm),
          ("no-named-relations", no_named_relations), ("no-structure", no_structure),
          ("no-min-over-group", no_min_over_group),
          ("no-parameters", no_parameters), ("no-labels", no_labels)]

for name, patch in PASSES:
    try:
        s, v = patch(GREEN_SRC, GREEN_VER)
    except ValueError as e:
        print(f"{name:20} PATCH-FAILED {e}"); continue
    open(SRC, "w").write(s); open(VER, "w").write(v)
    r = subprocess.run([PY_BIN, VER], capture_output=True, text=True)
    out = r.stdout + r.stderr
    sig = next((l for l in out.splitlines() if l.startswith("SIGNATURE")), "CRASH")
    bad = [l.strip()[5:] for l in out.splitlines() if l.strip().startswith("FAIL")]
    lines = len([l for l in s.splitlines() if l.strip() and not l.strip().startswith("#")])
    verdict = "GREEN" if "checks_failed=0" in sig else "RED"
    print(f"{name:20} {verdict:5} {lines:4}sl  {sig}")
    for b in bad[:4]: print(f"{'':20}   broke: {b}")
    if sig == "CRASH":
        print("\n".join("      " + l for l in out.strip().splitlines()[-4:]))

open(SRC, "w").write(GREEN_SRC); open(VER, "w").write(GREEN_VER)
