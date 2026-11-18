#!/usr/bin/env python3
"""Free test of the staged seed. Not a q5 run. No model calls.

Checks: digest, helper guard, import-as-seeded-module, the HUD-key D4 orbit
from the q3c frames, and the two docstring traps (point as string; colours mix).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

PROBE = Path(__file__).resolve().parent
SEED = PROBE / "seed" / "symmetry.py"
EXPECT = "7e0c1f1cdcd0d594b1bd21dee0fd74c9624a5a66141575d2d9f947fc298b433f"
Q3C = Path(
    "/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/"
    "tall-humerus/analysis/local-probes/20260904T-host-control-loop-q3/"
    "results/nemo_solver/20260904_140403_ls20_mdfiles_q3c"
)
NOOA_EX = Path("/home/pranav/projects/arc-agi-3-reference/nooa/examples/arc_agi_3")

failed = 0


def check(name, ok, detail=""):
    global failed
    if ok:
        print(f"PASS  {name}" + (f"  {detail}" if detail else ""))
    else:
        failed += 1
        print(f"FAIL  {name}" + (f"  {detail}" if detail else ""))


def key9(grid):
    return {(r, c): 9 for r in range(54, 61) for c in range(0, 12) if grid[r][c] == 9}


def load_sym_as_seeded():
    """Copy bytes to a temp helpers/symmetry.py and import under that name."""
    src = SEED.read_bytes()
    d = Path(tempfile.mkdtemp()) / "helpers"
    d.mkdir()
    dest = d / "symmetry.py"
    dest.write_bytes(src)
    spec = importlib.util.spec_from_file_location("symmetry", dest)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, dest, hashlib.sha256(dest.read_bytes()).hexdigest()


def main() -> int:
    raw = SEED.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    check("digest", digest == EXPECT, digest)
    check("name is symmetry.py", SEED.name == "symmetry.py")
    check("no wmlib", b"wmlib" not in raw)
    check("no run_python", b"run_python" not in raw)
    check("names load_helpers", b"load_helpers" in raw)

    sys.path.insert(0, str(NOOA_EX))
    from solver_agent import _HELPER_NAME, _check_helper_ast

    check("filename matches helper guard", bool(_HELPER_NAME.match(SEED.name)))
    err = _check_helper_ast(raw.decode())
    check("AST guard", err is None, repr(err))

    S, dest, dest_hash = load_sym_as_seeded()
    check("seeded copy digest", dest_hash == EXPECT)
    check("self.h.symmetry.canon exists", callable(getattr(S, "canon", None)))
    pubs = [n for n in dir(S) if not n.startswith("_") and callable(getattr(S, n))]
    check(
        "public API",
        set(pubs)
        >= {
            "compose",
            "det",
            "generate",
            "group",
            "proper",
            "act",
            "canon",
            "primitive",
            "stabiliser",
            "transporter",
            "orbit",
            "congruent",
            "directly_congruent",
            "similar",
            "mirror",
            "chiral",
            "enantiomorphic",
        },
        str(sorted(pubs)),
    )

    # Docstring trap: point='full' must work (string, not only the FULL object).
    L = {(0, 0): 1, (0, 1): 1, (1, 0): 1}
    try:
        a = S.canon(L, point="full")
        b = S.canon(L, S.FULL)
        check("point='full' string accepted", a == b)
    except Exception as e:
        check("point='full' string accepted", False, f"{type(e).__name__}: {e}")

    # Colours trap: mixing fixed and any is not a comparison.
    mixed = S.canon(L, colours="fixed") == S.canon(L, colours="any")
    check("mixed colours='fixed' vs 'any' is not equality", mixed is False)

    # HUD keys from q3c: same figure under D4, not under point='none'.
    g0 = json.loads((Q3C / "steps/step_0000.json").read_text())["grid"]
    g19 = json.loads((Q3C / "steps/step_0019.json").read_text())["grid"]
    k0, k19 = key9(g0), key9(g19)
    check("HUD keys extracted", len(k0) == 24 and len(k19) == 24, f"{len(k0)} {len(k19)}")
    check("D4 congruent (the orbit)", S.congruent(k0, k19) is True)
    check("not rigid (point='none')", S.canon(k0, "none") != S.canon(k19, "none"))
    core, k = S.primitive(k0)
    check("primitive k=2 core=6", k == 2 and len(core) == 6, f"k={k} n={len(core)}")

    # A 90° turn of a figure is congruent, a different shape is not.
    turned = S.act(S.R, k0)
    check("act(R) stays congruent", S.congruent(k0, turned) is True)
    other = {(0, 0): 1, (0, 1): 1, (0, 2): 1}
    check("unrelated figure not congruent", S.congruent(k0, other) is False)

    print()
    print("This is not a q5 run. C1–C4 are not in the live solver. Credit $2.75.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
