"""The verifier is now mathematics, not recordings.

Games can only ever say "this fired on the frames I happened to have". The group axioms,
orbit-stabiliser, and Burnside's lemma say whether the construction is right at all, and
they fail loudly when it is not. Every check below can fail.
"""
import random
from itertools import combinations, permutations
import symmetry as S


def eq(a, b, point=S.FULL, **kw):
    """Equivalence stated only as equality of canonical forms -- never through a named
    relation, so that deleting the name is a fair trial and deleting the quotient is not."""
    return S.canon(a, point, **kw) == S.canon(b, point, **kw)

FAIL = []
def check(name, cond, detail=""):
    if not cond: FAIL.append(f"{name}: {detail}")
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{'  ' + detail if detail and not cond else ''}")

def fig(cells, labels=None):
    return {c: (labels[i] if labels else 1) for i, c in enumerate(cells)}

def rand_fig(n=5, box=4, ncol=1, rng=random):
    cells = rng.sample([(y, x) for y in range(box) for x in range(box)], n)
    return {c: rng.randrange(ncol) + 1 for c in cells}

# --- 1. is it a group at all -------------------------------------------------------
print("group axioms")
G = S.FULL
check("order of D4 is 8", len(G) == 8, str(len(G)))
check("closure", all(S.compose(g, h) in G for g in G for h in G))
check("identity", all(S.compose(g, S.I) == g == S.compose(S.I, g) for g in G))
check("inverses exist", all(any(S.compose(g, h) == S.I for h in G) for g in G))
check("associativity", all(S.compose(S.compose(a, b), c) == S.compose(a, S.compose(b, c))
                           for a in G for b in G for c in G))
check("subgroups are closed", all(all(S.compose(g, h) in H for g in H for h in H)
                                  for H in (S.ROTATIONS, S.HALFTURN, S.NONE, S.proper(G))))
check("proper subgroup has index 2", len(S.proper(G)) * 2 == len(G))
check("determinants are +-1", all(abs(a*d - b*c) == 1 for (a, b), (c, d) in G))

# 1b. The model must be able to name the group ITS game's actions generate, which means
# closure from arbitrary generators -- not a menu of four. D4 has exactly 10 subgroups.
subs = set()
for g in G:
    for h in G:
        subs.add(S.generate([g, h]))
check("every generated subset is closed",
      all(all(S.compose(x, y) in H for x in H for y in H) for H in subs))
check("Lagrange: every subgroup order divides 8", all(8 % len(H) == 0 for H in subs))
check("the subgroup lattice of D4 has 10 members", len(subs) == 10, str(len(subs)))

# --- 2. is it an action ------------------------------------------------------------
print("group action")
rng = random.Random(0)
figs = [rand_fig(5, 4, 3, rng) for _ in range(200)]
check("act(I) is identity", all(S.act(S.I, f) == f for f in figs))
check("act(gh) == act(g)act(h)",
      all(S.act(S.compose(g, h), f) == S.act(g, S.act(h, f))
          for f in figs[:20] for g in G for h in G))

# 1c. Position is a quotient that can be declined. Comparing two figures where they SIT is a
# different geometry from comparing their shapes, and the library must be able to say both.
shifted = {(y + 3, x + 1): l for (y, x), l in figs[0].items()}
check("translation quotient identifies translates", eq(figs[0], shifted))
check("declining it distinguishes them", not eq(figs[0], shifted, translate=False))
check("absolute position is still invariant about the origin",
      all(S.canon(S.act(g, f), translate=False) == S.canon(f, translate=False)
          for f in figs for g in G))

# --- 3. is canon an invariant, and does it separate --------------------------------
print("canonical form")
for name, H in (("D4", S.FULL), ("C4", S.ROTATIONS), ("C2", S.HALFTURN)):
    check(f"invariant under {name}",
          all(S.canon(S.act(g, f), point=H) == S.canon(f, point=H) for f in figs for g in H))
check("complete (equal canon implies a carrying element exists)",
      all(bool(S.transporter(a, b)) for a in figs for b in figs
          if S.canon(a) == S.canon(b)))
check("sound (a carrying element implies equal canon)",
      all(S.canon(a) == S.canon(b) for a in figs[:40] for b in figs[:40]
          if S.transporter(a, b)))

# --- 4. orbit-stabiliser and Burnside ----------------------------------------------
print("counting theorems")
check("orbit-stabiliser |orbit|*|stab| == |G|",
      all(len(S.orbit(f, H)) * len(S.stabiliser(f, H)) == len(H)
          for f in figs for H in (S.FULL, S.ROTATIONS, S.HALFTURN)))
check("transporter is empty or a coset of the stabiliser",
      all(len(t) in (0, len(S.stabiliser(a))) for a in figs[:60] for b in figs[:60]
          for t in [S.transporter(a, b)]))

# Burnside: the number of orbits equals the average number of figures each element fixes.
box = [(y, x) for y in range(3) for x in range(3)]
univ = {S.canon(fig(c), point=S.NONE) for c in combinations(box, 4)}
univ = {tuple(u) for u in univ}
def as_fig(u): return {(y, x): l for y, x, l in u}
orbits = {S.canon(as_fig(u)) for u in univ}
fixed = sum(sum(1 for u in univ if S.canon(S.act(g, as_fig(u)), point=S.NONE) == u) for g in G)
check(f"Burnside on {len(univ)} tetromino-sized figures",
      len(orbits) * len(G) == fixed, f"{len(orbits)} orbits vs {fixed}/{len(G)}")

# --- 5. the equivalences refine each other correctly --------------------------------
print("refinement lattice")
check("rigid implies directly congruent",
      all(not eq(a, b, S.NONE) or eq(a, b, S.ROTATIONS) for a in figs for b in figs))
check("directly congruent implies congruent",
      all(not eq(a, b, S.ROTATIONS) or eq(a, b, S.FULL) for a in figs for b in figs))
check("congruent implies similar",
      all(not eq(a, b, S.FULL) or eq(a, b, S.FULL, scale=True) for a in figs for b in figs))
check("congruent implies same size and same colour multiset",
      all(not eq(a, b, S.FULL) or (len(a) == len(b) and
          sorted(a.values()) == sorted(b.values())) for a in figs for b in figs))

# --- 6. chirality, computed two independent ways ------------------------------------
print("chirality")
check("chiral iff the stabiliser contains no reflection",
      all(S.chiral(f) == (len(S.stabiliser(f, S.FULL) & (S.FULL - S.proper(S.FULL))) == 0)
          for f in figs))
Sp = fig([(0, 1), (0, 2), (1, 0), (1, 1)])
Z  = fig([(0, 0), (0, 1), (1, 1), (1, 2)])
T  = fig([(0, 0), (0, 1), (0, 2), (1, 1)])
O  = fig([(0, 0), (0, 1), (1, 0), (1, 1)])
check("S-tetromino is chiral", S.chiral(Sp))
check("T-tetromino is achiral", not S.chiral(T))
check("square is achiral", not S.chiral(O))
check("S and Z are enantiomorphic", S.enantiomorphic(Sp, Z))
check("S and Z are congruent", S.congruent(Sp, Z))
check("S and Z are not directly congruent", not S.directly_congruent(Sp, Z))
check("square has full stabiliser", len(S.stabiliser(O, S.FULL)) == 8)
check("S has stabiliser of order 2", len(S.stabiliser(Sp, S.FULL)) == 2,
      str(len(S.stabiliser(Sp, S.FULL))))

# --- 7. similarity and the primitive core -------------------------------------------
print("similarity")
def blow(f, k): return {(y*k + i, x*k + j): l for (y, x), l in f.items()
                        for i in range(k) for j in range(k)}
check("a magnified figure is similar but not congruent",
      all(eq(f, blow(f, 2), scale=True) and not eq(f, blow(f, 2)) for f in figs[:50]))
check("magnification by different factors is one class",
      all(eq(blow(f, 2), blow(f, 3), scale=True) for f in figs[:50]))
check("scale quotient is idempotent under further magnification",
      all(eq(f, blow(blow(f, 2), 2), scale=True) for f in figs[:50]))

# --- 8. colour as its own quotient ---------------------------------------------------
print("colour quotient")
a = fig([(0, 0), (0, 1), (1, 0)], [1, 1, 2])
b = fig([(0, 0), (0, 1), (1, 0)], [2, 2, 1])
check("recolouring differs under fixed labels", S.canon(a) != S.canon(b))
check("recolouring agrees under permutable labels",
      S.canon(a, colours="any") == S.canon(b, colours="any"))

# --- 9. parts do not determine the whole ---------------------------------------------
# The failure that three games took to find, stated once as mathematics: canonicalising
# each colour class on its own throws away the relative position of the classes, so it is
# a strictly coarser equivalence than canonicalising the labelled figure.
print("parts versus whole")
A = fig([(0, 0), (0, 1), (1, 0), (1, 1)], [1, 1, 2, 2])
B = fig([(0, 0), (0, 1), (1, 0), (1, 1)], [2, 2, 1, 1])
def parts_congruent(p, q):
    cols = {l for l in p.values()} | {l for l in q.values()}
    return all(S.canon({c: 1 for c, l in p.items() if l == k}) ==
               S.canon({c: 1 for c, l in q.items() if l == k}) for k in cols)
check("parts are congruent", parts_congruent(A, B))
check("the labelled wholes are not rigidly equal", S.canon(A, S.NONE) != S.canon(B, S.NONE))
check("but they ARE congruent under D4", S.congruent(A, B))

print()
print(f"SIGNATURE checks_failed={len(FAIL)}")
for f in FAIL: print("   ", f)
print("GREEN" if not FAIL else "RED")
