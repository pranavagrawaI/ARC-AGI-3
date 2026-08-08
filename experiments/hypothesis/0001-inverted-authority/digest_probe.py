"""Does a generic core-knowledge digest carry the game, before any model reads it?

The inverted bet says the program is the agent and the small model is a subroutine
that names things. That only works if the machinery hands the model a description in
which the game is already visible. This probe builds that description from an
admissible recording using generic mechanisms only, then scores it against ls20's
actual mechanic.

The scoring key is researcher knowledge. It never enters the digest.

    uv run python experiments/hypothesis/0001-inverted-authority/digest_probe.py

Verifier (each check can fail):

    V1  a controlled entity and its action->displacement map are recovered
    V2  an invariant non-controlled sprite is isolated          (the specification)
    V3  a mutable non-controlled sprite is isolated             (the workspace)
    V4  V3's mutation is attributed to contact above baseline   (the remote effect)
    V5  some given lens states a V2-vs-V3 mismatch that is false at spawn
        and true at the win                                     (the goal relation)
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
TRACE = REPO / "recordings/ae20a3e4-ca6e-4f29-b463-042fd0dd6710"

# Only these public fields may reach the digest. Game identity is deliberately absent.
ADMISSIBLE = {"state", "levels_completed", "win_levels", "available_actions", "full_reset"}


# --- admissible loading ------------------------------------------------------------


@dataclass
class Step:
    action: str
    grid: np.ndarray
    state: str
    levels_completed: int
    full_reset: bool


def load(trace_dir: Path) -> list[Step]:
    path = next(trace_dir.glob("*.jsonl"))
    steps: list[Step] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)["data"]
        public = {k: d[k] for k in ADMISSIBLE if k in d}
        steps.append(
            Step(
                action=str(d["action_input"]["id"]),
                grid=np.array(d["frame"][-1], dtype=np.int16),
                state=public["state"],
                levels_completed=public["levels_completed"],
                full_reset=bool(public.get("full_reset", False)),
            )
        )
    return steps


def level_slice(steps: list[Step], level: int) -> list[Step]:
    return [s for s in steps if s.levels_completed == level]


# --- objectness, as a family with one knob -----------------------------------------


@dataclass
class Sprite:
    cells: frozenset[tuple[int, int]]
    colors: tuple[tuple[int, int], ...]  # (color, count) sorted
    bbox: tuple[int, int, int, int]  # r0, r1, c0, c1 half-open

    @property
    def area(self) -> int:
        return len(self.cells)

    @property
    def extent(self) -> tuple[int, int]:
        r0, r1, c0, c1 = self.bbox
        return (r1 - r0, c1 - c0)

    @property
    def centroid(self) -> tuple[float, float]:
        rs = [r for r, _ in self.cells]
        cs = [c for _, c in self.cells]
        return (sum(rs) / len(rs), sum(cs) / len(cs))

    def mask(self) -> np.ndarray:
        r0, r1, c0, c1 = self.bbox
        m = np.zeros((r1 - r0, c1 - c0), dtype=bool)
        for r, c in self.cells:
            m[r - r0, c - c0] = True
        return m

    def paint(self, grid: np.ndarray) -> np.ndarray:
        r0, r1, c0, c1 = self.bbox
        out = np.full((r1 - r0, c1 - c0), -1, dtype=np.int16)
        for r, c in self.cells:
            out[r - r0, c - c0] = grid[r, c]
        return out

    def sig(self, grid: np.ndarray) -> tuple:
        """Appearance identity: colours + normalised shape. Position-free."""
        return (self.colors, self.extent, self.paint(grid).tobytes())


def _components(grid: np.ndarray, seed: np.ndarray, same_colour: bool, diag: bool) -> list[Sprite]:
    h, w = grid.shape
    seen = np.zeros_like(seed)
    offs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diag:
        offs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    out: list[Sprite] = []
    for sr in range(h):
        for sc in range(w):
            if not seed[sr, sc] or seen[sr, sc]:
                continue
            stack, cells = [(sr, sc)], []
            seen[sr, sc] = True
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in offs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and seed[nr, nc] and not seen[nr, nc]:
                        if same_colour and grid[nr, nc] != grid[r, c]:
                            continue
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rs = [r for r, _ in cells]
            cs = [c for _, c in cells]
            counts = Counter(int(grid[r, c]) for r, c in cells)
            out.append(
                Sprite(
                    cells=frozenset(cells),
                    colors=tuple(sorted(counts.items())),
                    bbox=(min(rs), max(rs) + 1, min(cs), max(cs) + 1),
                )
            )
    return out


def terrain_colours(grid: np.ndarray, k: int) -> list[int]:
    return [int(c) for c, _ in Counter(grid.ravel().tolist()).most_common(k)]


def segment(grid: np.ndarray, knob: str) -> list[Sprite]:
    """The objectness family. One knob, two settings, no game-specific content."""
    if knob == "colour":
        seed = grid != terrain_colours(grid, 1)[0]
        return _components(grid, seed, same_colour=True, diag=False)
    if knob == "adjacency":
        bg = terrain_colours(grid, 3)
        seed = ~np.isin(grid, bg)
        return _components(grid, seed, same_colour=False, diag=True)
    raise ValueError(knob)


# --- identity over time -------------------------------------------------------------


@dataclass
class Track:
    tid: int
    sprites: dict[int, Sprite] = field(default_factory=dict)  # step index -> sprite
    sigs: dict[int, tuple] = field(default_factory=dict)


def track(frames: list[np.ndarray], knob: str) -> list[Track]:
    """Persistence by appearance first, then overlap.

    Overlap alone fails whenever displacement >= extent, which is exactly the
    controlled object's case: a 5-cell sprite moving 5 cells shares no cell with
    itself. Appearance constancy and object persistence are both GIVEN priors.
    """
    tracks: list[Track] = []
    prev: list[tuple[Track, Sprite, tuple]] = []
    for i, grid in enumerate(frames):
        sprites = segment(grid, knob)
        sigs = [sp.sig(grid) for sp in sprites]
        used: set[int] = set()
        assigned: dict[int, Track] = {}

        # pass 1 - identical appearance, nearest centroid
        for j, sp in enumerate(sprites):
            cands = [(t, psp) for t, psp, ps in prev if ps == sigs[j] and t.tid not in used]
            if not cands:
                continue
            t, _ = min(cands, key=lambda tp: abs(tp[1].centroid[0] - sp.centroid[0])
                       + abs(tp[1].centroid[1] - sp.centroid[1]))
            used.add(t.tid)
            assigned[j] = t

        # pass 2 - changed appearance, greatest overlap
        for j, sp in enumerate(sprites):
            if j in assigned:
                continue
            best, best_ov = None, 0
            for t, psp, _ in prev:
                if t.tid in used:
                    continue
                ov = len(sp.cells & psp.cells)
                if ov > best_ov:
                    best, best_ov = t, ov
            if best is None:
                best = Track(tid=len(tracks))
                tracks.append(best)
            used.add(best.tid)
            assigned[j] = best

        cur = []
        for j, sp in enumerate(sprites):
            t = assigned[j]
            t.sprites[i] = sp
            t.sigs[i] = sigs[j]
            cur.append((t, sp, sigs[j]))
        prev = cur
    return tracks


# --- controllability, mutability, contact attribution -------------------------------


def controlled(tracks: list[Track], actions: list[str]) -> tuple[Track | None, dict[str, tuple]]:
    """A track whose displacement is a consistent non-zero function of the action."""
    best_n, best_t, best_m = 0, None, {}
    for t in tracks:
        by_action: dict[str, Counter] = defaultdict(Counter)
        for i in sorted(t.sprites):
            if i == 0 or i - 1 not in t.sprites:
                continue
            a, b = t.sprites[i - 1].centroid, t.sprites[i].centroid
            by_action[actions[i]][(round(b[0] - a[0]), round(b[1] - a[1]))] += 1
        # a modal displacement is admitted only if it dominates that action's outcomes
        model = {}
        for a, ds in by_action.items():
            (d, n), total = ds.most_common(1)[0], sum(ds.values())
            if n / total >= 0.6:
                model[a] = d
        moving = {a: d for a, d in model.items() if d != (0, 0)}
        if len(moving) >= 2 and len(moving) > best_n:
            best_n, best_t, best_m = len(moving), t, model
    return best_t, best_m


def resources(tracks: list[Track], actions: list[str], n: int) -> list[dict]:
    """Quantities that move with the action *count*, not with what the action did.

    Every game inspected so far renders one, and ls20 makes the stakes explicit: its
    `_step_counter_ui` is a per-level step budget whose exhaustion raises GAME_OVER.

    This needs its own channel for two reasons. Contact attribution scores such a meter
    at exactly 0.00 excess and discards it as noise, so the agent never learns it can run
    out of moves. And a meter that empties as the level proceeds looks like a quantity
    being driven to zero, which reads as an objective - an agent that adopts it optimises
    toward its own death. Reporting it here is what lets `classify` withhold it from the
    goal candidates, so that mistake cannot be represented rather than merely discouraged.
    """
    out = []
    for t in tracks:
        if len(t.sprites) < n * 0.6:
            continue
        deltas: list[int] = []
        by_action: dict[str, Counter] = defaultdict(Counter)
        for i in sorted(t.sprites):
            if i - 1 not in t.sprites:
                continue
            d = t.sprites[i].area - t.sprites[i - 1].area
            deltas.append(d)
            by_action[actions[i]][d] += 1
        moving = [d for d in deltas if d]
        if len(deltas) < 6 or len(moving) < len(deltas) * 0.5:
            continue
        step, hits = Counter(moving).most_common(1)[0]
        # a meter must be seen ticking repeatedly; a sprite that appears and vanishes
        # a few times satisfies every ratio test above on three observations
        if hits < 5 or hits < len(moving) * 0.6:
            continue
        # The discriminator against a controlled or reactive object. A meter answers to
        # the passage of actions rather than to what the action did, so *when it moves*
        # it moves the same way whichever action was taken. It need not move on every
        # action - game B's ticks on roughly two of three - so zero deltas are not
        # evidence either way and are excluded from the test rather than counted against.
        agree = True
        for a, c in by_action.items():
            nz = Counter({d: k for d, k in c.items() if d and (d > 0) == (step > 0)})
            if sum(nz.values()) >= 2 and nz.most_common(1)[0][0] != step:
                agree = False
        if not agree:
            continue
        refills = [i for i, d in enumerate(deltas) if d and (d > 0) != (step > 0)]
        sp = t.sprites[min(t.sprites)]
        out.append({"tid": t.tid, "extent": sp.extent, "step": step,
                    "observed": len(deltas), "refills": len(refills),
                    "spent": sum(1 for d in deltas if d == step)})
    return out


def classify(tracks: list[Track], ctrl: Track | None, n: int,
             exclude: set[int] = frozenset()) -> tuple[list[Track], list[Track]]:
    invariant, mutable = [], []
    for t in tracks:
        if t is ctrl or len(t.sprites) < n * 0.5 or t.tid in exclude:
            continue
        sigs = {s for s in t.sigs.values()}
        (invariant if len(sigs) == 1 else mutable).append(t)
    return invariant, mutable


def attribute(mut: Track, ctrl: Track, frames: list[np.ndarray]) -> dict:
    """Contact-conditioned change rate versus a same-level non-contact baseline."""
    change_steps, contact_at = [], {}
    for i in sorted(mut.sigs):
        if i - 1 in mut.sigs and mut.sigs[i] != mut.sigs[i - 1]:
            change_steps.append(i)
    for i in sorted(ctrl.sprites):
        if i - 1 not in ctrl.sprites:
            continue
        dest = ctrl.sprites[i]
        before = frames[i - 1]
        under = Counter(int(before[r, c]) for r, c in dest.cells)
        contact_at[i] = tuple(sorted(under.items()))
    tally: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])  # class -> [changes, opportunities]
    for i, cls in contact_at.items():
        tally[cls][1] += 1
        if i in change_steps:
            tally[cls][0] += 1
    total_changes = len(change_steps)
    total_ops = len(contact_at)
    base = total_changes / total_ops if total_ops else 0.0
    rows = []
    for cls, (ch, ops) in tally.items():
        if ch == 0:
            continue
        rows.append({"contact_class": cls, "changes": ch, "opportunities": ops,
                     "rate": ch / ops, "excess": ch / ops - base})
    rows.sort(key=lambda r: -r["excess"])
    return {"baseline_rate": base, "changes": total_changes, "rows": rows}


# --- the given comparison lenses ----------------------------------------------------


def _norm_scale(m: np.ndarray, size: int = 16) -> bytes:
    h, w = m.shape
    if h == 0 or w == 0:
        return b""
    ri = (np.arange(size) * h // size).clip(0, h - 1)
    ci = (np.arange(size) * w // size).clip(0, w - 1)
    return m[np.ix_(ri, ci)].tobytes()


def lenses(a: np.ndarray, b: np.ndarray) -> dict[str, bool]:
    am, bm = a >= 0, b >= 0
    out = {
        "raw": a.shape == b.shape and bool((a == b).all()),
        "translation": a.shape == b.shape and bool((a == b).all()),
        "colour_invariant": am.shape == bm.shape and bool((am == bm).all()),
        "rotation": any(a.shape == np.rot90(b, k).shape and bool((a == np.rot90(b, k)).all())
                        for k in range(4)),
        "scale_normalised": _norm_scale(am) == _norm_scale(bm),
    }
    return out


# --- the digest ---------------------------------------------------------------------


def digest(steps: list[Step], knob: str, horizon: int) -> dict:
    steps = steps[:horizon]
    frames = [s.grid for s in steps]
    actions = [s.action for s in steps]
    tracks = track(frames, knob)
    ctrl, model = controlled(tracks, actions)
    res = resources(tracks, actions, len(frames))
    if ctrl is not None:
        res = [r for r in res if r["tid"] != ctrl.tid]
    inv, mut = classify(tracks, ctrl, len(frames), {r["tid"] for r in res})
    d = {
        "knob": knob,
        "frames": len(frames),
        "sprites_frame0": len(segment(frames[0], knob)),
        "controlled": None,
        "resources": res,
        "invariant": [],
        "mutable": [],
        "attribution": [],
    }
    if ctrl is not None:
        sp = ctrl.sprites[min(ctrl.sprites)]
        d["controlled"] = {"tid": ctrl.tid, "area": sp.area, "extent": sp.extent,
                           "colours": sp.colors, "action_model": model}
    for t in sorted(inv, key=lambda t: -t.sprites[min(t.sprites)].area)[:6]:
        sp = t.sprites[min(t.sprites)]
        d["invariant"].append({"tid": t.tid, "area": sp.area, "extent": sp.extent,
                               "colours": sp.colors, "seen": len(t.sprites)})
    for t in sorted(mut, key=lambda t: -t.sprites[min(t.sprites)].area)[:6]:
        sp = t.sprites[min(t.sprites)]
        changes = sum(1 for i in sorted(t.sigs) if i - 1 in t.sigs and t.sigs[i] != t.sigs[i - 1])
        d["mutable"].append({"tid": t.tid, "area": sp.area, "extent": sp.extent,
                             "colours": sp.colors, "changes": changes, "seen": len(t.sprites)})
        if ctrl is not None and changes:
            a = attribute(t, ctrl, frames)
            if a["rows"]:
                d["attribution"].append({"tid": t.tid, **a})
    d["_tracks"] = {"all": tracks, "ctrl": ctrl, "inv": inv, "mut": mut, "frames": frames}
    return d


# --- the verifier -------------------------------------------------------------------


def verify(d: dict) -> list[tuple[str, bool, str]]:
    ctrl, inv, mut = d["_tracks"]["ctrl"], d["_tracks"]["inv"], d["_tracks"]["mut"]
    frames = d["_tracks"]["frames"]
    res = []

    ok = d["controlled"] is not None and len(d["controlled"]["action_model"]) >= 4
    res.append(("V1 controlled entity + action model", ok,
                str(d["controlled"]["action_model"]) if d["controlled"] else "none"))

    small_inv = [x for x in d["invariant"] if 3 <= x["area"] <= 60]
    res.append(("V2 invariant small sprite (specification)", bool(small_inv),
                f"{len(small_inv)} candidates, areas {[x['area'] for x in small_inv][:5]}"))

    small_mut = [x for x in d["mutable"] if 3 <= x["area"] <= 60 and x["changes"] > 0]
    res.append(("V3 mutable small sprite (workspace)", bool(small_mut),
                f"{len(small_mut)} candidates, areas {[x['area'] for x in small_mut][:5]}"))

    strong = [a for a in d["attribution"]
              if any(r["excess"] > 0.25 and r["changes"] >= 3 for r in a["rows"])]
    res.append(("V4 contact-attributed remote change", bool(strong),
                f"{len(strong)} attributed tracks"))

    # V5: over ALL invariant x mutable pairs and ALL lenses, does some relation go
    # False at spawn -> True at the level's end? The count of pairs examined is the
    # size of the choice the machinery cannot make for itself.
    pairs, hits = 0, []
    for it in inv:
        i0, i1 = min(it.sprites), max(it.sprites)
        if it.sprites[i0].area > 60:
            continue
        for mt in mut:
            m0, m1 = min(mt.sprites), max(mt.sprites)
            if mt.sprites[m0].area > 60:
                continue
            pairs += 1
            spawn = lenses(it.sprites[i0].paint(frames[i0]), mt.sprites[m0].paint(frames[m0]))
            end = lenses(it.sprites[i1].paint(frames[i1]), mt.sprites[m1].paint(frames[m1]))
            flip = [k for k in spawn if not spawn[k] and end[k]]
            if flip:
                hits.append((it.tid, mt.tid, flip))
    res.append(("V5 some lens states a mismatch that closes by the win", bool(hits),
                f"{len(hits)} of {pairs} candidate pairs flip: " +
                ("; ".join(f"inv{a} vs mut{b} under {f}" for a, b, f in hits) or "none")))
    res.append(("V5b the flip is unique (machinery could pick it alone)", len(hits) == 1,
                f"{len(hits)} flipping pairs out of {pairs} enumerated"))

    rs = d["resources"]
    res.append(("V6 a resource channel is identified", bool(rs),
                ", ".join(f"tid{r['tid']} {r['step']:+d}/action over {r['observed']}"
                          for r in rs) or "none"))
    leaked = {r["tid"] for r in rs} & ({t.tid for t in inv} | {t.tid for t in mut})
    res.append(("V6b no resource is offered as a goal candidate", not leaked,
                f"leaked {sorted(leaked)}" if leaked
                else f"{len(rs)} withheld from {pairs} enumerated pairs"))
    return res


def show(d: dict) -> None:
    print(f"\n{'=' * 78}\nDIGEST  knob={d['knob']}  frames={d['frames']}  "
          f"sprites@f0={d['sprites_frame0']}\n{'=' * 78}")
    c = d["controlled"]
    print(f"controlled : {('tid %d area %d extent %s colours %s' % (c['tid'], c['area'], c['extent'], c['colours'])) if c else 'NONE'}")
    if c:
        print(f"             action model {c['action_model']}")
    print("resources  : " + (", ".join(
        f"tid{r['tid']}(e={r['extent']},{r['step']:+d}/action,{r['spent']}/{r['observed']},"
        f"refills={r['refills']})" for r in d["resources"]) or "none"))
    print("invariant  : " + (", ".join(f"tid{x['tid']}(a={x['area']},e={x['extent']})"
                                       for x in d["invariant"]) or "none"))
    print("mutable    : " + (", ".join(f"tid{x['tid']}(a={x['area']},e={x['extent']},ch={x['changes']})"
                                       for x in d["mutable"]) or "none"))
    for a in d["attribution"]:
        print(f"attribution: tid{a['tid']} baseline={a['baseline_rate']:.3f} changes={a['changes']}")
        for r in a["rows"][:3]:
            print(f"             on-contact {r['contact_class']} -> {r['changes']}/{r['opportunities']} "
                  f"rate={r['rate']:.2f} excess={r['excess']:+.2f}")


def main() -> int:
    argv = sys.argv[1:]
    trace = TRACE
    if argv and argv[0] == "--trace":
        trace, argv = Path(argv[1]), argv[2:]
    steps = load(trace)
    levels = [int(x) for x in (argv or ["1"])]
    print(f"trace: {len(steps)} admissible steps, final state {steps[-1].state}, "
          f"levels_completed {steps[-1].levels_completed}")
    print("steps per level: " + str(Counter(s.levels_completed for s in steps).most_common()))
    failures = 0
    for level in levels:
        lvl = level_slice(steps, level)
        for knob in ("colour", "adjacency"):
            d = digest(lvl, knob, horizon=len(lvl))
            print(f"\n### level {level}")
            show(d)
            print("-" * 78)
            for name, ok, detail in verify(d):
                print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {detail}")
                failures += not ok
    print(f"\n{failures} failed checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
