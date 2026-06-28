"""0010 — Eager invariance-lattice perception (LR-0016 closed alphabet).

Run the offline checks (from the repo root):
    uv run python teach/lessons/0010-invariance-lattice-perception.py

Audit the same representation on reproducible live ls20 frames:
    uv run python teach/lessons/0010-invariance-lattice-perception.py --live

Perception emits MULTIPLE signatures per object, not one canonical hash. A single
signature silently hardcodes which transformations preserve identity — and "which
transform preserves identity" is a game rule in disguise. So we expose every
object under a lattice of comparison lenses and let the logic layer (clause
search, 0012) search over equality-under-which-invariance, paying a complexity
penalty for looser lenses.

GIVEN (LR-0016): primitive components, basic attributes, adjacency, equality,
motion comparison, and candidate invariances. LEARNED: repeated action-conditioned
co-motion can bind adjacent components into a composite object. Which comparison
lens matters remains undecided until downstream clause search.

Equality is never `a.symbol == b.symbol`. It is:

    equal("internal_pattern", a, b, under="rotation_color_invariant")

Non-goal (do not cross this line yet): no slot DISCOVERY. The five slots and the
six lattice nodes are the frozen alphabet. Only a proven slot-gap (0014) earns a
new letter.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from itertools import combinations
from typing import Hashable

import numpy as np

BG = -1  # background sentinel inside an object crop (a cell the object doesn't own)

# The lattice of comparison lenses, partially ordered by how much they forgive.
LATTICE_NODES = [
    "raw",
    "translation_normalized",
    "color_invariant",
    "rotation_invariant",
    "reflection_invariant",
    "rotation_color_invariant",
]

# Complexity penalty per lens — the search prior: prefer the LEAST invariant
# comparison that still explains the observations.
INVARIANCE_COST = {
    "raw": 0.0,
    "translation_normalized": 0.25,
    "color_invariant": 0.5,
    "rotation_invariant": 0.75,
    "reflection_invariant": 0.75,
    "rotation_color_invariant": 1.25,
}


def invariance_cost(under: str) -> float:
    if under not in INVARIANCE_COST:
        raise ValueError(f"unknown invariance lens {under!r}")
    return INVARIANCE_COST[under]


# ----------------------------------------------------------------------------
# Canonicalisation primitives (the candidate invariances, made concrete).
# ----------------------------------------------------------------------------
def _h(arr: np.ndarray) -> Hashable:
    """A hashable, comparable form of a 2D array (works across differing shapes)."""
    return tuple(map(tuple, arr.tolist()))


def _tighten(crop: np.ndarray) -> np.ndarray:
    """Translation-normalise: slice to the tight bbox of the object's own cells."""
    mask = crop != BG
    if not mask.any():
        return crop
    rs, cs = np.where(mask)
    return crop[rs.min() : rs.max() + 1, cs.min() : cs.max() + 1]


def _relabel(crop: np.ndarray) -> np.ndarray:
    """Colour-invariance: rename colours to first-seen order (row-major), keeping
    which cells share a colour. BG stays BG. (3,5,3)->(0,1,0)."""
    out = crop.copy()
    mapping: dict[int, int] = {}
    for v in crop.flat:
        v = int(v)
        if v != BG and v not in mapping:
            mapping[v] = len(mapping)
    for old, new in mapping.items():
        out[crop == old] = new
    return out


def _rotations(arr: np.ndarray) -> list[np.ndarray]:
    return [np.rot90(arr, k) for k in range(4)]


def _reflections(arr: np.ndarray) -> list[np.ndarray]:
    return [arr, np.fliplr(arr), np.flipud(arr)]


def _canon(arrays: list[np.ndarray]) -> Hashable:
    return min(_h(a) for a in arrays)


# ----------------------------------------------------------------------------
# The object view: one object, five slots, each a lattice of signatures.
# ----------------------------------------------------------------------------
@dataclass
class ObjectView:
    object_id_local: int
    frame: int
    pixels: frozenset[tuple[int, int]]
    bbox: tuple[int, int, int, int]  # (rmin, cmin, rmax, cmax), inclusive
    centroid: tuple[float, float]
    area: int

    color_lattice: dict[str, Hashable] = field(default_factory=dict)
    shape_lattice: dict[str, Hashable] = field(default_factory=dict)
    internal_pattern_lattice: dict[str, Hashable] = field(default_factory=dict)
    orientation_lattice: dict[str, Hashable] = field(default_factory=dict)
    part_structure_lattice: dict[str, Hashable] = field(default_factory=dict)

    def lattice(self, slot: str) -> dict[str, Hashable]:
        try:
            return getattr(self, f"{slot}_lattice")
        except AttributeError as e:
            raise ValueError(f"unknown slot {slot!r}") from e

    # -- builders ------------------------------------------------------------
    @classmethod
    def from_crop(
        cls,
        crop: np.ndarray,
        *,
        object_id_local: int = 0,
        frame: int = 0,
        origin: tuple[int, int] = (0, 0),
    ) -> "ObjectView":
        """Build a view from a tight-ish crop whose non-object cells are BG.
        `origin` is the crop's top-left in the source grid (for bbox/centroid)."""
        crop = _tighten(np.asarray(crop))
        h, w = crop.shape
        or_, oc = origin
        mask = crop != BG
        rs, cs = np.where(mask)
        pixels = frozenset((int(or_ + r), int(oc + c)) for r, c in zip(rs, cs))
        area = int(mask.sum())
        centroid = (
            (or_ + float(rs.mean()), oc + float(cs.mean())) if area else (0.0, 0.0)
        )
        bbox = (or_, oc, or_ + h - 1, oc + w - 1)

        v = cls(object_id_local, frame, pixels, bbox, centroid, area)
        v._build_lattices(crop, mask)
        return v

    @classmethod
    def from_grid(
        cls,
        grid: np.ndarray,
        pixels,
        *,
        object_id_local: int = 0,
        frame: int = 0,
    ) -> "ObjectView":
        """Build a view from a grid and the set of (row,col) cells the object owns
        (cells may carry different colours — that is the internal pattern)."""
        pix = list(pixels)
        rs = [p[0] for p in pix]
        cs = [p[1] for p in pix]
        rmin, cmin, rmax, cmax = min(rs), min(cs), max(rs), max(cs)
        crop = np.full((rmax - rmin + 1, cmax - cmin + 1), BG, dtype=int)
        for r, c in pix:
            crop[r - rmin, c - cmin] = int(grid[r, c])
        return cls.from_crop(
            crop, object_id_local=object_id_local, frame=frame, origin=(rmin, cmin)
        )

    # -- the five slots ------------------------------------------------------
    def _build_lattices(self, crop: np.ndarray, mask: np.ndarray) -> None:
        # internal_pattern — the full lattice; the slot clause search leans on.
        self.internal_pattern_lattice = {
            "raw": _h(crop),
            "translation_normalized": _h(crop),  # crop is already translated
            "color_invariant": _h(_relabel(crop)),
            "rotation_invariant": _canon(_rotations(crop)),
            "reflection_invariant": _canon(_reflections(crop)),
            "rotation_color_invariant": _canon([_relabel(r) for r in _rotations(crop)]),
        }

        # shape — colour stripped to a binary mask; no colour nodes.
        m = mask.astype(int)
        self.shape_lattice = {
            "raw": _h(m),
            "translation_normalized": _h(m),
            "rotation_invariant": _canon(_rotations(m)),
            "reflection_invariant": _canon(_reflections(m)),
        }

        # colour — content, not arrangement.
        vals, counts = np.unique(crop[mask], return_counts=True)
        pairs = tuple(sorted((int(v), int(c)) for v, c in zip(vals, counts)))
        self.color_lattice = {
            "raw": pairs,
            "color_invariant": tuple(sorted(int(c) for c in counts)),
        }

        # orientation — which rotation maps the shape to its canonical pose.
        k = min((_h(np.rot90(m, j)), j) for j in range(4))[1]
        self.orientation_lattice = {"raw": k, "rotation_invariant": 0}

        # part_structure — per-colour parts and their layout relative to centre.
        cy, cx = float(np.where(mask)[0].mean()), float(np.where(mask)[1].mean())
        parts = []
        for v in vals:
            pr, pc = np.where(crop == v)
            parts.append(
                (
                    int(v),
                    int(len(pr)),
                    round(float(pr.mean()) - cy, 1),
                    round(float(pc.mean()) - cx, 1),
                )
            )
        self.part_structure_lattice = {
            "raw": tuple(sorted(parts)),
            "translation_normalized": tuple(sorted(parts)),
        }


# ----------------------------------------------------------------------------
# Equality under a chosen lens — the only sanctioned way to compare objects.
# ----------------------------------------------------------------------------
def equal(slot: str, obj_a: ObjectView, obj_b: ObjectView, under: str) -> bool:
    if under not in INVARIANCE_COST:
        raise ValueError(f"unknown invariance lens {under!r}")
    la, lb = obj_a.lattice(slot), obj_b.lattice(slot)
    if under not in la or under not in lb:
        return False  # this slot does not expose this lens (e.g. colour under rotation)
    return la[under] == lb[under]


def cheapest_equal(slot: str, obj_a: ObjectView, obj_b: ObjectView) -> str | None:
    """The least-invariant lens under which a and b are equal in `slot`, or None.
    Encodes the search prior: don't reach for rotation+colour if raw suffices."""
    shared = obj_a.lattice(slot).keys() & obj_b.lattice(slot).keys()
    eq = [n for n in shared if equal(slot, obj_a, obj_b, n)]
    return min(eq, key=invariance_cost) if eq else None


# ----------------------------------------------------------------------------
# Live-data gate — apply the representation to ls20 instead of trusting toys.
# ----------------------------------------------------------------------------
def views_from_grid(grid: np.ndarray, frame: int = 0) -> tuple[int, list[ObjectView]]:
    """Segment a real frame into maximal same-colour 4-connected components.

    This is deliberately the objectness prior used in 0008, now retaining each
    component's cells so the richer 0010 slots can be built. The most frequent
    colour is treated as background; no ls20 colour is named.
    """
    grid = np.asarray(grid)
    vals, counts = np.unique(grid, return_counts=True)
    background = int(vals[counts.argmax()])
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    views: list[ObjectView] = []

    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(grid[r, c]) == background:
                continue
            colour = int(grid[r, c])
            cells: list[tuple[int, int]] = []
            queue = deque([(r, c)])
            seen[r, c] = True
            while queue:
                y, x = queue.popleft()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and not seen[ny, nx]
                        and int(grid[ny, nx]) == colour
                    ):
                        seen[ny, nx] = True
                        queue.append((ny, nx))
            views.append(
                ObjectView.from_grid(
                    grid, cells, object_id_local=len(views), frame=frame
                )
            )
    return background, views


def _appearance_key(view: ObjectView) -> Hashable:
    """Strict appearance key used only to align audit observations over time."""
    return view.internal_pattern_lattice["raw"]


def _describe(view: ObjectView) -> str:
    colours = "+".join(str(c) for c, _ in view.color_lattice["raw"])
    return f"colour={colours},area={view.area}"


def _strict_matches(
    previous: list[ObjectView], current: list[ObjectView], max_distance: float = 10.0
) -> list[tuple[ObjectView, ObjectView, tuple[float, float]]]:
    """Conservative audit correspondence: exact appearance, nearest position.

    This is not the learning agent's identity rule (0011 owns that problem). It
    keeps this audit honest: only compare movement when appearance did not change.
    """
    by_appearance: dict[Hashable, list[ObjectView]] = defaultdict(list)
    for view in current:
        by_appearance[_appearance_key(view)].append(view)
    used: set[int] = set()
    matches = []
    for old in previous:
        candidates = []
        for new in by_appearance[_appearance_key(old)]:
            if id(new) in used:
                continue
            distance = abs(new.centroid[0] - old.centroid[0]) + abs(
                new.centroid[1] - old.centroid[1]
            )
            if distance <= max_distance:
                candidates.append((distance, new))
        if not candidates:
            continue
        _, new = min(candidates, key=lambda item: item[0])
        used.add(id(new))
        displacement = (
            round(new.centroid[0] - old.centroid[0], 1),
            round(new.centroid[1] - old.centroid[1], 1),
        )
        matches.append((old, new, displacement))
    return matches


def _adjacent(a: ObjectView, b: ObjectView) -> bool:
    """True when two components share a 4-neighbour boundary."""
    smaller, other = (a, b) if len(a.pixels) <= len(b.pixels) else (b, a)
    other_pixels = other.pixels
    return any(
        (r + dr, c + dc) in other_pixels
        for r, c in smaller.pixels
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
    )


def _relation_key(a: ObjectView, b: ObjectView) -> tuple:
    """Canonical part classes plus their directed relative centroid offset."""
    ka, kb = _appearance_key(a), _appearance_key(b)
    oa, ob = repr(ka), repr(kb)
    if (oa, a.centroid) > (ob, b.centroid):
        a, b, ka, kb = b, a, kb, ka
    offset = (
        round(b.centroid[0] - a.centroid[0], 1),
        round(b.centroid[1] - a.centroid[1], 1),
    )
    return ka, kb, offset


class CompositeObjectGrouper:
    """Learn part-whole edges from repeated action-conditioned co-motion.

    GIVEN: adjacency, relative position, and equality of displacement.
    LEARNED: which strict part classes at which offset form one composite.
    """

    def __init__(self, min_evidence: int = 3, consistency: float = 0.8):
        if min_evidence < 1:
            raise ValueError("min_evidence must be >= 1")
        if not 0.5 <= consistency <= 1.0:
            raise ValueError("consistency must be between 0.5 and 1")
        self.min_evidence = min_evidence
        self.consistency = consistency
        self.evidence: dict[tuple, Counter] = defaultdict(Counter)

    def observe(self, previous: list[ObjectView], current: list[ObjectView]) -> None:
        matches = _strict_matches(previous, current)
        by_old = {id(old): (new, displacement) for old, new, displacement in matches}
        for a, b in combinations(previous, 2):
            if not _adjacent(a, b) or id(a) not in by_old or id(b) not in by_old:
                continue
            _, da = by_old[id(a)]
            _, db = by_old[id(b)]
            moving = abs(da[0]) + abs(da[1]) > 0 or abs(db[0]) + abs(db[1]) > 0
            if not moving:
                continue  # stationary scenery is not evidence of one body
            key = _relation_key(a, b)
            self.evidence[key]["support" if da == db else "contradict"] += 1

    def learned_relations(self) -> set[tuple]:
        learned = set()
        for key, counts in self.evidence.items():
            support = counts["support"]
            total = support + counts["contradict"]
            if support >= self.min_evidence and support / total >= self.consistency:
                learned.add(key)
        return learned

    def group(
        self, grid: np.ndarray, views: list[ObjectView], frame: int
    ) -> list[ObjectView]:
        """Merge only components whose current relation has earned support."""
        parent = list(range(len(views)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri

        learned = self.learned_relations()
        for i, j in combinations(range(len(views)), 2):
            if (
                _adjacent(views[i], views[j])
                and _relation_key(views[i], views[j]) in learned
            ):
                union(i, j)

        groups: dict[int, list[ObjectView]] = defaultdict(list)
        for i, view in enumerate(views):
            groups[find(i)].append(view)

        grouped = []
        for parts in groups.values():
            pixels = set().union(*(part.pixels for part in parts))
            grouped.append(
                ObjectView.from_grid(
                    grid,
                    pixels,
                    object_id_local=len(grouped),
                    frame=frame,
                )
            )
        return grouped


def audit_ls20(steps: int = 120, seed: int = 17) -> dict:
    """Random-walk ls20 and report what 0010's representation actually sees."""
    import arc_agi
    from arc_agi.base import OperationMode
    from arcengine import GameAction

    rng = random.Random(seed)
    env = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE).make("ls20")
    game_frame = env.reset()
    grid = np.asarray(game_frame.frame[-1])
    background, previous = views_from_grid(grid, frame=0)
    grouper = CompositeObjectGrouper()

    object_counts = [len(previous)]
    grouped_counts = [len(previous)]
    composites: Counter = Counter()
    backgrounds = Counter([background])
    lens_counts: Counter = Counter()
    lens_examples: dict[str, str] = {}
    co_motion: Counter = Counter()
    strict_match_count = 0
    moving_match_count = 0
    resets = 0

    def inspect_pairs(views: list[ObjectView]) -> None:
        for a, b in combinations(views, 2):
            lens = cheapest_equal("internal_pattern", a, b)
            if lens is None:
                continue
            lens_counts[lens] += 1
            lens_examples.setdefault(lens, f"{_describe(a)} ~ {_describe(b)}")

    def inspect_groups(views: list[ObjectView]) -> None:
        for view in views:
            if len(view.color_lattice["raw"]) > 1:
                composites[_describe(view)] += 1

    inspect_pairs(previous)
    for step in range(1, steps + 1):
        action = rng.choice(game_frame.available_actions)
        game_frame = env.step(GameAction.from_id(action))
        if game_frame.state.value in ("GAME_OVER", "WIN") or not game_frame.frame:
            resets += 1
            game_frame = env.reset()
            grid = np.asarray(game_frame.frame[-1])
            background, previous = views_from_grid(grid, frame=step)
            grouped = grouper.group(grid, previous, frame=step)
            object_counts.append(len(previous))
            grouped_counts.append(len(grouped))
            backgrounds[background] += 1
            inspect_pairs(previous)
            inspect_groups(grouped)
            continue

        grid = np.asarray(game_frame.frame[-1])
        background, current = views_from_grid(grid, frame=step)
        grouper.observe(previous, current)
        grouped = grouper.group(grid, current, frame=step)
        matches = _strict_matches(previous, current)
        strict_match_count += len(matches)
        moving = [m for m in matches if abs(m[2][0]) + abs(m[2][1]) >= 1.0]
        moving_match_count += len(moving)

        # Components that repeatedly move by the same displacement are evidence
        # that same-colour connectivity split one composite body into fragments.
        for left, right in combinations(moving, 2):
            if left[2] != right[2]:
                continue
            pair = tuple(sorted((_describe(left[0]), _describe(right[0]))))
            co_motion[pair] += 1

        object_counts.append(len(current))
        grouped_counts.append(len(grouped))
        backgrounds[background] += 1
        inspect_pairs(current)
        inspect_groups(grouped)
        previous = current

    return {
        "steps": steps,
        "seed": seed,
        "resets": resets,
        "backgrounds": backgrounds,
        "object_count_range": (min(object_counts), max(object_counts)),
        "grouped_count_range": (min(grouped_counts), max(grouped_counts)),
        "learned_relations": len(grouper.learned_relations()),
        "composites": composites,
        "strict_matches": strict_match_count,
        "moving_matches": moving_match_count,
        "lens_counts": lens_counts,
        "lens_examples": lens_examples,
        "co_motion": co_motion,
    }


def _print_live_audit(report: dict) -> None:
    print(
        f"\nLive ls20 audit ({report['steps']} random actions, seed={report['seed']}):"
    )
    print(f"  detected components/frame: {report['object_count_range']}")
    print(f"  grouped objects/frame:     {report['grouped_count_range']}")
    print(f"  inferred backgrounds: {dict(report['backgrounds'])}")
    print(
        f"  strict cross-frame matches: {report['strict_matches']} "
        f"({report['moving_matches']} moved)"
    )
    print("  cheapest equality lens for distinct components:")
    for lens in LATTICE_NODES:
        count = report["lens_counts"].get(lens, 0)
        if count:
            print(f"    {lens:28} {count:>5}  e.g. {report['lens_examples'][lens]}")

    repeated = [(pair, n) for pair, n in report["co_motion"].most_common() if n >= 3]
    if repeated:
        print("  repeated co-moving component pairs:")
        for pair, count in repeated[:8]:
            print(f"    {count:>3}x  {pair[0]} + {pair[1]}")
    print(f"  learned composite relations: {report['learned_relations']}")
    for description, count in report["composites"].most_common(5):
        print(f"    {count:>3} frames  {description}")

    print("\nAudit verdict:")
    print("  PASS — the lattice functions run on real ls20 detections.")
    print("  BASE — same-colour connectivity begins with monochrome fragments.")
    if repeated:
        print(
            "  EVIDENCE — repeated co-motion shows at least one body was split "
            "into multiple components."
        )
    print(
        "  WARN — random movement exercised raw/colour comparisons only; it did "
        "not validate rotation or reflection."
    )
    print(
        "  WARN — raw is built from an already-tight crop, so "
        "translation_normalized is currently redundant."
    )
    if report["learned_relations"] and report["composites"]:
        print(
            "  PASS — repeated non-zero co-motion + adjacency earned a "
            "multicolour composite; no colours were named."
        )
    else:
        print("  FAIL — no composite-object hypothesis was earned.")


# ----------------------------------------------------------------------------
# Offline checks — synthetic crops, instant, no engine. Do not edit below.
# ----------------------------------------------------------------------------
def _view(rows: list[list[int]]) -> ObjectView:
    """A view from a small pattern; 0 is treated as a real colour, BG fills nothing
    (the crop is fully owned). Use negative for holes if needed."""
    return ObjectView.from_crop(np.array(rows, dtype=int))


# An asymmetric, multi-colour pattern so rotation/reflection can't coincide.
P = [[3, 3, 5], [3, BG, BG], [7, BG, BG]]


def _rot(rows):
    return np.rot90(np.array(rows, dtype=int), 1).tolist()


def _recolor(rows, mapping):
    return [[mapping.get(v, v) for v in r] for r in rows]


def _check(name, ok, detail):
    print(f"  [{'✓' if ok else '✗'}] {name}: {detail}")
    return ok


def _run_checks() -> bool:
    a = _view(P)
    same = _view(P)
    rotated = _view(_rot(P))
    recolored = _view(_recolor(P, {3: 8, 5: 9, 7: 2}))
    rot_recolored = _view(_recolor(_rot(P), {3: 8, 5: 9, 7: 2}))
    S = "internal_pattern"

    results = []

    # 1. exact same pattern matches under raw.
    results.append(
        _check(
            "1 exact match under raw",
            equal(S, a, same, "raw"),
            "identical crops equal under raw",
        )
    )

    # 2. rotated fails raw, passes rotation_invariant.
    results.append(
        _check(
            "2 rotation",
            not equal(S, a, rotated, "raw")
            and equal(S, a, rotated, "rotation_invariant"),
            f"raw={equal(S, a, rotated, 'raw')} (exp F), "
            f"rotation_invariant={equal(S, a, rotated, 'rotation_invariant')} (exp T)",
        )
    )

    # 3. recoloured fails raw, passes color_invariant.
    results.append(
        _check(
            "3 colour",
            not equal(S, a, recolored, "raw")
            and equal(S, a, recolored, "color_invariant"),
            f"raw={equal(S, a, recolored, 'raw')} (exp F), "
            f"color_invariant={equal(S, a, recolored, 'color_invariant')} (exp T)",
        )
    )

    # 4. rotated+recoloured passes ONLY under rotation_color_invariant.
    passes = {n for n in LATTICE_NODES if equal(S, a, rot_recolored, n)}
    results.append(
        _check(
            "4 rotation+colour only at the top",
            passes == {"rotation_color_invariant"},
            f"passes under {sorted(passes)} (exp only rotation_color_invariant)",
        )
    )

    # 5. a broader lens never overwrites a stricter signature: the strict lens
    #    still discriminates the pair that only the broad lens unifies, and
    #    cheapest_equal reports the broad (true) cost, not a false cheaper match.
    strict_still_separates = not equal(S, a, rot_recolored, "raw")
    cheapest = cheapest_equal(S, a, rot_recolored)
    results.append(
        _check(
            "5 strict signature preserved",
            strict_still_separates and cheapest == "rotation_color_invariant",
            f"raw still F={strict_still_separates}, cheapest lens={cheapest} "
            f"(exp rotation_color_invariant)",
        )
    )

    # 6–8. Learn one composite from action-conditioned co-motion. The scene also
    # contains adjacent static parts and distant co-moving parts: neither pair
    # may be merged.
    def composite_scene(shift: int) -> np.ndarray:
        g = np.zeros((24, 24), dtype=int)
        # The intended composite: two differently-coloured adjacent parts.
        g[10:12, 8 + shift : 10 + shift] = 7
        g[9, 8 + shift : 10 + shift] = 8
        # Adjacent but static: zero motion must never earn part-whole evidence.
        g[2:4, 2:4] = 3
        g[2:4, 4:6] = 4
        # Same non-zero displacement, but far apart: shared control != one body.
        g[17, 3 + shift] = 9
        g[17, 18 + shift] = 10
        return g

    grouper = CompositeObjectGrouper(min_evidence=3)
    previous_grid = composite_scene(0)
    _, previous = views_from_grid(previous_grid, frame=0)
    current_grid, current = previous_grid, previous
    for step in range(1, 4):
        current_grid = composite_scene(step)
        _, current = views_from_grid(current_grid, frame=step)
        grouper.observe(previous, current)
        previous = current
    grouped = grouper.group(current_grid, current, frame=3)
    colour_sets = [frozenset(c for c, _ in v.color_lattice["raw"]) for v in grouped]
    results.append(
        _check(
            "6 co-moving adjacent parts become one object",
            frozenset({7, 8}) in colour_sets,
            f"composite colour sets={sorted(map(sorted, colour_sets))} (exp [7,8])",
        )
    )
    results.append(
        _check(
            "7 adjacent static parts stay separate",
            frozenset({3, 4}) not in colour_sets,
            f"[3,4] merged={frozenset({3, 4}) in colour_sets} (exp False)",
        )
    )
    results.append(
        _check(
            "8 distant co-moving parts stay separate",
            frozenset({9, 10}) not in colour_sets,
            f"[9,10] merged={frozenset({9, 10}) in colour_sets} (exp False)",
        )
    )

    return all(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="audit on live ls20 frames")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    print("Offline checks (synthetic crops, instant):\n")
    if not _run_checks():
        print("\nFix the checks above.")
        raise SystemExit(1)
    print(
        "\nAll checks pass. Perception now describes each object under six lenses"
        "\nand a per-lens cost. No lens is privileged; which one matters is the"
        "\nclause searcher's job (0012), and it pays for looseness."
    )
    if args.live:
        _print_live_audit(audit_ls20(steps=args.steps, seed=args.seed))
