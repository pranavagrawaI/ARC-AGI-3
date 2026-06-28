"""0011 — Persistent object identity under occlusion (LR-0016 closed alphabet).

Run the offline checks (from the repo root):
    uv run python teach/lessons/0011-object-identity-occlusion.py

Audit identity assignments on reproducible live ls20 frames:
    uv run python teach/lessons/0011-object-identity-occlusion.py --live

`moved`, `changed`, `appeared`, `disappeared` are all meaningless until we can say
WHICH object in frame t is WHICH in frame t+1. And single-layer rendering means an
avatar standing on a tile OCCLUDES it — mistaking that for DISAPPEARED would feed
the clause searcher a phantom event. So tracking emits an identity POSTERIOR, not
a hard label, and it is honest about uncertainty:

  - DISAPPEARED is only emitted when occlusion is implausible.
  - LOST means bounded identity memory expired; it is not a world event.
  - UNCERTAIN blocks hard falsification downstream (clause search, 0012, must not
    refute a law on a guessed identity).

GIVEN (LR-0016): object persistence, spatial relations, equality, basic
attributes. LEARNED: nothing — this layer only assigns identity. Non-goal: do not
make tracking perfect. Make it honest.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import importlib.util as _ilu
import sys as _sys

# Reuse lesson 0010's verified perception — load it by path (run from repo root).
_spec = _ilu.spec_from_file_location(
    "il_0010", "teach/lessons/0010-invariance-lattice-perception.py"
)
il = _ilu.module_from_spec(_spec)
_sys.modules[_spec.name] = il  # register so dataclasses can resolve annotations (3.14)
_spec.loader.exec_module(il)
ObjectView = il.ObjectView

# transition labels — the vocabulary of "what happened to this identity".
SAME = "SAME"
MOVED = "MOVED"
CHANGED_COLOR = "CHANGED_COLOR"
CHANGED_SHAPE = "CHANGED_SHAPE"
OCCLUDED = "OCCLUDED"
REAPPEARED = "REAPPEARED"
DISAPPEARED = "DISAPPEARED"
LOST = "LOST"
NEW_OBJECT = "NEW_OBJECT"
UNCERTAIN = "UNCERTAIN"

MATCH_THRESH = 0.45  # below this, a (track, view) pair is not a correspondence
UNCERTAIN_DELTA = 0.08  # two candidates this close => identity is ambiguous
OCC_THRESH = 0.5  # this fraction of a vanished track's pixels covered => occluded
MAX_OCCLUDED_FRAMES = 4  # bounded persistence prior; configurable per tracker
REAPPEAR_THRESH = 0.7  # hidden identities need stronger evidence than ordinary matches
MOVE_EPS = 1.0  # centroid shift over this many px counts as motion
DIST_SCALE = 10.0  # px scale of the position-continuity kernel


@dataclass
class TrackedObject:
    track_id: int | None
    frame: int
    transition: str
    visible: bool
    occluded: bool
    object_view: ObjectView | None
    identity_confidence: float
    possible_matches: list[tuple[int, float]] = field(default_factory=list)


def _manh(a: tuple[float, float], b: tuple[float, float]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _iou(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


class ObjectTracker:
    """Greedy appearance+continuity correspondence with explicit occlusion
    reasoning. State per track: its last ObjectView and whether it is occluded."""

    def __init__(
        self,
        *,
        max_occluded_frames: int = MAX_OCCLUDED_FRAMES,
        reappear_thresh: float = REAPPEAR_THRESH,
    ):
        if max_occluded_frames < 1:
            raise ValueError("max_occluded_frames must be >= 1")
        if not MATCH_THRESH <= reappear_thresh <= 1.0:
            raise ValueError("reappear_thresh must be between MATCH_THRESH and 1")
        self.max_occluded_frames = max_occluded_frames
        self.reappear_thresh = reappear_thresh
        self._tracks: dict[int, dict] = {}
        self._next_id = 0
        self.frame = -1

    # -- correspondence score between a track's last view and a new detection --
    def _score(self, tv: ObjectView, v: ObjectView) -> float:
        pos = math.exp(-_manh(tv.centroid, v.centroid) / DIST_SCALE)
        if tv.shape_lattice["raw"] == v.shape_lattice["raw"]:
            shape = 1.0
        elif (
            tv.shape_lattice["rotation_invariant"]
            == v.shape_lattice["rotation_invariant"]
        ):
            shape = 0.6
        else:
            shape = 0.0
        if tv.color_lattice["raw"] == v.color_lattice["raw"]:
            color = 1.0
        elif tv.color_lattice["color_invariant"] == v.color_lattice["color_invariant"]:
            color = 0.5
        else:
            color = 0.0
        overlap = _iou(tv.pixels, v.pixels)
        return 0.25 * pos + 0.35 * shape + 0.25 * color + 0.15 * overlap

    def _classify(self, tv: ObjectView, v: ObjectView, was_occluded: bool) -> str:
        if was_occluded:
            return REAPPEARED
        if _manh(tv.centroid, v.centroid) > MOVE_EPS:
            return MOVED
        if tv.color_lattice["raw"] != v.color_lattice["raw"]:
            return CHANGED_COLOR
        if tv.shape_lattice["raw"] != v.shape_lattice["raw"]:
            return CHANGED_SHAPE
        return SAME

    def update(self, views: list[ObjectView], action=None) -> list[TrackedObject]:
        """Advance one frame. Returns a TrackedObject per detection plus one for
        every prior track that went occluded or disappeared. `action` is accepted
        for the future motion prior (0012); appearance dominates for now."""
        self.frame += 1
        tids = list(self._tracks)
        out: list[TrackedObject] = []

        # rank existing tracks for each detection.
        view_rank: dict[int, list[tuple[int, float]]] = {}
        for j, v in enumerate(views):
            ranked = sorted(
                ((tid, self._score(self._tracks[tid]["view"], v)) for tid in tids),
                key=lambda x: -x[1],
            )
            view_rank[j] = [
                (tid, score)
                for tid, score in ranked
                if score
                >= (
                    self.reappear_thresh
                    if self._tracks[tid]["occluded"]
                    else MATCH_THRESH
                )
            ]

        # one-track-claimed-by-two-views collisions => those views are ambiguous.
        claims: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for j, ranked in view_rank.items():
            if ranked:
                claims[ranked[0][0]].append((j, ranked[0][1]))
        collision: set[int] = set()
        for tid, cs in claims.items():
            if len(cs) >= 2:
                cs.sort(key=lambda x: -x[1])
                if cs[0][1] - cs[1][1] < UNCERTAIN_DELTA:
                    collision.update(j for j, _ in cs)

        used: set[int] = set()
        ambiguous: set[int] = (
            set()
        )  # tracks held in suspense — neither matched nor gone
        for j, v in enumerate(views):
            ranked = view_rank[j]

            # ambiguous: one prior identity, several equally-good detections.
            if j in collision:
                if ranked:
                    ambiguous.add(ranked[0][0])
                out.append(
                    TrackedObject(
                        None,
                        self.frame,
                        UNCERTAIN,
                        True,
                        False,
                        v,
                        0.3,
                        possible_matches=ranked[:3],
                    )
                )
                continue
            # ambiguous: this detection fits two prior identities about equally.
            if len(ranked) >= 2 and ranked[0][1] - ranked[1][1] < UNCERTAIN_DELTA:
                ambiguous.update(tid for tid, _ in ranked[:2])
                out.append(
                    TrackedObject(
                        None,
                        self.frame,
                        UNCERTAIN,
                        True,
                        False,
                        v,
                        0.3,
                        possible_matches=ranked[:3],
                    )
                )
                continue
            # no prior identity is plausible => a genuinely new object.
            if not ranked or ranked[0][0] in used:
                tid = self._new_track(v)
                out.append(
                    TrackedObject(
                        tid,
                        self.frame,
                        NEW_OBJECT,
                        True,
                        False,
                        v,
                        0.8,
                        possible_matches=[],
                    )
                )
                continue
            # clean unique correspondence.
            tid, s = ranked[0]
            used.add(tid)
            was_occ = self._tracks[tid]["occluded"]
            trans = self._classify(self._tracks[tid]["view"], v, was_occ)
            self._tracks[tid] = {
                "view": v,
                "occluded": False,
                "occluded_since": None,
                "last_frame": self.frame,
            }
            out.append(
                TrackedObject(
                    tid,
                    self.frame,
                    trans,
                    True,
                    False,
                    v,
                    min(0.95, 0.6 + 0.35 * s),
                    possible_matches=[(tid, s)],
                )
            )

        # prior tracks with no detection this frame: occluded or gone?
        present_pixels = [v.pixels for v in views]
        for tid in tids:
            if tid in used or tid in ambiguous:
                continue  # ambiguous tracks stay alive, unresolved — not "gone"
            last = self._tracks[tid]["view"]
            covered = max(
                (
                    len(last.pixels & p) / len(last.pixels)
                    for p in present_pixels
                    if p != last.pixels
                ),
                default=0.0,
            )
            state = self._tracks[tid]
            if state["occluded"]:
                # Once the cover clears, continued absence refutes persistence.
                if covered < OCC_THRESH:
                    out.append(
                        TrackedObject(
                            tid,
                            self.frame,
                            DISAPPEARED,
                            False,
                            False,
                            last,
                            0.5,
                            possible_matches=[],
                        )
                    )
                    del self._tracks[tid]
                    continue

                age = self.frame - state["occluded_since"]
                if age >= self.max_occluded_frames:
                    # Memory budget expired while the identity remained hidden.
                    # LOST is epistemic, not a causal claim that the object died.
                    out.append(
                        TrackedObject(
                            tid,
                            self.frame,
                            LOST,
                            False,
                            False,
                            last,
                            0.0,
                            possible_matches=[],
                        )
                    )
                    del self._tracks[tid]
                # Still within the horizon: retain silently. OCCLUDED is an edge,
                # not an event to re-emit on every hidden frame.
                continue

            if covered >= OCC_THRESH:
                state["occluded"] = True
                state["occluded_since"] = self.frame
                out.append(
                    TrackedObject(
                        tid,
                        self.frame,
                        OCCLUDED,
                        False,
                        True,
                        last,
                        0.5,
                        possible_matches=[],
                    )
                )
            else:
                out.append(
                    TrackedObject(
                        tid,
                        self.frame,
                        DISAPPEARED,
                        False,
                        False,
                        last,
                        0.5,
                        possible_matches=[],
                    )
                )
                del self._tracks[tid]
        return out

    def _new_track(self, v: ObjectView) -> int:
        tid = self._next_id
        self._next_id += 1
        self._tracks[tid] = {
            "view": v,
            "occluded": False,
            "occluded_since": None,
            "last_frame": self.frame,
        }
        return tid


# ----------------------------------------------------------------------------
# Offline checks — synthetic frames, instant, no engine. Do not edit below.
# ----------------------------------------------------------------------------
import numpy as np  # noqa: E402


def _grid(objs: list[tuple[int, list[tuple[int, int]]]]) -> np.ndarray:
    """objs: [(colour, [(r,c), ...]), ...] painted onto a 32x32 background of 0.
    A list (not a dict) so two objects of the same colour don't collide."""
    g = np.zeros((32, 32), dtype=int)
    for color, cells in objs:
        for r, c in cells:
            g[r, c] = color
    return g


def _block(r0, c0, n, color):
    return (color, [(r0 + i, c0 + j) for i in range(n) for j in range(n)])


def _views(g: np.ndarray, frame: int) -> list[ObjectView]:
    """Segment by colour-connected blobs (good enough for the synthetic checks)."""
    from collections import deque

    H, W = g.shape
    seen = np.zeros_like(g, dtype=bool)
    out, k = [], 0
    for r in range(H):
        for c in range(W):
            if seen[r, c] or g[r, c] == 0:
                continue
            col = g[r, c]
            cells, q = [], deque([(r, c)])
            seen[r, c] = True
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < H
                        and 0 <= nx < W
                        and not seen[ny, nx]
                        and g[ny, nx] == col
                    ):
                        seen[ny, nx] = True
                        q.append((ny, nx))
            out.append(ObjectView.from_grid(g, cells, object_id_local=k, frame=frame))
            k += 1
    return out


def _by_color(tracked: list[TrackedObject], color: int):
    for t in tracked:
        if (
            t.object_view is not None
            and t.object_view.color_lattice["raw"][0][0] == color
        ):
            return t
    return None


def _check(name, ok, detail):
    print(f"  [{'✓' if ok else '✗'}] {name}: {detail}")
    return ok


def _run_checks() -> bool:
    results = []

    # 1. object moves one cell -> same identity, MOVED.
    tr = ObjectTracker()
    tr.update(_views(_grid([_block(5, 5, 2, 7)]), 0))
    out = tr.update(_views(_grid([_block(5, 8, 2, 7)]), 1))
    t = _by_color(out, 7)
    results.append(
        _check(
            "1 moved one cell",
            t and t.transition == MOVED and t.track_id == 0,
            f"transition={t and t.transition}, id={t and t.track_id} (exp MOVED, 0)",
        )
    )

    # 2. avatar steps onto tile -> tile OCCLUDED, not DISAPPEARED.
    tr = ObjectTracker()
    avatar = _block(10, 10, 2, 7)  # colour 7
    tile = _block(10, 14, 1, 5)  # colour 5, one cell to the right
    tr.update(_views(_grid([avatar, tile]), 0))
    # avatar moves onto the tile's cell; single-layer => tile not rendered.
    out = tr.update(_views(_grid([_block(10, 13, 2, 7)]), 1))
    tile_t = _by_color(out, 5)
    results.append(
        _check(
            "2 tile occluded not gone",
            tile_t and tile_t.transition == OCCLUDED and tile_t.occluded,
            f"tile transition={tile_t and tile_t.transition} (exp OCCLUDED)",
        )
    )

    # 3. avatar leaves -> tile REAPPEARED, same identity.
    out = tr.update(_views(_grid([_block(10, 10, 2, 7), tile]), 2))
    tile_t = _by_color(out, 5)
    results.append(
        _check(
            "3 tile reappeared",
            tile_t and tile_t.transition == REAPPEARED,
            f"tile transition={tile_t and tile_t.transition} (exp REAPPEARED)",
        )
    )

    # 4. object genuinely removed (nothing covers it) -> DISAPPEARED.
    tr = ObjectTracker()
    tr.update(_views(_grid([_block(2, 2, 1, 5), _block(20, 20, 2, 7)]), 0))
    out = tr.update(_views(_grid([_block(20, 20, 2, 7)]), 1))  # colour 5 gone
    gone = _by_color(out, 5)
    results.append(
        _check(
            "4 disappeared",
            gone and gone.transition == DISAPPEARED,
            f"transition={gone and gone.transition} (exp DISAPPEARED)",
        )
    )

    # 5. colour change, same position/shape -> same identity, CHANGED_COLOR.
    tr = ObjectTracker()
    tr.update(_views(_grid([_block(8, 8, 2, 7)]), 0))
    out = tr.update(_views(_grid([_block(8, 8, 2, 9)]), 1))  # 7 -> 9, in place
    t = out[0]
    results.append(
        _check(
            "5 changed colour in place",
            t.transition == CHANGED_COLOR and t.track_id == 0,
            f"transition={t.transition}, id={t.track_id} (exp CHANGED_COLOR, 0)",
        )
    )

    # 6. ambiguous segmentation -> UNCERTAIN, no hard refutation.
    tr = ObjectTracker()
    tr.update(_views(_grid([_block(15, 15, 2, 7)]), 0))
    # two identical detections equidistant from the prior identity.
    out = tr.update(
        _views(
            _grid(
                [
                    (7, [(15, 11), (15, 12), (16, 11), (16, 12)]),
                    (7, [(15, 19), (15, 20), (16, 19), (16, 20)]),
                ]
            ),
            1,
        )
    )
    all_uncertain = all(
        t.transition == UNCERTAIN and t.identity_confidence < 0.5 for t in out
    )
    results.append(
        _check(
            "6 ambiguous -> uncertain",
            all_uncertain and len(out) == 2,
            f"transitions={[t.transition for t in out]} (exp both UNCERTAIN)",
        )
    )

    # 7. A covered track is one transition, not an immortal event generator.
    tr = ObjectTracker(max_occluded_frames=2)
    tile = _block(10, 10, 1, 5)
    cover = _block(10, 10, 2, 7)
    tr.update(_views(_grid([tile]), 0))
    first = tr.update(_views(_grid([cover]), 1))
    middle = tr.update(_views(_grid([cover]), 2))
    expired = tr.update(_views(_grid([cover]), 3))
    tile_first = _by_color(first, 5)
    tile_middle = _by_color(middle, 5)
    tile_expired = _by_color(expired, 5)
    results.append(
        _check(
            "7 bounded occlusion emits once then expires",
            tile_first
            and tile_first.transition == OCCLUDED
            and tile_middle is None
            and tile_expired
            and tile_expired.transition == LOST,
            f"tile transitions={[t and t.transition for t in (tile_first, tile_middle, tile_expired)]} "
            "(exp OCCLUDED, None, LOST)",
        )
    )

    # 8. If the cover clears and the object is still absent, persistence is
    # disconfirmed immediately; do not wait for the horizon.
    tr = ObjectTracker(max_occluded_frames=5)
    tr.update(_views(_grid([tile]), 0))
    tr.update(_views(_grid([cover]), 1))
    gone = tr.update(_views(_grid([_block(20, 20, 2, 7)]), 2))
    tile_gone = _by_color(gone, 5)
    results.append(
        _check(
            "8 uncovered absence disconfirms occlusion",
            tile_gone and tile_gone.transition == DISAPPEARED,
            f"tile transition={tile_gone and tile_gone.transition} (exp DISAPPEARED)",
        )
    )

    # 9. A merely plausible lookalike elsewhere is NEW, not positive evidence
    # that a hidden identity reappeared.
    tr = ObjectTracker(max_occluded_frames=5, reappear_thresh=0.7)
    old = _block(5, 5, 2, 5)
    cover = _block(5, 5, 2, 7)
    lookalike = _block(5, 15, 2, 5)  # same appearance, too far to resurrect old
    tr.update(_views(_grid([old]), 0))
    tr.update(_views(_grid([cover]), 1))
    out = tr.update(_views(_grid([cover, lookalike]), 2))
    colour5 = _by_color(out, 5)
    results.append(
        _check(
            "9 reappearance needs stronger evidence",
            colour5 and colour5.transition == NEW_OBJECT and colour5.track_id != 0,
            f"lookalike transition={colour5 and colour5.transition}, "
            f"id={colour5 and colour5.track_id} (exp NEW_OBJECT, not 0)",
        )
    )

    return all(results)


# ----------------------------------------------------------------------------
# Live-data gate — compare tracker decisions with conservative correspondences.
# ----------------------------------------------------------------------------
def _tracked_by_view(tracked: list[TrackedObject]) -> dict[int, TrackedObject]:
    return {
        id(t.object_view): t for t in tracked if t.visible and t.object_view is not None
    }


def audit_ls20_identity(
    steps: int = 120, seed: int = 17, *, use_composites: bool = True
) -> dict:
    """Random-walk ls20 and audit identity on exact-appearance matches.

    The oracle is intentionally narrower than ObjectTracker: 0010's strict
    appearance + nearest-position matcher. It cannot judge recolour, occlusion,
    or genuine appearance change, but it gives a high-confidence continuity set.
    """
    import arc_agi
    from arc_agi.base import OperationMode
    from arcengine import GameAction

    rng = random.Random(seed)
    env = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE).make("ls20")
    game_frame = env.reset()
    grid = np.asarray(game_frame.frame[-1])
    _, previous_parts = il.views_from_grid(grid, frame=0)
    grouper = il.CompositeObjectGrouper()
    previous = (
        grouper.group(grid, previous_parts, frame=0)
        if use_composites
        else previous_parts
    )
    tracker = ObjectTracker()
    initial = tracker.update(previous)
    previous_tracked = _tracked_by_view(initial)
    initial_track_count = tracker._next_id

    transitions: Counter = Counter()
    transition_objects: dict[str, Counter] = defaultdict(Counter)
    examples: dict[str, str] = {}
    oracle_total = 0
    continuity_correct = 0
    transition_correct = 0
    both_correct = 0
    oracle_unjudgeable = 0
    new_after_initial = 0
    occluded_by_track: Counter = Counter()
    occluded_examples: dict[int, str] = {}
    last_occluded_emission: dict[int, int] = {}
    consecutive_occluded_reemits = 0
    max_hidden_age = 0
    max_events_per_step = 0
    max_live_tracks = len(tracker._tracks)
    final_visible_tracks = len(previous_tracked)
    resets = 0

    for step in range(1, steps + 1):
        action = rng.choice(game_frame.available_actions)
        game_frame = env.step(GameAction.from_id(action))
        if game_frame.state.value in ("GAME_OVER", "WIN") or not game_frame.frame:
            resets += 1
            game_frame = env.reset()
            grid = np.asarray(game_frame.frame[-1])
            _, previous_parts = il.views_from_grid(grid, frame=step)
            previous = (
                grouper.group(grid, previous_parts, frame=step)
                if use_composites
                else previous_parts
            )
            tracker = ObjectTracker()
            previous_tracked = _tracked_by_view(tracker.update(previous))
            initial_track_count += tracker._next_id
            continue

        grid = np.asarray(game_frame.frame[-1])
        _, current_parts = il.views_from_grid(grid, frame=step)
        if use_composites:
            grouper.observe(previous_parts, current_parts)
            current = grouper.group(grid, current_parts, frame=step)
        else:
            current = current_parts
        oracle = il._strict_matches(previous, current)
        tracked = tracker.update(current, action=action)
        current_tracked = _tracked_by_view(tracked)

        for t in tracked:
            transitions[t.transition] += 1
            if t.object_view is not None:
                examples.setdefault(t.transition, il._describe(t.object_view))
                transition_objects[t.transition][il._describe(t.object_view)] += 1
            if t.transition == NEW_OBJECT:
                new_after_initial += 1
            if t.transition == OCCLUDED and t.track_id is not None:
                occluded_by_track[t.track_id] += 1
                if last_occluded_emission.get(t.track_id) == step - 1:
                    consecutive_occluded_reemits += 1
                last_occluded_emission[t.track_id] = step
                if t.object_view is not None:
                    occluded_examples.setdefault(
                        t.track_id, il._describe(t.object_view)
                    )

        max_events_per_step = max(max_events_per_step, len(tracked))
        max_live_tracks = max(max_live_tracks, len(tracker._tracks))
        max_hidden_age = max(
            max_hidden_age,
            max(
                (
                    tracker.frame - state["occluded_since"]
                    for state in tracker._tracks.values()
                    if state["occluded"]
                ),
                default=0,
            ),
        )

        for old, new, displacement in oracle:
            oracle_total += 1
            old_t = previous_tracked.get(id(old))
            new_t = current_tracked.get(id(new))
            if (
                old_t is None
                or new_t is None
                or old_t.track_id is None
                or new_t.track_id is None
            ):
                oracle_unjudgeable += 1
                continue
            same_identity = old_t.track_id == new_t.track_id
            expected = (
                MOVED
                if abs(displacement[0]) + abs(displacement[1]) > MOVE_EPS
                else SAME
            )
            right_transition = new_t.transition == expected
            continuity_correct += int(same_identity)
            transition_correct += int(right_transition)
            both_correct += int(same_identity and right_transition)

        previous = current
        previous_parts = current_parts
        previous_tracked = current_tracked
        final_visible_tracks = sum(
            1 for t in tracked if t.visible and t.track_id is not None
        )

    judgeable = oracle_total - oracle_unjudgeable
    return {
        "steps": steps,
        "seed": seed,
        "representation": "composite" if use_composites else "primitive",
        "learned_composite_relations": len(grouper.learned_relations()),
        "resets": resets,
        "initial_tracks": initial_track_count,
        "tracks_created": initial_track_count + new_after_initial,
        "live_tracks": len(tracker._tracks),
        "transitions": transitions,
        "transition_objects": transition_objects,
        "examples": examples,
        "new_after_initial": new_after_initial,
        "occluded_by_track": occluded_by_track,
        "occluded_examples": occluded_examples,
        "consecutive_occluded_reemits": consecutive_occluded_reemits,
        "max_hidden_age": max_hidden_age,
        "occlusion_horizon": tracker.max_occluded_frames,
        "max_events_per_step": max_events_per_step,
        "max_live_tracks": max_live_tracks,
        "final_visible_tracks": final_visible_tracks,
        "ghost_tracks": max(0, len(tracker._tracks) - final_visible_tracks),
        "oracle_total": oracle_total,
        "oracle_judgeable": judgeable,
        "oracle_unjudgeable": oracle_unjudgeable,
        "continuity_correct": continuity_correct,
        "transition_correct": transition_correct,
        "both_correct": both_correct,
    }


def _rate(numerator: int, denominator: int) -> str:
    return (
        f"{numerator}/{denominator} ({numerator / denominator:.1%})"
        if denominator
        else "n/a"
    )


def _print_live_audit(report: dict) -> None:
    print(
        f"\nLive ls20 identity audit ({report['steps']} random actions, "
        f"seed={report['seed']}, views={report['representation']}):"
    )
    print(f"  learned composite relations: {report['learned_composite_relations']}")
    print("  emitted transitions:")
    for transition, count in report["transitions"].most_common():
        example = report["examples"].get(transition, "")
        print(f"    {transition:16} {count:>5}  e.g. {example}")
    for transition in (NEW_OBJECT, LOST, MOVED):
        common = report["transition_objects"].get(transition, Counter()).most_common(3)
        if common:
            print(f"  top {transition} descriptions:")
            for description, count in common:
                print(f"    {count:>4}x  {description}")
    print(
        f"  conservative oracle matches: {report['oracle_total']} "
        f"({report['oracle_unjudgeable']} unjudgeable)"
    )
    judgeable = report["oracle_judgeable"]
    print("  identity continuity: " + _rate(report["continuity_correct"], judgeable))
    print("  SAME/MOVED label:    " + _rate(report["transition_correct"], judgeable))
    print("  both correct:        " + _rate(report["both_correct"], judgeable))
    print(
        f"  track churn: {report['new_after_initial']} NEW_OBJECT after initial frame; "
        f"{report['tracks_created']} IDs created, {report['live_tracks']} still live"
    )
    print(
        f"  lifecycle pressure: max {report['max_events_per_step']} events in one frame, "
        f"max {report['max_live_tracks']} retained tracks, "
        f"{report['ghost_tracks']} non-visible tracks at the end"
    )
    repeated_occlusions = report["occluded_by_track"].most_common(5)
    if repeated_occlusions:
        print("  tracks most often entering occlusion:")
        for track_id, count in repeated_occlusions:
            print(
                f"    id={track_id:<3} {count:>3} entries  "
                f"{report['occluded_examples'].get(track_id, '')}"
            )
    print(
        f"  consecutive OCCLUDED re-emissions: "
        f"{report['consecutive_occluded_reemits']}; "
        f"max hidden age {report['max_hidden_age']}/"
        f"{report['occlusion_horizon']} frames"
    )

    continuity_rate = report["continuity_correct"] / judgeable if judgeable else 0.0
    print("\nAudit verdict:")
    if continuity_rate >= 0.95:
        print("  PASS (narrow) — identity persists on exact-appearance matches.")
    else:
        print(
            "  FAIL — identity continuity is unreliable even when appearance is unchanged."
        )
    if report["oracle_unjudgeable"]:
        print(
            "  WARN — some exact matches became UNCERTAIN or otherwise lacked a "
            "usable track ID."
        )
    if report["new_after_initial"]:
        print(
            "  WARN — post-initial NEW_OBJECT events reveal track churn; inspect "
            "before treating appearance as creation."
        )
    if report["consecutive_occluded_reemits"]:
        print(
            "  FAIL — occluded identities have no expiry or disconfirmation path; "
            "ghost tracks accumulate and re-emit OCCLUDED every frame."
        )
    else:
        print(
            "  PASS — OCCLUDED is edge-triggered and hidden identities stay "
            "within the configured memory horizon."
        )
    if report["ghost_tracks"]:
        print(
            f"  BOUNDED — {report['ghost_tracks']} identities are currently hidden, "
            "all still within the horizon."
        )
    print(
        "  NOT VERIFIED — the oracle cannot score CHANGED_COLOR, CHANGED_SHAPE, "
        "OCCLUDED, or REAPPEARED."
    )
    if report["representation"] == "composite":
        print(
            "  COMPOSITE — learned part-whole views are active; remaining churn "
            "must be attributed from the descriptions above, not to the merged body."
        )
    else:
        print(
            "  BASELINE — primitive views still split the co-moving body found by 0010."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="audit on live ls20 frames")
    parser.add_argument(
        "--primitive",
        action="store_true",
        help="use same-colour components instead of learned composites",
    )
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    print("Offline checks (synthetic frames, instant):\n")
    if not _run_checks():
        print("\nFix the checks above.")
        raise SystemExit(1)
    print(
        "\nAll checks pass. Identity survives occlusion (occluded != disappeared),"
        "\nhidden memory expires as LOST, attribute changes keep identity, and"
        "\ngenuine ambiguity is UNCERTAIN so 0012 won't learn a law from a guess."
    )
    if args.live:
        _print_live_audit(
            audit_ls20_identity(
                steps=args.steps,
                seed=args.seed,
                use_composites=not args.primitive,
            )
        )
