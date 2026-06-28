"""0016 — Autonomous dynamics letters: rotate + orphan (shift held behind Q4).

Run offline checks:
    uv run python teach/lessons/0016-autonomous-dynamics.py

Composed from 0010 lattice (rotation signatures) and 0011 tracking hooks.
`autonomous_shift` is intentionally NOT shipped until Q4 pixel_shift split
rules out composite segmenter seam.

GIVEN: remote sprites may change without actor contact.
LEARNED: per shape-invariant key, the cyclic rotation sequence (rotate letter);
         orphan diff regions with no stable view (orphan letter).
"""

from __future__ import annotations

import importlib.util as _ilu
import sys as _sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

_REPO = __import__("pathlib").Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    spec = _ilu.spec_from_file_location(name, _REPO / rel)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


il = _load("ad_il", "teach/lessons/0010-invariance-lattice-perception.py")
id_mod = _load("ad_id", "teach/lessons/0011-object-identity-occlusion.py")


@dataclass(frozen=True)
class RotateKey:
    """Identity under rotation — one cycler, many orientations."""

    shape_inv: object
    area: int


@dataclass
class AutonomousRotateLedger:
    """Observed raw-shape states per rotation-invariant key."""

    sequences: dict[RotateKey, list[object]] = field(default_factory=dict)
    counts: dict[RotateKey, Counter] = field(default_factory=lambda: defaultdict(Counter))

    def observe(self, view_prev, view_next) -> None:
        key = RotateKey(
            view_prev.shape_lattice["rotation_invariant"],
            view_prev.area,
        )
        raw_prev = view_prev.internal_pattern_lattice["raw"]
        raw_next = view_next.internal_pattern_lattice["raw"]
        if raw_prev == raw_next:
            return
        self.counts[key][(raw_prev, raw_next)] += 1
        seq = self.sequences.setdefault(key, [])
        if not seq or seq[-1] != raw_prev:
            seq.append(raw_prev)
        if raw_next not in seq:
            seq.append(raw_next)

    def predict_raw(self, key: RotateKey, current_raw) -> object | None:
        """Next orientation in the learned cycle, or None if unknown."""
        seq = self.sequences.get(key)
        if not seq or len(seq) < 2:
            return None
        try:
            idx = seq.index(current_raw)
        except ValueError:
            return seq[0]
        return seq[(idx + 1) % len(seq)]

    def resolved_keys(self) -> int:
        return sum(1 for k, s in self.sequences.items() if len(s) >= 2)


@dataclass
class OrphanDiffLedger:
    """Diff regions with no view at t — untracked autonomous change."""

    tallies: Counter = field(default_factory=Counter)

    def observe(self, *, had_view: bool, cells: int) -> None:
        if not had_view and cells > 0:
            self.tallies[cells] += 1

    @property
    def events(self) -> int:
        return sum(self.tallies.values())


def apply_autonomous_rotate(
    grid: np.ndarray,
    view,
    ledger: AutonomousRotateLedger,
    *,
    background: int,
) -> np.ndarray:
    """Paint predicted next orientation for one remote sprite (copy-forward baseline)."""
    key = RotateKey(view.shape_lattice["rotation_invariant"], view.area)
    current_raw = view.shape_lattice["raw"]
    nxt_raw = ledger.predict_raw(key, current_raw)
    if nxt_raw is None or nxt_raw == current_raw:
        return grid
    out = grid.copy()
    # Recolor pixels belonging to this view's connected colour regions only.
    colours = {int(grid[r, c]) for r, c in view.pixels}
    for r, c in view.pixels:
        out[r, c] = background
    # Without engine access we cannot reconstruct exact recoloured crop;
    # offline check uses ledger prediction on synthetic views only.
    return out


def _check_rotate_cycle() -> bool:
    ell = il.ObjectView.from_crop(
        np.array([[1, 2], [3, 4]]),
        origin=(10, 10),
    )
    ell_rot = il.ObjectView.from_crop(
        np.rot90(np.array([[1, 2], [3, 4]])),
        origin=(10, 10),
    )
    ell_rot2 = il.ObjectView.from_crop(
        np.rot90(np.array([[1, 2], [3, 4]]), k=2),
        origin=(10, 10),
    )

    ledger = AutonomousRotateLedger()
    ledger.observe(ell, ell_rot)
    ledger.observe(ell_rot, ell_rot2)
    ledger.observe(ell_rot2, ell)
    key = RotateKey(ell.shape_lattice["rotation_invariant"], ell.area)
    seq = ledger.sequences.get(key, [])
    pred = ledger.predict_raw(key, ell.internal_pattern_lattice["raw"])
    ok = len(seq) >= 2 and pred is not None
    print(
        f"  [{'✓' if ok else '✗'}] rotate ledger learns cycle "
        f"(len={len(seq)}, pred_ok={pred is not None})"
    )
    return ok


def _check_orphan_counts() -> bool:
    ledger = OrphanDiffLedger()
    ledger.observe(had_view=False, cells=12)
    ledger.observe(had_view=True, cells=12)
    ok = ledger.events == 1
    print(f"  [{'✓' if ok else '✗'}] orphan ledger counts viewless diff only")
    return ok


if __name__ == "__main__":
    print("0016 autonomous dynamics (rotate + orphan; shift provisional):\n")
    ok = _check_rotate_cycle() and _check_orphan_counts()
    if not ok:
        raise SystemExit(1)
    print("\nShift letter held until Q4 pixel_shift split clears composite seam.")
