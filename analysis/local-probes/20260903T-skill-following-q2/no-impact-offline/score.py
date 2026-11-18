#!/usr/bin/env python3
"""HUD no-impact predicate, scored on a recorded NOOA run.

Discovers HUD rows as rows that change on ≥ min_frac of consecutive pairs.
Does not mention a colour or a row index. Classifies each pair as
blocked / hud_only / playfield.

    python3 score.py <run-dir>
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def load_grids(run: Path) -> tuple[list[np.ndarray], list[dict]]:
    path = next(run.glob("agent_logs/**/events.jsonl"))
    env = [
        json.loads(l)
        for l in path.read_text().splitlines()
        if l.strip() and json.loads(l).get("event") == "env_step"
    ]
    env.sort(key=lambda e: e["step"])
    grids = [np.array(e["grid_data"], dtype=np.int16) for e in env]
    return grids, env


def load_gameplay_steps(run: Path) -> list[dict]:
    gp = json.loads((run / "gameplay.json").read_text())
    return [e for e in gp["log"] if e.get("type") == "step"]


def discover_hud_rows(frames: list[np.ndarray], min_frac: float = 0.8) -> list[int]:
    n = len(frames) - 1
    hits = np.zeros(frames[0].shape[0], dtype=int)
    for a, b in zip(frames[:-1], frames[1:]):
        hits[(a != b).any(axis=1)] += 1
    return [int(r) for r, h in enumerate(hits) if n and h / n >= min_frac]


def classify(a: np.ndarray, b: np.ndarray, hud_rows: list[int]) -> str:
    m = a != b
    if not m.any():
        return "blocked"
    rows = set(np.where(m.any(axis=1))[0].tolist())
    if rows <= set(hud_rows):
        return "hud_only"
    return "playfield"


def score(run: Path) -> dict:
    grids, env = load_grids(run)
    steps = load_gameplay_steps(run)
    assert len(grids) >= 121 and grids[0].shape == (64, 64), "invalid trajectory"
    hud = discover_hud_rows(grids)
    by_state: dict[str, list[int]] = {"blocked": [], "hud_only": [], "playfield": []}
    rows = []
    for i in range(1, len(grids)):
        label = classify(grids[i - 1], grids[i], hud)
        n = int((grids[i - 1] != grids[i]).sum())
        gp = steps[i - 1] if i - 1 < len(steps) else {}
        rec = {
            "step": env[i]["step"],
            "action": env[i].get("action_name"),
            "label": label,
            "n_changed": n,
            "levels_completed": env[i].get("levels_completed"),
            "state": gp.get("state"),
            "reward": gp.get("reward"),
        }
        rows.append(rec)
        by_state[label].append(i)
    tiny = [r for r in rows if r["n_changed"] <= 4]
    death_hud = [
        r
        for r in rows
        if r["label"] in ("hud_only", "blocked") and "GAME_OVER" in str(r["state"])
    ]
    tiny_not_hud = [r for r in tiny if r["label"] == "playfield"]
    hud_not_tiny = [
        r
        for r in rows
        if r["label"] == "hud_only" and r["n_changed"] > 4 and "GAME_OVER" not in str(r["state"])
    ]
    return {
        "run": run.name,
        "n_pairs": len(rows),
        "hud_rows": hud,
        "counts": {k: len(v) for k, v in by_state.items()},
        "tiny_n_le_4": len(tiny),
        "tiny_labelled_playfield": tiny_not_hud,
        "hud_only_n_gt_4_non_death": hud_not_tiny,
        "hud_only_game_over": death_hud,
        "share_non_playfield": round(
            (len(by_state["blocked"]) + len(by_state["hud_only"])) / len(rows), 3
        ),
        "by_level": {
            "lv0": dict(
                Counter(r["label"] for r in rows if r["levels_completed"] == 0)
            ),
            "lv1": dict(
                Counter(r["label"] for r in rows if r["levels_completed"] == 1)
            ),
        },
        "hud_only_steps": [r["step"] for r in rows if r["label"] == "hud_only"],
        "blocked_steps": [r["step"] for r in rows if r["label"] == "blocked"],
    }


if __name__ == "__main__":
    print(json.dumps(score(Path(sys.argv[1])), indent=2))
