#!/usr/bin/env python3
"""Score a q3b scribe-port run against EXPERIMENT-CARD-q3b.md.

Frozen before launch. Reads only run artifacts; recomputes every label from the
recorded frames rather than trusting the host's own observation record, so a bug
in the host machinery cannot score itself as a success.

    python3 score_q3.py <run-dir>

Predictions (card):
  P1 batches len>1 carrying a predicted_path with >=0.6 expect-accuracy   >= 0.50
  P2 completed levels with a MODEL-WRITTEN consolidation note             == 1.00
     A note carrying the marker below was written by the host fallback, which is a
     P2 failure, not a save. P5 (helper refresh) is RECORDED, NOT GATED: the scribe
     makes the helper rewrite conditional on the evidence changing the model.
  P3 submitted actions re-entering a known no-impact/terminal (state,act) <= 0.05
     Counted over ATTEMPTS, not executions: the host now rejects a re-entry at
     submit time, so scoring only executed actions would make P3 vacuously 0.
     Rejected attempts are read from ipc/rejections.jsonl.
  P4 waste share of executed env actions (HUD-only or blocked)            <= 0.15

The unit for P1 and P2 is the AGENT TURN, as in the Q2 precommit. Do not
re-denominate to env steps.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HUD_MIN_FRAC = 0.9  # matches the host; on q2d frames rows 61-62 sit at 0.984, next row 0.228
HUD_WARMUP = 5
EXPECT_ACC_THRESHOLD = 0.6

P1_MIN, P2_MIN, P3_MAX, P4_MAX = 0.50, 1.00, 0.05, 0.15
FALLBACK_MARKER = "Deterministic fallback note:"


# ----------------------------------------------------------------- artifacts

def load_env_steps(run: Path) -> list[dict]:
    path = next(run.glob("agent_logs/**/events.jsonl"))
    ev = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    env = sorted((e for e in ev if e.get("event") == "env_step"), key=lambda e: e["step"])
    return env


def load_agent_turns(run: Path) -> list[dict]:
    path = next(run.glob("agent_logs/**/events.jsonl"))
    ev = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return [e for e in ev if e.get("event") == "agent_turn"]


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


# ------------------------------------------------------------------- labels

def discover_hud_rows(frames: list[np.ndarray]) -> list[int]:
    """Rows changing on >= HUD_MIN_FRAC of consecutive pairs. Never names a colour."""
    n = len(frames) - 1
    if n <= 0:
        return []
    hits = np.zeros(frames[0].shape[0], dtype=int)
    for a, b in zip(frames[:-1], frames[1:]):
        hits[(a != b).any(axis=1)] += 1
    return [int(r) for r, h in enumerate(hits) if h / n >= HUD_MIN_FRAC]


def classify(a: np.ndarray, b: np.ndarray, hud: list[int]) -> str:
    m = a != b
    if not m.any():
        return "blocked"
    rows = set(np.where(m.any(axis=1))[0].tolist())
    return "hud_only" if rows <= set(hud) else "playfield"


def outcome_class(label: str, state: str, level_up: bool) -> str:
    """The four classes a predicted_path entry's `expect` field may name."""
    if state in ("GAME_OVER", "WIN"):
        return "terminal"
    if level_up:
        return "level_up"
    return "no_impact" if label in ("hud_only", "blocked") else "move"


def playfield_key(grid: np.ndarray, hud: list[int]) -> bytes:
    keep = [r for r in range(grid.shape[0]) if r not in set(hud)]
    return grid[keep, :].tobytes()


# -------------------------------------------------------------------- score

def score(run: Path) -> dict:
    env = load_env_steps(run)
    grids = [np.array(e["grid_data"], dtype=np.int16) for e in env]
    if len(grids) < 2:
        return {"invalid": f"only {len(grids)} recorded frames"}

    hud = discover_hud_rows(grids)
    # labels[k] describes env step k+1 (the transition grids[k] -> grids[k+1])
    labels = [classify(grids[i], grids[i + 1], hud) for i in range(len(grids) - 1)]

    states = load_jsonl(run / "ipc/states.jsonl")
    batches = load_jsonl(run / "ipc/actions.jsonl")
    turns = load_agent_turns(run)

    # ---- per-action outcome, walked over the AUTHORITATIVE executed record.
    # states.jsonl action_results includes actions the harness force-advanced,
    # which have no actions.jsonl entry. Walking batches instead drifts the frame
    # index by one after every force-advance and mislabels the rest of the run.
    # A state with turn N carries the results of the batch submitted at turn N-1.
    by_submit_turn = {b.get("turn"): b for b in batches}
    per_action: list[dict] = []
    for st in states:
        results = st.get("action_results") or []
        if not results:
            continue
        end = st.get("step")  # env_steps AFTER this turn's actions
        k = len(results)
        b = by_submit_turn.get((st.get("turn") or 0) - 1)
        path = (b or {}).get("predicted_path")
        for j, r in enumerate(results):
            idx = end - k + j  # labels[idx] describes grids[idx] -> grids[idx+1]
            if idx < 0 or idx >= len(labels):
                continue
            per_action.append(
                {
                    "turn": st.get("turn"),
                    "submit_turn": (st.get("turn") or 0) - 1,
                    "idx": j,
                    "step": idx + 1,
                    "action": r.get("action"),
                    "forced": b is None,
                    "label": labels[idx],
                    "before": grids[idx],
                    "outcome": outcome_class(
                        labels[idx], r.get("state", ""), bool(r.get("level_completed"))
                    ),
                    "path_entry": path[j] if isinstance(path, list) and j < len(path) else None,
                }
            )

    # ---- P4 waste share of EXECUTED env actions
    executed = len(per_action)
    waste = sum(1 for a in per_action if a["label"] in ("hud_only", "blocked"))
    p4 = waste / executed if executed else 0.0
    # The card defines P4 over all executed actions and is the gate. Harness
    # force-advances (agent silent past the turn timeout) inject actions the model
    # did not choose; report the model-only figure ALONGSIDE, never in place of it.
    model_rows = [a for a in per_action if not a.get("forced")]
    p4_model = (
        sum(1 for a in model_rows if a["label"] in ("hud_only", "blocked")) / len(model_rows)
        if model_rows
        else None
    )

    # ---- P1 expect-accuracy on batches longer than 1
    long_batches, long_ok = 0, 0
    per_batch = []
    for b in batches:
        acts = b.get("actions", [])
        if len(acts) <= 1:
            continue
        long_batches += 1
        path = b.get("predicted_path")
        rows = [a for a in per_action if a["submit_turn"] == b.get("turn")]
        if not path or len(path) != len(acts) or not rows:
            per_batch.append({"turn": b.get("turn"), "n": len(acts), "acc": None})
            continue
        checked = min(len(rows), len(path))
        matched = sum(
            1
            for i in range(checked)
            if isinstance(path[i], dict) and path[i].get("expect") == rows[i]["outcome"]
        )
        acc = matched / checked if checked else 0.0
        long_ok += acc >= EXPECT_ACC_THRESHOLD
        per_batch.append({"turn": b.get("turn"), "n": len(acts), "acc": round(acc, 3)})
    p1 = long_ok / long_batches if long_batches else None

    # ---- P2 helper rewritten within 2 agent invocations of a level change.
    # Message files are step_<ENV STEP>_round_<N>; the earlier version read the
    # ROUND number as if it were an agent turn, which compared unrelated integers.
    level_up_steps = [a["step"] for a in per_action if a["outcome"] == "level_up"]
    invocation_steps, write_steps = agent_invocations(run)

    # ---- P2 (GATE): a MODEL-written consolidation for every completed level.
    completed = int(result_levels(run))
    notes_dir = next(iter(run.glob("**/knowledge")), None)
    p2_rows = []
    for lvl in range(completed):
        p = (notes_dir / f"level_{lvl}_insights.md") if notes_dir else None
        if p is None or not p.exists():
            p2_rows.append({"level": lvl, "state": "missing"})
            continue
        body = p.read_text()
        p2_rows.append(
            {
                "level": lvl,
                "state": "fallback" if FALLBACK_MARKER in body else "model_written",
                "bytes": len(body),
            }
        )
    p2 = (
        sum(1 for r in p2_rows if r["state"] == "model_written") / len(p2_rows)
        if p2_rows
        else None
    )

    # ---- P5 (RECORDED, NOT GATED): helper refreshed at the boundary.
    p5_hits = []
    for s_up in level_up_steps:
        later = [s for s in invocation_steps if s >= s_up]
        p5_hits.append(bool(set(later[:2]) & write_steps))
    p5 = (sum(p5_hits) / len(p5_hits)) if p5_hits else None

    # ---- P3 re-entry into a known no-impact / terminal (playfield state, action)
    known: dict[tuple, str] = {}
    reentries = []
    for a in per_action:
        key = (playfield_key(a["before"], hud), a["action"])
        if key in known:
            reentries.append({"step": a["step"], "action": a["action"], "was": known[key]})
        if a["outcome"] in ("no_impact", "terminal"):
            known[key] = a["outcome"]
    # P3 over ATTEMPTS: executed re-entries plus the ones the host refused.
    rej_path = run / "ipc/rejections.jsonl"
    rejections = load_jsonl(rej_path) if rej_path.exists() else []
    rejected_reentry_actions = sum(
        len(r.get("actions", [])) for r in rejections if r.get("reason") == "reentry"
    )
    rejected_actions_total = sum(len(r.get("actions", [])) for r in rejections)
    attempted = executed + rejected_actions_total
    p3 = ((len(reentries) + rejected_reentry_actions) / attempted) if attempted else 0.0

    result = json.loads((run / "result.json").read_text())
    sizes = [len(b.get("actions", [])) for b in batches]
    median_batch = float(np.median(sizes)) if sizes else 0.0

    return {
        "hud_rows": hud,
        "agent_turns": len(batches),
        "env_actions_executed": executed,
        "P1_search_batches": p1,
        "P1_long_batches": long_batches,
        "P1_detail": per_batch,
        "P2_scribe": p2,
        "P2_consolidations": p2_rows,
        "P5_helper_refresh_RECORDED_NOT_GATED": p5,
        "P5_level_up_steps": level_up_steps,
        "P5_helper_write_steps": sorted(write_steps),
        "forced_actions": sum(1 for a in per_action if a.get("forced")),
        "P3_reentry_rate": p3,
        "P3_executed_reentries": len(reentries),
        "P3_rejected_reentries": rejected_reentry_actions,
        "P3_actions_attempted": attempted,
        "P3_reentries": reentries[:20],
        "rejections_by_reason": {
            k: sum(1 for r in rejections if r.get("reason") == k)
            for k in sorted({r.get("reason") for r in rejections})
        },
        "P4_waste_share": p4,
        "P4_waste_share_model_only": p4_model,
        "P4_forced_actions_excluded": executed - len(model_rows),
        "median_batch": median_batch,
        "levels_completed": result.get("levels_completed"),
        "game_overs": result.get("game_overs"),
        "rhae_game_score": result.get("rhae_game_score"),
        "termination": result.get("termination"),
    }


def agent_invocations(run: Path) -> tuple[list[int], set[int]]:
    """(every agent invocation's env step, the subset that wrote a helper).

    Message files are ``step_<ENV STEP>_round_<N>_assistant.md``. The env step is
    the invocation identity; the round is a cell within it.
    """
    steps: set[int] = set()
    wrote: set[int] = set()
    for p in sorted(run.glob("agent_logs/**/messages/*_assistant.md")):
        try:
            st = int(p.name.split("step_")[1].split("_")[0])
        except (IndexError, ValueError):
            continue
        steps.add(st)
        try:
            if "write_helper" in p.read_text():
                wrote.add(st)
        except Exception:
            continue
    return sorted(steps), wrote


def result_levels(run: Path) -> int:
    try:
        return json.loads((run / "result.json").read_text()).get("levels_completed") or 0
    except Exception:
        return 0


def verdict(s: dict) -> tuple[bool, list[str]]:
    lines, ok = [], True

    unmeasured: list[str] = []

    def check(name, val, cmp, thr):
        nonlocal ok
        if val is None:
            # An unmeasured gate is NOT a pass. The card classifies a clause with no
            # qualifying units as INCONCLUSIVE; letting it fall through produced a
            # "PASS" verdict on q3b while its primary clause never ran.
            unmeasured.append(name)
            lines.append(f"{name}: INCONCLUSIVE — no qualifying units")
            return
        good = val >= thr if cmp == ">=" else val <= thr
        ok = ok and good
        lines.append(f"{name}: {val:.3f} {cmp} {thr:.2f} -> {'pass' if good else 'FAIL'}")

    check("P1 search-backed batches", s.get("P1_search_batches"), ">=", P1_MIN)
    check("P2 model-written consolidation", s.get("P2_scribe"), ">=", P2_MIN)
    check("P3 re-entry rate", s.get("P3_reentry_rate"), "<=", P3_MAX)
    check("P4 waste share", s.get("P4_waste_share"), "<=", P4_MAX)

    # q3a's median-batch counter-outcome is NOT carried forward: it fired on a
    # statistic that cannot describe a bimodal policy (9 long batches, median 1).
    fb = [r for r in (s.get("P2_consolidations") or []) if r.get("state") == "fallback"]
    if fb:
        lines.append(
            f"HOST SAVE: {len(fb)} consolidation(s) written by the fallback, not the "
            "model — a P2 failure, not a rescue (see card Result map)."
        )
    lines.append(
        f"P5 helper refresh (recorded, not gated): "
        f"{s.get('P5_helper_refresh_RECORDED_NOT_GATED')}"
    )
    if unmeasured:
        lines.append(
            "VERDICT IS INCONCLUSIVE: " + ", ".join(unmeasured) + " could not be measured."
        )
        return None, lines
    return ok, lines


def main() -> None:
    run = Path(sys.argv[1]).resolve()
    s = score(run)
    if "invalid" in s:
        print("INVALID:", s["invalid"])
        sys.exit(2)
    ok, lines = verdict(s)
    print(json.dumps({k: v for k, v in s.items() if not k.endswith("_detail")}, indent=2, default=str))
    print()
    for l in lines:
        print(" ", l)
    print()
    print("VERDICT:", "INCONCLUSIVE" if ok is None else ("PASS" if ok else "FAIL"))
    (run / "score_q3.json").write_text(json.dumps(s, indent=2, default=str))


if __name__ == "__main__":
    main()
