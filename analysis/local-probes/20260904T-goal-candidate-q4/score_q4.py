#!/usr/bin/env python3
"""Score a q4 goal-candidate run against EXPERIMENT-CARD.md.

Frozen before launch. Reads only run artifacts; recomputes labels from frames
rather than trusting ipc/goal_candidate.json's status field, so a bug in the
host machinery cannot score itself as a success.

    python3 score_q4.py <run-dir>

P1–P4 hold the q3c control loop. P6/P7 are the frontier-4 clauses.
P5 and P8 are recorded, not gated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HUD_MIN_FRAC = 0.9
HUD_WARMUP = 5
EXPECT_ACC_THRESHOLD = 0.6

P1_MIN, P2_MIN, P3_MAX, P4_MAX = 0.50, 1.00, 0.05, 0.15
P6_MIN, P7_MIN = 1.00, 1.00
FALLBACK_NOTE = "Deterministic fallback note:"
FALLBACK_CANDIDATE = "Deterministic fallback candidate:"
TERMINAL_EXPECTS = frozenset({"level_up", "terminal"})
DEADLINE_FLOOR = 16
DISCRIMINATOR_MAX_ACTIONS = 5


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
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def discover_hud_rows(frames: list[np.ndarray]) -> list[int]:
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
    if state in ("GAME_OVER", "WIN"):
        return "terminal"
    if level_up:
        return "level_up"
    return "no_impact" if label in ("hud_only", "blocked") else "move"


def playfield_key(grid: np.ndarray, hud: list[int]) -> bytes:
    keep = [r for r in range(grid.shape[0]) if r not in set(hud)]
    return grid[keep, :].tobytes()


def path_has_terminal_expect(path) -> bool:
    if not isinstance(path, list):
        return False
    return any(isinstance(e, dict) and e.get("expect") in TERMINAL_EXPECTS for e in path)


def is_discriminator_batch(actions, path) -> bool:
    return (
        isinstance(actions, list)
        and 1 <= len(actions) <= DISCRIMINATOR_MAX_ACTIONS
        and path_has_terminal_expect(path)
        and (not isinstance(path, list) or len(path) == len(actions))
    )


def terminal_expect_hit(path, outcomes: list[str]) -> bool | None:
    """True if a terminal-class expect matched; False if checked and missed; None if none executed."""
    if not isinstance(path, list) or not outcomes:
        return None
    n = min(len(path), len(outcomes))
    checked = 0
    matched = 0
    for i in range(n):
        e = path[i] if isinstance(path[i], dict) else {}
        want = e.get("expect")
        if want in TERMINAL_EXPECTS:
            checked += 1
            if want == outcomes[i]:
                matched += 1
    if checked == 0:
        return None
    return matched > 0


def score(run: Path) -> dict:
    env = load_env_steps(run)
    grids = [np.array(e["grid_data"], dtype=np.int16) for e in env]
    if len(grids) < 2:
        return {"invalid": f"only {len(grids)} recorded frames"}

    hud = discover_hud_rows(grids)
    labels = [classify(grids[i], grids[i + 1], hud) for i in range(len(grids) - 1)]

    states = load_jsonl(run / "ipc/states.jsonl")
    batches = load_jsonl(run / "ipc/actions.jsonl")
    turns = load_agent_turns(run)

    by_submit_turn = {b.get("turn"): b for b in batches}
    per_action: list[dict] = []
    for st in states:
        results = st.get("action_results") or []
        if not results:
            continue
        end = st.get("step")
        k = len(results)
        b = by_submit_turn.get((st.get("turn") or 0) - 1)
        path = (b or {}).get("predicted_path")
        for j, r in enumerate(results):
            idx = end - k + j
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

    executed = len(per_action)
    waste = sum(1 for a in per_action if a["label"] in ("hud_only", "blocked"))
    p4 = waste / executed if executed else 0.0
    model_rows = [a for a in per_action if not a.get("forced")]
    p4_model = (
        sum(1 for a in model_rows if a["label"] in ("hud_only", "blocked")) / len(model_rows)
        if model_rows
        else None
    )

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

    level_up_steps = [a["step"] for a in per_action if a["outcome"] == "level_up"]
    invocation_steps, write_steps = agent_invocations(run)

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
                "state": "fallback" if FALLBACK_NOTE in body else "model_written",
                "bytes": len(body),
            }
        )
    p2 = (
        sum(1 for r in p2_rows if r["state"] == "model_written") / len(p2_rows)
        if p2_rows
        else None
    )

    p5_hits = []
    for s_up in level_up_steps:
        later = [s for s in invocation_steps if s >= s_up]
        p5_hits.append(bool(set(later[:2]) & write_steps))
    p5 = (sum(p5_hits) / len(p5_hits)) if p5_hits else None

    known: dict[tuple, str] = {}
    reentries = []
    for a in per_action:
        key = (playfield_key(a["before"], hud), a["action"])
        if key in known:
            reentries.append({"step": a["step"], "action": a["action"], "was": known[key]})
        if a["outcome"] in ("no_impact", "terminal"):
            known[key] = a["outcome"]
    rej_path = run / "ipc/rejections.jsonl"
    rejections = load_jsonl(rej_path)
    rejected_reentry_actions = sum(
        len(r.get("actions", [])) for r in rejections if r.get("reason") == "reentry"
    )
    rejected_actions_total = sum(len(r.get("actions", [])) for r in rejections)
    attempted = executed + rejected_actions_total
    p3 = ((len(reentries) + rejected_reentry_actions) / attempted) if attempted else 0.0

    result = json.loads((run / "result.json").read_text())
    sizes = [len(b.get("actions", [])) for b in batches]
    median_batch = float(np.median(sizes)) if sizes else 0.0

    # ---- P6 / P7: recomputed from frames + actions.jsonl, not host status.
    cand_live = run / "ipc/goal_candidate.json"
    cand_log = load_jsonl(run / "ipc/goal_candidate.jsonl")
    port_fired = cand_live.exists() or bool(cand_log)
    p6_rows, p7_rows = [], []
    p8_rows = []

    for i, s_up in enumerate(level_up_steps):
        term_path = run / "ipc" / f"level_{i}_terminal.json"
        spent = None
        if term_path.exists():
            try:
                spent = json.loads(term_path.read_text()).get("actions_spent_on_level")
            except Exception:
                spent = None
        if spent is None:
            spent = s_up - (level_up_steps[i - 1] if i else 0)
        deadline = max(DEADLINE_FLOOR, int(spent))
        window_end = s_up + deadline
        window_batches = []
        for b in batches:
            rows = [a for a in per_action if a["submit_turn"] == b.get("turn")]
            if not rows:
                continue
            first = min(a["step"] for a in rows)
            if s_up < first <= window_end:
                window_batches.append((b, rows, first))
        fallback_hit = False
        for rec in cand_log:
            if rec.get("level") == i + 1 and rec.get("fallback") and FALLBACK_CANDIDATE in str(
                rec.get("fallback")
            ):
                fallback_hit = True
        if cand_live.exists():
            try:
                live = json.loads(cand_live.read_text())
            except Exception:
                live = {}
            if live.get("level") == i + 1 and live.get("fallback") and FALLBACK_CANDIDATE in str(
                live.get("fallback")
            ):
                fallback_hit = True

        disc = None
        for b, rows, first in window_batches:
            path = b.get("predicted_path")
            if not is_discriminator_batch(b.get("actions") or [], path):
                continue
            outcomes = [a["outcome"] for a in rows]
            hit = terminal_expect_hit(path, outcomes)
            if hit is None:
                continue
            disc = {
                "turn": b.get("turn"),
                "step": first,
                "actions": tuple(b.get("actions") or []),
                "hit": bool(hit),
            }
            break

        p6_ok = disc is not None and not fallback_hit
        p6_rows.append(
            {
                "completed_level": i,
                "level_up_step": s_up,
                "deadline_actions": deadline,
                "discriminator": None
                if disc is None
                else {k: (list(v) if k == "actions" else v) for k, v in disc.items()},
                "fallback": fallback_hit,
                "ok": p6_ok,
            }
        )

        # P8 recorded: predicted_terminal vs actual winning action class on the
        # *next* completion, if any. Not gated.
        if i + 1 < len(level_up_steps) and disc is not None:
            p8_rows.append(
                {
                    "from_level": i,
                    "discriminator_hit_on_later_completion": disc["hit"],
                }
            )

        # P7 only when a discriminator missed, or the deadline expired.
        expired = disc is None
        contradicted = disc is not None and not disc["hit"]
        if not (expired or contradicted):
            p7_rows.append({"completed_level": i, "applicable": False, "ok": None})
            continue
        t_star = disc["step"] if contradicted else window_end
        extra_end = t_star + deadline
        later = []
        for b in batches:
            rows = [a for a in per_action if a["submit_turn"] == b.get("turn")]
            if not rows:
                continue
            first = min(a["step"] for a in rows)
            if t_star < first <= extra_end:
                later.append(b)
        missed_tuple = disc["actions"] if contradicted else None
        different = False
        for b in later:
            path = b.get("predicted_path")
            acts = tuple(b.get("actions") or [])
            if is_discriminator_batch(list(acts), path) and acts != missed_tuple:
                different = True
                break
        all_len1 = bool(later) and all(len(b.get("actions") or []) <= 1 for b in later)
        p7_ok = different or all_len1
        p7_rows.append(
            {
                "completed_level": i,
                "applicable": True,
                "t_star": t_star,
                "later_batches": len(later),
                "different_discriminator": different,
                "all_length_1": all_len1,
                "ok": p7_ok,
            }
        )

    n_transfer = len(p6_rows)
    p6 = (sum(1 for r in p6_rows if r["ok"]) / n_transfer) if n_transfer else None
    p7_app = [r for r in p7_rows if r.get("applicable")]
    p7 = (sum(1 for r in p7_app if r["ok"]) / len(p7_app)) if p7_app else None

    truncated_cells = 0
    for p in run.glob("agent_logs/**/messages/*.md"):
        try:
            if "<truncated>" in p.read_text(errors="ignore"):
                truncated_cells += 1
        except Exception:
            continue

    out = {
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
        "port_fired": port_fired,
        "P6_discriminator_before_deadline": p6,
        "P6_detail": p6_rows,
        "P7_revision_after_miss_or_expiry": p7,
        "P7_detail": p7_rows,
        "P8_terminal_signature_RECORDED_NOT_GATED": p8_rows,
        "truncated_cell_count": truncated_cells,
        "agent_turn_events": len(turns),
    }
    if completed >= 1 and not port_fired:
        # q4 invalid-if: the port did not fire. Scoring the q3c incumbent is a
        # separate use: that run has transfers and no file, which is P6=0.
        out["q4_port_missing_after_transfer"] = True
    return out


def agent_invocations(run: Path) -> tuple[list[int], set[int]]:
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


def verdict(s: dict, *, require_port: bool) -> tuple[bool | None, list[str]]:
    lines, ok = [], True
    unmeasured: list[str] = []

    if require_port and s.get("q4_port_missing_after_transfer"):
        lines.append("INVALID: ipc/goal_candidate.json absent after a level-transfer")
        return False, lines

    def check(name, val, cmp, thr):
        nonlocal ok
        if val is None:
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
    check("P6 discriminator before deadline", s.get("P6_discriminator_before_deadline"), ">=", P6_MIN)
    # P7 is gated only when applicable (a miss or expiry existed).
    if s.get("P7_revision_after_miss_or_expiry") is None:
        lines.append("P7 revision after miss/expiry: N/A (no contradicted or expired candidate)")
    else:
        check(
            "P7 revision after miss/expiry",
            s.get("P7_revision_after_miss_or_expiry"),
            ">=",
            P7_MIN,
        )

    fb = [r for r in (s.get("P2_consolidations") or []) if r.get("state") == "fallback"]
    if fb:
        lines.append(
            f"HOST SAVE: {len(fb)} consolidation(s) written by the fallback, not the "
            "model — a P2 failure, not a rescue."
        )
    lines.append(
        f"P5 helper refresh (recorded, not gated): "
        f"{s.get('P5_helper_refresh_RECORDED_NOT_GATED')}"
    )
    lines.append(f"P8 terminal signature (recorded, not gated): {s.get('P8_terminal_signature_RECORDED_NOT_GATED')}")
    lines.append(f"<truncated> cells: {s.get('truncated_cell_count')}")
    if unmeasured:
        lines.append(
            "VERDICT IS INCONCLUSIVE: " + ", ".join(unmeasured) + " could not be measured."
        )
        return None, lines
    return ok, lines


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--incumbent"]
    require_port = "--incumbent" not in sys.argv
    if not args:
        print("usage: score_q4.py [--incumbent] <run-dir>", file=sys.stderr)
        sys.exit(2)
    run = Path(args[0]).resolve()
    s = score(run)
    if "invalid" in s:
        print("INVALID:", s["invalid"])
        sys.exit(2)
    ok, lines = verdict(s, require_port=require_port)
    skip = {"P1_detail", "P3_reentries"}
    print(json.dumps({k: v for k, v in s.items() if k not in skip}, indent=2, default=str))
    print()
    for l in lines:
        print(" ", l)
    print()
    print("VERDICT:", "INCONCLUSIVE" if ok is None else ("PASS" if ok else "FAIL"))
    (run / "score_q4.json").write_text(json.dumps(s, indent=2, default=str))


if __name__ == "__main__":
    main()
