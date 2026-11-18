#!/usr/bin/env python3
"""Recompute q6 endpoints from raw proposals, executed frames and authored bases.

No authority verdict (actual, abandon_if_satisfied, resolution, stall_escaped)
is used to award a pass. Incomplete K=20 follow-up is censored, never an escape.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "patched-source/after"))
from host_contract import ObservationRecord  # noqa: E402
from stall_contract import (  # noqa: E402
    detect_stall,
    falsifier_satisfied,
    resolution_error,
    transition_outcomes,
    valid_basis,
)

K = 20


def rows(path: Path, required: bool = False) -> list[dict]:
    if not path.exists():
        if required:
            raise ValueError(f"missing {path.name}")
        return []
    content = path.read_text()
    if content and not content.endswith("\n"):
        raise ValueError(f"incomplete record in {path.name}")
    result = [json.loads(line) for line in content.splitlines() if line.strip()]
    if any(not isinstance(r, dict) for r in result):
        raise ValueError(f"non-object record in {path.name}")
    return result


def fraction(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def score_run(run: Path) -> dict:
    try:
        return _score_run(run)
    except (ValueError, KeyError, TypeError, IndexError, OSError) as error:
        return {"run": str(run), "status": "INVALID", "errors": [str(error)]}


def _score_run(run: Path) -> dict:
    ipc = run / "ipc"
    launch = json.loads((run / "launch.json").read_text())
    arm = launch["arm"]
    if arm not in ("control", "treatment"):
        raise ValueError("arm must be control or treatment")
    states = rows(ipc / "states.jsonl", True)
    batches = rows(ipc / "actions.jsonl", True)
    attempts = rows(ipc / "policy_attempts.jsonl", True)
    transitions = rows(ipc / "transitions.jsonl", True)
    authority = rows(ipc / "authority.jsonl")
    errors = []
    if not states or not transitions:
        raise ValueError("no initial state or executed frames")
    by_turn = {s["turn"]: s for s in states}
    if len(by_turn) != len(states):
        errors.append("duplicate published state turn")
    if len({b["turn"] for b in batches}) != len(batches):
        errors.append("duplicate committed batch turn")
    if len({a["turn"] for a in attempts}) != len(attempts):
        errors.append("duplicate detector attempt turn")
    initial = states[0]
    if initial["step"] != 0:
        errors.append("missing initial frame at step zero")
    initial_grid = [[int(v, 16) for v in row] for row in initial["grid_rows"]]
    frame_at = {
        0: {
            "grid": initial_grid,
            "state": initial["state"],
            "levels_completed": initial["levels_completed"],
            "available_actions": initial["available_actions"],
        }
    }
    for expected, transition in enumerate(transitions, 1):
        if transition["step"] != expected:
            errors.append("executed transition steps are not contiguous")
        previous = frame_at.get(expected - 1)
        if previous is not None and transition["before"] != previous:
            errors.append(f"frame continuity fails at step {expected}")
        frame_at[transition["step"]] = transition["after"]
    for state in states:
        frame = frame_at.get(state["step"])
        if (
            frame is None
            or frame["grid"]
            != [[int(v, 16) for v in row] for row in state["grid_rows"]]
            or any(
                frame[key] != state[key]
                for key in ("state", "levels_completed", "available_actions")
            )
        ):
            errors.append(
                f"published state differs from raw frame at turn {state['turn']}"
            )
    for batch in batches:
        if (
            batch["turn"] not in by_turn
            or batch.get("step") != by_turn[batch["turn"]]["step"]
        ):
            errors.append("batch is not joined to its starting observation")
    attempt_turns = [a["turn"] for a in attempts]
    if attempt_turns != sorted(attempt_turns):
        errors.append("detector attempts are out of order")
    if [b["turn"] for b in batches] != sorted(b["turn"] for b in batches):
        errors.append("committed batches are out of order")
    for batch in batches:
        if batch["turn"] not in attempt_turns:
            errors.append("committed batch has no detector attempt")
        actual = [
            r["action"]
            for r in transitions
            if r["turn"] == batch["turn"] and not r.get("automatic")
        ]
        if not actual or actual != batch["actions"][: len(actual)]:
            errors.append(
                "executed actions are not a nonempty prefix of the committed batch"
            )
    for transition in transitions:
        if not transition.get("automatic") and transition["turn"] not in {
            b["turn"] for b in batches
        }:
            errors.append("executed action has no committed batch")
    detections = []
    for attempt in attempts:
        state = by_turn.get(attempt["turn"])
        if state is None or state["step"] != attempt["step"]:
            errors.append("detector attempt is not joined to its observation")
            continue
        history = [b for b in batches if b["turn"] < attempt["turn"]]
        result = detect_stall(history, state, attempt["rationale"])
        if result:
            detections.append({"stall_id": len(detections) + 1, **result})
    logged = [r for r in authority if r["event"] == "STALL_DETECTED"]
    if [(r.get("stall_id"), r.get("turn"), r.get("step")) for r in logged] != [
        (r["stall_id"], r["turn"], r["step"]) for r in detections
    ]:
        errors.append("logged detections differ from recomputed proposals/frames")
    max_step = transitions[-1]["step"]
    outcomes = transition_outcomes(transitions)
    by_step = {r["step"]: (r, outcome) for r, outcome in zip(transitions, outcomes)}
    reviews = [r for r in authority if r["event"] == "AUTHORITY_REVIEWED"]
    resolved = [r for r in authority if r["event"] == "AUTHORITY_RESOLVED"]
    consequences = [r for r in authority if r["event"] == "CONSEQUENCE_OBSERVED"]
    if arm == "control" and (reviews or resolved or consequences):
        errors.append("control received authority treatment")
    for kind, records in [
        ("review", reviews),
        ("resolution", resolved),
        ("consequence", consequences),
    ]:
        if len({r.get("stall_id") for r in records}) != len(records):
            errors.append(f"duplicate {kind} stall id")
        if any(
            r.get("stall_id") not in {d["stall_id"] for d in detections}
            for r in records
        ):
            errors.append(f"{kind} has no recomputed detection")
    p9, p10, p11, p12 = [], [], [], []
    details = []
    for index, detection in enumerate(detections):
        sid, step = detection["stall_id"], detection["step"]
        next_detection = detections[index + 1] if index + 1 < len(detections) else None
        recurs = next_detection is not None and next_detection["step"] - step <= K
        escape = False if recurs else (True if max_step >= step + K else None)
        if escape is not None:
            p9.append(escape)
        detail = {
            "stall_id": sid,
            "step": step,
            "escape": escape,
            "censored": escape is None,
        }
        if arm == "treatment":
            review = next((r for r in reviews if r.get("stall_id") == sid), None)
            resolution = next((r for r in resolved if r.get("stall_id") == sid), None)
            consequence = next(
                (r for r in consequences if r.get("stall_id") == sid), None
            )
            if review is None:
                errors.append(f"stall {sid} has no committed review")
                details.append(detail)
                continue
            error = resolution_error(
                review, by_turn[detection["turn"]]["available_actions"]
            )
            if error:
                errors.append(f"stall {sid}: {error}")
            p11.append(
                review.get("step") == step and review.get("turn") == detection["turn"]
            )
            batch = next((b for b in batches if b["turn"] == detection["turn"]), None)
            if batch is None or batch["actions"] != [review.get("action")]:
                errors.append(f"stall {sid} did not commit exactly its reviewed action")
            executed = by_step.get(step + 1)
            if executed is None:
                errors.append(f"stall {sid} has no executed consequence")
            else:
                transition, outcome = executed
                if transition["turn"] != detection["turn"] or transition[
                    "action"
                ] != review.get("action"):
                    errors.append(f"stall {sid} executed a different action")
                satisfied = falsifier_satisfied(
                    review.get("abandon_if"), transition, outcome
                )
                changed = (
                    resolution is not None
                    and valid_basis(resolution.get("active_basis_after"))
                    and resolution["active_basis_after"] != review.get("active_basis")
                )
                detail.update(
                    disposition=review["disposition"],
                    actual=outcome,
                    abandon_if_satisfied=satisfied,
                    authority_changed=changed,
                )
                if consequence is None or consequence.get("step") != step + 1:
                    errors.append(f"stall {sid} missing consequence trace")
                if resolution is None:
                    detail["authority_censored"] = True
                    if any(b["step"] > step for b in batches):
                        errors.append(
                            f"stall {sid} played on without authored authority after consequence"
                        )
                elif resolution.get("step") != step + 1 or not valid_basis(
                    resolution.get("active_basis_after")
                ):
                    errors.append(f"stall {sid} malformed authority after consequence")
                # Unknown continue falsifiers count as misses, never vanish from P12.
                if review["disposition"] == "continue":
                    if satisfied is True or satisfied is None:
                        p12.append(satisfied is True and changed)
                    if escape is not None:
                        next_review = next(
                            (
                                r
                                for r in reviews
                                if next_detection
                                and r.get("stall_id") == next_detection["stall_id"]
                            ),
                            None,
                        )
                        if recurs and next_review is None:
                            errors.append(
                                f"stall {sid} recurrence has no authored basis"
                            )
                        p10.append(
                            bool(
                                recurs
                                and next_review
                                and next_review["previous_basis"]
                                == review["active_basis"]
                            )
                        )
        details.append(detail)
    if not all(p11):
        errors.append("nonzero review latency")
    holds = control_loop_metrics(run, batches, transitions)
    if (ipc / "ABORT").exists():
        errors.append("host ABORT exists")
    integrity = rows(ipc / "integrity.jsonl", True)
    for record in integrity:
        if (
            record.get("event") == "NOTE_WRITE"
            and record["authored"] != record["stored"]
        ):
            errors.append("authored-to-stored mutation")
        if record.get("event") == "INTEGRITY_FAILED":
            errors.append("runtime integrity failure")
        if record.get("event") == "NOTE_RENDER" and (
            record["stored"] != record["rendered"]
            or record.get("expected_event") != record.get("rendered_event")
        ):
            errors.append("stored-to-rendered mutation")
    if not any(r.get("event") == "GUARD_INSTALLED" for r in integrity):
        errors.append("runtime integrity guard was not installed")
    # Actual request/usage logging is mandatory before any research interpretation.
    usage = rows(ipc / "model_calls.jsonl")
    if launch.get("qualification") is not True:
        if not usage:
            errors.append("no model-call evidence")
        if any(
            r.get("reasoning_tokens", 0) > 0
            or (r.get("usage") or {}).get("reasoning_tokens", 0) > 0
            for r in usage
        ):
            errors.append("reasoning tokens returned")
    manifest = launch.get("manifest")
    if launch.get("qualification") is not True:
        if launch.get("platform") == "kaggle" and (
            not launch.get("run_id") or not launch.get("session_id")
        ):
            errors.append("missing independent run or worker session identity")
        if not isinstance(manifest, dict):
            errors.append("missing candidate provenance")
        else:
            identity = hashlib.sha256(
                json.dumps(
                    {"files": manifest["files"], "sources": manifest["sources"]},
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            if identity != launch.get("contract_sha256"):
                errors.append("candidate provenance hash mismatch")
            scorer_key = (
                "analysis/local-probes/20260905T-stall-authority-q6/score_q6.py"
            )
            if (
                manifest["sources"].get(scorer_key)
                != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            ):
                errors.append("scorer differs from frozen candidate")
            for module_name in ("stall_contract", "host_contract"):
                loaded = Path(sys.modules[module_name].__file__)
                expected = manifest["files"].get(f"chassis/{module_name}.py")
                if hashlib.sha256(loaded.read_bytes()).hexdigest() != expected:
                    errors.append(f"loaded {module_name} differs from frozen candidate")
        if not (run / "stop.json").exists() and launch.get("platform") == "kaggle":
            errors.append("worker session has no normal stop artifact")
        if not launch.get("model_config", {}).get("ARC_LLM_MODEL"):
            errors.append("missing model identity")
        game = launch.get("game_id", "")
        if game and any(
            game in call.get("arguments", "")
            for record in usage
            for call in record.get("tool_calls", [])
        ):
            errors.append("game id appeared in emitted code")
    summary = ancillary_metrics(run, authority, transitions, batches, integrity, usage)
    return {
        "run": str(run),
        "arm": arm,
        "status": "INVALID" if errors else "SCORED",
        "errors": sorted(set(errors)),
        "contract_sha256": launch.get("contract_sha256"),
        "platform": launch.get("platform"),
        "qualification": launch.get("qualification", False),
        "game_id": launch.get("game_id"),
        "started_at": launch.get("started_at"),
        "run_id": launch.get("run_id"),
        "session_id": launch.get("session_id"),
        "model_config": launch.get("model_config"),
        "agent_turns": len(attempts),
        "steps": max_step,
        "levels_completed": max(r["after"]["levels_completed"] for r in transitions),
        "stall_events": len(detections),
        "evaluable_events": len(p9),
        "escaped_events": sum(p9),
        "P10_failures": sum(p10),
        "P12_changes": sum(p12),
        "P9": fraction(p9),
        "P9b": len(detections) * 100 / max_step,
        "P10": fraction(p10),
        "P10_denominator": len(p10),
        "P11_zero_latency": all(p11) if p11 else None,
        "P12": fraction(p12),
        "P12_denominator": len(p12),
        "holds": holds,
        "events": details,
        **summary,
        "model_calls": len(usage),
        "model_wall_seconds": sum(r.get("wall_seconds", 0) for r in usage),
        "spend": sum(r.get("cost", 0) for r in usage)
        if usage and all("cost" in r for r in usage)
        else None,
    }


def ancillary_metrics(run, authority, transitions, batches, integrity, usage):
    from collections import Counter

    reviews = [r for r in authority if r["event"] == "AUTHORITY_REVIEWED"]
    resolved = [r for r in authority if r["event"] == "AUTHORITY_RESOLVED"]
    outcomes = transition_outcomes(transitions)
    by_step = {r["step"]: (r, outcome) for r, outcome in zip(transitions, outcomes)}
    latencies = []
    for review in reviews:
        item = by_step.get(review["step"] + 1)
        if (
            item is None
            or falsifier_satisfied(review.get("abandon_if"), *item) is not True
        ):
            continue
        changed = [
            r["step"]
            for r in resolved
            if r["step"] >= review["step"] + 1
            and r.get("active_basis_after") != review["active_basis"]
        ]
        latencies.append(
            {
                "stall_id": review["stall_id"],
                "actions": min(changed) - review["step"] - 1 if changed else None,
            }
        )
    launch = json.loads((run / "launch.json").read_text())
    stop = (
        json.loads((run / "stop.json").read_text())
        if (run / "stop.json").exists()
        else {}
    )
    wall = (
        stop.get("ended_at", 0) - launch["started_at"]
        if launch.get("started_at") and stop.get("ended_at")
        else None
    )
    return {
        "dispositions": dict(Counter(r["disposition"] for r in reviews)),
        "authority_change_latency": latencies,
        "force_advances": sum(bool(r.get("forced")) for r in transitions),
        "truncated_batches": sum("truncated_from" in b for b in batches),
        "truncated_text_count": sum(
            str(r.get("response_content", "")).count("<truncated>") for r in usage
        ),
        "wall_seconds": wall,
        "rhae": json.loads((run / "result.json").read_text()).get("rhae_game_score")
        if (run / "result.json").exists()
        else None,
        "integrity_coverage": dict(
            Counter(r.get("coverage") for r in integrity if r.get("coverage"))
        ),
    }


def score_game(run: Path) -> dict:
    """A Kaggle game may contain multiple workers; retain each episode's denominator."""
    paths = (
        [run]
        if (run / "ipc").is_dir()
        else [p.parent for p in run.rglob("launch.json")]
    )
    if not paths:
        return {
            "run": str(run),
            "status": "INVALID",
            "errors": ["no worker sessions found"],
        }
    sessions = [score_run(path) for path in paths]
    if any(s["status"] != "SCORED" for s in sessions):
        return {
            "run": str(run),
            "status": "INVALID",
            "errors": ["invalid worker session"],
            "sessions": sessions,
        }
    keys = {
        (
            s.get("run_id"),
            s["arm"],
            s["contract_sha256"],
            s["platform"],
            s["game_id"],
            json.dumps(s["model_config"], sort_keys=True),
        )
        for s in sessions
    }
    if len(keys) != 1:
        return {
            "run": str(run),
            "status": "INVALID",
            "errors": [
                "sessions differ in run, game, arm, contract, platform or model config"
            ],
        }
    result = dict(sessions[0])
    result["run"] = str(run)
    result["session_count"] = len(sessions)
    for key in (
        "steps",
        "stall_events",
        "evaluable_events",
        "escaped_events",
        "P10_denominator",
        "P10_failures",
        "P12_denominator",
        "P12_changes",
        "agent_turns",
        "model_calls",
        "model_wall_seconds",
    ):
        result[key] = sum(s[key] for s in sessions)
    result["P9"] = (
        result["escaped_events"] / result["evaluable_events"]
        if result["evaluable_events"]
        else None
    )
    result["P9b"] = result["stall_events"] * 100 / result["steps"]
    result["P10"] = (
        result["P10_failures"] / result["P10_denominator"]
        if result["P10_denominator"]
        else None
    )
    result["P12"] = (
        result["P12_changes"] / result["P12_denominator"]
        if result["P12_denominator"]
        else None
    )
    result["P11_zero_latency"] = all(
        s["P11_zero_latency"] is not False for s in sessions
    )
    result["levels_completed"] = max(s["levels_completed"] for s in sessions)
    counts = {
        key: sum(s["holds"]["counts"][key] for s in sessions)
        for key in sessions[0]["holds"]["counts"]
    }
    result["holds"] = holds_from_counts(counts)
    result["sessions"] = sessions
    if not result["qualification"] and result["agent_turns"] < 15:
        result["status"] = "INVALID"
        result["errors"] = ["fewer than 15 agent turns per game"]
    for key in ("wall_seconds", "spend"):
        result[key] = (
            sum(s[key] for s in sessions)
            if all(s[key] is not None for s in sessions)
            else None
        )
    from collections import Counter

    for key in ("dispositions", "integrity_coverage"):
        total = Counter()
        for session in sessions:
            total.update(session[key])
        result[key] = dict(total)
    for key in ("force_advances", "truncated_batches", "truncated_text_count"):
        result[key] = sum(session[key] for session in sessions)
    result["authority_change_latency"] = [
        {"session": session["run"], **row}
        for session in sessions
        for row in session["authority_change_latency"]
    ]
    result["events"] = [
        {"session": s["run"], **event} for s in sessions for event in s["events"]
    ]
    return result


def holds_from_counts(counts):
    def ratio(n, d):
        return counts[n] / counts[d] if counts[d] else None

    p1, p2, p3, p4 = (
        ratio("long_ok", "long_batches"),
        ratio("model_notes", "completed"),
        ratio("reentries", "attempted"),
        ratio("waste", "executed"),
    )
    status = (
        "FAIL"
        if (p1 is not None and p1 < 0.5)
        or (p2 is not None and p2 < 1)
        or (p3 is not None and p3 > 0.05)
        or (p4 is not None and p4 > 0.15)
        else "INCONCLUSIVE"
        if p1 is None or p2 is None
        else "PASS"
    )
    return {"P1": p1, "P2": p2, "P3": p3, "P4": p4, "status": status, "counts": counts}


def control_loop_metrics(
    run: Path, batches: list[dict], transitions: list[dict]
) -> dict:
    """P1–P4 use q4's full-run HUD and executed-action denominators.

    Use frames directly; IPC outcome/score labels cannot award credit.
    """
    grids = [transitions[0]["before"]["grid"]] + [
        r["after"]["grid"] for r in transitions
    ]
    n = len(transitions)
    hud = [
        i
        for i in range(len(grids[0]))
        if sum(a[i] != b[i] for a, b in zip(grids, grids[1:])) / n >= 0.9
    ]
    labels, outcomes = [], []
    for row in transitions:
        a, b = row["before"], row["after"]
        changed = [i for i, (x, y) in enumerate(zip(a["grid"], b["grid"])) if x != y]
        label = (
            "blocked"
            if not changed
            else ("hud_only" if set(changed) <= set(hud) else "playfield")
        )
        labels.append(label)
        outcomes.append(
            ObservationRecord().outcome(
                label, b["state"], b["levels_completed"] > a["levels_completed"]
            )
        )
    passed = []
    known, reentries = {}, 0
    for row, outcome in zip(transitions, outcomes):
        key = (
            tuple(
                tuple(line)
                for i, line in enumerate(row["before"]["grid"])
                if i not in hud
            ),
            row["action"],
        )
        reentries += key in known
        if outcome in ("no_impact", "terminal"):
            known[key] = outcome
    for batch in batches:
        if len(batch["actions"]) <= 1:
            continue
        actual = [
            outcome
            for row, outcome in zip(transitions, outcomes)
            if row["turn"] == batch["turn"] and not row.get("automatic")
        ]
        path = batch.get("predicted_path")
        ok = False
        if isinstance(path, list) and len(path) == len(batch["actions"]) and actual:
            checked = min(len(actual), len(path))
            ok = (
                sum(
                    isinstance(path[i], dict) and path[i].get("expect") == actual[i]
                    for i in range(checked)
                )
                / checked
                >= 0.6
            )
        passed.append(ok)
    levels = max(row["after"]["levels_completed"] for row in transitions)
    notes = run / "team_nemo/shared/knowledge"
    consolidated = []
    for level in range(levels):
        path = notes / f"level_{level}_insights.md"
        consolidated.append(
            path.exists()
            and bool(path.read_text().strip())
            and "Deterministic fallback note:" not in path.read_text()
        )
    rejections = rows(run / "ipc/rejections.jsonl")
    rejected_total = sum(len(r.get("actions", [])) for r in rejections)
    rejected_reentry = sum(
        len(r.get("actions", [])) for r in rejections if r.get("reason") == "reentry"
    )
    return holds_from_counts(
        {
            "long_ok": sum(passed),
            "long_batches": len(passed),
            "model_notes": sum(consolidated),
            "completed": len(consolidated),
            "reentries": reentries + rejected_reentry,
            "attempted": n + rejected_total,
            "waste": sum(label in ("blocked", "hud_only") for label in labels),
            "executed": n,
        }
    )


def score_arm(runs: list[dict]) -> dict:
    if any(r["status"] != "SCORED" for r in runs):
        return {"status": "INVALID", "reason": "one or more runs invalid"}
    identities = [
        str(Path(session["run"]).resolve())
        for run in runs
        for session in run.get("sessions", [run])
    ]
    if len(set(identities)) != len(identities):
        return {"status": "INVALID", "reason": "runs reuse or overlap worker sessions"}
    if {r.get("game_id") for r in runs} != {"ls20"}:
        return {"status": "INVALID", "reason": "all experimental runs must be ls20"}
    if any(r["qualification"] for r in runs):
        return {
            "status": "INCONCLUSIVE",
            "reason": "scripted qualification is not a research arm",
        }
    if any(r.get("platform") == "kaggle" for r in runs):
        run_ids = [r.get("run_id") for r in runs]
        session_ids = [
            s.get("session_id") for r in runs for s in r.get("sessions", [r])
        ]
        if (
            not all(run_ids)
            or len(set(run_ids)) != len(run_ids)
            or not all(session_ids)
            or len(set(session_ids)) != len(session_ids)
        ):
            return {
                "status": "INVALID",
                "reason": "missing or reused run/session identity",
            }
    groups = {
        arm: [r for r in runs if r["arm"] == arm] for arm in ("control", "treatment")
    }
    if any(len(group) != 3 for group in groups.values()):
        return {
            "status": "INCONCLUSIVE",
            "reason": "requires exactly three runs per arm",
        }
    if (
        not runs[0]["contract_sha256"]
        or len({r["contract_sha256"] for r in runs}) != 1
        or len({r["platform"] for r in runs}) != 1
        or len({json.dumps(r.get("model_config"), sort_keys=True) for r in runs}) != 1
    ):
        return {
            "status": "INVALID",
            "reason": "arms do not share a pinned contract/platform",
        }
    if any(r["evaluable_events"] < 3 for r in runs):
        return {
            "status": "INCONCLUSIVE",
            "reason": "fewer than three evaluable stall events in a run",
        }
    separation = min(r["P9"] for r in groups["treatment"]) > max(
        r["P9"] for r in groups["control"]
    )
    if any(r["P11_zero_latency"] is False for r in runs):
        return {"status": "INVALID", "reason": "nonzero review latency"}
    anti_ritual = all(r["P10"] is None or r["P10"] <= 0.5 for r in groups["treatment"])
    falsifiers = all(r["P12"] is None or r["P12"] == 1 for r in groups["treatment"])
    holds = all(r["holds"]["status"] == "PASS" for r in runs)
    return {
        "status": "PASS"
        if separation and anti_ritual and falsifiers and holds
        else "FAIL",
        "P9_complete_separation": separation,
        "P10_holds": anti_ritual,
        "P12_holds": falsifiers,
        "P1_P4_hold": holds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    scores = [score_game(run) for run in args.runs]
    report = {"runs": scores, "arm": score_arm(scores)}
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0 if all(r["status"] == "SCORED" for r in scores) else 1


if __name__ == "__main__":
    raise SystemExit(main())
