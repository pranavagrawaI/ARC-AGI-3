"""Recompute workflow closures from host frames, ignoring recorded verdict fields.
Control loops require separate reading of the same raw requests and transitions:
absence of this intervention's convenience log is not a zero control score.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def rows(path):
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.exists()
        else []
    )


def score(root):
    result = []
    requests = rows(root / "requests.jsonl")
    responses = rows(root / "responses.jsonl")
    for run in json.loads((root / "RESULTS.json").read_text()):
        arm = run["arm"]
        loops = []
        errors = []
        for session in (root / arm / "workspaces").iterdir():
            transitions = rows(session / "ipc/transitions.jsonl")
            by_step = {r["step"]: r for r in transitions}
            events = rows(session / "ipc/experiments.jsonl")
            seen = set()
            for event in events:
                if event["event"] != "CLOSED":
                    continue
                step = event["step"] + 1
                t = by_step.get(step)
                commits = [
                    r
                    for r in events
                    if r["event"] == "COMMITTED" and r["step"] == event["step"]
                ]
                if (
                    t is None
                    or t["action"] != event["action"]
                    or step in seen
                    or len(commits) != 1
                ):
                    errors.append("missing/duplicate experiment action")
                    continue
                if any(
                    event.get(key) != commits[0].get(key)
                    for key in (
                        "step",
                        "turn",
                        "action",
                        "region",
                        "expect_change",
                        "hypothesis",
                        "alternative",
                        "rationale",
                    )
                ):
                    errors.append("closure differs from committed prediction")
                    continue
                seen.add(step)
                r0, r1, c0, c1 = event["region"]
                a, b = t["before"]["grid"], t["after"]["grid"]
                changed = any(
                    a[r][c] != b[r][c]
                    for r in range(r0, r1 + 1)
                    for c in range(c0, c1 + 1)
                )
                matched = changed == event["expect_change"]
                valid = event["conclusion"] == ("supported" if matched else "rejected")
                revised = (
                    event["next_hypothesis"].strip() != event["hypothesis"].strip()
                )
                valid = (
                    valid and (matched or revised) and bool(event["next_test"].strip())
                )
                loops.append(
                    {
                        "session": session.name,
                        "step": step,
                        "matched": matched,
                        "revised": revised,
                        "valid_closure": valid,
                        "hypothesis": event["hypothesis"],
                        "next_hypothesis": event["next_hypothesis"],
                        "next_test": event["next_test"],
                    }
                )
        selected = [r for r in responses if r["arm"] == arm]
        first_action = None
        for r in requests:
            if r["arm"] != arm:
                continue
            # Requests after a submission contain its tool feedback and raw code;
            # exact first-action call is adjudicated from responses below.
        for r in selected:
            text = json.dumps(r.get("response", {}).get("choices", []))
            if any(
                term in text for term in ("self.experiment(", "self.submit_actions(")
            ):
                first_action = r["call"]
                break
        result.append(
            {
                **run,
                "loop_verification_errors": errors,
                "verified_workflow_closures": sum(
                    loop["valid_closure"] for loop in loops
                )
                if arm.startswith("workflow")
                else None,
                "control_requires_raw_trace_adjudication": arm.startswith("control"),
                "loops": loops,
                "first_submission_attempt_call": first_action,
                "reported_cost": sum(
                    r.get("response", {}).get("usage", {}).get("cost", 0)
                    for r in selected
                ),
            }
        )
    return result


if __name__ == "__main__":
    result = score(HERE)
    (HERE / "SCORE.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
