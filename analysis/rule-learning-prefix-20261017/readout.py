"""Summarize frozen transition receipts without changing the candidate or run."""
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def summary(rows):
    good = [r for r in rows if r["status"] == "OK"]
    changed = [r for r in good if r["raw_changed"]]
    ev = sum(r["score"]["events"] for r in good)
    pred = sum(r["score"]["predictions"] for r in good)
    explained = sum(r["score"]["events_explained"] for r in good)
    right = sum(r["score"]["predictions_right"] for r in good)
    return dict(status_counts=dict(collections.Counter(r["status"] for r in rows)), scored=len(good),
                raw_changed=len(changed), raw_unchanged=len(good)-len(changed),
                exact=sum(r["score"]["exact"] for r in good),
                exact_on_raw_changed=sum(r["score"]["exact"] for r in changed),
                no_change_baseline_exact=sum(r["score"]["events"] == 0 for r in good),
                no_change_baseline_exact_on_raw_changed=sum(r["score"]["events"] == 0 for r in changed),
                predictions=pred, right_predictions=right, observed_events=ev, explained_events=explained,
                precision=right/pred if pred else None, event_recall=explained/ev if ev else None,
                nonempty_prediction_transitions=sum(r["score"]["predictions"] > 0 for r in good),
                informative_exact=sum(r["score"]["exact"] and r["score"]["predictions"] > 0 for r in good),
                raw_changed_without_encoded_events=sum(r["score"]["events"] == 0 for r in changed),
                top10_differs=sum(r["score"] != r["visible_top10_score"] for r in good),
                scoring_wall_s=sum(r.get("elapsed_s", 0) for r in rows))


def main():
    rows = [json.loads(line) for line in (HERE/"transitions.jsonl").read_text().splitlines()]
    assert len(rows) == 258 and len({(r["track"], r["local_action"]) for r in rows}) == 258
    assert all(r["status"] in {"OK", "LEVEL_BOUNDARY_NOT_SCORED", "RESET_NOT_SCORED"} for r in rows)
    sentinels = {}
    for track, action in [("tn36-b1-r1", 41), ("s5i5-b1-r1", 16), ("cn04-b1-r2", 8)]:
        row = next(r for r in rows if r["track"] == track and r["local_action"] == action)
        sentinels[track] = {k:v for k,v in row.items() if k not in {"objects_before", "objects_after", "code_before", "code_after"}}
    report = dict(scope="Six selected level trajectories; donor representation-relative event predictions, not native-state or gameplay accuracy.",
                  total=summary(rows),
                  per_track={t:summary([r for r in rows if r["track"] == t]) for t in dict.fromkeys(r["track"] for r in rows)},
                  per_game={g:summary([r for r in rows if r["game"] == g]) for g in ["cn04", "s5i5", "tn36"]},
                  sentinels=sentinels)
    (HERE/"RESULT.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in {"sentinels", "per_track"}}, indent=2))


if __name__ == "__main__":
    main()
