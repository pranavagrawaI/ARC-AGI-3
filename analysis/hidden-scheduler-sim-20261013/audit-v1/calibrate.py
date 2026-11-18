"""Leave-one-c14-run-out public calibration, benchmark clock, fixed slot capacity."""
import hashlib
import json
import random
import statistics

import heldout as h


def main():
    target = h.HERE / "CALIBRATION.json"
    assert not target.exists(), target
    data = json.loads((h.PARENT / "attempts.json").read_text())
    runs = [r for r, m in data["runs"].items() if m["slots"] == 14]
    raw_holds = json.loads((h.PARENT / "holds.json").read_text())
    results = []
    for run in runs:
        donors = [r for r in runs if r != run]
        train = h.select_data(data, runs=donors)
        world = h.TaggedWorld(train, train, tail_scale=1.0)
        holds = h.TrainingHolds(world.games)
        holds.first = [v[0] for r in donors for v in raw_holds[r].values() if len(v) > 1]
        holds.later = [x for r in donors for v in raw_holds[r].values() for x in v[1:-1] if x > 0]
        rows = []
        for rep in range(50):
            rng = random.Random(f"public-calibration-v1:{rep}")
            insts = [world.draw(g, rng) for g in world.games]
            rows.append(h.simulate(insts, h.sim.make_dfranzen(), horizon=7254, rate=h.RATE,
                                   holds=holds, seed=f"public-calibration-holds:{rep}", record=True))
        actual = data["games"][run]
        times = sorted(a["wall_end"] for a in data["attempts"] if a["run"] == run and a["solved"])
        obs_curve = [sum(t <= m * 60 for t in times) for m in h.CHECKPOINTS]
        pred_curve = [statistics.mean(sum(t <= m * 60 for t in row["solve_times"]) for row in rows)
                      for m in h.CHECKPOINTS]
        obs_score = statistics.mean(g["score"] for g in actual.values())
        obs_levels = sum(g["done"] for g in actual.values())
        pred_score = statistics.mean(row["score"] for row in rows)
        pred_levels = statistics.mean(row["levels"] for row in rows)
        results.append({"run": run, "donor_runs": donors, "observed_score": obs_score,
                        "predicted_score": pred_score, "observed_levels": obs_levels,
                        "predicted_levels": pred_levels,
                        "score_relative_error": pred_score / obs_score - 1,
                        "level_relative_error": pred_levels / obs_levels - 1,
                        "observed_curve": obs_curve, "predicted_curve": pred_curve,
                        "curve_relative_errors": [p / o - 1 for p, o in zip(pred_curve, obs_curve)],
                        "observed_tokens": sum(g["tokens"] for g in actual.values()),
                        "predicted_tokens": statistics.mean(row["tokens"] for row in rows)})
        print(json.dumps(results[-1]), flush=True)
    score_mae = statistics.mean(abs(r["score_relative_error"]) for r in results)
    level_mae = statistics.mean(abs(r["level_relative_error"]) for r in results)
    curve_mae = statistics.mean(abs(e) for r in results for e in r["curve_relative_errors"])
    out = {"clock": "benchmark ActionRecord.wallclock_seconds since Game.start_game; queue waiting included",
           "reps": 50, "fixed_rate": h.RATE, "rows": results,
           "score_relative_mae": score_mae, "level_relative_mae": level_mae,
           "curve_relative_mae": curve_mae,
           "diagnostic_gate": "score, level and curve mean absolute relative error each <=10%; frozen before this calibration",
           "passes_diagnostic_gate": max(score_mae, level_mae, curve_mae) <= 0.10,
           "code_sha256": hashlib.sha256(h.Path(__file__).read_bytes()).hexdigest()}
    target.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
