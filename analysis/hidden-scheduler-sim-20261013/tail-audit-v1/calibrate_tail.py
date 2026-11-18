"""Out-of-fold late-completion forecasts against recorded continuations.

CPU only. Evaluation identities never contribute to fitted rates. Artificial
observation stops at a landmark; only recorded follow-up scores the forecast.
Natural censoring is preserved, not relabeled as a failed solve.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT / "audit-v1"))
import heldout as h

THRESHOLDS = (60_000, 80_000, 100_000, 120_000)
HORIZONS = (20_000, 40_000, 80_000)
PRIMARY_THRESHOLD = 80_000
PSEUDO_EVENTS = 2.0
BOOTSTRAPS = 2000


def forecast_at_horizon(lam, residual, event, horizon):
    probability = -math.expm1(-lam * horizon)
    label = 1 if event and residual <= horizon else (0 if residual >= horizon else None)
    return {"p": probability, "observed_label": label}


def rate_fit(attempts, threshold, class_index=None):
    survivors = [a for a in attempts if a["tokens"] > threshold]
    exposure = sum(a["tokens"] - threshold for a in survivors)
    events = sum(a["solved"] for a in survivors)
    pooled = events / exposure if exposure else 0.0
    if class_index is None:
        return {"rate": pooled, "events": events, "exposure": exposure,
                "survivors": len(survivors), "shrinkage_pseudo_events": 0.0}
    subset = [a for a in survivors if min(a["level"], 4) == class_index]
    sub_exposure = sum(a["tokens"] - threshold for a in subset)
    sub_events = sum(a["solved"] for a in subset)
    # A frozen diagnostic alternative: two pooled-rate pseudo-events. This is
    # a forecast comparison, not a change to the submitted actor or scheduler.
    shrink_exposure = PSEUDO_EVENTS / pooled if pooled else 0.0
    smoothed = (sub_events + PSEUDO_EVENTS) / (sub_exposure + shrink_exposure) if pooled else 0.0
    return {"rate": smoothed, "events": sub_events, "exposure": sub_exposure,
            "survivors": len(subset), "shrinkage_pseudo_events": PSEUDO_EVENTS}


def forecasts(data, cohort, threshold):
    folds = h.folds_for(next(iter(data["games"].values())).keys())
    c14_runs = [r for r, m in data["runs"].items() if m["slots"] == 14]
    rows, fits = [], []
    for fold, test_games in enumerate(folds):
        train_games = set(sum(folds, [])) - set(test_games)
        if cohort == "c14_only":
            scale_value, pairs = 1.0, 0
            runs = c14_runs
        else:
            scale_value, pairs = h.training_scale(data, train_games)
            runs = list(data["runs"])
        scale = {r: scale_value for r, m in data["runs"].items() if m["slots"] == 10}
        training = h.select_data(data, train_games, runs, scale)
        evaluation = h.select_data(data, test_games, runs, scale)
        pooled = rate_fit(training["attempts"], threshold)
        levels = {k: rate_fit(training["attempts"], threshold, k) for k in range(5)}
        assert set(a["game"] for a in training["attempts"]) == train_games
        assert not set(test_games).intersection(a["game"] for a in training["attempts"])
        fits.append({"fold": fold, "training_games": sorted(train_games), "heldout_games": test_games,
                     "scale": scale_value, "scale_pairs": pairs, "pooled": pooled, "by_level": levels})
        for a in evaluation["attempts"]:
            if a["tokens"] <= threshold:
                continue
            residual = a["tokens"] - threshold
            rate = pooled["rate"]
            level_rate = levels[min(a["level"], 4)]["rate"]
            rates = {"pooled": rate, "half_pooled": 0.5 * rate, "level_conditioned": level_rate}
            event = bool(a["solved"])
            row = {"fold": fold, "run": a["run"], "game": a["game"], "level": a["level"],
                   "threshold": threshold, "residual_tokens": residual, "observed_solve": event,
                   "rates": rates, "forecast": {}, "survival_scores": {}}
            for name, lam in rates.items():
                # Loglikelihood in units of 20K tokens: same Jacobian for every
                # model, enabling interpretable comparison without tiny densities.
                dimensionless_rate = lam * 20_000
                nll = lam * residual - (math.log(dimensionless_rate) if event and dimensionless_rate > 0 else 0.0)
                if event and lam == 0:
                    nll = None
                row["survival_scores"][name] = {"expected_events": lam * residual, "nll": nll,
                                                "event_cdf": -math.expm1(-lam * residual) if event else None}
                row["forecast"][name] = {}
                for horizon in HORIZONS:
                    row["forecast"][name][str(horizon)] = forecast_at_horizon(lam, residual, event, horizon)
            rows.append(row)
    return rows, fits


def km_probability(rows, horizon):
    samples = sorted((r["residual_tokens"], r["observed_solve"]) for r in rows)
    n, survival, i = len(samples), 1.0, 0
    while i < n:
        t = samples[i][0]
        if t > horizon:
            break
        count, events = 0, 0
        while i < n and samples[i][0] == t:
            count += 1
            events += samples[i][1]
            i += 1
        risk = n - (i - count)
        if events:
            survival *= 1 - events / risk
    return 1 - survival


def grouped(rows, key):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[str(key(r))].append(r)
    return dict(groups)


def summary(rows, bootstrap=True):
    if not rows:
        return {"n": 0}
    n = len(rows)
    events = sum(r["observed_solve"] for r in rows)
    out = {"n": n, "games": len(set(r["game"] for r in rows)), "observed_events": events,
           "right_censored": n - events,
           "followup_tokens": sum(r["residual_tokens"] for r in rows),
           "max_followup_tokens": max(r["residual_tokens"] for r in rows), "models": {}}
    for name in rows[0]["rates"]:
        expected = sum(r["survival_scores"][name]["expected_events"] for r in rows)
        nlls = [r["survival_scores"][name]["nll"] for r in rows]
        forecasts_by_horizon = {}
        for horizon in HORIZONS:
            fs = [r["forecast"][name][str(horizon)] for r in rows]
            known = [f for f in fs if f["observed_label"] is not None]
            solved = sum(f["observed_label"] == 1 for f in known)
            unknown = n - len(known)
            forecasts_by_horizon[str(horizon)] = {
                "prediction_mean_all_at_risk": statistics.mean(f["p"] for f in fs),
                "observed_known": len(known), "unknown_due_to_censoring": unknown,
                "observed_positive": solved,
                "observable_fraction_bounds": [solved / n, (solved + unknown) / n],
                "km_observed_probability": km_probability(rows, horizon),
                "prediction_mean_known_only": statistics.mean(f["p"] for f in known) if known else None,
                "observed_known_only_fraction": solved / len(known) if known else None,
                "brier_known_only": statistics.mean((f["p"] - f["observed_label"]) ** 2 for f in known) if known else None,
            }
        out["models"][name] = {"expected_events_during_observed_exposure": expected,
                                "observed_over_expected": events / expected if expected else None,
                                "mean_survival_nll": statistics.mean(nlls) if all(x is not None for x in nlls) else None,
                                "infinite_nll_cases": sum(x is None for x in nlls),
                                "forecasts": forecasts_by_horizon}
    if bootstrap:
        by_game = grouped(rows, lambda r: r["game"])
        games = sorted(by_game)
        rng = random.Random("tail-audit-v1-game-bootstrap")
        ratios, nll_diff = [], []
        for _ in range(BOOTSTRAPS):
            selected = [rng.choice(games) for _ in games]
            samples = [r for g in selected for r in by_game[g]]
            observed = sum(r["observed_solve"] for r in samples)
            expected = sum(r["survival_scores"]["pooled"]["expected_events"] for r in samples)
            ratios.append(observed / expected if expected else 0.0)
            if all(r["survival_scores"][m]["nll"] is not None for r in samples for m in ("pooled", "half_pooled")):
                nll_diff.append(statistics.mean(r["survival_scores"]["half_pooled"]["nll"] -
                                               r["survival_scores"]["pooled"]["nll"] for r in samples))
        out["game_cluster_bootstrap"] = {
            "reps": BOOTSTRAPS, "observed_over_expected_p2.5_p50_p97.5": [float(np.percentile(ratios, p)) for p in (2.5, 50, 97.5)],
            "half_minus_pooled_nll_p2.5_p50_p97.5": [float(np.percentile(nll_diff, p)) for p in (2.5, 50, 97.5)] if nll_diff else None,
            "meaning": "Resamples observed game clusters with already-fitted out-of-fold rates; does not include refitting uncertainty or unseen hidden-game shift",
        }
    return out


def run_manifest():
    files = [PARENT / "attempts.json", PARENT / "audit-v1" / "heldout.py", PARENT / "audit-v1" / "MANIFEST.json",
             PARENT / "audit-v1" / "RESULT.json", PARENT / "audit-v1" / "DECISION.json"]
    return {"experiment": "tail-calibration-v1", "authority": "Pranav: try it",
            "primary_threshold_tokens": PRIMARY_THRESHOLD, "sensitivity_thresholds": THRESHOLDS,
            "forecast_horizons_tokens": HORIZONS, "primary_cohort": "c14_only",
            "secondary_cohort": "all_runs_train_only_scaled", "folds": h.folds_for(next(iter(json.loads(files[0].read_text())["games"].values())).keys()),
            "fit": "Constant hazard events/exposure beyond landmark, on other20 game identities; full training follow-up is permitted",
            "alternatives": "Half pooled hazard; level-conditioned hazard with2 pooled pseudo-events; diagnostic only, no policy rewrite",
            "evaluation": "Artificially hide future after landmark. Score naturally censored follow-up; unknown horizon outcomes remain unknown",
            "decision_rule": "Gross overprediction if observed/expected point ratio<0.5 or >2; evidence of miscalibration if game-cluster interval excludes1. No finite observed-range result licenses indefinite extrapolation or a live scheduler trial",
            "cluster_bootstrap_reps": BOOTSTRAPS, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "input_sha256": {str(p.relative_to(h.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            "scope": "CPU only; existing logs; no inference/training/GPU/solver/queue/submission changes"}


def main():
    target = HERE / "RESULT.json"
    assert not target.exists() and not (HERE / "MANIFEST.json").exists()
    manifest = run_manifest()
    (HERE / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    data = json.loads((PARENT / "attempts.json").read_text())
    results = []
    for cohort in ("c14_only", "all_runs_train_only_scaled"):
        for threshold in THRESHOLDS:
            rows, fits = forecasts(data, cohort, threshold)
            result = {"cohort": cohort, "threshold": threshold, "fit": fits, "summary": summary(rows),
                      "by_level": {k: summary(v, False) for k, v in grouped(rows, lambda r: min(r["level"], 4)).items()},
                      "by_game": {k: summary(v, False) for k, v in grouped(rows, lambda r: r["game"]).items()},
                      "by_fold": {k: summary(v, False) for k, v in grouped(rows, lambda r: r["fold"]).items()},
                      "by_run": {k: summary(v, False) for k, v in grouped(rows, lambda r: r["run"]).items()},
                      "rows": rows}
            results.append(result)
            s = result["summary"]
            a = s["models"]["pooled"]
            print(f"{cohort:27s} threshold={threshold:6d} n={s['n']:3d} solves={s['observed_events']:3d} "
                  f"expected={a['expected_events_during_observed_exposure']:.2f} obs/exp={a['observed_over_expected']:.3f} "
                  f"cluster={s['game_cluster_bootstrap']['observed_over_expected_p2.5_p50_p97.5']} "
                  f"nll(pooled,half,level)={[round(s['models'][m]['mean_survival_nll'],3) if s['models'][m]['mean_survival_nll'] is not None else None for m in s['models']]}", flush=True)
    target.write_text(json.dumps({"manifest": manifest, "evaluations": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
