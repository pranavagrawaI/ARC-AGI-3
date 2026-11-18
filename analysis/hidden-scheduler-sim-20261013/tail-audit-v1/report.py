"""Write the finite-tail calibration decision, preserving previous audit outputs."""
import json

import calibrate_tail as t


def main():
    findings = t.HERE / "FINDINGS.md"
    decision_path = t.HERE / "DECISION.json"
    assert not findings.exists() and not decision_path.exists()
    results = json.loads((t.HERE / "RESULT.json").read_text())
    primary = next(e for e in results["evaluations"] if e["cohort"] == "c14_only" and e["threshold"] == 80_000)
    pooled = next(e for e in results["evaluations"] if e["cohort"] == "all_runs_train_only_scaled" and e["threshold"] == 80_000)
    bounds = json.loads((t.HERE / "SUPPORTED-RANGE.json").read_text())
    verification = json.loads((t.HERE / "VERIFICATION.json").read_text())
    s = primary["summary"]
    p = s["models"]["pooled"]
    decision = {
        "status": "FINITE_RANGE_CALIBRATION_NOT_REJECTED_INDEFINITE_TAIL_UNQUALIFIED",
        "primary_at_risk_attempts": s["n"], "primary_recorded_late_solves": s["observed_events"],
        "expected_events_over_observed_followup": p["expected_events_during_observed_exposure"],
        "observed_over_expected": p["observed_over_expected"],
        "game_cluster_ratio_interval": s["game_cluster_bootstrap"]["observed_over_expected_p2.5_p50_p97.5"],
        "supported_range_scheduler_sensitivities": {k: v["relative_mean_gain"] for k, v in bounds["comparisons"].items()},
        "interpretation": "Ordinary late solves are supported in these recorded public continuations. The proposed scheduler still relies on indefinite extrapolation beyond the observed token range; disabling only those extrapolated long solves reverses its modeled gain.",
        "next": "A bounded continuation probe of still-unsolved levels near 200K tokens, with a fixed extra token budget, is needed to test the genuinely unobserved tail. Do not promote or launch this scheduler from finite-range calibration alone.",
        "scope": "CPU only; existing evidence; no model calls, GPU launch, training, solver edit, queue change, submission or promotion",
    }
    decision_path.write_text(json.dumps(decision, indent=2) + "\n")
    lines = [
        "# Late-solve calibration, 2026-10-13",
        "",
        "The constant late solve rate is not rejected within the range covered by the logs. The scheduler remains unqualified because its simulated benefit depends on extrapolated solves beyond that range. This narrows the diagnosis from 'assumed late solves may be fictitious' to 'the indefinite continuation assumption is untested'.",
        "",
        "## Observed continuations",
        "",
        f"At the fixed 80K-token landmark, the three c14 cohorts contain {s['n']} eligible attempts across {s['games']} game identities. There are {s['observed_events']} later solves and {s['right_censored']} naturally right-censored attempts. Each game is held out of the fit, using the same five identity folds, fitting on 20 games and evaluating on five as the prior audit. Artificially hiding a test attempt's future does not change its forecast.",
        "",
        f"The fitted rate implies {p['expected_events_during_observed_exposure']:.2f} events over the actual {s['followup_tokens']:,.0f} follow-up tokens; {s['observed_events']} occurred. This is integrated-hazard calibration over observed exposure, not a forecast that guessed each attempt's eventual exposure. Actual follow-up durations enter scoring only; they do not fit that attempt's rate. Fixed-window forecasts below are made solely from training-game rates.",
        "",
        "| Extra token window after 80K | Mean forecast | Observed KM estimate | Known outcomes | Early-censored unknown outcomes |",
        "|---|---:|---:|---:|---:|",
    ]
    for horizon in t.HORIZONS:
        f = p["forecasts"][str(horizon)]
        lines.append(f"| {horizon:,} | {f['prediction_mean_all_at_risk']:.1%} | {f['km_observed_probability']:.1%} | {f['observed_known']} | {f['unknown_due_to_censoring']} |")
    lo, _, hi = s["game_cluster_bootstrap"]["observed_over_expected_p2.5_p50_p97.5"]
    q = pooled["summary"]
    qp = q["models"]["pooled"]
    lines += [
        "",
        f"The game-cluster bootstrap interval for observed/expected events is [{lo:.2f}, {hi:.2f}], including 1. This rules out neither moderate underprediction nor moderate overprediction. The pooled ten-cohort sensitivity gives {q['observed_events']} events versus {qp['expected_events_during_observed_exposure']:.2f} expected. Its first 20K window forecast underpredicts the observed estimate ({qp['forecasts']['20000']['prediction_mean_all_at_risk']:.1%} versus {qp['forecasts']['20000']['km_observed_probability']:.1%}), so aggregate event-count agreement is not exact time calibration.",
        "",
        "KM estimates assume sufficiently noninformative censoring. Scheduler-dependent censoring can violate that assumption. RESULT.json also provides raw lower/upper fractions compatible with unknown outcomes, known-only Brier scores, coverage, and every forecast. Known-only metrics are diagnostics on a selected subset and must not be treated as unconditional performance.",
        "",
        "The half-rate alternative underpredicts aggregate late events and has worse mean held-out survival log score. However, its game-cluster log-score difference interval includes 0, so these logs do not establish a statistically decisive advantage for the full rate. A level-conditioned, shrunk-rate alternative does not materially improve the primary mean log score. No fitted alternative was installed in the actor or scheduler.",
        "",
        "| Landmark | Primary survivors | Later solves | Expected events over observed exposure | Observed / expected |",
        "|---|---:|---:|---:|---:|",
    ]
    for e in results["evaluations"]:
        if e["cohort"] != "c14_only":
            continue
        es = e["summary"]
        ep = es["models"]["pooled"]
        lines.append(f"| {e['threshold']:,} | {es['n']} | {es['observed_events']} | {ep['expected_events_during_observed_exposure']:.2f} | {ep['observed_over_expected']:.3f} |")
    lines += [
        "",
        "## Connecting calibration to the scheduler claim",
        "",
        "The longest observed per-level attempt in the primary cohorts is 204,490 tokens; no genuinely observed continuation beyond that point exists in these inputs. The original latent-world extrapolation can create much longer solves. The following supplementary CPU test reproduces the original 100 mixed-set bootstraps, exact latent draws and identical hold tapes, changing only extrapolated costs. Every empirical completion is retained, including recorded completions above a chosen cap.",
        "",
        "| Extrapolated completions allowed | Shipped score | Proposed score | Relative gain |",
        "|---|---:|---:|---:|",
    ]
    labels = {"None": "No upper limit", "160000": "Only at <=160K total level tokens", "200000": "Only at <=200K total level tokens"}
    for cap, b in bounds["comparisons"].items():
        lines.append(f"| {labels[cap]} | {b['shipped_score']:.2f} | {b['hazard_score']:.2f} | {b['relative_mean_gain']:+.1%} |")
    lines += [
        "",
        "These caps are conservative sensitivity worlds, not claims that a level becomes impossible at 160K or 200K. A sampled extrapolated solve beyond the cap is replaced with an unresolved level. The candidate's beliefs are unchanged, so this tests transfer when its indefinite-tail belief is wrong. The no-limit result reproduces the prior audit exactly. All three worlds use equal token budgets, the same benchmark/deadline clock, and an unchanged out-of-fold policy.",
        "",
        f"Across all sampled latent extrapolated level costs, {bounds['all_latent_extrapolated_costs_over_200k_fraction']:.1%} exceed 200K; their 10/50/90 percentiles are {[round(v) for v in bounds['all_latent_extrapolated_cost_quantiles']]}. These are all latent costs, not just completed levels. Thus agreement on the ordinary observed tail cannot by itself validate the counterfactual benefit from very long or deeper unseen completions.",
        "",
        "## Interpretation and next action",
        "",
        "The prior zero-tail sensitivity was deliberately pessimistic. Real data show substantial late progress, so that world should not be mistaken for an empirical finding that old levels never solve. At the same time, a well-matched rate over 80K to 200K does not imply that the rate remains constant indefinitely, nor that it applies to deeper levels never reached in the logs. The proposed scheduler's gain still changes sign when the untested extension is removed.",
        "",
        "Keep the current allocator. The next discriminating experiment is a bounded continuation of still-unsolved levels near 200K tokens, with a fixed additional token budget and recorded rule/plan state. It should measure late completion yield rather than tune another priority curve against assumed future solves. This would require a separately bounded inference run; this CPU task did not launch one.",
        "",
        "## Verification and artifacts",
        "",
        f"Five behavioral tests pass, including future-outcome mutation proving held-out forecasts do not change, unknown-label handling, exact horizon boundaries, a worked censored-exposure rate, and a KM oracle. Verified identity isolation and censoring labels for {verification['identity_isolation_and_censoring_rows_checked']} forecast cases across both cohorts and all four landmarks. Previous audit artifacts and all hashed inputs remain unchanged.",
        "",
        "MANIFEST.json freezes landmarks, horizons, folds, models and decision rules before the main forecast evaluation. SUPPORTED-RANGE-MANIFEST.json separately records the subsequent range sensitivity. RESULT.json contains predictions, actual outcomes, exposure, fit metadata and stratified summaries. SUPPORTED-RANGE.json retains all 300 paired range simulations. VERIFICATION.json, tests.log, run.log and supported-range.log preserve checks and execution evidence. DECISION.json records the supported inference.",
        "",
        "Reproduction commands from this directory: `python3 -B -m unittest -v test_tail.py`, `python3 -B calibrate_tail.py`, then `python3 -B supported_range.py`. Completed-output scripts refuse overwrites; use a fresh copy/destination. CPU only, with no inference, training, GPU launch, queue write, submission, solver change or promotion.",
    ]
    findings.write_text("\n".join(lines) + "\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
