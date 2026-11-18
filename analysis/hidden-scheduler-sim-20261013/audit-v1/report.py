"""Render a reproducible decision capsule from the completed audit artifacts."""
import json

import heldout as h


def main():
    findings = h.HERE / "FINDINGS.md"
    decision_file = h.HERE / "DECISION.json"
    assert not findings.exists() and not decision_file.exists()
    result = json.loads((h.HERE / "RESULT.json").read_text())
    calibration = json.loads((h.HERE / "CALIBRATION.json").read_text())
    verification = json.loads((h.HERE / "VERIFICATION.json").read_text())
    primary = {c["scenario"]: c["summary"] for c in result["comparisons"]
               if c["cohort"] == "c14_only" and c["fold"] == "mixed"}
    secondary = {c["scenario"]: c["summary"] for c in result["comparisons"]
                 if c["cohort"] == "all_runs_train_only_scaled" and c["fold"] == "mixed"}
    assert primary["base"]["relative_mean_gain"] >= 0.10
    assert primary["empirical_only"]["relative_mean_gain"] < 0
    assert primary["base"]["score_gain_without_extrapolation_dependency"] < 0
    decision = {
        "status": "FAIL_ROBUSTNESS_DO_NOT_ADVANCE_THIS_POLICY",
        "scope": "This implementation of the parent HazardIndex, not all allocation policies",
        "public_calibration_pass": calibration["passes_diagnostic_gate"],
        "heldout_base_gain": primary["base"]["relative_mean_gain"],
        "heldout_empirical_only_gain": primary["empirical_only"]["relative_mean_gain"],
        "heldout_no_extrapolated_tail_gain": primary["tail_zero"]["relative_mean_gain"],
        "score_gain_without_extrapolation_dependency": primary["base"]["score_gain_without_extrapolation_dependency"],
        "observed": "Positive base-world gain survives holding out game identities, but net gain is entirely carried by paths requiring extrapolated completions; pessimistic worlds reverse it.",
        "interpretation": "The current index reallocates time away from recorded productive paths toward presumed late/deeper solves. Its tail model is the decisive uncertainty.",
        "next": "Measure out-of-sample late-completion calibration with pseudo-censored historical runs before revising the allocator or spending GPU quota on it.",
        "authority_used": "CPU test only; no GPU/model call, training, queue change, submission or promotion",
        "paired_simulations": verification["paired_runs_checked"],
    }
    decision_file.write_text(json.dumps(decision, indent=2) + "\n")
    lines = [
        "# Held-out-game scheduler test, 2026-10-13",
        "",
        "The current hazard-index scheduler fails the robustness test. Its base-world gain survives holding out game identities, but relies on extrapolated completions. Keep the shipped allocator; this result does not rule out other allocation policies.",
        "",
        "## Observation",
        "",
        "Primary results use the three c14 cohorts. The pooled sensitivity uses all ten cohorts with the 10-to-14-stream cost correction estimated only on training games. Every comparison has 110 simulated games, 14 slots, a shared 31,840-second deadline, and 53.2 effective tokens/second per slot. Initial queue waiting counts against the deadline.",
        "",
        "| Evaluation world | Shipped score | Held-out index score | Relative gain | Pooled-cohort gain |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {"base": "Fitted constant late hazard", "tail_half": "Late hazard halved",
              "tail_zero": "No extrapolated tail completions", "harder_x2": "Twice the token costs",
              "empirical_only": "Recorded trajectories only"}
    for scenario in h.SCENARIOS:
        p = primary[scenario]
        lines.append(f"| {labels[scenario]} | {p['shipped_score']:.2f} | {p['hazard_score']:.2f} | {p['relative_mean_gain']:+.1%} | {secondary[scenario]['relative_mean_gain']:+.1%} |")
    lines += [
        "",
        "These scores describe resampled public-game worlds, not predictions of the actual hidden score. The base model, half-hazard world, no-tail world and empirical-only world are different possible futures of censored levels. Empirical-only stops permanently at the selected recorded trajectory's first unresolved level; it is a pessimistic sensitivity, not proof those levels cannot be solved.",
        "",
        "## Why the apparent win is fragile",
        "",
    ]
    p = primary["base"]
    a, b = p["shipped"], p["hazard"]
    delta = p["hazard_score"] - p["shipped_score"]
    dependent_delta = b["depends_on_extrapolation_score"] - a["depends_on_extrapolation_score"]
    lines += [
        f"The base-world net gain is {delta:.2f} score points. Score carried by paths requiring at least one extrapolated completion rises from {a['depends_on_extrapolation_score']:.2f} to {b['depends_on_extrapolation_score']:.2f}, a {dependent_delta:.2f}-point increase. The rest of the score falls by {-p['score_gain_without_extrapolation_dependency']:.2f} points. This is provenance accounting within the simulated outcomes, not a causal estimate of deleting those levels.",
        "",
        f"Score from directly recorded trajectory levels falls from {a['provenance_score']['observed_trajectory']:.2f} to {b['provenance_score']['observed_trajectory']:.2f}. The index leaves an average {b['starved_50k']:.1f} unfinished games below 50K total tokens, versus {a['starved_50k']:.1f} for the shipped allocator. With no extrapolated tail, its paired win rate is {primary['tail_zero']['paired_win_rate']:.0%} over 100 mixed-set bootstraps. That supports the interpretation that it pursues assumed late rewards at the expense of recorded productive paths.",
        "",
        "All five primary folds gain in the base world; holding out game identities does not remove the effect. Sensitivity is heterogeneous. Halving late hazard loses in two folds. Doubling costs loses in three folds; the mixed primary mean is slightly positive, while the pooled mean is slightly negative. Thus neither a simple in-sample-fit explanation nor a stable improvement across harder games is supported.",
        "",
        "| Fold | Held-out identities | Base gain | Half-hazard gain | Double-cost gain |",
        "|---|---|---:|---:|---:|",
    ]
    for fold, games in enumerate(result["manifest"]["folds"]):
        def gain(sc):
            return next(c["summary"]["relative_mean_gain"] for c in result["comparisons"]
                        if c["cohort"] == "c14_only" and c["scenario"] == sc and c["fold"] == fold)
        lines.append(f"| {fold} | {', '.join(games)} | {gain('base'):+.1%} | {gain('tail_half'):+.1%} | {gain('harder_x2'):+.1%} |")
    lines += [
        "",
        "## Measurement and validation",
        "",
        "Five deterministic, outcome-independent folds each contain five identities. Policy hazard, score-per-level estimates, late hazard, and handover duration distributions use only the other 20 identities. Evaluation-world trajectories come only from the five held-out identities; no evaluation outcome fits the deployed priority. The mixed-set test resamples from all 25 identities, but dispatches each instance to its own out-of-fold policy. This retains game diversity while preserving identity holdout. Isolated-fold results use only that fold's five identities resampled to 110 instances; they are heterogeneity checks, not five independent hidden populations.",
        "",
        "Two cohorts × five world assumptions × six evaluation groups produce 3,500 paired runs, or 7,000 individual simulations. Each mixed group uses 100 bootstraps; each isolated fold uses 50. Quantiles in RESULT.json are descriptive bootstrap outcome quantiles, not confidence intervals over the unseen hidden-game distribution.",
        "",
        f"Every paired run consumed {result['manifest']['equal_token_capacity']:,.1f} tokens per arm. Maximum paired floating-point difference was {verification['max_paired_token_difference']:.3g} tokens. A per-instance hold-duration tape is identical between policies regardless of dispatch order. Fixed slot rates remove the parent's startup-rate bonus, where early holds retained higher rates after other slots filled.",
        "",
        "The common clock is benchmark ActionRecord.wallclock_seconds since Game.start_game. The older transcript curve starts at the first agent turn and omits initial waiting. In c14_E2 the benchmark counts 21/44 completions at minutes 15/30, versus 40/55 on transcript-local clocks. FT09 starts at 09:58:00.713257 but its first transcript turn is 10:10:59. Transcript turn headers also provide only approximate completion timing, especially for a final turn with no following header.",
        "",
        f"Leave-one-c14-run-out public calibration, 50 realizations per run, gives mean absolute relative errors of {calibration['score_relative_mae']:.1%} for score, {calibration['level_relative_mae']:.1%} for final level count, and {calibration['curve_relative_mae']:.1%} for the 15/30/60/90/121-minute curve. It passes the stated 10% aggregate diagnostic gate. This is aggregate calibration; B1_c14's early 15/30-minute predictions still err by about 20%/27%, so it is not exact trajectory reconstruction.",
        "",
        f"Eight behavior tests passed. The baseline priority port matched {verification['priority_cases_matched']:,} cases against preserved runtime source. The scorer reconciled {verification['saved_game_scores_matched']} saved game scores, and completion segmentation matched all {verification['level_completion_event_boundaries_matched']} primary-cohort event boundaries. All protected parent input/result hashes remained unchanged.",
        "",
        "## Remaining limits and next action",
        "",
        "The 25 public identities are the entire observed support; resampling them cannot establish transfer to different hidden games. The three primary cohorts include a B1 variant, and the pooled cohort mixes other harness changes. Fixed effective slot speed abstracts tool/prefill gaps and cache interactions. Handover lengths are resampled independently of the chosen level. The inherited baseline action trajectory is interpolated from realized token/action costs; it is not a replay of every observed request. The latent-world model extrapolates pooled late hazard and action efficiency, and some deeper empirical costs are borrowed from other held-out game identities. Provenance tags distinguish these sources. Calibration of aggregate public outcomes does not validate any of these counterfactual tails.",
        "",
        "Do not advance this HazardIndex to a GPU smoke or hidden draw on these results. Next, pseudo-censor held-out historical runs at a fixed token threshold and score out-of-sample late-completion forecasts by level and game difficulty. If it cannot predict observed continuations, replace the constant-tail assumption before revising the priority. If it can, the genuinely unobserved tail still requires a bounded continuation experiment. Allocation remains an open direction; this particular policy is unqualified.",
        "",
        "## Artifacts and reproduction",
        "",
        "- MANIFEST.json records folds, assumptions, rates, decision rule, and source hashes before the full run.",
        "- RESULT.json contains all 60 comparisons and per-fold fit metadata; runs/*.json retain every paired observation and score provenance.",
        "- CALIBRATION.json and VERIFICATION.json contain independent checks and clock receipts.",
        "- DECISION.json records the scope of the failed robustness result.",
        "- tests.log, calibration.log, verification.log and run.log preserve execution evidence.",
        "",
        "From this directory: `python3 -B -m unittest -v test_heldout.py`. Run scripts refuse to overwrite completed outputs; use a fresh copy/destination for a reproduction. Main command was `python3 -B heldout.py --reps 50 --mixed-reps 100 --workers 8`. No model inference, training, GPU run, queue write, submission, solver change or promotion occurred.",
        "",
        "Shared record: board thread 682, advance-research CPU test authorized by Pranav's 'do the test'.",
    ]
    findings.write_text("\n".join(lines) + "\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
