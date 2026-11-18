"""Produce inspectable synthetic witnesses, without a model service or ARC runtime."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import fixtures
from construction import WorkItem, construct
from kernel import (FrozenModel, Observation, Prefix, Step, Targets, assess,
                    initialize, predict, rebuild, score_open_loop)


def observation(**values):
    return Observation.make(values, values)


def main() -> None:
    output = Path(__file__).parent / "results"
    output.mkdir(exist_ok=True)
    xs = Targets.make({"x": range(5)})
    lengths = Targets.make({"length": range(65)})
    wrong = FrozenModel(fixtures.WRONG_BUT_ASSIMILATES)
    f = predict(wrong, initialize(wrong, observation(x=1)), "right", xs)
    actual = observation(x=1)
    report = assess(f, actual, xs)
    posterior = wrong.call("observe_one", "always_move", {"x": 2}, {"x": 1})
    posterior_witness = {
        "predicted_x": 2, "observed_x": 1,
        "assimilated_x": posterior["x"],
        "incorrect_posterior_equality_would_accept": posterior["x"] == 1,
        "frozen_prediction_verdict": report.status,
        "forecast_seal": f.seal,
        "forecast": asdict(f), "assessment": asdict(report),
    }
    joint_targets = Targets.make({"x": [0, 1], "meter": range(11)})
    joint_model = FrozenModel(fixtures.CORRELATED_FIELDS)
    jf = predict(joint_model, initialize(joint_model, Observation.make({}, {})), "go", joint_targets)
    jv = assess(jf, observation(x=1, meter=8), joint_targets)
    joint_witness = {
        "permitted_joint_outcomes": [{"x": 1, "meter": 9}, {"x": 0, "meter": 8}],
        "observation": {"x": 1, "meter": 8},
        "all_marginal_membership_checks_pass": all(v["hit"] for v in json.loads(jv.fields_json).values()),
        "joint_verdict": jv.status,
    }
    alt = FrozenModel(fixtures.ALTERNATING)
    iid = FrozenModel(fixtures.INDEPENDENT)
    sequence = (Step.make("click", observation(length=1)),
                Step.make("click", observation(length=2)))
    alternating = score_open_loop(alt, initialize(alt, observation(length=0)), sequence, lengths)
    independent = score_open_loop(iid, initialize(iid, observation(length=0)), sequence, lengths)
    alternating_witness = {
        "observed_lengths": [0, 1, 2], "observed_increments": [1, 1],
        "alternating_phase_model": [r.status for r in alternating],
        "independent_growth_model": [r.status for r in independent],
        "note": "An invented example, not the S5I5 rule or trace.",
    }
    prefix = Prefix(Observation.make({"x": 0, "limit": 1}, {"x": 0}),
                    (Step.make("right", observation(x=1)), Step.make("right", observation(x=1))))
    work = WorkItem("Predict another move", "A retained move contradicts the incumbent",
                    "contrast", "Movement applicability", (0, 1, 2), ("x", "limit"),
                    ((1, ("x",)), (2, ("x",))), ("Only this synthetic boundary tested",))
    proposals = iter((fixtures.WRONG_BUT_ASSIMILATES, fixtures.BOUNDARY))
    result = construct(work, prefix, wrong, xs, lambda *args: next(proposals))
    assert result.model is not None
    # Only the evaluator below receives the suffix. The builder was scripted,
    # so this is an integration witness, not evidence of successful induction.
    suffix = tuple(Step.make("right", observation(x=1)) for _ in range(4))
    rollout = score_open_loop(result.model, rebuild(result.model, prefix, xs), suffix, xs)
    construction_witness = {
        "builder_type": "scripted fixture, no LLM", "visible_frontier": prefix.frontier,
        "status": result.status, "attempts": [asdict(a) for a in result.attempts],
        "synthetic_suffix_verdicts": [r.status for r in rollout],
    }
    results = {
        "evidence_class": "synthetic engine tests, not ARC results",
        "posterior_leakage": posterior_witness, "joint_outcomes": joint_witness,
        "temporal_correlation": alternating_witness, "construction_plumbing": construction_witness,
    }
    (output / "witnesses.json").write_text(json.dumps(results, indent=2) + "\n")
    (output / "scripted_candidate.py").write_text(result.model.source)
    summary = {
        "posterior_leakage": report.status,
        "impossible_joint_outcome": jv.status,
        "alternating_model_on_plus_one_plus_one": alternating[-1].status,
        "independent_model_on_plus_one_plus_one": independent[-1].status,
        "scripted_construction_status": result.status,
        "scripted_suffix_steps_consistent": len(rollout),
        "real_ARC_traces_evaluated": 0, "LLM_construction_calls": 0,
    }
    print(json.dumps(summary, indent=2))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
