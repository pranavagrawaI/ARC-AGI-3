"""Build the mechanical comparison index for iteration 0002."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any


ITERATION_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = ITERATION_ROOT.parents[1]
PROJECT_ROOT = ITERATION_ROOT.parents[4]
P9_ROOT = EXPERIMENT_ROOT / "p9"
CONTROL_ROOT = P9_ROOT / "artifacts/runs/pair-20260729T080256Z-663e125e/control"
V0_ROOT = (
    ITERATION_ROOT.parent
    / "0001-minimum-empirical-loop/artifacts/live-treatment-cd82-v0"
)
V1_FAILED_ROOT = ITERATION_ROOT / "artifacts/live-treatment-cd82-v1"
V2_ROOT = ITERATION_ROOT / "artifacts/live-treatment-cd82-v2"
HUMAN_REVIEW_PATH = ITERATION_ROOT / "human-review.json"

sys.path.insert(0, str(EXPERIMENT_ROOT))

from p9.arm_diff import compare_arms  # noqa: E402
from p9.arms import arm_manifest, build_arm, control_spec  # noqa: E402
from p9.capabilities import authorized_manifest_paths  # noqa: E402
from p9.freeze import _SilentClient, control_manifest_digest, runtime_digest  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_events(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (root / "process.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_summary(root: Path) -> dict[str, Any]:
    manifest = read_json(root / "manifest.json")
    return {
        key: manifest.get(key)
        for key in (
            "status",
            "stop_reason",
            "actions",
            "levels_completed",
            "state",
            "model_calls",
            "decision_repairs",
            "usage",
            "friction",
        )
    }


def main() -> None:
    control = read_json(CONTROL_ROOT / "manifest.json")
    v0 = read_json(V0_ROOT / "manifest.json")
    v2 = read_json(V2_ROOT / "manifest.json")
    v1_events = read_events(V1_FAILED_ROOT)
    v2_events = read_events(V2_ROOT)
    human_review = (
        read_json(HUMAN_REVIEW_PATH)
        if HUMAN_REVIEW_PATH.exists()
        else {
            "status": "pending",
            "authority": "complete trace in the existing output viewer",
            "decisive_question": (
                "After each objective falsification, did the model materially change its "
                "objective hypothesis and action strategy rather than rephrase and continue?"
            ),
        }
    )

    spec = control_spec(
        max_actions=44,
        max_context_turns=6,
        max_model_rounds_per_action=10,
        max_model_calls=400,
        max_reported_tokens=12_000_000,
        workspace_timeout_seconds=20.0,
        transport_max_retries=6,
        transport_backoff_seconds=4.0,
    )
    with TemporaryDirectory(prefix="p11-v2-control-") as scratch:
        arm = build_arm(
            spec,
            run_dir=Path(scratch) / "control",
            inner_client=_SilentClient(),
        )
        try:
            current_control = arm_manifest(arm)
        finally:
            arm.close()

    diff = compare_arms(
        current_control,
        v2["arm_manifest"],
        authorized_paths=authorized_manifest_paths(("learning_loop",)),
    )

    condition_keys = (
        "endpoint",
        "env_seed",
        "game",
        "mode",
        "model",
        "model_profile",
        "model_request",
        "model_seed",
    )
    conditions: dict[str, Any] = {}
    for key in condition_keys:
        values = {
            "control": control["environment"].get(key),
            "v0": v0["environment"].get(key),
            "v2": v2["environment"].get(key),
        }
        conditions[key] = {**values, "matched": len({json.dumps(value, sort_keys=True) for value in values.values()}) == 1}

    budget_keys = (
        "color_permutation_seed",
        "max_actions",
        "max_model_calls",
        "max_reported_tokens",
    )
    budget_values = {
        name: manifest["arm_manifest"]["host_only"]["spec"]
        for name, manifest in (("control", control), ("v0", v0), ("v2", v2))
    }
    conditions["arm_budgets"] = {
        **budget_values,
        "matched": all(
            budget_values["control"].get(key)
            == budget_values["v0"].get(key)
            == budget_values["v2"].get(key)
            for key in budget_keys
        ),
    }
    retry_values = {
        name: manifest["arm_manifest"]["host_only"]["transport_retry_policy"]
        for name, manifest in (("control", control), ("v0", v0), ("v2", v2))
    }
    conditions["transport_retry_policy"] = {
        **retry_values,
        "matched": retry_values["control"] == retry_values["v0"] == retry_values["v2"],
    }

    decisions = [event for event in v2_events if event.get("type") == "decision"]
    results = [event for event in v2_events if event.get("type") == "prediction_result"]
    mechanism_outcomes = Counter(event.get("outcome") for event in results)
    objective_outcomes = Counter(
        (event.get("objective_result") or {}).get("outcome") for event in results
    )
    falsified_objective_claims = Counter(
        (event.get("objective_result") or {}).get("claim")
        for event in results
        if (event.get("objective_result") or {}).get("outcome") == "falsified"
    )
    actions = Counter(event.get("action") for event in decisions)

    report = {
        "schema": "0006-iteration-evidence-v1",
        "iteration": "0002-bounded-objective-falsifier",
        "runs": {
            "control_anchor": str(CONTROL_ROOT.relative_to(PROJECT_ROOT)),
            "rejected_v0": str(V0_ROOT.relative_to(PROJECT_ROOT)),
            "failed_pre_run_v1": str(V1_FAILED_ROOT.relative_to(PROJECT_ROOT)),
            "treatment_v2": str(V2_ROOT.relative_to(PROJECT_ROOT)),
        },
        "frozen_control": {
            "runtime_digest": runtime_digest(),
            "control_manifest_digest": control_manifest_digest(),
            "anchor_model_visible_equals_current_control": (
                control["arm_manifest"]["model_visible"]
                == current_control["model_visible"]
            ),
        },
        "matched_conditions": conditions,
        "current_control_to_v2_manifest_diff": diff.as_json(),
        "quantitative": {
            "control_anchor": run_summary(CONTROL_ROOT),
            "rejected_v0": run_summary(V0_ROOT),
            "treatment_v2": run_summary(V2_ROOT),
            "official_rhae_game_score": {
                name: 0.0 if manifest.get("levels_completed") == 0 else None
                for name, manifest in (
                    ("control_anchor", control),
                    ("rejected_v0", v0),
                    ("treatment_v2", v2),
                )
            },
        },
        "mechanical_v2_checks": {
            "decision_count": len(decisions),
            "transition_count": sum(
                event.get("type") == "transition" for event in v2_events
            ),
            "prediction_result_count": len(results),
            "mechanism_outcomes": dict(sorted(mechanism_outcomes.items())),
            "objective_outcomes": dict(sorted(objective_outcomes.items())),
            "falsified_objective_claims": dict(
                sorted(falsified_objective_claims.items())
            ),
            "action_ids": {str(key): value for key, value in sorted(actions.items())},
            "reconciliations_authored": sum(
                bool(
                    (event.get("reasoning") or {})
                    .get("commitment", {})
                    .get("reconciliation")
                )
                for event in decisions
            ),
        },
        "failed_pre_run_v1": {
            "decision_repairs": [
                event.get("error")
                for event in v1_events
                if event.get("type") == "decision_repair"
            ],
            "failure": next(
                (
                    event.get("error")
                    for event in reversed(v1_events)
                    if event.get("type") == "run_failed"
                ),
                None,
            ),
            "root_cause": (
                "final-round validation invoked the treatment's mutating accept path twice; "
                "fixed by idempotent acceptance of the identical pending response"
            ),
        },
        "human_verdict": {
            **human_review,
            "authority": "complete trace in the existing output viewer",
            "mechanical_uptake_is_acceptance": False,
        },
    }

    if not all(item["matched"] for item in conditions.values()):
        raise SystemExit("control, v0, and v2 conditions do not match")
    if diff.unauthorized:
        raise SystemExit(f"unauthorized v2 arm differences: {diff.unauthorized}")
    if not report["frozen_control"]["anchor_model_visible_equals_current_control"]:
        raise SystemExit("recorded control anchor no longer matches current control surface")
    if len(decisions) != v2.get("actions") or len(results) != v2.get("actions"):
        raise SystemExit("v2 decision/result events do not cover every action")
    if any(event.get("objective_result") is None for event in results):
        raise SystemExit("v2 objective results do not cover every action")

    output = ITERATION_ROOT / "evidence.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "evidence": str(output.relative_to(PROJECT_ROOT)),
                "matched_conditions": True,
                "unauthorized_manifest_differences": [],
                "mechanical_v2_checks": report["mechanical_v2_checks"],
                "quantitative": report["quantitative"],
                "human_verdict": report["human_verdict"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
