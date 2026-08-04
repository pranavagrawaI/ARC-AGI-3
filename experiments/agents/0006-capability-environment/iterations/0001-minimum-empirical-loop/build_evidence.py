"""Build the mechanical comparison index for iteration 0001.

The mechanical report incorporates a separate human-authored verdict when present.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


ITERATION_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = ITERATION_ROOT.parents[1]
PROJECT_ROOT = ITERATION_ROOT.parents[4]
P9_ROOT = EXPERIMENT_ROOT / "p9"
ANCHOR_ROOT = P9_ROOT / "artifacts/runs/pair-20260729T080256Z-663e125e/control"
TREATMENT_ROOT = ITERATION_ROOT / "artifacts/live-treatment-cd82-v0"

sys.path.insert(0, str(EXPERIMENT_ROOT))

from p9.arm_diff import compare_arms  # noqa: E402
from p9.arms import (  # noqa: E402
    arm_manifest,
    build_arm,
    control_spec,
)
from p9.capabilities import authorized_manifest_paths  # noqa: E402
from p9.freeze import (  # noqa: E402
    _SilentClient,
    control_manifest_digest,
    runtime_digest,
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    anchor = read_json(ANCHOR_ROOT / "manifest.json")
    treatment = read_json(TREATMENT_ROOT / "manifest.json")
    review_path = ITERATION_ROOT / "human-review.json"
    human_verdict = (
        read_json(review_path)
        if review_path.exists()
        else {"status": "pending"}
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
    with TemporaryDirectory(prefix="p11-control-manifest-") as scratch:
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
        treatment["arm_manifest"],
        authorized_paths=authorized_manifest_paths(("learning_loop",)),
    )
    post_run_source_drift = tuple(
        item
        for item in diff.unauthorized
        if item.path.startswith("host_only.source_sha256.")
    )
    non_source_unauthorized = tuple(
        item for item in diff.unauthorized if item not in post_run_source_drift
    )
    historical_diff = diff.as_json()
    historical_diff["unauthorized"] = [
        item.as_json() for item in non_source_unauthorized
    ]
    historical_diff["post_run_source_drift"] = [
        item.as_json() for item in post_run_source_drift
    ]
    historical_diff["ok"] = not non_source_unauthorized
    events = [
        json.loads(line)
        for line in (TREATMENT_ROOT / "process.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    results = [event for event in events if event.get("type") == "prediction_result"]
    decisions = [event for event in events if event.get("type") == "decision"]
    outcomes = Counter(event.get("outcome") for event in results)

    anchor_env = anchor["environment"]
    treatment_env = treatment["environment"]
    matched_fields = (
        "endpoint",
        "env_seed",
        "game",
        "mode",
        "model",
        "model_profile",
        "model_request",
        "model_seed",
    )
    conditions = {
        key: {
            "anchor": anchor_env.get(key),
            "treatment": treatment_env.get(key),
            "matched": anchor_env.get(key) == treatment_env.get(key),
        }
        for key in matched_fields
    }
    anchor_spec = anchor["arm_manifest"]["host_only"]["spec"]
    treatment_spec = treatment["arm_manifest"]["host_only"]["spec"]
    budget_keys = (
        "color_permutation_seed",
        "max_actions",
        "max_model_calls",
        "max_reported_tokens",
    )
    conditions["arm_budgets"] = {
        "anchor": anchor_spec,
        "treatment": treatment_spec,
        "matched": all(anchor_spec.get(key) == treatment_spec.get(key) for key in budget_keys),
    }
    anchor_retry = anchor["arm_manifest"]["host_only"]["transport_retry_policy"]
    treatment_retry = treatment["arm_manifest"]["host_only"]["transport_retry_policy"]
    conditions["transport_retry_policy"] = {
        "anchor": anchor_retry,
        "treatment": treatment_retry,
        "matched": anchor_retry == treatment_retry,
    }

    report = {
        "schema": "0006-iteration-evidence-v0",
        "iteration": "0001-minimum-empirical-loop",
        "anchor_run": str(ANCHOR_ROOT.relative_to(PROJECT_ROOT)),
        "treatment_run": str(TREATMENT_ROOT.relative_to(PROJECT_ROOT)),
        "frozen_control": {
            "runtime_digest": runtime_digest(),
            "control_manifest_digest": control_manifest_digest(),
            "agent_sha256": current_control["host_only"]["source_sha256"][
                "experiments/agents/0006-capability-environment/p9/agent.py"
            ],
            "anchor_model_visible_equals_current_control": (
                anchor["arm_manifest"]["model_visible"] == current_control["model_visible"]
            ),
        },
        "matched_conditions": conditions,
        "current_control_to_treatment_manifest_diff": historical_diff,
        "quantitative": {
            "anchor": {
                key: anchor.get(key)
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
            },
            "treatment": {
                key: treatment.get(key)
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
            },
            "official_rhae_game_score": {
                "anchor": 0.0 if anchor.get("levels_completed") == 0 else None,
                "treatment": 0.0 if treatment.get("levels_completed") == 0 else None,
                "note": (
                    "Zero is established only by zero completed levels; nonzero scoring "
                    "is not computed here."
                ),
            },
        },
        "mechanical_treatment_checks": {
            "decision_count": len(decisions),
            "transition_count": sum(event.get("type") == "transition" for event in events),
            "prediction_result_count": len(results),
            "prediction_outcomes": dict(sorted(outcomes.items())),
            "reconciliations_authored": sum(
                bool(
                    (event.get("reasoning") or {})
                    .get("commitment", {})
                    .get("reconciliation")
                )
                for event in decisions
            ),
            "decisions_selecting_history": sum(
                bool(
                    (event.get("reasoning") or {})
                    .get("commitment", {})
                    .get("history_steps")
                )
                for event in decisions
            ),
        },
        "human_verdict": {
            "authority": "full trace in existing viewer",
            "mechanical_uptake_is_acceptance": False,
            "review_template": "README.md#human-review-record",
            **human_verdict,
        },
    }
    if not all(item["matched"] for item in conditions.values()):
        raise SystemExit("anchor and treatment conditions do not match")
    if non_source_unauthorized:
        raise SystemExit(f"unauthorized arm differences: {non_source_unauthorized}")
    if report["frozen_control"]["anchor_model_visible_equals_current_control"] is not True:
        raise SystemExit("recorded anchor no longer matches the current control surface")
    if len(results) != treatment.get("actions"):
        raise SystemExit("prediction results do not cover every treatment action")

    output = ITERATION_ROOT / "evidence.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "evidence": str(output.relative_to(PROJECT_ROOT)),
                "matched_conditions": True,
                "unauthorized_manifest_differences": [],
                "mechanical_treatment_checks": report["mechanical_treatment_checks"],
                "quantitative": report["quantitative"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
