"""Prepare and execute one live learning_loop treatment against the recorded control.

The comparison this exists to make is single-variable. Iteration 0004 established
that uniformly naming the six input events reopens action choice — 44/44 action-6
fixation became a spread across 1/2/3/4 — and that reopened choice still completed
zero levels. The open question is whether a model-owned learning loop converts
choice into progress.

So the treatment arm is the recorded corrected control, derived from the same
``fixed_args()`` object that ran it, plus exactly one capability. Budgets, seeds,
model, provider pin, and action ceiling are not restated here; restating them is
how a matched pair silently stops matching.

``max_decision_repairs`` is deliberately left at the control's value. The
learning_loop contract asks for a six-field commitment every action, and a model
that cannot produce it will exhaust the repair budget and stop. That is a result
about the capability, not a defect to tune away — raising the cap here would buy a
completed run by breaking the match that makes it mean anything.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from itertools import pairwise
import json
import os
from pathlib import Path
from typing import Any

from p9 import freeze
from p9.arms import participating_sources, treatment_spec
from p9.run_arm import run_one, write_json
from uniform_action_live import (
    ENDPOINTS_URL,
    PROVIDER_TAG,
    corrected_live_spec,
    fetch_modelrun_endpoint,
    fixed_args,
)


REPO = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
ITERATION_ROOT = EXPERIMENT_ROOT / "iterations" / "0005-learning-loop-treatment"
ARTIFACTS_ROOT = ITERATION_ROOT / "artifacts"
PLAN_PATH = ARTIFACTS_ROOT / "live-treatment-cd82-v1-plan.json"
RUN_DIR = ARTIFACTS_ROOT / "live-treatment-cd82-v1"

# The control half of the pair. It already ran; this run is measured against it, so
# its summary is pinned by digest rather than re-derived.
BASELINE_SUMMARY = (
    EXPERIMENT_ROOT
    / "iterations"
    / "0004-uniform-action-interface-control"
    / "artifacts"
    / "live-control-cd82-v1"
    / "uniform-action-summary.json"
)

TREATMENT_ARM_ID = "0006-learning-loop-v1"
CAPABILITY = "learning_loop"
CONFIRM_EXECUTION = "EXECUTE LEARNING LOOP TREATMENT CD82 V1"
APPROVAL_QUOTE = "write the treatment runner"


def treatment_live_spec():
    """The recorded corrected control, plus one capability and a new id."""

    control = corrected_live_spec()
    if control.treatment_capabilities:
        raise RuntimeError("the corrected control acquired a capability; pair is void")
    return treatment_spec(
        control,
        arm_id=TREATMENT_ARM_ID,
        treatment_capabilities=(CAPABILITY,),
    )


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _source_commitments() -> dict[str, str]:
    committed = dict(participating_sources())
    for path in (
        EXPERIMENT_ROOT / "uniform_action_control.py",
        EXPERIMENT_ROOT / "uniform_action_live.py",
        EXPERIMENT_ROOT / "p9" / "learning_loop.py",
        EXPERIMENT_ROOT / "p9" / "learning_loop_agent.py",
        Path(__file__).resolve(),
    ):
        committed[path.relative_to(REPO).as_posix()] = _sha256(path)
    return dict(sorted(committed.items()))


def require_unmoved_control() -> dict[str, Any]:
    """A treatment measured against a drifted control measures nothing.

    The freeze is checked here and again before the paid call. Between preparation
    and execution is exactly where an edit lands.
    """

    report = freeze.verify()
    if not report["pinned"]:
        raise RuntimeError("0002-R is unpinned; run p9.freeze")
    if report["drift"]:
        raise RuntimeError(
            "0002-R moved; the pair would compare against a different control:\n"
            + json.dumps(report["drift"], indent=2)
        )
    return {
        "runtime_digest": report["runtime_digest"],
        "control_manifest_digest": report["control_manifest_digest"],
    }


def build_plan(endpoint: dict[str, Any]) -> dict[str, Any]:
    args = fixed_args()
    spec = treatment_live_spec()
    control = corrected_live_spec()
    return {
        "schema": "0005-learning-loop-live-plan-v1",
        "stage": "one learning_loop treatment live run",
        "question": (
            "With action choice already reopened by uniform input naming, does a "
            "model-owned learning loop convert choice into level progress?"
        ),
        "anti_actions": [
            "no control rerun",
            "no second treatment",
            "no automatic successor run",
            "no re-pin of 0002-R",
            "no repair-budget or ceiling change to force completion",
        ],
        "approval": {
            "authority": "user",
            "quote": APPROVAL_QUOTE,
            "confirmation_required": CONFIRM_EXECUTION,
        },
        "arm_id": spec.arm_id,
        "treatment_capabilities": list(spec.treatment_capabilities),
        # The single variable, stated as a check rather than a claim. Everything a
        # reader would otherwise have to take on trust about the match is here.
        "matched_against_control": {
            "control_arm_id": control.arm_id,
            "baseline_summary": {
                "path": BASELINE_SUMMARY.relative_to(REPO).as_posix(),
                "sha256": _sha256(BASELINE_SUMMARY),
            },
            "shared_fields_identical": _shared_field_diff(control, spec) == {},
            "differing_fields": sorted(_shared_field_diff(control, spec)),
        },
        "frozen_control": require_unmoved_control(),
        "game": args.game,
        "game_files": require_game_files(args.game).relative_to(REPO).as_posix(),
        "model": args.model,
        "model_profile": args.model_profile,
        "endpoint_url": ENDPOINTS_URL,
        "endpoint": endpoint,
        "provider_only": [PROVIDER_TAG],
        "allow_fallbacks": False,
        "env_seed": args.env_seed,
        "model_seed": args.model_seed,
        "temperature": args.temperature,
        "max_completion_tokens": args.max_completion_tokens,
        "max_actions": args.max_actions,
        "max_context_turns": args.max_context_turns,
        "max_model_rounds_per_action": args.max_model_rounds_per_action,
        "max_decision_repairs": spec.max_decision_repairs,
        "max_model_calls": args.max_model_calls,
        "max_reported_tokens": args.max_reported_tokens,
        "workspace_timeout_seconds": args.workspace_timeout,
        "transport_max_retries": args.transport_retries,
        "transport_backoff_seconds": args.transport_backoff,
        "source_sha256": _source_commitments(),
        "run_dir": RUN_DIR.relative_to(REPO).as_posix(),
    }


def _shared_field_diff(control: Any, treatment: Any) -> dict[str, list[Any]]:
    """Every spec field the two arms must agree on, and any that do not.

    ``arm_id``, ``treatment_capabilities`` and ``system_prompt`` are the authorized
    difference: the capability's prompt addendum is appended to the control's prompt
    by the shared factory, and is hashed separately in the manifest.
    """

    authorized = {"arm_id", "treatment_capabilities", "system_prompt"}
    differing: dict[str, list[Any]] = {}
    for field_name in control.__dataclass_fields__:
        if field_name in authorized:
            continue
        left = getattr(control, field_name)
        right = getattr(treatment, field_name)
        if left != right:
            differing[field_name] = [left, right]
    return differing


def require_confirmation(value: str) -> None:
    if value != CONFIRM_EXECUTION:
        raise ValueError(f"confirmation must be exactly {CONFIRM_EXECUTION!r}")


def require_game_files(game: str) -> Path:
    """The game must be resolvable before anything reaches the network.

    ``arc_agi.Arcade`` scans ``environment_files/`` relative to the working
    directory, so running this from its own directory finds no games at all. The
    first attempt failed that way — harmlessly, before any model call, but only
    after ``Arcade`` had already opened a scorecard against the service. Checking
    the directory here is a filesystem read and reaches nothing.
    """

    game_dir = REPO / "environment_files" / game
    if not game_dir.is_dir():
        raise RuntimeError(
            f"no environment files for {game} at {game_dir}; "
            "the game cannot be created and the run would abort after opening a scorecard"
        )
    return game_dir


def prepare() -> dict[str, Any]:
    if RUN_DIR.exists():
        raise RuntimeError(f"run directory already exists: {RUN_DIR}")
    plan = build_plan(fetch_modelrun_endpoint())
    if not plan["matched_against_control"]["shared_fields_identical"]:
        raise RuntimeError(
            "treatment and control differ outside the authorized fields: "
            + json.dumps(plan["matched_against_control"]["differing_fields"])
        )
    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(PLAN_PATH, plan)
    return plan


def _load_and_revalidate_plan() -> dict[str, Any]:
    if not PLAN_PATH.exists():
        raise RuntimeError(f"prepare first: {PLAN_PATH}")
    frozen = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    current = build_plan(fetch_modelrun_endpoint())
    if current != frozen:
        raise RuntimeError("live plan drifted after preparation")
    return frozen


def _learning_loop_measures(events: list[dict[str, Any]]) -> dict[str, Any]:
    """What the capability did, read only from what the host mechanically recorded.

    Every field here is a count or an echo of a model-authored string. The host
    scored no hypothesis and ranked no objective; a reader deciding whether the loop
    worked is reading the model's own commitments and their mechanical outcomes.
    """

    results = [event for event in events if event.get("type") == "prediction_result"]
    decisions = [event for event in events if event.get("type") == "decision"]
    commitments = [
        event["reasoning"]["commitment"]
        for event in decisions
        if isinstance(event.get("reasoning"), dict)
        and isinstance(event["reasoning"].get("commitment"), dict)
    ]

    objective_results = [
        result["objective_result"]
        for result in results
        if isinstance(result.get("objective_result"), dict)
    ]
    objectives = [
        commitment["objective"]
        for commitment in commitments
        if isinstance(commitment.get("objective"), dict)
    ]
    return {
        "commitments_recorded": len(commitments),
        "mechanism_outcomes": dict(
            sorted(Counter(str(r.get("outcome")) for r in results).items())
        ),
        "objective_outcomes": dict(
            sorted(Counter(str(r.get("outcome")) for r in objective_results).items())
        ),
        "objectives_authored": len(objectives),
        "objective_claims": [str(o.get("claim")) for o in objectives],
        "reconciliations_written": sum(
            1 for c in commitments if c.get("reconciliation") not in (None, "")
        ),
        "history_steps_requested": sum(
            len(c.get("history_steps") or []) for c in commitments
        ),
        "working_state_changed": sum(
            1
            for previous, current in pairwise(commitments)
            if previous.get("working_state") != current.get("working_state")
        ),
    }


def _audit_run(manifest: dict[str, Any]) -> dict[str, Any]:
    events = [
        json.loads(line)
        for line in (RUN_DIR / "process.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    decisions = [event for event in events if event.get("type") == "decision"]
    action_counts = Counter(str(event["action"]) for event in decisions)

    providers: set[str] = set()
    for path in sorted((RUN_DIR / "response-bodies").glob("*.raw")):
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        provider = body.get("provider")
        if provider:
            providers.add(str(provider))

    requested_routes: set[str] = set()
    for path in sorted((RUN_DIR / "request-payloads").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        provider = payload.get("provider") or {}
        for tag in provider.get("only", []):
            requested_routes.add(str(tag))

    baseline = json.loads(BASELINE_SUMMARY.read_text(encoding="utf-8"))
    summary = {
        "schema": "0005-learning-loop-live-summary-v1",
        "run_dir": RUN_DIR.relative_to(REPO).as_posix(),
        "arm_id": manifest["arm_id"],
        "treatment_capabilities": manifest["treatment_capabilities"],
        "status": manifest["status"],
        "stop_reason": manifest["stop_reason"],
        "actions": manifest["actions"],
        "action_counts": dict(sorted(action_counts.items())),
        "action_sequence": [event["action"] for event in decisions],
        "first_eight_actions": [event["action"] for event in decisions[:8]],
        "levels_completed": manifest["levels_completed"],
        "model_calls": manifest["model_calls"],
        # The contract is demanding and the repair cap is the control's. A run that
        # stops here stops on the capability, and this is the field that says so.
        "decision_repairs": manifest["decision_repairs"],
        "transport_attempts": manifest["transport_attempts"],
        "transport_retries": manifest["transport_retries"],
        "usage": manifest["usage"],
        "learning_loop": _learning_loop_measures(events),
        "control_comparison": {
            "baseline_sha256": _sha256(BASELINE_SUMMARY),
            "control_actions": baseline["actions"],
            "control_action_counts": baseline["action_counts"],
            "control_levels_completed": baseline["levels_completed"],
            "control_model_calls": baseline["model_calls"],
            "treatment_levels_completed": manifest["levels_completed"],
            "levels_delta": manifest["levels_completed"]
            - baseline["levels_completed"],
        },
        "response_providers": sorted(providers),
        "requested_provider_only": sorted(requested_routes),
        "plan_sha256": _sha256(PLAN_PATH),
        "process_sha256": _sha256(RUN_DIR / "process.jsonl"),
        "trace_sha256": _sha256(RUN_DIR / "trace.jsonl"),
    }
    write_json(RUN_DIR / "learning-loop-summary.json", summary)
    return summary


def execute(confirmation: str) -> dict[str, Any]:
    from dotenv import load_dotenv

    require_confirmation(confirmation)
    _load_and_revalidate_plan()
    require_unmoved_control()
    require_game_files(fixed_args().game)
    if RUN_DIR.exists():
        raise RuntimeError(f"run directory already exists: {RUN_DIR}")
    load_dotenv(REPO / ".env")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not present in the process environment")

    os.environ["OPENROUTER_PROVIDER_ONLY"] = PROVIDER_TAG
    # Resolved against the repository, not the caller's directory, so the run does
    # not depend on where the command was typed. See require_game_files.
    os.chdir(REPO)
    manifest = run_one(treatment_live_spec(), RUN_DIR, fixed_args())
    summary = _audit_run(manifest)
    if summary["requested_provider_only"] != [PROVIDER_TAG]:
        raise RuntimeError("persisted requests did not retain the provider pin")
    if summary["response_providers"] != ["ModelRun"]:
        raise RuntimeError(
            f"unexpected response providers: {summary['response_providers']}"
        )
    if summary["treatment_capabilities"] != [CAPABILITY]:
        raise RuntimeError(
            f"run recorded unexpected capabilities: {summary['treatment_capabilities']}"
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    execute_parser = sub.add_parser("execute")
    execute_parser.add_argument("--confirm", required=True)
    args = parser.parse_args(argv)
    result = prepare() if args.command == "prepare" else execute(args.confirm)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
