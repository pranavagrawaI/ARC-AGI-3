"""Build compact evidence for the uniform-action corrected live control."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[5]
ITERATION_ROOT = Path(__file__).resolve().parent
NEW_RUN = ITERATION_ROOT / "artifacts" / "live-control-cd82-v1"
OLD_RUN = (
    REPO
    / "experiments"
    / "agents"
    / "0006-capability-environment"
    / "p9"
    / "artifacts"
    / "runs"
    / "pair-20260729T080256Z-663e125e"
    / "control"
)
PLAN = ITERATION_ROOT / "artifacts" / "live-control-cd82-v1-plan.json"
OUTPUT = ITERATION_ROOT / "evidence.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def action_evidence(run: Path) -> dict[str, Any]:
    events = load_jsonl(run / "process.jsonl")
    decisions = [event for event in events if event.get("type") == "decision"]
    transitions = [event for event in events if event.get("type") == "transition"]
    counts = Counter(str(event["action"]) for event in decisions)
    reasons = {
        str(event["action_step"]): event["reasoning"]["reason"]
        for event in decisions
    }
    return {
        "counts": {str(action): counts.get(str(action), 0) for action in range(1, 7)},
        "sequence": [event["action"] for event in decisions],
        "first_action": decisions[0]["action"],
        "changed_pixels": [event["changed_pixels"] for event in transitions],
        "zero_change_steps": [
            event["step"] for event in transitions if event["changed_pixels"] == 0
        ],
        "selected_reasons": {
            step: reasons[step] for step in ("1", "2", "39", "44")
        },
    }


def recursive_diff(old: Any, new: Any, path: str = "") -> list[dict[str, Any]]:
    if type(old) is not type(new):
        return [{"path": path, "old": old, "new": new}]
    if isinstance(old, dict):
        result: list[dict[str, Any]] = []
        for key in sorted(set(old) | set(new)):
            child = f"{path}.{key}" if path else key
            if key not in old or key not in new:
                result.append({"path": child, "old": old.get(key), "new": new.get(key)})
            else:
                result.extend(recursive_diff(old[key], new[key], child))
        return result
    if isinstance(old, list):
        return [] if old == new else [{"path": path, "old": old, "new": new}]
    return [] if old == new else [{"path": path, "old": old, "new": new}]


def verify_manifest_hashes(run: Path, manifest: dict[str, Any]) -> list[str]:
    failures = []
    for relative, expected in manifest["evidence_hashes"].items():
        path = run / relative
        if not path.exists() or digest(path) != expected:
            failures.append(relative)
    return failures


def provider_evidence(run: Path) -> dict[str, Any]:
    providers = set()
    for path in (run / "response-bodies").glob("*.raw"):
        providers.add(str(load_json(path).get("provider")))
    routes = set()
    for path in (run / "request-payloads").glob("*.json"):
        provider = load_json(path).get("provider") or {}
        routes.update(str(value) for value in provider.get("only", []))
    return {
        "response_providers": sorted(providers),
        "requested_provider_only": sorted(routes),
    }


def main() -> int:
    old_manifest = load_json(OLD_RUN / "manifest.json")
    new_manifest = load_json(NEW_RUN / "manifest.json")
    plan = load_json(PLAN)
    old_actions = action_evidence(OLD_RUN)
    new_actions = action_evidence(NEW_RUN)
    model_visible_diffs = recursive_diff(
        old_manifest["arm_manifest"]["model_visible"],
        new_manifest["arm_manifest"]["model_visible"],
        "model_visible",
    )
    old_sources = old_manifest["arm_manifest"]["host_only"]["source_sha256"]
    new_sources = new_manifest["arm_manifest"]["host_only"]["source_sha256"]
    changed_shared_sources = sorted(
        path
        for path in set(old_sources) & set(new_sources)
        if old_sources[path] != new_sources[path]
    )
    source_failures = [
        relative
        for relative, expected in plan["source_sha256"].items()
        if digest(REPO / relative) != expected
    ]
    evidence = {
        "schema": "0004-uniform-action-live-evidence-v1",
        "status": "complete_valid_run_pending_human_acceptance",
        "question": (
            "Does uniformly naming all six input events remove excessive action-6 "
            "affordance under the prior cd82 control conditions?"
        ),
        "answer": (
            "Yes for action-6 monopoly: 44/44 became 0/44. No for broad exploration "
            "or progress: the model used only actions 1-4, completed no levels, and "
            "deferred actions 5 and 6 despite mentioning them."
        ),
        "conditions": {
            "held_constant": [
                "game cd82",
                "environment seed 0",
                "model google/gemma-4-31b-it",
                "model seed 0",
                "44-action budget",
                "400-model-call ceiling",
                "12000000-reported-token ceiling",
                "6-turn context",
                "10 model rounds per action",
                "temperature 1.0",
                "4096 completion-token ceiling",
                "6 transport retries with 4-second backoff",
            ],
            "changed": [
                "system prompt appended uniform input-event names",
                "arm id 0002-R -> 0006-uniform-actions-v1",
                "historically unpinned routing that returned ModelRun -> modelrun/fp4 only, fallbacks disabled",
            ],
            "model_visible_diffs": model_visible_diffs,
            "host_source_comparison": {
                "byte_identical": False,
                "changed_shared_sources": changed_shared_sources,
                "added_sources": sorted(set(new_sources) - set(old_sources)),
                "removed_sources": sorted(set(old_sources) - set(new_sources)),
                "qualification": (
                    "Freeze verification and manifest comparison constrain the model-visible "
                    "difference to the system prompt, but the historical and corrected host "
                    "source closures are not byte-identical."
                ),
            },
        },
        "old_control": {
            "run": OLD_RUN.relative_to(REPO).as_posix(),
            "actions": old_actions,
            "levels_completed": old_manifest["levels_completed"],
            "model_calls": old_manifest["model_calls"],
            "usage": old_manifest["usage"],
        },
        "corrected_control": {
            "run": NEW_RUN.relative_to(REPO).as_posix(),
            "actions": new_actions,
            "levels_completed": new_manifest["levels_completed"],
            "model_calls": new_manifest["model_calls"],
            "decision_repairs": new_manifest["decision_repairs"],
            "transport_retries": new_manifest["transport_retries"],
            "stop_reason": new_manifest["stop_reason"],
            "usage": new_manifest["usage"],
            "provider": provider_evidence(NEW_RUN),
        },
        "post_run_oracle": {
            "source": "environment_files/cd82/fb555c5d/cd82.py:531-753",
            "not_model_visible": True,
            "findings": [
                "actions 1-4 navigate among selectable positions",
                "action 6 selects a color or activates a directional piece",
                "action 5 applies the selected color/shape",
                "completion compares the constructed 10x10 output to the reference",
            ],
        },
        "interpretation": {
            "observation": (
                "The first RIGHT action changed 201 pixels and the model subsequently "
                "used genuine directional transitions as evidence for a movable object."
            ),
            "hypothesis": (
                "Uniform names removed the click-interface prior, but immediate directional "
                "feedback plus a pretrained movement-game prior closed attention around a "
                "player-to-target story before actions 5 and 6 were tested."
            ),
            "alternative": (
                "The result may be model-seed variance; one run does not establish the "
                "population effect or prove that the names-only contract is itself biased."
            ),
            "falsifier": (
                "Matched corrected-control replications that retain heavy action-6 use, or "
                "sample actions 5/6 and solve without the proposed closure pattern."
            ),
            "conclusion": (
                "The old asymmetric interface was a material cause of action-6 fixation. "
                "The next observed bottleneck is experiment coverage and objective-role "
                "closure, not perception of directional consequences."
            ),
        },
        "integrity": {
            "old_manifest_sha256": digest(OLD_RUN / "manifest.json"),
            "new_manifest_sha256": digest(NEW_RUN / "manifest.json"),
            "plan_sha256": digest(PLAN),
            "new_manifest_hash_failures": verify_manifest_hashes(NEW_RUN, new_manifest),
            "source_hash_failures": source_failures,
        },
    }
    if evidence["integrity"]["new_manifest_hash_failures"]:
        raise RuntimeError("new manifest evidence hash verification failed")
    if source_failures:
        raise RuntimeError("participating sources drifted after execution")
    if evidence["corrected_control"]["provider"] != {
        "response_providers": ["ModelRun"],
        "requested_provider_only": ["modelrun/fp4"],
    }:
        raise RuntimeError("provider identity audit failed")
    OUTPUT.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
