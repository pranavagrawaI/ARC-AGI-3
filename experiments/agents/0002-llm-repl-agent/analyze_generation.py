"""Recompute the frozen four-component profile from three valid included pairs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Mapping


AGENT_ROOT = Path(__file__).resolve().parent
REPO = AGENT_ROOT.parents[2]
for import_root in (REPO, AGENT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from llm_repl_agent.evidence import validate_run_evidence
from llm_repl_agent.authority import (
    resolve_artifact_reference,
    validate_final_freeze,
    validate_included_pair_result,
)
from llm_repl_agent.experiment_protocol import (
    PairTelemetry,
    RunOutcome,
    assess_component_profile,
    assess_interface_damage,
    assess_pair,
    endpoint_metadata_sha256,
    participating_source_hashes,
    resolve_run_annotation,
)
from llm_repl_agent.intervention import validate_shadow_bundle_result


ANALYSIS_INPUT_SCHEMA = "scientist-game-designer-player-analysis-input-v2"
ANALYSIS_RESULT_SCHEMA = "scientist-game-designer-player-analysis-result-v2"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _path(base: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is missing")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _reported_tokens(usage: Any) -> int | None:
    if not isinstance(usage, Mapping):
        return None
    total = usage.get("total_tokens")
    if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
        return total
    values: list[int] = []
    for key in ("prompt_tokens", "completion_tokens"):
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            values.append(value)
    return sum(values) if values else None


def _outcome(manifest: Mapping[str, Any]) -> RunOutcome:
    levels = _integer(manifest.get("levels_completed"), "levels_completed")
    if levels > 2:
        raise ValueError("included gameplay outcome exposes a held-out level")
    completions = manifest.get("completion_action_counts")
    if not isinstance(completions, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in completions
    ):
        raise ValueError("completion action counts are malformed")
    if len(completions) != levels:
        raise ValueError("completion action counts do not match levels completed")
    return RunOutcome(
        levels_completed=levels,
        level_1_completion_action=completions[0] if levels >= 1 else None,
        level_2_completion_action=completions[1] if levels >= 2 else None,
    )


def _telemetry(manifest: Mapping[str, Any]) -> PairTelemetry:
    actions = _integer(manifest.get("actions"), "actions")
    stop_reason = manifest.get("stop_reason")
    rounds = manifest.get("model_rounds_per_action")
    if not isinstance(stop_reason, str) or not stop_reason:
        raise ValueError("run stop reason is missing")
    if not isinstance(rounds, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in rounds
    ):
        raise ValueError("model rounds per action are malformed")
    return PairTelemetry(
        actions=actions,
        stop_reason=stop_reason,
        reported_tokens=_reported_tokens(manifest.get("usage")),
        model_rounds=tuple(rounds),
    )


def _run_evidence_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    experiment = manifest.get("experiment")
    if not isinstance(experiment, Mapping):
        experiment = {}
    return {
        field: manifest.get(field)
        for field in (
            "run_id",
            "model",
            "endpoint",
            "model_profile",
            "status",
            "stop_reason",
            "actions",
            "levels_completed",
            "model_calls",
            "python_calls",
            "decision_repairs",
            "usage",
            "provider_usage",
            "evidence_hashes",
        )
    } | {
        "endpoint_metadata": experiment.get("endpoint_metadata"),
        "endpoint_metadata_sha256": experiment.get("endpoint_metadata_sha256"),
    }


def analyze(input_path: Path) -> dict[str, Any]:
    source = _load(input_path)
    if source.get("schema") != ANALYSIS_INPUT_SCHEMA:
        raise ValueError("analysis input schema mismatch")
    generation = source.get("generation")
    rows = source.get("pairs")
    if not isinstance(generation, str) or not generation:
        raise ValueError("analysis generation is missing")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("analysis requires exactly three included pairs")
    if {row.get("pair_id") for row in rows if isinstance(row, Mapping)} != {"1", "2", "3"}:
        raise ValueError("analysis input must contain included pairs 1, 2, and 3 exactly once")

    base = input_path.parent
    artifact_hashes: dict[str, str] = {str(input_path.resolve()): _hash(input_path)}
    final_freeze_path = _path(base, source.get("final_freeze"), "analysis final freeze")
    if source.get("final_freeze_sha256") != _hash(final_freeze_path):
        raise ValueError("analysis final freeze hash mismatch")
    freeze_document = _load(final_freeze_path)
    freeze_artifacts = freeze_document.get("artifacts")
    if not isinstance(freeze_artifacts, Mapping):
        raise ValueError("analysis final freeze has no artifact closure")
    release_results_path = resolve_artifact_reference(
        freeze_artifacts.get("release_gate_results"),
        label="analysis release gate results",
    )
    endpoint_hash = freeze_document.get("endpoint_metadata_sha256")
    if not isinstance(endpoint_hash, str) or not endpoint_hash:
        raise ValueError("analysis final freeze has no endpoint commitment")
    current_sources = participating_source_hashes()
    validate_final_freeze(
        final_freeze_path,
        generation=generation,
        release_gate_dir=release_results_path.parent,
        source_hashes=current_sources,
        endpoint_metadata_hash=endpoint_hash,
    )
    artifact_hashes[str(final_freeze_path)] = _hash(final_freeze_path)
    analyzed_pairs: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: str(item["pair_id"])):
        pair_id = str(row["pair_id"])
        pair_path = _path(base, row.get("pair_result"), f"pair {pair_id} result")
        validate_included_pair_result(
            pair_path,
            final_freeze_path=final_freeze_path,
            generation=generation,
            release_gate_dir=release_results_path.parent,
            source_hashes=current_sources,
            endpoint_metadata_hash=endpoint_hash,
        )
        pair = _load(pair_path)
        artifact_hashes[str(pair_path)] = _hash(pair_path)
        if (
            pair.get("schema") != "scientist-game-designer-player-pair-v1"
            or pair.get("pair_id") != pair_id
            or pair.get("generation") != generation
            or pair.get("included") is not True
            or pair.get("gameplay_executed") is not True
            or pair.get("status") != "pair_complete"
        ):
            raise ValueError(f"pair {pair_id} is not one valid included gameplay pair")
        valid_attempt_number = pair.get("valid_attempt")
        attempts = pair.get("attempts")
        if not isinstance(attempts, list):
            raise ValueError(f"pair {pair_id} has no attempt ledger")
        matching = [
            attempt
            for attempt in attempts
            if isinstance(attempt, Mapping)
            and attempt.get("attempt") == valid_attempt_number
            and attempt.get("valid") is True
        ]
        if len(matching) != 1:
            raise ValueError(f"pair {pair_id} does not identify exactly one valid attempt")
        members = matching[0].get("members")
        if not isinstance(members, list) or len(members) != 2:
            raise ValueError(f"pair {pair_id} valid attempt does not contain two members")
        by_condition = {
            member.get("condition"): member
            for member in members
            if isinstance(member, Mapping)
            and member.get("condition") in {"enabled", "disabled"}
        }
        if set(by_condition) != {"enabled", "disabled"}:
            raise ValueError(f"pair {pair_id} is missing a condition member")
        annotation_paths = row.get("annotations")
        if not isinstance(annotation_paths, Mapping) or set(annotation_paths) != {
            "enabled",
            "disabled",
        }:
            raise ValueError(f"pair {pair_id} must provide both completed annotations")

        manifests: dict[str, dict[str, Any]] = {}
        resolved_annotations: dict[str, dict[str, Any]] = {}
        for condition in ("enabled", "disabled"):
            member = by_condition[condition]
            run_dir = _path(base, member.get("run_dir"), f"pair {pair_id} {condition} run")
            manifest_path = run_dir / "manifest.json"
            manifest = _load(manifest_path)
            artifact_hashes[str(manifest_path)] = _hash(manifest_path)
            if member.get("manifest_sha256") != artifact_hashes[str(manifest_path)]:
                raise ValueError(f"pair {pair_id} {condition} manifest hash mismatch")
            experiment = manifest.get("experiment")
            evidence = manifest.get("evidence_hashes")
            if (
                member.get("validity") != "valid_contestant_result"
                or not isinstance(experiment, Mapping)
                or experiment.get("generation") != generation
                or experiment.get("pair_id") != pair_id
                or experiment.get("condition") != condition
                or experiment.get("included") is not True
                or not isinstance(evidence, Mapping)
                or not validate_run_evidence(run_dir, evidence)
            ):
                raise ValueError(f"pair {pair_id} {condition} raw gameplay evidence is invalid")
            annotation_path = _path(
                base,
                annotation_paths[condition],
                f"pair {pair_id} {condition} annotation",
            )
            annotation = _load(annotation_path)
            artifact_hashes[str(annotation_path)] = _hash(annotation_path)
            resolved_annotations[condition] = resolve_run_annotation(
                annotation,
                condition=condition,
                evidence_hashes=evidence,
            )
            manifests[condition] = manifest

        recomputed_sign = assess_pair(_outcome(manifests["enabled"]), _outcome(manifests["disabled"]))
        if pair.get("pair_sign") != recomputed_sign:
            raise ValueError(f"pair {pair_id} sign does not match raw outcomes")
        damage = assess_interface_damage(
            _telemetry(manifests["disabled"]),
            _telemetry(manifests["enabled"]),
            annotated_damage=resolved_annotations["disabled"]["criterion_3_interface_damage"],
        )
        disabled_status = resolved_annotations["disabled"]["terminal_status"]
        clean_disabled = disabled_status == "compliant" and damage.severe_interface_damage is False

        shadow_pass = False
        shadow_path_value = row.get("shadow_result")
        if shadow_path_value is not None:
            shadow_path = _path(base, shadow_path_value, f"pair {pair_id} shadow result")
            shadow = _load(shadow_path)
            artifact_hashes[str(shadow_path)] = _hash(shadow_path)
            if shadow.get("generation") != generation or shadow.get("run_id") != manifests[
                "enabled"
            ].get("run_id"):
                raise ValueError(f"pair {pair_id} shadow result belongs to another run")
            experiment = manifests["enabled"]["experiment"]
            if endpoint_metadata_sha256(shadow.get("endpoint_metadata", {})) != experiment.get(
                "endpoint_metadata_sha256"
            ):
                raise ValueError(f"pair {pair_id} shadow endpoint differs from gameplay")
            shadow_pass = validate_shadow_bundle_result(shadow) == "demonstrated"

        annotated_enabled_status = resolved_annotations["enabled"]["terminal_status"]
        if annotated_enabled_status == "demonstrated" and not shadow_pass:
            raise ValueError(f"pair {pair_id} claims demonstration without a passing shadow bundle")
        enabled_status = (
            "demonstrated"
            if shadow_pass
            else annotated_enabled_status
        )
        analyzed_pairs.append({
            "pair_id": pair_id,
            "pair_result_sha256": artifact_hashes[str(pair_path)],
            "pair_sign": recomputed_sign,
            "enabled_status": enabled_status,
            "disabled_status": disabled_status,
            "interface_damage": asdict(damage),
            "clean_disabled_run": clean_disabled,
            "local_content_dependence": shadow_pass,
            "outcomes": {
                condition: asdict(_outcome(manifests[condition]))
                for condition in ("enabled", "disabled")
            },
            "run_evidence": {
                condition: _run_evidence_summary(manifests[condition])
                for condition in ("enabled", "disabled")
            },
        })

    profile = assess_component_profile(
        enabled_statuses=[row["enabled_status"] for row in analyzed_pairs],
        clean_disabled_runs=[row["clean_disabled_run"] for row in analyzed_pairs],
        intervention_passes=[row["local_content_dependence"] for row in analyzed_pairs],
        pair_signs=[row["pair_sign"] for row in analyzed_pairs],
    )
    return {
        "schema": ANALYSIS_RESULT_SCHEMA,
        "generation": generation,
        "component_profile": profile,
        "pairs": analyzed_pairs,
        "input_artifact_hashes": dict(sorted(artifact_hashes.items())),
        "rejected_claims": [
            "conventional significance or a population effect",
            "equivalence when the gameplay threshold is missed",
            "the selected response caused the gameplay difference",
            "beneficial local dependence",
            "source-faithful reconstruction",
            "deterministic same-seed control",
            "transfer to levels 3-7 or cross-game generalization",
            "researcher blindness",
            "general Python, world-model, reconstruction, or Three-Hat benefit",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"analysis output already exists: {args.output}")
    result = analyze(args.input.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "component_profile": result["component_profile"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
