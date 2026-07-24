"""Preflight and execute one frozen adjacent matched pair.

Gameplay is impossible without both ``--execute-gameplay`` and an exact
``--confirm-pair`` value. The default operation is a non-gameplay preflight.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


AGENT_ROOT = Path(__file__).resolve().parent
REPO = AGENT_ROOT.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from llm_repl_agent.experiment_protocol import (
    EXPERIMENT_FILES,
    ExperimentCondition,
    canonical_endpoint_metadata,
    compose_condition_prompt,
    endpoint_metadata_sha256,
    fetch_endpoint_metadata,
    frozen_pair_schedule,
    gameplay_attempt_validity,
    load_frozen_runtime,
    participating_source_hashes,
    RunOutcome,
    assess_pair,
    validate_endpoint_metadata,
    validate_pair_commitments,
    validate_prompt_pair,
)
from llm_repl_agent.authority import (
    consume_pair_authority,
    validate_final_freeze,
    validate_pilot_approval,
    validate_release_gate,
)


PAIR_SCHEMA = "scientist-game-designer-player-pair-v1"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_completed_pair_result(
    *,
    preflight: Mapping[str, Any],
    valid_attempt: int,
    pair_sign: str,
    attempts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Materialize the pair-result contract consumed by the final analyzer."""

    if (
        preflight.get("schema") != PAIR_SCHEMA
        or not isinstance(preflight.get("included"), bool)
        or isinstance(valid_attempt, bool)
        or valid_attempt < 1
        or pair_sign not in {"enabled", "disabled", "tie"}
    ):
        raise ValueError("completed pair result inputs violate the pair contract")
    if len(attempts) != valid_attempt:
        raise ValueError("completed pair result must stop at the first valid attempt")
    for index, attempt in enumerate(attempts, start=1):
        expected_valid = index == valid_attempt
        if (
            attempt.get("attempt") != index
            or attempt.get("valid") is not expected_valid
        ):
            raise ValueError("completed pair result attempt ledger is inconsistent")
    return {
        **dict(preflight),
        "status": "pair_complete",
        "gameplay_executed": True,
        "valid_attempt": valid_attempt,
        "pair_sign": pair_sign,
        "attempts": [dict(attempt) for attempt in attempts],
    }


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def validate_release_gate_bundle(
    root: Path,
    release_gate: Mapping[str, Any],
    release_candidate: Mapping[str, Any],
) -> None:
    """Verify every supporting release artifact against the candidate commitment."""

    if not isinstance(release_gate, Mapping) or not isinstance(release_candidate, Mapping):
        raise ValueError("release gate bundle contains a non-object manifest")
    artifacts = release_gate.get("artifacts")
    committed = release_candidate.get("generated_artifact_hashes")
    if not isinstance(artifacts, Mapping) or not isinstance(committed, Mapping):
        raise ValueError("release gate is missing supporting artifact commitments")
    expected_keys = {
        "endpoint_metadata",
        "sandbox_results",
        "synthetic_test_results",
        "intervention_test_vectors",
    }
    if set(artifacts) != expected_keys:
        raise ValueError("release gate supporting artifact inventory mismatch")
    resolved_root = root.resolve()
    for logical_name in sorted(expected_keys):
        relative = artifacts.get(logical_name)
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError(f"invalid release supporting artifact path: {logical_name}")
        path = (root / relative).resolve()
        if path.parent != resolved_root or not path.is_file():
            raise ValueError(f"missing release supporting artifact: {logical_name}")
        if committed.get(Path(relative).name) != _file_hash(path):
            raise ValueError(f"release supporting artifact hash mismatch: {logical_name}")


def build_pair_commitments(
    *,
    generation: str,
    pair_id: str,
    endpoint_metadata: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not generation.strip():
        raise ValueError("generation must not be blank")
    config = load_frozen_runtime()
    validate_endpoint_metadata(endpoint_metadata, config)
    try:
        schedule = next(pair for pair in frozen_pair_schedule() if pair.pair_id == pair_id)
    except StopIteration as error:
        raise ValueError(f"unknown pair: {pair_id}") from error
    prompt_report = validate_prompt_pair(
        (EXPERIMENT_FILES / "reconstruction-enabled.txt").read_bytes(),
        (EXPERIMENT_FILES / "reconstruction-disabled.txt").read_bytes(),
    )
    if not prompt_report["valid"]:
        raise ValueError("condition prompts do not match the authorized frozen diff")
    runtime_hash = _file_hash(EXPERIMENT_FILES / "runtime-config.yaml")
    common = {
        "schema": PAIR_SCHEMA,
        "generation": generation,
        "pair_id": pair_id,
        "included": schedule.included,
        "model": config.model,
        "endpoint": config.endpoint,
        "model_profile": config.model_profile,
        "runtime_sha256": runtime_hash,
        "source_hashes": dict(sorted(source_hashes.items())),
        "endpoint_metadata_sha256": endpoint_metadata_sha256(endpoint_metadata),
        "environment_seed": schedule.environment_seed,
        "requested_seed": schedule.requested_seed,
        "color_mapping": list(range(16)),
        "pair_order": list(schedule.order),
        "authorized_prompt_diff_sha256": prompt_report["authorized_diff_sha256"],
    }
    by_condition: dict[str, dict[str, Any]] = {}
    for condition_name in ("enabled", "disabled"):
        block_path = EXPERIMENT_FILES / f"reconstruction-{condition_name}.txt"
        full_prompt = compose_condition_prompt(ExperimentCondition(condition_name))
        by_condition[condition_name] = {
            **common,
            "condition": condition_name,
            "condition_label": f"reconstruction-{condition_name}",
            "prompt_sha256": sha256(full_prompt.encode("utf-8")).hexdigest(),
            "condition_block_sha256": _file_hash(block_path),
        }
    validate_pair_commitments(by_condition["enabled"], by_condition["disabled"])
    return tuple(by_condition[condition] for condition in schedule.order)  # type: ignore[return-value]


def _outcome(manifest: Mapping[str, Any]) -> RunOutcome:
    completions = manifest.get("completion_action_counts")
    completion_actions = completions if isinstance(completions, list) else []
    return RunOutcome(
        levels_completed=int(manifest.get("levels_completed") or 0),
        level_1_completion_action=(
            int(completion_actions[0]) if len(completion_actions) >= 1 else None
        ),
        level_2_completion_action=(
            int(completion_actions[1]) if len(completion_actions) >= 2 else None
        ),
    )


def _run_member(
    *,
    generation: str,
    pair_id: str,
    condition: str,
    endpoint_metadata_path: Path,
    release_gate_results_path: Path,
    final_freeze_path: Path | None,
    authority_consumption_path: Path,
    run_dir: Path,
) -> tuple[int, dict[str, Any] | None]:
    command = [
        sys.executable,
        str(AGENT_ROOT / "run.py"),
        "--game",
        "ls20",
        "--experiment-condition",
        condition,
        "--experiment-pair",
        pair_id,
        "--experiment-generation",
        generation,
        "--endpoint-metadata",
        str(endpoint_metadata_path),
        "--release-gate-results",
        str(release_gate_results_path),
        "--authority-consumption",
        str(authority_consumption_path),
        "--run-dir",
        str(run_dir),
    ]
    if final_freeze_path is not None:
        command.extend(("--final-freeze-manifest", str(final_freeze_path)))
    completed = subprocess.run(command, check=False)
    manifest_path = run_dir / "manifest.json"
    manifest = None
    if manifest_path.is_file():
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = value if isinstance(value, dict) else None
    return completed.returncode, manifest


def _refresh_endpoint(
    *,
    baseline: Mapping[str, Any],
    raw_path: Path,
    canonical_path: Path,
) -> dict[str, Any]:
    config = load_frozen_runtime()
    raw = fetch_endpoint_metadata(config)
    canonical = canonical_endpoint_metadata(
        raw,
        model=config.model,
        tag=config.provider.only[0],
    )
    validate_endpoint_metadata(canonical, config)
    _write_json(raw_path, raw)
    _write_json(canonical_path, canonical)
    if endpoint_metadata_sha256(canonical) != endpoint_metadata_sha256(baseline):
        raise SystemExit(
            "endpoint generation differs from the compatibility baseline; gameplay aborted"
        )
    return canonical


def _validate_generation_sequence(generation_root: Path, pair_id: str) -> None:
    if (generation_root / "generation-aborted.json").exists():
        raise SystemExit("generation is permanently aborted after a second invalid pair attempt")
    completed: set[str] = set()
    attempted: set[str] = set()
    for result_path in sorted(generation_root.glob("pair-*/pair-result.json")):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise SystemExit(f"unreadable prior pair ledger entry: {result_path}") from None
        if not isinstance(result, dict) or result.get("gameplay_executed") is not True:
            continue
        prior = str(result.get("pair_id"))
        attempted.add(prior)
        if result.get("status") == "pair_complete":
            completed.add(prior)
        if result.get("status") == "experiment_aborted_after_second_pair_invalidation":
            raise SystemExit("generation is permanently aborted after a second invalid pair attempt")
    expected_completed = {
        "P": set(),
        "1": {"P"},
        "2": {"P", "1"},
        "3": {"P", "1", "2"},
    }[pair_id]
    if pair_id in attempted or completed != expected_completed:
        raise SystemExit(
            f"pair {pair_id} violates frozen generation order; completed={sorted(completed)}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--pair", required=True, choices=("P", "1", "2", "3"))
    parser.add_argument("--baseline-endpoint-metadata", required=True, type=Path)
    parser.add_argument("--release-gate-dir", required=True, type=Path)
    parser.add_argument("--pilot-approval", type=Path)
    parser.add_argument("--final-freeze-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute-gameplay", action="store_true")
    parser.add_argument("--confirm-pair")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.execute_gameplay and args.pair == "P" and args.pilot_approval is None:
        raise SystemExit("pilot gameplay requires --pilot-approval")
    if (
        args.execute_gameplay
        and args.pair != "P"
        and args.final_freeze_manifest is None
    ):
        raise SystemExit("included gameplay requires --final-freeze-manifest")
    baseline = json.loads(args.baseline_endpoint_metadata.read_text(encoding="utf-8"))
    if not isinstance(baseline, dict):
        raise SystemExit("baseline endpoint metadata must be a JSON object")
    validate_endpoint_metadata(baseline)
    release_gate_results_path = args.release_gate_dir / "release-gate-results.json"
    release_candidate_path = args.release_gate_dir / "release-candidate-manifest.json"
    if not release_gate_results_path.is_file() or not release_candidate_path.is_file():
        raise SystemExit("release gate directory is incomplete")
    release_gate = json.loads(release_gate_results_path.read_text(encoding="utf-8"))
    release_candidate = json.loads(release_candidate_path.read_text(encoding="utf-8"))
    try:
        validate_release_gate_bundle(args.release_gate_dir, release_gate, release_candidate)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    current_sources = participating_source_hashes()
    if (
        not isinstance(release_gate, dict)
        or release_gate.get("passed") is not True
        or release_gate.get("pilot_authorized_by_gate") is not True
        or release_gate.get("source_hashes") != current_sources
        or not isinstance(release_candidate, dict)
        or release_candidate.get("release_gate_passed") is not True
        or release_candidate.get("release_gate_results_sha256")
        != _file_hash(release_gate_results_path)
        or not isinstance(release_gate.get("compatibility"), dict)
        or release_gate["compatibility"].get("endpoint_metadata_sha256")
        != endpoint_metadata_sha256(baseline)
    ):
        raise SystemExit("release gate does not authorize these exact sources and endpoint")
    endpoint_hash = endpoint_metadata_sha256(baseline)
    try:
        validate_release_gate(
            args.release_gate_dir,
            source_hashes=current_sources,
            endpoint_metadata_hash=endpoint_hash,
        )
        if args.pair == "P":
            if args.final_freeze_manifest is not None:
                raise ValueError("the excluded pilot must precede the final freeze")
            if args.execute_gameplay and args.pilot_approval is None:
                raise ValueError("pilot gameplay requires --pilot-approval")
            approval = (
                validate_pilot_approval(
                    args.pilot_approval,
                    generation=args.generation,
                    release_gate_dir=args.release_gate_dir,
                    source_hashes=current_sources,
                    endpoint_metadata_hash=endpoint_hash,
                )
                if args.pilot_approval is not None
                else None
            )
            maximum_pair_attempts = (
                int(approval["maximum_pair_attempts"]) if approval is not None else 1
            )
            authority_path = args.pilot_approval
            authority_kind = "pilot_approval"
        else:
            if args.pilot_approval is not None:
                raise ValueError("pilot approval cannot authorize included gameplay")
            if args.final_freeze_manifest is None:
                raise ValueError("included gameplay requires --final-freeze-manifest")
            final_freeze = validate_final_freeze(
                args.final_freeze_manifest,
                generation=args.generation,
                release_gate_dir=args.release_gate_dir,
                source_hashes=current_sources,
                endpoint_metadata_hash=endpoint_hash,
            )
            exposure = final_freeze["authorization"]["maximum_exposure"]
            maximum_pair_attempts = int(exposure["maximum_pair_attempts_per_pair"])
            authority_path = args.final_freeze_manifest
            authority_kind = "final_freeze"
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pair_root = args.output_dir or (
        AGENT_ROOT / "artifacts" / "generations" / args.generation / f"pair-{args.pair}-{timestamp}"
    )
    generation_root = AGENT_ROOT / "artifacts" / "generations" / args.generation
    if args.execute_gameplay:
        if pair_root.parent.resolve() != generation_root.resolve():
            raise SystemExit("gameplay output must use the canonical generation ledger directory")
        _validate_generation_sequence(generation_root, args.pair)
    if pair_root.exists():
        raise SystemExit(f"pair output directory already exists: {pair_root}")
    pair_root.mkdir(parents=True)
    endpoint_path = pair_root / "endpoint-metadata.canonical.json"
    current = _refresh_endpoint(
        baseline=baseline,
        raw_path=pair_root / "endpoint-metadata.raw.json",
        canonical_path=endpoint_path,
    )
    source_hashes = participating_source_hashes()
    commitments = build_pair_commitments(
        generation=args.generation,
        pair_id=args.pair,
        endpoint_metadata=current,
        source_hashes=source_hashes,
    )
    pair_commitments_path = pair_root / "pair-commitments.json"
    _write_json(pair_commitments_path, {
        "schema": PAIR_SCHEMA,
        "commitments": list(commitments),
    })
    preflight = {
        "schema": PAIR_SCHEMA,
        "generation": args.generation,
        "pair_id": args.pair,
        "included": bool(commitments[0]["included"]),
        "status": "preflight_passed",
        "gameplay_executed": False,
        "endpoint_metadata_sha256": endpoint_metadata_sha256(current),
        "source_hashes": source_hashes,
        "order": [row["condition"] for row in commitments],
        "pair_commitments_sha256": _file_hash(pair_commitments_path),
        "release_gate_results_sha256": _file_hash(release_gate_results_path),
        "release_candidate_manifest_sha256": _file_hash(release_candidate_path),
    }
    _write_json(pair_root / "pair-result.json", preflight)
    if not args.execute_gameplay:
        print(json.dumps({**preflight, "output_dir": str(pair_root)}, indent=2, sort_keys=True))
        return
    if args.confirm_pair != args.pair:
        raise SystemExit(
            f"gameplay requires --confirm-pair {args.pair}; no environment action was run"
        )

    assert authority_path is not None
    try:
        authority_consumption_path = consume_pair_authority(
            generation_root=generation_root,
            pair_root=pair_root,
            generation=args.generation,
            pair_id=args.pair,
            authority_path=authority_path,
            authority_kind=authority_kind,
            maximum_pair_attempts=maximum_pair_attempts,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    preflight["authority_consumption"] = str(authority_consumption_path)
    preflight["authority_consumption_sha256"] = _file_hash(authority_consumption_path)
    _write_json(pair_root / "pair-result.json", preflight)

    attempts: list[dict[str, Any]] = []
    for attempt_number in range(1, maximum_pair_attempts + 1):
        if attempt_number == 2:
            current = _refresh_endpoint(
                baseline=baseline,
                raw_path=pair_root / "endpoint-metadata.attempt-2.raw.json",
                canonical_path=endpoint_path,
            )
        attempt: dict[str, Any] = {
            "attempt": attempt_number,
            "members": [],
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        attempt_valid = True
        for commitment in commitments:
            condition = str(commitment["condition"])
            run_dir = pair_root / f"attempt-{attempt_number}" / condition
            return_code, manifest = _run_member(
                generation=args.generation,
                pair_id=args.pair,
                condition=condition,
                endpoint_metadata_path=endpoint_path,
                release_gate_results_path=release_gate_results_path,
                final_freeze_path=args.final_freeze_manifest,
                authority_consumption_path=authority_consumption_path,
                run_dir=run_dir,
            )
            validity = (
                gameplay_attempt_validity(manifest, commitment, run_dir=run_dir)
                if manifest is not None
                else "infrastructure_invalid"
            )
            member = {
                "condition": condition,
                "return_code": return_code,
                "run_dir": str(run_dir),
                "validity": validity,
                "started_at": manifest.get("started_at") if manifest else None,
                "finished_at": manifest.get("finished_at") if manifest else None,
                "manifest_sha256": (
                    _file_hash(run_dir / "manifest.json") if manifest is not None else None
                ),
            }
            attempt["members"].append(member)
            if return_code != 0 or validity != "valid_contestant_result":
                attempt_valid = False
        attempt["finished_at"] = datetime.now(timezone.utc).isoformat()
        attempt["valid"] = attempt_valid and len(attempt["members"]) == 2
        attempts.append(attempt)
        if attempt["valid"]:
            manifests_by_condition = {
                member["condition"]: json.loads(
                    (Path(member["run_dir"]) / "manifest.json").read_text(encoding="utf-8")
                )
                for member in attempt["members"]
            }
            result = build_completed_pair_result(
                preflight=preflight,
                valid_attempt=attempt_number,
                pair_sign=assess_pair(
                    _outcome(manifests_by_condition["enabled"]),
                    _outcome(manifests_by_condition["disabled"]),
                ),
                attempts=attempts,
            )
            _write_json(pair_root / "pair-result.json", result)
            print(json.dumps({**result, "output_dir": str(pair_root)}, indent=2, sort_keys=True))
            return
    result = {
        **preflight,
        "status": "experiment_aborted_after_pair_invalidation_limit",
        "gameplay_executed": True,
        "attempts": attempts,
    }
    _write_json(pair_root / "pair-result.json", result)
    _write_json(generation_root / "generation-aborted.json", {
        "schema": PAIR_SCHEMA,
        "generation": args.generation,
        "pair_id": args.pair,
        "reason": "pair_invalidation_attempt_limit_reached",
        "pair_result_sha256": _file_hash(pair_root / "pair-result.json"),
    })
    raise SystemExit("gameplay-pair invalidation limit reached; experiment aborted")


if __name__ == "__main__":
    main()
