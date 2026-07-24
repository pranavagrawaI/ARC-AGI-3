"""Run the pre-pilot machine-readable release gate without gameplay."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping


AGENT_ROOT = Path(__file__).resolve().parent
REPO = AGENT_ROOT.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from llm_repl_agent.experiment_protocol import (
    EXPERIMENT_FILES,
    canonical_endpoint_metadata,
    endpoint_metadata_sha256,
    experiment_artifact_templates,
    frozen_pair_schedule,
    frozen_request_options,
    load_frozen_runtime,
    participating_source_hashes,
    validate_endpoint_metadata,
    validate_prompt_pair,
)
from llm_repl_agent.intervention import (
    build_branch_payloads,
    build_shadow_schedule,
    response_signature,
)
from llm_repl_agent.sandbox_admissibility import run_sandbox_admissibility_suite
from llm_repl_agent.workspace import PythonResult



RELEASE_SCHEMA = "scientist-game-designer-player-release-gate-v1"
TEST_MODULES = (
    "experiments.harness.tests.test_harness",
    "experiments.agents.tests.test_llm_repl_agent",
    "experiments.agents.tests.test_falsifiable_experiment",
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _run_synthetic_tests() -> dict[str, Any]:
    command = [sys.executable, "-m", "unittest", "-v", *TEST_MODULES]
    completed = subprocess.run(
        command,
        cwd=REPO,
        capture_output=True,
        check=False,
        text=True,
    )
    output = completed.stdout + completed.stderr
    match = re.search(r"Ran (\d+) tests?", output)
    return {
        "passed": completed.returncode == 0,
        "return_code": completed.returncode,
        "command": command,
        "modules": list(TEST_MODULES),
        "tests_run": int(match.group(1)) if match else None,
        "output": output,
    }


def _compatibility_result(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"provided": False, "passed": False, "reason": "not run"}
    result_path = path / "probe-result.json"
    raw_path = path / "endpoint-metadata.raw.json"
    canonical_path = path / "endpoint-metadata.canonical.json"
    evidence_paths = {
        "probe_result": result_path,
        "endpoint_metadata_raw": raw_path,
        "endpoint_metadata_canonical": canonical_path,
        "request_1": path / "request-1.json",
        "response_1": path / "response-1.raw.json",
        "tool_result": path / "tool-result.json",
        "request_2": path / "request-2.json",
        "response_2": path / "response-2.raw.json",
    }
    missing = [str(item) for item in evidence_paths.values() if not item.is_file()]
    if missing:
        return {"provided": True, "passed": False, "reason": "missing files", "missing": missing}
    result = json.loads(result_path.read_text(encoding="utf-8"))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    if not all(isinstance(value, dict) for value in (result, raw, canonical)):
        return {"provided": True, "passed": False, "reason": "non-object probe artifact"}
    config = load_frozen_runtime()
    expected_canonical = canonical_endpoint_metadata(
        raw,
        model=config.model,
        tag=config.provider.only[0],
    )
    try:
        validate_endpoint_metadata(canonical, config)
    except ValueError as error:
        return {"provided": True, "passed": False, "reason": str(error)}
    canonical_hash = endpoint_metadata_sha256(canonical)
    request_1 = json.loads(evidence_paths["request_1"].read_text(encoding="utf-8"))
    response_1 = json.loads(evidence_paths["response_1"].read_text(encoding="utf-8"))
    tool_result = json.loads(evidence_paths["tool_result"].read_text(encoding="utf-8"))
    request_2 = json.loads(evidence_paths["request_2"].read_text(encoding="utf-8"))
    response_2 = json.loads(evidence_paths["response_2"].read_text(encoding="utf-8"))
    artifact_hashes = {key: _hash(value) for key, value in evidence_paths.items()}
    committed_hashes = result.get("artifact_hashes")
    archive_commitments_valid = isinstance(committed_hashes, Mapping) and all(
        committed_hashes.get(key) == artifact_hashes[artifact_key]
        for key, artifact_key in {
            "request_1_sha256": "request_1",
            "response_1_sha256": "response_1",
            "tool_result_sha256": "tool_result",
            "request_2_sha256": "request_2",
            "response_2_sha256": "response_2",
        }.items()
    )
    config = load_frozen_runtime()
    request_contract = all(
        request.get("seed") == 0
        and request.get("top_k") == config.request.top_k
        and request.get("max_tokens") == config.request.max_completion_tokens
        and request.get("provider") == config.provider.request_value()
        and isinstance(request.get("tools"), list)
        and request["tools"][0].get("function", {}).get("name") == "python"
        for request in (request_1, request_2)
    )
    first_message = response_1.get("choices", [{}])[0].get("message", {})
    second_message = response_2.get("choices", [{}])[0].get("message", {})
    calls = first_message.get("tool_calls")
    response_contract = (
        isinstance(calls, list)
        and len(calls) == 1
        and calls[0].get("function", {}).get("name") == "python"
        and bool(first_message.get("reasoning_content") or first_message.get("reasoning"))
        and tool_result.get("ok") is True
        and str(tool_result.get("stdout", "")).strip() == "42"
        and isinstance(second_message.get("content"), str)
        and bool(second_message["content"].strip())
        and not second_message.get("tool_calls")
        and any(message.get("role") == "tool" for message in request_2.get("messages", []))
    )
    passed = (
        result.get("passed") is True
        and result.get("request_contract_valid") is True
        and result.get("has_reasoning_content") is True
        and result.get("arguments_valid") is True
        and result.get("tool_execution_valid") is True
        and result.get("continuation_valid") is True
        and result.get("tool_names") == ["python"]
        and result.get("endpoint_metadata_sha256") == canonical_hash
        and result.get("source_hashes") == participating_source_hashes()
        and expected_canonical == canonical
        and archive_commitments_valid
        and request_contract
        and response_contract
    )
    return {
        "provided": True,
        "passed": passed,
        "result": result,
        "endpoint_metadata_sha256": canonical_hash,
        "archive_commitments_valid": archive_commitments_valid,
        "request_contract_recomputed": request_contract,
        "response_contract_recomputed": response_contract,
        "artifact_hashes": artifact_hashes,
    }


def _tokenizer_counts() -> dict[str, Any]:
    path = EXPERIMENT_FILES / "prompt-token-counts.json"
    if not path.is_file():
        return {"provided": False, "passed": False, "reason": "prompt token counts not generated"}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        return {"provided": True, "passed": False, "reason": "non-object token count artifact"}
    expected_hashes = {
        "enabled": _hash(EXPERIMENT_FILES / "reconstruction-enabled.txt"),
        "disabled": _hash(EXPERIMENT_FILES / "reconstruction-disabled.txt"),
    }
    expected_tokenizer = {
        "add_special_tokens": False,
        "model_repository": "Qwen/Qwen3.6-27B",
        "model_revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        "tokenizer_json_sha256": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
        "tokenizer_url": (
            "https://huggingface.co/Qwen/Qwen3.6-27B/resolve/"
            "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9/tokenizer.json"
        ),
    }
    expected_counts = {
        "enabled": {
            "bytes": 2848,
            "tokens": 515,
            "full_prompt_bytes": 4339,
            "full_prompt_tokens": 810,
            "full_prompt_sha256": "5d357591c193499e0c248b1c2acba57f31664894e91162da8ae06e828aa5032d",
        },
        "disabled": {
            "bytes": 3040,
            "tokens": 552,
            "full_prompt_bytes": 4531,
            "full_prompt_tokens": 847,
            "full_prompt_sha256": "1f112934f5f4f386401af5e9575608d04a9d499d0968b7fd15724fa3d4a58d06",
        },
    }
    records = value.get("conditions")
    passed = (
        value.get("schema") == "scientist-game-designer-player-prompt-token-counts-v1"
        and value.get("tokenizer") == expected_tokenizer
        and isinstance(records, Mapping)
        and all(
        isinstance(records.get(condition), Mapping)
        and records[condition].get("sha256") == prompt_hash
        and all(
            records[condition].get(field) == expected
            for field, expected in expected_counts[condition].items()
        )
        for condition, prompt_hash in expected_hashes.items()
        )
    )
    return {
        "provided": True,
        "passed": passed,
        "artifact_sha256": _hash(path),
        "artifact": value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compatibility-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or AGENT_ROOT / "artifacts" / "release-gates" / timestamp
    if output_dir.exists():
        raise SystemExit(f"release-gate output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    config = load_frozen_runtime()
    prompt_result = validate_prompt_pair(
        (EXPERIMENT_FILES / "reconstruction-enabled.txt").read_bytes(),
        (EXPERIMENT_FILES / "reconstruction-disabled.txt").read_bytes(),
    )
    request_options = frozen_request_options(config)
    static_contract = {
        "runtime_config_sha256": _hash(EXPERIMENT_FILES / "runtime-config.yaml"),
        "runtime_values": {
            "model": config.model,
            "endpoint": config.endpoint,
            "model_profile": config.model_profile,
            "request": request_options,
            "runtime": vars(config.runtime),
            "budgets": vars(config.budgets),
        },
        "prompt_validation": prompt_result,
        "schedule": [vars(pair) for pair in frozen_pair_schedule()],
    }
    synthetic = _run_synthetic_tests()
    sandbox = run_sandbox_admissibility_suite(
        timeout_seconds=float(config.runtime.workspace_timeout_seconds)
    )
    compatibility = _compatibility_result(args.compatibility_dir)
    token_counts = _tokenizer_counts()
    source_vector = {
        "model": "release-vector",
        "messages": [{
            "role": "tool",
            "tool_call_id": "release-vector-call",
            "content": PythonResult(ok=True, stdout="A", stderr="").model_text(),
        }],
        "seed": 1,
    }
    intact_vector, perturbed_vector = build_branch_payloads(
        source_vector,
        selected_result_content_path="/messages/0/content/stdout",
        intact_content="A",
        perturbed_content="B",
        seed=2,
    )
    shadow_executor_path = AGENT_ROOT / "run_shadow_intervention.py"
    intervention_vectors = {
        "schema": "reconstruction-content-intervention-test-vectors-v1",
        "schedule": list(build_shadow_schedule(candidate_id="release-vector", generation="release-vector")),
        "signatures": {
            "action": response_signature(
                {"content": '{"action":1,"data":null,"reason":"x"}'},
                decision_state={
                    "available_actions": [1, 2],
                    "frame_width": 8,
                    "frame_height": 6,
                },
            ),
            "continue": response_signature({
                "content": None,
                "tool_calls": [{
                    "function": {"name": "python", "arguments": '{"code":"print(1)"}'},
                }],
            }),
            "unidentifiable": response_signature({"content": "not an action"}),
        },
        "artifact_templates": experiment_artifact_templates(
            "enabled",
            evidence_hashes={"process.jsonl": "release-vector"},
        ),
        "branch_payloads": {
            "intact": intact_vector,
            "perturbed": perturbed_vector,
            "selected_result_content_path": "/messages/0/content/stdout",
        },
        "shadow_executor_sha256": (
            sha256(shadow_executor_path.read_bytes()).hexdigest()
            if shadow_executor_path.is_file()
            else None
        ),
    }
    _write_json(output_dir / "sandbox-results.json", sandbox)
    _write_json(output_dir / "synthetic-test-results.json", synthetic)
    _write_json(output_dir / "intervention-test-vectors.json", intervention_vectors)
    endpoint_metadata_source = args.compatibility_dir / "endpoint-metadata.canonical.json"
    endpoint_metadata_artifact = output_dir / "endpoint-metadata.json"
    if endpoint_metadata_source.is_file():
        endpoint_metadata_artifact.write_bytes(endpoint_metadata_source.read_bytes())

    source_hashes = participating_source_hashes()
    intervention_valid = (
        len(intervention_vectors["schedule"]) == 10
        and intervention_vectors["signatures"]
        == {
            "action": 'ACT:{"action":1,"data":null}',
            "continue": "CONTINUE",
            "unidentifiable": "UNIDENTIFIABLE",
        }
        and intervention_vectors["artifact_templates"]["annotation"]["authority"]
        == "exact_independent_agreement_without_adjudication"
        and intervention_vectors["artifact_templates"]["candidate"]["selector"][
            "selection_rule"
        ]
        == "earliest_eligible_candidate"
        and intervention_vectors["artifact_templates"]["candidate"][
            "second_candidate_permitted"
        ]
        is False
        and intervention_vectors["shadow_executor_sha256"]
        == source_hashes.get(shadow_executor_path.relative_to(REPO).as_posix())
        and json.loads(
            intervention_vectors["branch_payloads"]["intact"]["messages"][0]["content"]
        )["stdout"]
        == "A"
        and json.loads(
            intervention_vectors["branch_payloads"]["perturbed"]["messages"][0]["content"]
        )["stdout"]
        == "B"
    )
    checks = {
        "frozen_runtime_loaded": True,
        "authorized_prompt_diff": prompt_result["valid"],
        "pair_schedule_materialized": len(static_contract["schedule"]) == 4,
        "synthetic_tests": synthetic["passed"],
        "sandbox_admissibility": sandbox["passed"],
        "prompt_token_counts": token_counts["passed"],
        "compatibility_probe": compatibility["passed"],
        "intervention_implementation": intervention_valid,
        "participating_sources_hashed": all(
            path in source_hashes
            for path in (
                "pyproject.toml",
                "uv.lock",
                "experiments/agents/tests/test_falsifiable_experiment.py",
                "experiments/harness/tests/test_harness.py",
            )
        ),
        "endpoint_metadata_archived": endpoint_metadata_artifact.is_file(),
    }
    release_passed = all(checks.values())
    results = {
        "schema": RELEASE_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": release_passed,
        "pilot_authorized_by_gate": release_passed,
        "gameplay_or_environment_executed": False,
        "checks": checks,
        "static_contract": static_contract,
        "tokenizer_counts": token_counts,
        "compatibility": compatibility,
        "sandbox_summary": {
            "passed": sandbox["passed"],
            "test_inventory": sandbox["test_inventory"],
            "suite_source_sha256": sandbox["evidence"]["suite_source_sha256"],
        },
        "synthetic_summary": {
            "passed": synthetic["passed"],
            "tests_run": synthetic["tests_run"],
            "modules": synthetic["modules"],
        },
        "source_hashes": source_hashes,
        "artifacts": {
            "sandbox_results": "sandbox-results.json",
            "synthetic_test_results": "synthetic-test-results.json",
            "intervention_test_vectors": "intervention-test-vectors.json",
            "endpoint_metadata": "endpoint-metadata.json",
        },
    }
    _write_json(output_dir / "release-gate-results.json", results)
    candidate_manifest = {
        "schema": "scientist-game-designer-player-release-candidate-v1",
        "final_freeze": False,
        "reason": "the final freeze occurs only after an accepted excluded pilot",
        "release_gate_passed": release_passed,
        "release_gate_results_sha256": _hash(output_dir / "release-gate-results.json"),
        "participating_source_hashes": source_hashes,
        "generated_artifact_hashes": {
            path.name: _hash(path)
            for path in sorted(output_dir.glob("*.json"))
        },
    }
    _write_json(output_dir / "release-candidate-manifest.json", candidate_manifest)
    print(json.dumps({
        "passed": release_passed,
        "pilot_authorized_by_gate": release_passed,
        "checks": checks,
        "synthetic_tests_run": synthetic["tests_run"],
        "sandbox_tests_run": len(sandbox["tests"]),
        "output_dir": str(output_dir),
    }, indent=2, sort_keys=True))
    if not release_passed:
        raise SystemExit("release gate did not pass; inspect release-gate-results.json")


if __name__ == "__main__":
    main()
