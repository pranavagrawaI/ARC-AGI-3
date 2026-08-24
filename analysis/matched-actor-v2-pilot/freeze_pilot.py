"""Regenerate the disabled resource-safe V2 pilot manifest."""

from __future__ import annotations

import json
from pathlib import Path

from tycho_nooa.matched_actor import CODEACT_SYSTEM_PROMPT, NATIVE_SYSTEM_PROMPT
from tycho_nooa.matched_actor_manifest import sha256_file, sha256_text
from tycho_nooa.matched_actor_pilot_manifest_v2 import (
    nooa_request_preparation_sha256,
)

repo = Path(__file__).resolve().parents[2]
base = Path(__file__).parent
sample_path = repo / "analysis/matched-actor-v1-pilot/sample.json"
sample = json.loads(sample_path.read_text())


def bind(path: str) -> dict[str, str]:
    return {"path": path, "sha256": sha256_file(repo / path)}


manifest = {
    "schema_version": 2,
    "experiment_id": "matched-actor-v2-pilot",
    "execution_authorized": False,
    "manifest_schema": bind(
        "solvers/tycho_nooa/src/tycho_nooa/matched_actor_pilot_manifest_v2.py"
    ),
    "dependency_lock": bind("solvers/tycho_nooa/uv.lock"),
    "nooa_request_preparation_sha256": nooa_request_preparation_sha256(),
    "base_treatment": bind("analysis/matched-actor-v1/experiment.json"),
    "sample": bind("analysis/matched-actor-v1-pilot/sample.json"),
    "selection_driver": bind("analysis/matched-actor-v1-pilot/select_pilot.py"),
    "sample_sha256": sample["sample_sha256"],
    "runner": bind("solvers/tycho_nooa/src/tycho_nooa/matched_actor_pilot_run_v2.py"),
    "launcher": bind(
        "solvers/tycho_nooa/src/tycho_nooa/matched_actor_pilot_launch_v2.py"
    ),
    "authorization_schema": bind(
        "solvers/tycho_nooa/src/tycho_nooa/matched_actor_pilot_authorization_v2.py"
    ),
    "durable_ledger": bind(
        "solvers/tycho_nooa/src/tycho_nooa/compatibility_authorization_v2.py"
    ),
    "provider_wrapper": bind(
        "solvers/tycho_nooa/src/tycho_nooa/compatibility_launch_v2.py"
    ),
    "actor_runner": bind(
        "solvers/tycho_nooa/src/tycho_nooa/matched_actor_experiment_v2.py"
    ),
    "provider_wrapper_base": bind(
        "solvers/tycho_nooa/src/tycho_nooa/compatibility_launch.py"
    ),
    "pilot_runner_base": bind(
        "solvers/tycho_nooa/src/tycho_nooa/matched_actor_pilot_run.py"
    ),
    "pilot_launcher_base": bind(
        "solvers/tycho_nooa/src/tycho_nooa/matched_actor_pilot_launch.py"
    ),
    "evidence_service": bind("solvers/tycho_nooa/src/tycho_nooa/evidence_v2.py"),
    "request_admission": bind(
        "solvers/tycho_nooa/src/tycho_nooa/compatibility_authorization_v2.py"
    ),
    "replay_driver": bind("analysis/matched-actor-v2-pilot/replay_call_40.py"),
    "replay_report": bind("analysis/matched-actor-v2-pilot/call-40-replay.json"),
    "after_action": bind("analysis/matched-actor-v2-pilot/AFTER-ACTION.md"),
    "protocol": bind("analysis/matched-actor-v2-pilot/PROTOCOL.md"),
    "experiment_card": bind("analysis/matched-actor-v2-pilot/EXPERIMENT-CARD.md"),
    "native_prompt_sha256": sha256_text(NATIVE_SYSTEM_PROMPT),
    "codeact_prompt_sha256": sha256_text(CODEACT_SYSTEM_PROMPT),
    "model_route": {
        "model": "openai/deepseek/deepseek-v4-flash-0731",
        "reported_model": "deepseek/deepseek-v4-flash-0731",
        "provider": "DeepInfra",
        "reasoning_effort": "enabled",
        "temperature": None,
    },
    "caps": {
        "sessions": 12,
        "calls_per_session": 6,
        "calls_total": 72,
        "input_tokens_per_call": 16384,
        "output_tokens_per_call": 4096,
        "wall_seconds_per_session": 120.0,
        "max_cost_usd": 3.0,
    },
    "advancement": {
        "incumbent": "native",
        "codeact_minimum_packet_wins": 2,
        "codeact_losses_may_not_exceed_wins": True,
        "equal_correctness_resource_reduction_percent": 20,
        "full_treatment_requires_complete_pairs": 6,
    },
    "invalidity_conditions": [
        "committed input drift",
        "unequal treatment semantics or authority",
        "answer-key disclosure",
        "missing trace or usage",
        "provider route drift",
        "evidence payload or request admission drift",
        "canonical or ARC mutation",
    ],
    "stop_conditions": [
        "12 sessions complete",
        "72 calls",
        "$3 actual, observed, or reserved",
        "120 seconds in one session",
        "integrity or provider failure",
    ],
    "output_root": "analysis/matched-actor-v2-pilot/runs",
}
(base / "experiment.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
