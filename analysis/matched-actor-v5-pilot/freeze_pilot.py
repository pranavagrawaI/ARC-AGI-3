"""Regenerate the disabled checkpoint-repaired V5 pilot manifest."""

from __future__ import annotations

import json
from pathlib import Path

from tycho_nooa.matched_actor_context import (
    CONTEXT_CODEACT_SYSTEM_PROMPT,
    CONTEXT_NATIVE_SYSTEM_PROMPT,
)
from tycho_nooa.matched_actor_manifest import sha256_file, sha256_text
from tycho_nooa.matched_actor_pilot_manifest_v3 import (
    nooa_request_preparation_sha256,
)

repo = Path(__file__).resolve().parents[2]
base = Path(__file__).resolve().parent
sample_path = repo / "analysis/matched-actor-v1-pilot/sample.json"
sample = json.loads(sample_path.read_text())


def bind(path: str) -> dict[str, str]:
    return {"path": path, "sha256": sha256_file(repo / path)}


manifest = {
    "schema_version": 5,
    "experiment_id": "matched-actor-v5-pilot",
    "execution_authorized": False,
    "context_plane_base_revision": "9747693",
    "manifest_schema": bind(
        "solvers/tycho_nooa/src/tycho_nooa/matched_actor_pilot_manifest_v5.py"
    ),
    "dependency_lock": bind("solvers/tycho_nooa/uv.lock"),
    "nooa_request_preparation_sha256": nooa_request_preparation_sha256(),
    "base_treatment": bind("analysis/matched-actor-v1/experiment.json"),
    "base_v3_package": bind("analysis/matched-actor-v3-pilot/experiment.json"),
    "base_v4_package": bind("analysis/matched-actor-v4-pilot/experiment.json"),
    "sample": bind("analysis/matched-actor-v1-pilot/sample.json"),
    "selection_driver": bind("analysis/matched-actor-v1-pilot/select_pilot.py"),
    "sample_sha256": sample["sample_sha256"],
    "runner": bind("solvers/tycho_nooa/src/tycho_nooa/matched_actor_pilot_run_v4.py"),
    "launcher": bind(
        "solvers/tycho_nooa/src/tycho_nooa/matched_actor_pilot_launch_v5.py"
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
        "solvers/tycho_nooa/src/tycho_nooa/matched_actor_experiment_v4.py"
    ),
    "evidence_service": bind("solvers/tycho_nooa/src/tycho_nooa/evidence_v2.py"),
    "context_evidence": bind("solvers/tycho_nooa/src/tycho_nooa/context_evidence.py"),
    "context_projection": bind(
        "solvers/tycho_nooa/src/tycho_nooa/context_projection.py"
    ),
    "context_execution": bind(
        "solvers/tycho_nooa/src/tycho_nooa/context_execution.py"
    ),
    "context_controller": bind(
        "solvers/tycho_nooa/src/tycho_nooa/matched_actor_context.py"
    ),
    "audit_driver": bind(
        "solvers/tycho_nooa/src/tycho_nooa/matched_actor_context_audit.py"
    ),
    "checkpoint_loader": bind(
        "solvers/tycho_nooa/src/tycho_nooa/repaired_checkpoint.py"
    ),
    "checkpoint_commitments": bind(
        "analysis/matched-actor-v5-pilot/checkpoints.json"
    ),
    "freeze_driver": bind("analysis/matched-actor-v5-pilot/freeze_pilot.py"),
    "replay_driver": bind(
        "analysis/matched-actor-v5-pilot/replay_six_packets.py"
    ),
    "replay_report": bind("analysis/matched-actor-v5-pilot/offline-replay.json"),
    "replay_artifact_index": bind(
        "analysis/matched-actor-v5-pilot/offline-replay-files.json"
    ),
    "protocol": bind("analysis/matched-actor-v5-pilot/PROTOCOL.md"),
    "experiment_card": bind("analysis/matched-actor-v5-pilot/EXPERIMENT-CARD.md"),
    "native_prompt_sha256": sha256_text(CONTEXT_NATIVE_SYSTEM_PROMPT),
    "codeact_prompt_sha256": sha256_text(CONTEXT_CODEACT_SYSTEM_PROMPT),
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
        "committed input or repaired checkpoint drift",
        "unequal treatment semantics or authority",
        "answer-key disclosure",
        "missing trace or usage",
        "provider route drift",
        "context/evidence plane drift",
        "canonical or ARC mutation",
    ],
    "stop_conditions": [
        "12 sessions complete",
        "72 calls",
        "$3 actual, observed, or reserved",
        "120 seconds in one session",
        "integrity or provider failure",
    ],
    "output_root": "analysis/matched-actor-v5-pilot/runs",
}
(base / "experiment.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
