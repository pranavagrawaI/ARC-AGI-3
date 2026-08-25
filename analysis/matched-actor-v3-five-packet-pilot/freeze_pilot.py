"""Regenerate the disabled five-packet pilot with congruent receipt semantics."""

from __future__ import annotations

import json
from pathlib import Path

from tycho_nooa.matched_actor_context_v2 import (
    CONTEXT_CODEACT_SYSTEM_PROMPT,
    CONTEXT_NATIVE_SYSTEM_PROMPT,
)
from tycho_nooa.matched_actor_manifest import sha256_file, sha256_text
from tycho_nooa.matched_actor_pilot_manifest_v3 import (
    nooa_request_preparation_sha256,
)

repo = Path(__file__).resolve().parents[2]
base = Path(__file__).resolve().parent
sample = json.loads((base / "sample.json").read_text())


def bind(path: str) -> dict[str, str]:
    return {"path": path, "sha256": sha256_file(repo / path)}


manifest = {
    "schema_version": 7,
    "experiment_id": "matched-actor-v3-five-packet-pilot",
    "execution_authorized": False,
    "base_contract_repair": bind(
        "analysis/matched-actor-contract-repair-v1/manifest.json"
    ),
    "base_receipt_pilot": bind(
        "analysis/matched-actor-v2-five-packet-pilot/experiment.json"
    ),
    "dependency_lock": bind("solvers/tycho_nooa/uv.lock"),
    "sample_schema": bind(
        "solvers/tycho_nooa/src/tycho_nooa/matched_actor_pilot_v2.py"
    ),
    "sample": bind("analysis/matched-actor-v3-five-packet-pilot/sample.json"),
    "sample_sha256": sample["sample_sha256"],
    "runner": bind("solvers/tycho_nooa/src/tycho_nooa/matched_actor_pilot_run_v7.py"),
    "launcher": bind(
        "solvers/tycho_nooa/src/tycho_nooa/matched_actor_pilot_launch_v7.py"
    ),
    "manifest_schema": bind(
        "solvers/tycho_nooa/src/tycho_nooa/matched_actor_pilot_manifest_v7.py"
    ),
    "authorization_schema": bind(
        "solvers/tycho_nooa/src/tycho_nooa/matched_actor_pilot_authorization_v3.py"
    ),
    "durable_ledger": bind(
        "solvers/tycho_nooa/src/tycho_nooa/compatibility_authorization_v2.py"
    ),
    "provider_wrapper": bind(
        "solvers/tycho_nooa/src/tycho_nooa/compatibility_launch_v2.py"
    ),
    "actor_runner": bind(
        "solvers/tycho_nooa/src/tycho_nooa/matched_actor_experiment_v7.py"
    ),
    "evidence_service": bind("solvers/tycho_nooa/src/tycho_nooa/evidence_v2.py"),
    "context_evidence": bind(
        "solvers/tycho_nooa/src/tycho_nooa/context_evidence_v2.py"
    ),
    "context_projection": bind(
        "solvers/tycho_nooa/src/tycho_nooa/context_projection.py"
    ),
    "context_computational_evidence": bind(
        "solvers/tycho_nooa/src/tycho_nooa/context_computational_evidence_v3.py"
    ),
    "context_computational_evidence_base": bind(
        "solvers/tycho_nooa/src/tycho_nooa/context_computational_evidence_v2.py"
    ),
    "context_execution": bind(
        "solvers/tycho_nooa/src/tycho_nooa/context_execution_v3.py"
    ),
    "context_execution_base": bind(
        "solvers/tycho_nooa/src/tycho_nooa/context_execution_v2.py"
    ),
    "context_worker": bind(
        "solvers/tycho_nooa/src/tycho_nooa/_context_computational_worker_v3.py"
    ),
    "context_worker_base": bind(
        "solvers/tycho_nooa/src/tycho_nooa/_context_computational_worker_v2.py"
    ),
    "context_controller": bind(
        "solvers/tycho_nooa/src/tycho_nooa/matched_actor_context_v4.py"
    ),
    "context_controller_base": bind(
        "solvers/tycho_nooa/src/tycho_nooa/matched_actor_context_v2.py"
    ),
    "acquisition_controller_base": bind(
        "solvers/tycho_nooa/src/tycho_nooa/matched_actor_context_v3.py"
    ),
    "audit_driver": bind(
        "solvers/tycho_nooa/src/tycho_nooa/matched_actor_context_audit_v3.py"
    ),
    "replay_driver": bind(
        "analysis/matched-actor-v3-five-packet-pilot/replay_five_packets.py"
    ),
    "replay_report": bind(
        "analysis/matched-actor-v3-five-packet-pilot/offline-replay.json"
    ),
    "replay_artifact_index": bind(
        "analysis/matched-actor-v3-five-packet-pilot/offline-replay-files.json"
    ),
    "contract_probe": bind("analysis/context-receipt-contract-v3/probe.py"),
    "contract_probe_report": bind("analysis/context-receipt-contract-v3/report.json"),
    "checkpoint_loader": bind(
        "solvers/tycho_nooa/src/tycho_nooa/repaired_checkpoint.py"
    ),
    "checkpoint_commitments": bind("analysis/matched-actor-v5-pilot/checkpoints.json"),
    "freeze_driver": bind(
        "analysis/matched-actor-v3-five-packet-pilot/freeze_pilot.py"
    ),
    "protocol": bind("analysis/matched-actor-v3-five-packet-pilot/PROTOCOL.md"),
    "experiment_card": bind(
        "analysis/matched-actor-v3-five-packet-pilot/EXPERIMENT-CARD.md"
    ),
    "native_prompt_sha256": sha256_text(CONTEXT_NATIVE_SYSTEM_PROMPT),
    "codeact_prompt_sha256": sha256_text(CONTEXT_CODEACT_SYSTEM_PROMPT),
    "nooa_request_preparation_sha256": nooa_request_preparation_sha256(),
    "model_route": {
        "model": "openai/deepseek/deepseek-v4-flash-0731",
        "reported_model": "deepseek/deepseek-v4-flash-0731",
        "provider": "DeepInfra",
        "reasoning_effort": "enabled",
        "temperature": None,
    },
    "caps": {
        "sessions": 10,
        "calls_per_session": 6,
        "calls_total": 60,
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
        "complete_pairs_required": 5,
    },
    "invalidity_conditions": [
        "committed input or repaired checkpoint drift",
        "remote-history stress packet enters the matched sample",
        "unequal treatment semantics or authority",
        "answer-key disclosure",
        "missing trace or usage",
        "provider route drift",
        "canonical or ARC mutation",
    ],
    "stop_conditions": [
        "10 sessions complete",
        "60 calls",
        "$3 actual, observed, or reserved",
        "120 seconds in one session",
        "integrity or provider failure",
    ],
    "output_root": "analysis/matched-actor-v3-five-packet-pilot/runs",
}
(base / "experiment.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
