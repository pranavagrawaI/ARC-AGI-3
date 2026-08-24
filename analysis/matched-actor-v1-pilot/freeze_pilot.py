"""Regenerate the disabled pilot manifest after reviewed package changes."""

from __future__ import annotations

import json
from pathlib import Path

from tycho_nooa.matched_actor import CODEACT_SYSTEM_PROMPT, NATIVE_SYSTEM_PROMPT
from tycho_nooa.matched_actor_manifest import sha256_file, sha256_text

repo = Path(__file__).resolve().parents[2]
base = Path(__file__).parent
sample = json.loads((base / "sample.json").read_text())


def bind(path: str) -> dict[str, str]:
    return {"path": path, "sha256": sha256_file(repo / path)}


manifest = {
    "schema_version": 1,
    "experiment_id": "matched-actor-v1-pilot",
    "execution_authorized": False,
    "base_treatment": bind("analysis/matched-actor-v1/experiment.json"),
    "sample": bind("analysis/matched-actor-v1-pilot/sample.json"),
    "selection_driver": bind("analysis/matched-actor-v1-pilot/select_pilot.py"),
    "sample_sha256": sample["sample_sha256"],
    "runner": bind("solvers/tycho_nooa/src/tycho_nooa/matched_actor_pilot_run.py"),
    "launcher": bind("solvers/tycho_nooa/src/tycho_nooa/matched_actor_pilot_launch.py"),
    "authorization_schema": bind("solvers/tycho_nooa/src/tycho_nooa/matched_actor_pilot_authorization.py"),
    "durable_ledger": bind("solvers/tycho_nooa/src/tycho_nooa/compatibility_authorization.py"),
    "provider_wrapper": bind("solvers/tycho_nooa/src/tycho_nooa/compatibility_launch.py"),
    "protocol": bind("analysis/matched-actor-v1-pilot/PROTOCOL.md"),
    "experiment_card": bind("analysis/matched-actor-v1-pilot/EXPERIMENT-CARD.md"),
    "native_prompt_sha256": sha256_text(NATIVE_SYSTEM_PROMPT),
    "codeact_prompt_sha256": sha256_text(CODEACT_SYSTEM_PROMPT),
    "model_route": {
        "model": None,
        "reported_model": None,
        "provider": None,
        "reasoning_effort": None,
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
        "canonical or ARC mutation",
    ],
    "stop_conditions": [
        "12 sessions complete",
        "72 calls",
        "$3 observed or reserved",
        "120 seconds in one session",
        "integrity or provider failure",
    ],
    "output_root": "analysis/matched-actor-v1-pilot/runs",
}
(base / "experiment.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
