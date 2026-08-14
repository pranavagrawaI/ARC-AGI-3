"""Paid one-artifact qualification without contacting the ARC environment."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import run_live_ls20 as base
from host.canonical import content_sha256, write_atomic_json
from host.ipc import ArtifactBridge
from host.protocol import grounding_claim_from_draft
from run_candidate_ls20 import CANDIDATES, configure

HISTORICAL_REQUESTS = (
    base.EXPERIMENT
    / "artifacts/live-ls20/20260813T155603Z-qwen36-provisional-heuristics-44"
    / "worker/ipc/requests.jsonl"
)
PREFLIGHT_CARDS = {
    "deepseek-v4-flash": base.REPO / "experiments/cards/0158-deepseek-v4-flash-coreweave-preflight-v32.md",
    "qwen37-flash": base.REPO / "experiments/cards/0158-qwen37-flash-preflight-v32.md",
    "muse-glimmer-30b": base.REPO / "experiments/cards/0158-muse-glimmer-30b-preflight-v32.md",
}


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), required=True)
    args = parser.parse_args()
    candidate = CANDIDATES[args.candidate]
    configure(candidate)
    key = os.environ.get("ARC_LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("ARC_LLM_API_KEY or OPENROUTER_API_KEY is required")
    card = PREFLIGHT_CARDS[candidate.key]
    if len(card.read_bytes()) > 1024:
        raise RuntimeError("preflight experiment card exceeds 1 KiB")
    historical = json.loads(HISTORICAL_REQUESTS.read_text(encoding="utf-8").splitlines()[0])
    artifact_input = historical["artifact_input"]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_root = base.EXPERIMENT / "artifacts/model-preflight" / f"{stamp}-{candidate.key}"
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "EXPERIMENT-CARD.md").write_bytes(card.read_bytes())
    bridge = ArtifactBridge(
        runtime_root=base.RUNTIME,
        run_root=run_root / "worker",
        provider_config=base._provider_config(),
        worker_env=base._worker_environment(key),
        timeout_seconds=900.0,
        run_policy=base.RUN_POLICY,
    )
    claim = bridge.request(
        phase=artifact_input["phase"],
        observation_id=artifact_input["current_observation_id"],
        evidence=artifact_input["evidence"],
        legal_actions=artifact_input["legal_actions"],
        accepted_claim=artifact_input.get("accepted_claim"),
    )
    claim_response = bridge.latest_response
    claim_record = grounding_claim_from_draft(
        claim,
        parent=None,
        latest_probe_result=None,
    ).to_record()
    commitment = bridge.request(
        phase="commitment",
        observation_id=artifact_input["current_observation_id"],
        evidence=artifact_input["evidence"],
        legal_actions=artifact_input["legal_actions"],
        accepted_claim=claim_record,
    )
    commitment_response = bridge.latest_response
    responses = [claim_response, commitment_response]
    if not all(isinstance(response, dict) for response in responses):
        raise RuntimeError("preflight worker returned no response audit")
    audits = [response.get("audit") for response in responses]  # type: ignore[union-attr]
    if not all(isinstance(audit, dict) for audit in audits):
        raise RuntimeError("preflight response has no audit")
    model_calls = [call for audit in audits for call in audit.get("model_calls", [])]
    usage = [record for audit in audits for record in audit.get("usage", [])]
    result = {
        "schema": "0158-model-candidate-preflight-v1",
        "candidate": candidate.key,
        "model": candidate.model,
        "provider_route": candidate.provider_route,
        "historical_request_sha256": historical["request_sha256"],
        "preflight_card_sha256": _sha(card),
        "claim_sha256": content_sha256(claim),
        "commitment_sha256": content_sha256(commitment),
        "audit_sha256": content_sha256(audits),
        "model_calls": len(model_calls),
        "usage": usage,
        "status": "pass",
        "run_root": str(run_root),
    }
    write_atomic_json(run_root / "result.json", result)
    qualification = {
        **result,
        "schema": "0158-model-candidate-qualification-v1",
        "substrate_qualification_sha256": _sha(
            base.EXPERIMENT / "qualification-interface-v32.json"
        ),
    }
    write_atomic_json(base.QUALIFICATION, qualification)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
