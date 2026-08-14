"""Create one content-bound live approval from Pranav's explicit run authority."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256

import run_live_ls20 as base
from host.canonical import content_sha256
from run_candidate_ls20 import CANDIDATES, configure

APPROVAL_TEXT = "let's do the same 44 action run across models."


def _sha(path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), required=True)
    args = parser.parse_args()
    candidate = CANDIDATES[args.candidate]
    configure(candidate)
    qualification = json.loads(base.QUALIFICATION.read_text(encoding="utf-8"))
    if qualification.get("status") != "pass" or qualification.get("model") != candidate.model:
        raise RuntimeError("exact candidate has not passed the provider preflight")
    if base.APPROVAL.exists() or base.CONSUMPTION.exists():
        raise FileExistsError("candidate approval or consumption already exists")
    card_bytes = base.CARD.read_bytes()
    if len(card_bytes) > 1024:
        raise RuntimeError("live experiment card exceeds 1 KiB")
    manifest = base._candidate_manifest()
    approval = {
        "schema": "0158-live-approval-v1",
        "approval_text": APPROVAL_TEXT,
        "authorization_basis": (
            "User explicitly requested fresh matched 44-action runs across the three named models."
        ),
        "execution_authority": True,
        "approved_runs": 1,
        "approved_game": "ls20",
        "approved_model": candidate.model,
        "approved_provider_route": candidate.provider_route_id,
        "approved_action_cap": 44,
        "approved_codeact_iterations": base.MAX_CODEACT_ITERATIONS,
        "approved_codeact_errors": base.MAX_CODEACT_ERRORS,
        "approved_evidence_access_policy": base.EVIDENCE_ACCESS_POLICY,
        "approved_commitment_surface": base.COMMITMENT_SURFACE,
        "approved_submission_grammar": base.SUBMISSION_GRAMMAR,
        "approved_max_output_tokens": base.MAX_OUTPUT_TOKENS,
        "approved_transport_max_retries": base.TRANSPORT_MAX_RETRIES,
        "approved_model_call_cap": base.MAX_MODEL_CALLS,
        "approved_reported_token_cap": base.MAX_REPORTED_TOKENS,
        "approved_active_seconds_cap": base.MAX_ACTIVE_SECONDS,
        "approved_run_policy": base.RUN_POLICY["mode"],
        "approved_model_frame_window": base.MODEL_FRAME_WINDOW,
        "approved_environment_read_timeout_seconds": base.ENVIRONMENT_READ_TIMEOUT_SECONDS,
        "approved_environment_post_retries": 0,
        "approved_card_sha256": sha256(card_bytes).hexdigest(),
        "qualification_summary_sha256": _sha(base.QUALIFICATION),
        "approved_candidate_manifest_sha256": manifest["manifest_sha256"],
    }
    base.APPROVAL.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(approval, sort_keys=True, separators=(",", ":")) + "\n"
    with base.APPROVAL.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    print(
        json.dumps(
            {
                "approval": str(base.APPROVAL),
                "approval_sha256": content_sha256(approval),
                "candidate_manifest_sha256": manifest["manifest_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
