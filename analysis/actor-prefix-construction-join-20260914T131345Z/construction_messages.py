"""Constructor prompt packets from an Actor-prefix case. No model calls."""

from __future__ import annotations

import json
from typing import Any

from prefix_adapter import ActorPrefixCase

SYSTEM = """You are the model constructor, not the game-playing policy.

Objective: Construct the smallest model that supports the next useful decision, while preserving what is known and exposing what remains unresolved.

You must discover the work item from the visible evidence. The host is a checker, not an oracle. A mismatch does not identify its semantic cause.

Candidate Python ABI:
- initialize(observation) -> JSON state
- predict(state, action) -> nonempty list of branches
- assimilate(predicted_state, observation, verdict) -> JSON state

ABI observations contain grid (list of integer rows). The chronological visible prefix is accessible at /evidence/case.json. Finish with submit_candidate or request_evidence.
"""


def construction_messages(case: ActorPrefixCase) -> list[dict[str, str]]:
    user: dict[str, Any] = {
        "visible_case": case.compact_view(),
        "required_submission": {
            "work_item": {
                "decision": "decision blocked by the current model",
                "obstacle": "candidate explanation, not host-established diagnosis",
                "operation": "ground | contrast | close | compose | test",
                "claim": "narrow model component being constructed",
                "evidence_refs": "nonempty list of integers: visible step numbers only",
                "dependencies": "nonempty list of strings naming declared quantities and their suppliers",
                "completion_checks": "nonempty list of integers: visible step numbers to check",
                "remaining_scope": "unsupported conditions",
                "reasoning_need": "needs_computation | needs_observation | does_not_block_decision",
            },
            "candidate_source": "complete Python source implementing the ABI",
        },
    }
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(user, separators=(",", ":"))},
    ]
