"""Validated human review whose effects apply only to future runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .canonical import append_jsonl, content_sha256, read_jsonl, write_atomic_json
from .failure import verify_packet

DECISIONS = frozenset({"probe", "revise", "keep", "demote", "delete", "terminate"})


def validate_review(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"schema", "reviewer", "decision", "rationale", "next_run_changes"}:
        raise ValueError("review requires schema, reviewer, decision, rationale, and next_run_changes")
    if value["schema"] != "0158-review-v1" or value["decision"] not in DECISIONS:
        raise ValueError("unsupported review schema or decision")
    if value["reviewer"] != "Pranav":
        raise ValueError("initial promotion and demotion decisions require reviewer Pranav")
    if not isinstance(value["rationale"], str) or not value["rationale"].strip():
        raise ValueError("review rationale is required")
    if not isinstance(value["next_run_changes"], dict):
        raise TypeError("next_run_changes must be an object")
    return value


def apply_review(
    packet: Path,
    review: dict[str, Any],
    *,
    run_root: Path,
    mode: str,
) -> dict[str, Any]:
    if mode not in {"live", "offline"}:
        raise ValueError("review mode must be live or offline")
    verify_packet(packet)
    current = json.loads((packet / "review.yaml").read_text(encoding="utf-8"))
    if current.get("status") != "pending":
        raise RuntimeError("failure packet has already been reviewed")
    validated = validate_review(review)
    finalized = {
        **validated,
        "status": "final",
        "packet_id": packet.name,
        "mode": mode,
        "resume_scored_run": False,
    }
    finalized["review_digest"] = content_sha256(finalized)
    next_run = {
        "schema": "0158-next-run-v1",
        "source_packet_id": packet.name,
        "review_digest": finalized["review_digest"],
        "decision": finalized["decision"],
        "configuration": finalized["next_run_changes"],
        "requires_new_run_id": True,
        "resume_source_run": False,
    }
    next_run_path = packet / "next-run.json"
    if next_run_path.exists():
        if json.loads(next_run_path.read_text(encoding="utf-8")) != next_run:
            raise RuntimeError("existing next-run decision conflicts with this review")
    else:
        write_atomic_json(next_run_path, next_run)
    history_path = run_root / "promotion-history.jsonl"
    history = read_jsonl(history_path)
    matching = [item for item in history if item.get("review_digest") == finalized["review_digest"]]
    if len(matching) > 1:
        raise RuntimeError("review digest appears more than once in promotion history")
    if not matching:
        append_jsonl(
            history_path,
            {
                "event": "review_finalized",
                "packet_id": packet.name,
                "review_digest": finalized["review_digest"],
                "reviewer": finalized["reviewer"],
                "decision": finalized["decision"],
                "applies_to": "future_run_only" if mode == "live" else "offline_qualification",
            },
        )
    write_atomic_json(packet / "review.yaml", finalized)
    return finalized


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("review", type=Path, help="JSON-compatible YAML review file")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("live", "offline"), required=True)
    args = parser.parse_args()
    review = json.loads(args.review.read_text(encoding="utf-8"))
    print(json.dumps(apply_review(args.packet, review, run_root=args.run_root, mode=args.mode), sort_keys=True))


if __name__ == "__main__":
    main()
