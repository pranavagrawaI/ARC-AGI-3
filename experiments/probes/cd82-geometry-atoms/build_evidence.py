"""Extract integrity and mechanical selections from a completed geometry bundle."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "artifacts" / "bundle-v1"
OUTPUT = ROOT / "evidence.json"
REVIEW = ROOT / "semantic-review.json"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> int:
    result = load(BUNDLE / "result.json")
    review = load(REVIEW) if REVIEW.exists() else None
    failures = [
        relative
        for relative, expected in result["evidence_hashes"].items()
        if digest(BUNDLE / relative) != expected
    ]
    records = []
    semantic_keys = (
        "geometry_used",
        "global_translation_rule_withdrawn",
        "nonrigid_transformation_considered",
        "action_semantic_hypothesis_reopened",
        "discriminating_action_semantics_probe",
        "target_navigation_preserved",
        "notes",
    )
    for record in result["records"]:
        semantic = {key: None for key in semantic_keys}
        if review is not None:
            reviewed = review["conditions"].get(record["condition"])
            if not isinstance(reviewed, dict):
                raise ValueError(f"missing review: {record['condition']}")
            if reviewed.get("response_sha256") != record["response_sha256"]:
                raise ValueError(f"review response drift: {record['condition']}")
            semantic = {key: reviewed.get(key) for key in semantic_keys}
        records.append(
            {
                "condition": record["condition"],
                "information": record["information"],
                "selection": record["selection"],
                "usage": record["usage"],
                "response_metadata": record["response_metadata"],
                "semantic_review": semantic,
            }
        )
    if failures:
        raise RuntimeError(f"bundle evidence hash failures: {failures}")
    evidence = {
        "schema": "cd82-geometry-atoms-evidence-v1",
        "status": "complete_with_human_review" if review else "mechanically_complete_pending_human_review",
        "source_payload_sha256": result["source_payload_sha256"],
        "public_measurement": result["public_measurement"],
        "model_calls": result["model_calls"],
        "environment_actions": result["environment_actions"],
        "records": records,
        "integrity_failures": failures,
        "semantic_review_sha256": digest(REVIEW) if review else None,
        "semantic_verdict": review.get("semantic_verdict") if review else None,
        "licensed_next_move": review.get("licensed_next_move") if review else None,
    }
    OUTPUT.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
