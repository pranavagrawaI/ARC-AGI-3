"""Summarize the crude micro-pair without pretending process judgments are automatic."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def _records(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if isinstance(value, dict):
            values.append(value)
    return values


def _signature(record: Mapping[str, Any]) -> str:
    return json.dumps(
        [record.get("action"), record.get("action_data")],
        separators=(",", ":"),
        sort_keys=True,
    )


def _condition_summary(root: Path, condition: str) -> dict[str, Any]:
    member = root / condition
    manifest = json.loads((member / "manifest.json").read_text(encoding="utf-8"))
    records = _records(member / "process.jsonl")
    decisions = [record for record in records if record.get("type") == "decision"]
    transitions = [record for record in records if record.get("type") == "transition"]
    signatures = [_signature(record) for record in transitions]
    repeated_unchanged = 0
    for previous, current in zip(transitions, transitions[1:]):
        if (
            _signature(previous) == _signature(current)
            and current.get("changed_pixels") == 0
        ):
            repeated_unchanged += 1

    structured = []
    for record in decisions:
        reasoning = record.get("reasoning")
        if isinstance(reasoning, Mapping) and reasoning.get("policy") == "crude_0006_potency_prototype":
            structured.append(reasoning)
    hypothesis_sets = [
        {
            item.get("claim")
            for item in reasoning.get("hypotheses", [])
            if isinstance(item, Mapping) and isinstance(item.get("claim"), str)
        }
        for reasoning in structured
    ]
    retained_claim_links = sum(
        len(before & after)
        for before, after in zip(hypothesis_sets, hypothesis_sets[1:])
    )
    return {
        "condition": condition,
        "run_dir": str(member),
        "status": manifest.get("status"),
        "actions": manifest.get("actions"),
        "levels_completed": manifest.get("levels_completed"),
        "state": manifest.get("state"),
        "stop_reason": manifest.get("stop_reason"),
        "usage": manifest.get("usage"),
        "model_calls": manifest.get("model_calls"),
        "python_calls": manifest.get("python_calls"),
        "reset_count": manifest.get("reset_count"),
        "game_over_count": manifest.get("game_over_count"),
        "mechanical_process_diagnostics": {
            "transition_count": len(transitions),
            "unchanged_transition_count": sum(
                record.get("changed_pixels") == 0 for record in transitions
            ),
            "repeated_identical_action_after_unchanged_count": repeated_unchanged,
            "distinct_action_data_count": len(set(signatures)),
            "action_frequency": dict(sorted(Counter(signatures).items())),
            "structured_commitment_count": len(structured),
            "prediction_coverage": (
                len(structured) / len(decisions) if decisions else None
            ),
            "successive_exact_claim_retention_links": retained_claim_links,
        },
    }


def _review_template(summaries: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "crude-0006-human-process-review-v0",
        "warning": (
            "The intervention is structurally visible, so this review is not blind. Cite exact "
            "process-event steps. Mechanical diagnostics are proxies, not causal attributions."
        ),
        "runs": {
            str(summary["condition"]): {
                "process_log": str(Path(str(summary["run_dir"])) / "process.jsonl"),
                "remembered_earlier_evidence": {
                    "rating_0_to_3": None,
                    "cited_steps": [],
                    "notes": "",
                },
                "reconciled_contradictions": {
                    "rating_0_to_3": None,
                    "cited_steps": [],
                    "notes": "",
                },
                "selected_informative_probes": {
                    "rating_0_to_3": None,
                    "cited_steps": [],
                    "notes": "",
                },
                "avoidable_actions": {
                    "count": None,
                    "cited_steps": [],
                    "counterfactual": "",
                },
            }
            for summary in summaries
        },
        "comparative_verdict": {
            "behavioral_signal": "unset: none | weak | strong | mixed",
            "which_condition": "unset: baseline | treatment | neither",
            "central_observation": "",
            "next_discriminating_action": "",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", type=Path, required=True)
    args = parser.parse_args()
    summaries = [
        _condition_summary(args.pair_root, condition)
        for condition in ("baseline", "treatment")
    ]
    report = {
        "schema": "crude-0006-micro-pair-analysis-v0",
        "claim_limit": (
            "One micro-pair can reveal a behavioral signal worth pursuing; it cannot establish "
            "a package effect or identify which mechanism caused any difference."
        ),
        "runs": summaries,
    }
    (args.pair_root / "analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.pair_root / "human-review.json").write_text(
        json.dumps(_review_template(summaries), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "analysis_complete",
        "analysis": str(args.pair_root / "analysis.json"),
        "human_review": str(args.pair_root / "human-review.json"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
