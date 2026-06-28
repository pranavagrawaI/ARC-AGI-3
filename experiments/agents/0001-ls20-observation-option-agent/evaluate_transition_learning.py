"""Prequentially evaluate generic transition prediction on neutral events."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
AGENT_ROOT = Path(__file__).resolve().parent
for path in (REPO, AGENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from observation_option_agent import (
    PrequentialTransitionLearner,
    ProbePolicy,
    RelationEffectLearner,
    ReplayInterventionAssay,
    contact_effect_symbols_from_record,
    transition_symbols_from_record,
)


DEFAULT_EVENTS = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "human-trace-events.jsonl"
)
DEFAULT_OUTPUT = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "transition-learning-summary.json"
)


def _load_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _result_record(result) -> dict[str, object]:
    return {
        "examples": result.examples,
        "non_empty_examples": result.non_empty_examples,
        "mean_observed_probability": round(result.mean_observed_probability, 6),
        "top1_accuracy": round(result.top1_accuracy, 6),
        "average_symbol_log_loss": round(result.average_symbol_log_loss, 6),
    }


def _relation_hypothesis_record(hypothesis) -> dict[str, object]:
    return {
        "contact_signature": [
            {
                "value": value,
                "area": area,
                "shape_id": shape_id,
            }
            for value, area, shape_id in hypothesis.contact_signature
        ],
        "support": hypothesis.support,
        "top_effects": [
            {"effect": effect, "rate": rate, "lift": lift}
            for effect, rate, lift in hypothesis.top_effects
        ],
    }


def _probe_record(proposal, result) -> dict[str, object]:
    return {
        "contact_signature": [
            {
                "value": value,
                "area": area,
                "shape_id": shape_id,
            }
            for value, area, shape_id in proposal.contact_signature
        ],
        "predicted_effect": proposal.predicted_effect,
        "support": proposal.support,
        "effect_rate": proposal.effect_rate,
        "lift": proposal.lift,
        "positive_attempts": result.positive_attempts,
        "positive_successes": result.positive_successes,
        "negative_controls": result.negative_controls,
        "negative_control_violations": result.negative_control_violations,
        "reproduced": result.reproduced,
        "negative_control_passed": result.negative_control_passed,
    }


def _evaluate_models(records, *, symbol_extractor) -> dict[str, object]:
    learners = {
        "global": PrequentialTransitionLearner(
            contexts=("global",),
            symbol_extractor=symbol_extractor,
        ),
        "action": PrequentialTransitionLearner(
            contexts=("action", "global"),
            symbol_extractor=symbol_extractor,
        ),
        "action_contact": PrequentialTransitionLearner(
            contexts=("action_contact", "action", "global"),
            symbol_extractor=symbol_extractor,
        ),
    }
    results = {
        name: _result_record(learner.score_records(records))
        for name, learner in learners.items()
    }
    return {
        "models": results,
        "deltas_vs_global": {
            name: {
                "mean_observed_probability": round(
                    result["mean_observed_probability"]
                    - results["global"]["mean_observed_probability"],
                    6,
                ),
                "top1_accuracy": round(
                    result["top1_accuracy"]
                    - results["global"]["top1_accuracy"],
                    6,
                ),
                "average_symbol_log_loss": round(
                    result["average_symbol_log_loss"]
                    - results["global"]["average_symbol_log_loss"],
                    6,
                ),
            }
            for name, result in results.items()
            if name != "global"
        },
    }


def evaluate(events_path: Path) -> dict[str, object]:
    records = _load_records(events_path)
    contact_records = [
        record
        for record in records
        if (record.get("interaction") or {}).get("contact_candidates")
    ]
    relation_learner = RelationEffectLearner()
    relation_result = relation_learner.score_records(contact_records)
    relation_hypotheses = relation_learner.hypotheses(minimum_support=3)
    split_index = int(len(contact_records) * 2 / 3)
    train_contacts = contact_records[:split_index]
    holdout_contacts = contact_records[split_index:]
    probe_learner = RelationEffectLearner()
    for record in train_contacts:
        probe_learner.observe(record)
    probe_policy = ProbePolicy(minimum_support=3, minimum_lift=1.5)
    probe_proposals = probe_policy.propose(
        probe_learner.hypotheses(minimum_support=3)
    )
    assay = ReplayInterventionAssay()
    probe_results = [
        (proposal, assay.evaluate(proposal, holdout_contacts))
        for proposal in probe_proposals[:12]
    ]
    return {
        "event_artifact": str(events_path),
        "protocol": "prequential: predict current transition before observing it",
        "broad_visual_symbols": _evaluate_models(
            records,
            symbol_extractor=transition_symbols_from_record,
        ),
        "contact_effect_symbols": _evaluate_models(
            contact_records,
            symbol_extractor=contact_effect_symbols_from_record,
        ) | {
            "examples": len(contact_records),
            "target": "generic effects on non-controlled objects during contact",
        },
        "relation_effects": {
            "target": "contact signature to generic effect distribution",
            "score": _result_record(relation_result),
            "hypothesis_count": len(relation_hypotheses),
            "top_hypotheses": [
                _relation_hypothesis_record(item)
                for item in relation_hypotheses[:12]
            ],
        },
        "replay_probe_assay": {
            "protocol": (
                "learn proposals on first two-thirds of contact records; "
                "test reproduction and negative controls on final third"
            ),
            "train_contact_examples": len(train_contacts),
            "holdout_contact_examples": len(holdout_contacts),
            "proposal_count": len(probe_proposals),
            "top_results": [
                _probe_record(proposal, result)
                for proposal, result in probe_results
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = evaluate(args.events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
