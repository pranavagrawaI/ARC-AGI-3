"""Run the zero-cost matched-world audit into a new evidence directory."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from tycho_nooa.matched_actor_audit import audit_matched_actor_packet
from tycho_nooa.matched_actor_corpus import load_matched_actor_corpus

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = Path(__file__).with_name("corpus.json")


async def run(destination: Path) -> None:
    manifest, packets = load_matched_actor_corpus(
        CORPUS_PATH, repository_root=REPOSITORY_ROOT
    )
    destination.mkdir(parents=True, exist_ok=False)
    audits = [
        await audit_matched_actor_packet(
            packet,
            repository_root=REPOSITORY_ROOT,
            destination=destination / packet.task.packet_id,
        )
        for packet in packets
    ]
    summary = {
        "schema_version": 1,
        "corpus_id": manifest.corpus_id,
        "packet_count": len(audits),
        "all_evidence_disclosures_identical": all(
            value.evidence_disclosures_identical for value in audits
        ),
        "all_computation_stdout_identical": all(
            value.computation_stdout_identical for value in audits
        ),
        "all_native_answers_correct": all(value.native_answer_correct for value in audits),
        "all_codeact_answers_correct": all(
            value.codeact_answer_correct for value in audits
        ),
        "authority": {
            "native_python_evidence": any(
                value.native_python_has_evidence_authority for value in audits
            ),
            "canonical_writes": any(value.canonical_write_authority for value in audits),
            "arc_actions": any(value.arc_action_authority for value in audits),
        },
        "packets": [value.model_dump(mode="json") for value in audits],
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    asyncio.run(run(args.destination))


if __name__ == "__main__":
    main()
