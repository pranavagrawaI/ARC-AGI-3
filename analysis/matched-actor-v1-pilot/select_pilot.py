"""Derive the pilot from named roles in the frozen V1 corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
base = Path(__file__).parent
source_path = repo / "analysis/matched-actor-v1/corpus.json"
source = json.loads(source_path.read_text())

# Declared before paid model behavior: two contrastive roles per stratum.
selection = [
    ("action-success-grounded", "easy direct action"),
    ("action-failure-illegal-bait", "historically failure-prone action"),
    ("history-success-level1-noop", "deterministic equality simplifies judgment"),
    ("objective-success-level-boundary", "direct ordering may suffice"),
    ("cross-level-success-action-response", "successful cross-level composition"),
    ("remote-failure-composition", "historically failure-prone remote composition"),
]
by_id = {record["packet_id"]: record for record in source["records"]}
packets = []
for index, (packet_id, role) in enumerate(selection):
    record = by_id[packet_id]
    packets.append(
        {
            "packet_id": packet_id,
            "stratum": record["stratum"],
            "selection_role": role,
            "arm_order": (
                ["native", "codeact"]
                if index % 2 == 0
                else ["codeact", "native"]
            ),
        }
    )
value = {
    "schema_version": 1,
    "pilot_id": "matched-actor-v1-pilot",
    "source_corpus": {
        "path": "analysis/matched-actor-v1/corpus.json",
        "sha256": "sha256:" + hashlib.sha256(source_path.read_bytes()).hexdigest(),
    },
    "source_corpus_sha256": source["corpus_sha256"],
    "selection_rule": (
        "Within each frozen stratum select two predeclared contrasting roles: "
        "easy versus historical failure for A, deterministic-computation versus "
        "direct-ordering for B, and successful cross-level versus historical "
        "remote failure for C; alternate arm order in listed packet order."
    ),
    "packets": packets,
    "sample_sha256": "sha256:"
    + hashlib.sha256(
        json.dumps(packets, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest(),
}
(base / "sample.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
