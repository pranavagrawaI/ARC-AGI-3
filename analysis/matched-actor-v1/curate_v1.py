"""Rebuild only the A/B/C selection overlay; source packets stay immutable."""

import hashlib
import json
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
base = Path(__file__).parent
source = repo / "analysis/codeact-cognitive-parity/v1/packets"
selections = [
    ("action-failure-illegal-bait", "computation_unnecessary", "The task input alone determines the only legal action."),
    ("objective-failure-insufficient", "computation_unnecessary", "Absence of terminal evidence settles insufficiency without calculation."),
    ("cross-level-failure-insufficient", "computation_unnecessary", "Only one recorded level settles insufficiency without transformation."),
    ("action-success-grounded", "computation_unnecessary", "The control asks for a legal grounded proposal, not a derived numeric fact."),
    ("remote-success-old-up", "computation_useful", "Programmatic filtering can locate the earliest matching action."),
    ("history-success-level1-noop", "computation_useful", "A deterministic before/after equality check supports the judgment."),
    ("history-failure-duplicate-click", "computation_useful", "Grid equality plus recorded action distinguishes history from appearance."),
    ("objective-success-level-boundary", "computation_useful", "Ordering the last transition and next-level frame supports the boundary claim."),
    ("remote-failure-composition", "composition", "Requires pagination, grid differencing, exact citations, and a complete result set."),
    ("discriminate-success-left-levels", "composition", "Requires selecting cross-level LEFT records, differencing, comparing, and citing."),
    ("discriminate-failure-variable-right", "composition", "Requires selecting multiple RIGHT records, differencing, falsifying, and citing."),
    ("cross-level-success-action-response", "composition", "Requires cross-level retrieval, numeric transformation, provenance, and abstraction."),
]
records = []
for packet_id, stratum, rationale in selections:
    path = source / f"{packet_id}.json"
    records.append({
        "packet_id": packet_id,
        "packet_path": str(path.relative_to(repo)),
        "packet_sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        "stratum": stratum,
        "selection_rationale": rationale,
    })
manifest = {
    "schema_version": 1,
    "corpus_id": "matched-actor-v1",
    "selection_rule": "Twelve reviewed real-checkpoint packets, four each where computation is unnecessary, useful, or must compose with retrieval and provenance; no synthetic compatibility gates.",
    "packet_count": len(records),
    "records": records,
    "corpus_sha256": "sha256:" + hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
}
(base / "corpus.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(manifest["corpus_sha256"])
