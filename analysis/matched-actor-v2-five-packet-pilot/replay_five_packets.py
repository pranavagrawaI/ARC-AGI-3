"""Generate the zero-provider five-packet reachability proof."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from pathlib import Path

from tycho_nooa.matched_actor_context_audit_v2 import audit_context_actor_packet
from tycho_nooa.matched_actor_pilot_v2 import load_matched_actor_pilot_sample_v2

repo = Path(__file__).resolve().parents[2]
base = Path(__file__).resolve().parent
output = base / "offline-replay"
summary_path = base / "offline-replay.json"
index_path = base / "offline-replay-files.json"


async def main() -> None:
    if output.exists():
        shutil.rmtree(output)
    sample, packets = load_matched_actor_pilot_sample_v2(
        base / "sample.json", repository_root=repo
    )
    audits = []
    for scheduled, packet in zip(sample.packets, packets, strict=True):
        audit = await audit_context_actor_packet(
            packet,
            stratum=scheduled.stratum,
            repository_root=repo,
            destination=output / scheduled.packet_id,
        )
        audits.append(audit.model_dump(mode="json"))
    checks = {
        field: all(record[field] is expected for record in audits)
        for field, expected in {
            "evidence_disclosures_identical": True,
            "computation_stdout_identical": True,
            "native_answer_correct": True,
            "codeact_answer_correct": True,
            "bulk_evidence_absent_from_provider_context": True,
            "immutable_handle_semantics_shared": True,
            "native_python_has_evidence_acquisition_authority": False,
            "canonical_write_authority": False,
            "arc_action_authority": False,
        }.items()
    }
    summary = {
        "schema_version": 1,
        "experiment_id": "matched-actor-v2-five-packet-pilot-offline-replay",
        "provider_calls": 0,
        "arc_actions": 0,
        "packet_count": len(audits),
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "audits": audits,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    files = [
        {
            "path": str(path.relative_to(repo)),
            "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file()
    ]
    index_path.write_text(
        json.dumps({"schema_version": 1, "files": files}, indent=2, sort_keys=True)
        + "\n"
    )
    if not summary["all_checks_passed"]:
        raise SystemExit("Matched Actor V2 offline replay failed")


if __name__ == "__main__":
    asyncio.run(main())
