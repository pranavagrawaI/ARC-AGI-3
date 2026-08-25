"""Prove the repaired receipt contract through the real sterile worker and provider projection."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from tycho_nooa.context_execution_v3 import (
    ContextComputationalSterileExecutor,
    ContextExecutionManifest,
)
from tycho_nooa.context_projection import ContextProjector
from tycho_nooa.execution_events import ExecutionProvenanceLog, InvocationProvenance
from tycho_nooa.native_parity_v2 import NativeEvidenceArguments

base = Path(__file__).resolve().parent
trace = base / "probe-events.jsonl"
report_path = base / "report.json"


def receipt() -> dict:
    return {
        "schema_version": "evidence_receipt.v2",
        "handle": "ev_fixed",
        "content_hash": "sha256:" + "a" * 64,
        "query_hash": "sha256:" + "b" * 64,
        "refs": ["L0:T1", "L0:T2"],
        "record_count": 2,
        "record_fields": ["action", "ref"],
        "encoding": "rle",
        "frontier": {"level": 1, "turn": 2},
        "next_after": None,
        "truncated": False,
        "truncation_reason": None,
        "serialized_bytes": 512,
    }


def query_contract() -> dict:
    return {
        "schema_version": "evidence_query.v1",
        "parameters": NativeEvidenceArguments.model_json_schema(),
        "example": {
            "kind": "transitions",
            "level": 0,
            "limit": 16,
            "projection": "metadata",
        },
        "result_schema": "evidence_receipt.v2",
    }


async def main() -> None:
    trace.unlink(missing_ok=True)
    log = ExecutionProvenanceLog(trace, actor_session_id="receipt-contract-v3")
    provenance = InvocationProvenance(log, invocation_id="receipt-contract-v3:1")
    manifest = ContextExecutionManifest(
        invocation_id=provenance.invocation_id,
        role="actor",
        capabilities=("evidence.query", "evidence.read"),
        capability_contracts={"evidence.query": query_contract()},
    )
    source = (
        "receipt=evidence.query(kind='transitions', level=0, limit=16, "
        "projection='metadata')\n"
        "print(repr(receipt))\n"
        "print('record_count' in dir(receipt), 'records' in dir(receipt))\n"
        "page=evidence_cache.load(receipt.handle)\n"
        "print(page.records[0].ref)"
    )
    async with ContextComputationalSterileExecutor(
        manifest,
        return_type=dict,
        capability_handlers={
            "evidence.query": lambda _: receipt(),
            "evidence.read": lambda _: {
                "records": [{"ref": "L0:T1"}, {"ref": "L0:T2"}],
                "frontier": {"level": 1, "turn": 2},
            },
        },
        provenance=provenance,
    ) as executor:
        result = await executor.run_cell(source)
    messages = ContextProjector(recent_exchanges=2).codeact(log.events)
    rendered = json.dumps(messages, sort_keys=True)
    checks = {
        "worker_completed": result.status == "complete",
        "receipt_names_real_count": "record_count=2" in result.stdout,
        "receipt_omits_false_records_attribute": (
            "records=2" not in result.stdout and "True False" in result.stdout
        ),
        "receipt_advertises_load_path": "evidence_cache.load('ev_fixed')"
        in result.stdout,
        "load_reaches_exact_value": result.stdout.endswith("L0:T1\n"),
        "provider_projection_preserves_real_contract": "record_count=2" in rendered,
        "provider_projection_omits_false_contract": "records=2" not in rendered,
        "provider_projection_preserves_load_path": (
            "evidence_cache.load" in rendered and "ev_fixed" in rendered
        ),
    }
    report = {
        "schema_version": 1,
        "experiment_id": "context-receipt-contract-v3",
        "provider_calls": 0,
        "arc_actions": 0,
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "stdout": result.stdout,
        "provider_messages": messages,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["all_checks_passed"]:
        raise SystemExit("receipt contract probe failed")


if __name__ == "__main__":
    asyncio.run(main())
