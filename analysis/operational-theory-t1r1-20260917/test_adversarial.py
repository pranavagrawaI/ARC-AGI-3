"""T2's real Actor attacks, with raw request/receipt/canonical evidence retained."""

import json
import time
from pathlib import Path

import pytest

from experiment import evidence_rows
from fixtures import Server, response, treatment_codes
from supervisor import run_arm
from transport import Transport, initial_ledger, Stop
from identity import sha

FINDINGS = json.loads((Path(__file__).parent / "reviewer-fixtures.json").read_text())


def run(tmp_path, arm, codes, ids=None):
    ledger = tmp_path / "ledger.json"
    initial_ledger(ledger)
    replies = [response(c, i) for i, c in enumerate(codes)]
    if ids:
        for r, identity in zip(replies, ids):
            r["choices"][0]["message"]["tool_calls"][0]["id"] = identity
    with Server(replies) as server:
        result = run_arm(
            arm,
            tmp_path / arm,
            Transport(
                arm,
                tmp_path / "raw",
                ledger,
                time.monotonic() + 60,
                fixture_url=server.url,
            ),
            wall_seconds=40,
        )
    return result, server.requests


def outputs(requests):
    return "\n".join(
        m.get("content") or ""
        for r in requests
        for m in r["messages"]
        if m["role"] == "user" and "PythonOutput(" in (m.get("content") or "")
    )


def test_reviewer_changed_helper_rejected_before_execution(tmp_path):
    result, requests = run(tmp_path, "treatment", FINDINGS[0]["fixture_code"])
    assert result["metrics"]["construction_completion"]
    out = outputs(requests)
    assert "canonical helper is sealed" in out
    assert "MUTATED_ACTION DOWN" in out and "MUTATED_ACTION UP" not in out
    expected = FINDINGS[0]["observed"]["canonical_helper_sha256"]
    assert sha(tmp_path / "treatment/host-helpers/policy.py") == expected
    records = [
        json.loads(s)
        for s in (tmp_path / "treatment/canonical/lineage.jsonl")
        .read_text()
        .splitlines()
    ]
    assert all(r["theory"]["procedures"][0]["sha256"] == expected for r in records)


@pytest.mark.parametrize("arm", ["control", "treatment"])
def test_reviewer_transient_evidence_never_changes_api(tmp_path, arm):
    probe = FINDINGS[1]["fixture_code"][0].split("self.submit_actions")[0]
    probe += """
for p in ['/evidence/states.jsonl','/evidence/transitions.jsonl']:
    try:
        with open(p,'w') as f: f.write('fabricated')
        print('AUTHORITATIVE_WRITE_SUCCEEDED')
    except OSError:
        print('AUTHORITATIVE_WRITE_DENIED',p)
print('AUTHORITATIVE_TRANSITION',self.last_transition())
print('EVIDENCE_QUEUE',self.queue_manager.status())
"""
    codes = treatment_codes() if arm == "treatment" else FINDINGS[1]["fixture_code"][:]
    if arm == "treatment":
        codes[1] = probe + codes[1]
    else:
        codes[0] = probe + "self.submit_actions(['UP'],rationale='after probe')"
    result, requests = run(tmp_path, arm, codes)
    assert result["metrics"]["construction_completion"]
    out = outputs(requests)
    expected = evidence_rows()[0]["after"]["levels_completed"]
    assert f"TAMPERED_OBSERVATION {expected}" in out
    assert "TAMPERED_OBSERVATION 777" not in out
    assert "AUTHORITATIVE_WRITE_SUCCEEDED" not in out
    assert out.count("AUTHORITATIVE_WRITE_DENIED") >= 2
    assert "1 active background job" in out
    assert "game-state tail" in out
    assert json.loads((tmp_path / arm / "EVIDENCE.json").read_text()) == evidence_rows()


@pytest.mark.parametrize(
    "identity",
    ["prefill_deadbeef_suffix", "prefill_00000000", "local_prefill_00000000_suffix"],
)
def test_real_prefill_preserves_provider_ids_and_literal_text(tmp_path, identity):
    literal = "literal prefill_deadbeef and local_prefill_00000000 must remain text"
    codes = [
        f'print({literal!r}); self.submit_actions(["UP"],rationale="fixture")',
        'self.submit_actions(["UP"],rationale="second")',
    ]
    result, requests = run(tmp_path, "control", codes, [identity, "next_provider_id"])
    assert result["metrics"]["construction_completion"]
    first_ids = [
        c["id"] for m in requests[0]["messages"] for c in m.get("tool_calls", [])
    ]
    assert "local_prefill_00000000" in first_ids
    second = requests[1]["messages"]
    assert any(c["id"] == identity for m in second for c in m.get("tool_calls", []))
    assert any(m.get("tool_call_id") == identity for m in second)
    assert literal in outputs(requests)
    from protocol import project_messages

    assert project_messages(second) == second


def test_real_duplicate_local_id_stops_before_execution(tmp_path):
    with pytest.raises(Stop, match="duplicate/invalid returned tool-call ID"):
        run(
            tmp_path,
            "control",
            ['open("/work/SHOULD_NOT_EXIST","w").write("executed")'],
            ["local_prefill_00000000"],
        )
    assert not (tmp_path / "control/actor/SHOULD_NOT_EXIST").exists()
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert ledger["stopped"]
    assert len(list((tmp_path / "raw").glob("*.receipt.json"))) == 1


def test_owned_id_allocator_skips_provider_collision():
    from prefill import CallIDs

    ids = CallIDs()
    ids.accept_provider(["local_prefill_00000000"])
    assert ids.allocate() == "local_prefill_00000001"
    with pytest.raises(ValueError, match="duplicate"):
        ids.accept_provider(["local_prefill_00000001"])
    with pytest.raises(ValueError, match="duplicate"):
        ids.accept_provider(["local_prefill_00000000"])


def test_malformed_procedure_binding_keeps_bounded_repair(tmp_path):
    codes = treatment_codes()
    codes[1] = (
        "print(self.operational_cycle('revise', {'procedures':None}))\n" + codes[1]
    )
    result, requests = run(tmp_path, "treatment", codes)
    assert result["metrics"]["construction_completion"]
    assert "procedures must be a list" in outputs(requests)
