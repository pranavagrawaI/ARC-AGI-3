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


@pytest.mark.parametrize(
    "defect", ["missing", "unregistered", "wrong_hash", "stale_registered"]
)
def test_proposed_binding_repairs_before_any_publication(tmp_path, defect):
    from fixtures import theory_one, decision_code
    from theory import digest

    invalid = theory_one()
    invalid["procedures"] = [
        dict(
            id="p1",
            path="helpers/missing.py@" + "0" * 64,
            sha256="0" * 64,
            applicability="fixture",
            claim_refs=["c1"],
        )
    ]
    source = "def action(): return 'UP'\n"
    first = "print(self.operational_cycle('observe',['transition:1']))\n"
    first += "print(self.operational_cycle('reconcile',{'evidence_refs':['transition:1'],'claim_refs':[],'disposition':'initial'}))\n"
    unbound_source = (
        "def action():\n    print('INVALID_HELPER_EXECUTED')\n    return 'DOWN'\n"
    )
    if defect in ("unregistered", "stale_registered"):
        import hashlib

        value = hashlib.sha256(unbound_source.encode()).hexdigest()
        invalid["procedures"][0].update(
            path="helpers/missing.py@" + value, sha256=value
        )
    if defect != "missing":
        first += f"print(self.write_helper('missing.py',{unbound_source!r}))\n"
    if defect in ("wrong_hash", "stale_registered"):
        first += "print(self.register_operational_artifact('missing.py'))\n"
    if defect == "stale_registered":
        first += f"print(self.write_helper('missing.py',{source!r}))\n"
    first += f"print(self.operational_cycle('revise',{invalid!r}))\n"
    first += "print('AFTER_INVALID',self.operational_cycle('read'))\nprint(self.load_helpers())\n"
    # Deliberately do not execute any proposed helper before a valid binding.
    repair = f"print(self.write_helper('valid.py',{source!r}))\na=self.register_operational_artifact('valid.py')\nt={theory_one()!r}\n"
    repair += "t['procedures']=[dict(id='p1',applicability='fixture',claim_refs=['c1'],**a)]\nprint(self.operational_cycle('revise',t))\nprint(self.load_helpers())\nprint('VALID_EXECUTION',self.h.valid.action())\n"
    repair += decision_code().replace(
        "'kind':'uncertainty','id':'u1'", "'kind':'procedure','id':'p1'"
    )
    result, requests = run(tmp_path, "treatment", [first, repair, treatment_codes()[2]])
    assert result["metrics"]["construction_completion"]
    assert len(requests) == 3
    out = outputs(requests)
    assert "REPAIR" in out and "repairs_remaining" in out
    assert "AFTER_INVALID" in out and "'theory': None" in out
    assert "VALID_EXECUTION UP" in out
    records = [
        json.loads(s)
        for s in (tmp_path / "treatment/canonical/lineage.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [r["theory"]["version"] for r in records] == [1, 2, 3, 4, 5]
    assert all(r["sha256"] != digest(invalid) for r in records)
    assert all("missing.py" not in json.dumps(r) for r in records)
    events = [
        json.loads(s)
        for s in (tmp_path / "treatment/host-events.jsonl").read_text().splitlines()
    ]
    rejected = [
        e
        for e in events
        if e["op"] == "theory"
        and e["request"]["operation"] == "revise"
        and e["result"].get("status") == "REPAIR"
    ]
    assert len(rejected) == 1 and rejected[0]["result"]["phase"] == "reconciled"
    assert rejected[0]["result"]["repairs_remaining"] == 2
    assert "register" in rejected[0]["result"]["error"]
    # No unbound helper callable is invoked in the executed fixture or recorded output.
    assert "self.h.missing.action()" not in json.dumps(requests)
    assert "INVALID_HELPER_EXECUTED" not in out
    assert not (tmp_path / "treatment/FAILURE.json").exists()
    assert not json.loads((tmp_path / "ledger.json").read_text())["stopped"]


def test_missing_binding_exhausts_only_bounded_repair(tmp_path):
    finding = json.loads(
        (Path(__file__).parent / "binding-review-fixture.json").read_text()
    )
    cell = finding["actual_fixture_code"][1]
    # Preserve the exact reviewer observe/reconcile and invalid revise; omit its recovery.
    cut = cell.index(
        "print(self.operational_cycle('revise',",
        cell.index("print(self.operational_cycle('revise',") + 1,
    )
    initial = cell[:cut]
    bad = initial[initial.index("print(self.operational_cycle('revise',") :]
    result, requests = run(
        tmp_path, "treatment", [initial, bad, bad, bad, 'print("MUST_NOT_RUN")']
    )
    assert not result["metrics"]["construction_completion"]
    assert len(requests) == 4
    cycle = json.loads((tmp_path / "treatment/canonical/cycle.json").read_text())
    assert cycle["errors"] == 4 and cycle["phase"] == "reconciled"
    assert not (tmp_path / "treatment/canonical/lineage.jsonl").exists()
    assert not json.loads((tmp_path / "ledger.json").read_text())["stopped"]
    assert json.loads((tmp_path / "treatment/SUPERVISION.json").read_text())["reaped"]


@pytest.mark.parametrize("mutation", ["replace", "remove"])
def test_accepted_helper_drift_remains_hard_stop(tmp_path, mutation):
    from fixtures import theory_one, decision_code
    from theory import IntegrityError

    source = "def action(): return 'UP'\n"
    code = f"self.write_helper('bound.py',{source!r})\na=self.register_operational_artifact('bound.py')\nt={theory_one()!r}\n"
    code += "t['procedures']=[dict(id='p1',applicability='fixture',claim_refs=['c1'],**a)]\nself.operational_cycle('observe',['transition:1'])\nself.operational_cycle('reconcile',{'evidence_refs':['transition:1'],'claim_refs':[],'disposition':'initial'})\nprint(self.operational_cycle('revise',t))\n"
    ledger = tmp_path / "ledger.json"
    initial_ledger(ledger)
    with Server([response(code, 0), response(decision_code(), 1)]) as server:
        transport = Transport(
            "treatment",
            tmp_path / "raw",
            ledger,
            time.monotonic() + 60,
            fixture_url=server.url,
        )
        calls = 0

        def tampering_transport(body):
            nonlocal calls
            reply = transport(body)
            calls += 1
            if calls == 2:
                path = tmp_path / "treatment/host-helpers/bound.py"
                if mutation == "remove":
                    path.unlink()
                else:
                    path.write_text("def action(): return 'DOWN'\n")
            return reply

        with pytest.raises(IntegrityError, match="active helper differs"):
            run_arm(
                "treatment",
                tmp_path / "treatment",
                tampering_transport,
                wall_seconds=40,
            )
    assert not (tmp_path / "treatment/RESULT.json").exists()
    assert json.loads((tmp_path / "treatment/SUPERVISION.json").read_text())["reaped"]
    assert (tmp_path / "treatment/FAILURE.json").exists()
