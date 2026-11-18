"""Host fault injection through the real Actor, never provider/gameplay calls."""

import json
import time
from pathlib import Path

import pytest

from fixtures import Server, response, treatment_codes
from supervisor import run_arm
from transport import Transport, initial_ledger
from theory import IntegrityError
from evidence_authority import Evidence

FINDING = json.loads(
    (Path(__file__).parent / "evidence-review-fixture.json").read_text()
)


def corrupt(output, target, mutation):
    path = (
        output / "host-evidence/ipc" / target
        if target != "EVIDENCE.json"
        else output / target
    )
    raw = path.read_bytes()
    if mutation == "pixel":
        rows = [json.loads(s) for s in raw.decode().splitlines()]
        rows[-1]["grid_rows"][0] = "0" + rows[-1]["grid_rows"][0][1:]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    elif mutation == "value":
        data = json.loads(raw)
        if target == "transitions.jsonl":
            data["after"]["grid"][0][0] = 0
        else:
            data["catalog"]["transition:1"] = "0" * 64
        path.write_text(json.dumps(data) + "\n")
    elif mutation == "truncate":
        path.write_bytes(raw[:20])
    elif mutation == "replace":
        other = path.with_suffix(".replacement")
        other.write_bytes(raw + b" ")
        other.replace(path)
    elif mutation == "same_bytes_relink":
        other = path.with_suffix(".replacement")
        other.write_bytes(raw)
        other.replace(path)
    elif mutation == "hardlink_restore":
        other = output / "retained-inode"
        other.hardlink_to(path)
        path.unlink()
        path.hardlink_to(other)
        other.unlink()
    elif mutation == "symlink":
        other = output / "substituted"
        other.write_bytes(raw)
        path.unlink()
        path.symlink_to(other)


def assert_stopped(output):
    assert not (output / "RESULT.json").exists()
    assert json.loads((output / "SUPERVISION.json").read_text())["reaped"]
    assert (
        json.loads((output / "FAILURE.json").read_text())["error_type"]
        == "IntegrityError"
    )
    lineage = output / "canonical/lineage.jsonl"
    assert not lineage.exists() or not lineage.read_text().strip()
    assert not (output / "actor/CORRUPT_EXECUTION").exists()


@pytest.mark.parametrize(
    "target,mutation",
    [
        ("states.jsonl", "pixel"),
        ("transitions.jsonl", "value"),
        ("catalog.json", "value"),
        ("states.jsonl", "replace"),
        ("transitions.jsonl", "truncate"),
        ("states.jsonl", "symlink"),
        ("states.jsonl", "hardlink_restore"),
        ("transitions.jsonl", "same_bytes_relink"),
        ("EVIDENCE.json", "replace"),
    ],
)
def test_host_drift_before_codeact_is_rejected(tmp_path, target, mutation):
    ledger = tmp_path / "ledger.json"
    initial_ledger(ledger)
    output = tmp_path / "treatment"
    codes = FINDING["executed_fixture_code"][:]
    codes[1] += "\nopen('/work/CORRUPT_EXECUTION','w').write('executed')\n"
    with Server([response(c, i) for i, c in enumerate(codes)]) as server:
        transport = Transport(
            "treatment",
            tmp_path / "raw",
            ledger,
            time.monotonic() + 60,
            fixture_url=server.url,
        )
        calls = 0

        def fault(body):
            nonlocal calls
            reply = transport(body)
            calls += 1
            if calls == 2:
                corrupt(output, target, mutation)
            return reply

        with pytest.raises(IntegrityError, match="evidence"):
            run_arm("treatment", output, fault, wall_seconds=40)
    assert calls == 2
    assert_stopped(output)


def test_drift_after_verified_actor_read_before_publication(tmp_path):
    ledger = tmp_path / "ledger.json"
    initial_ledger(ledger)
    output = tmp_path / "treatment"
    codes = [
        "print('READ_BEFORE_DRIFT',self.latest_state()['grid_rows'][0][0]); print(self.last_transition()); print(self.decision_packet()); print(self.observations())",
        treatment_codes()[1],
    ]
    with Server([response(c, i) for i, c in enumerate(codes)]) as server:
        transport = Transport(
            "treatment",
            tmp_path / "raw",
            ledger,
            time.monotonic() + 60,
            fixture_url=server.url,
        )
        calls = 0

        def fault(body):
            nonlocal calls
            reply = transport(body)
            calls += 1
            if calls == 2:
                assert "READ_BEFORE_DRIFT 5" in json.dumps(body)
                corrupt(output, "states.jsonl", "pixel")
            return reply

        with pytest.raises(IntegrityError, match="evidence"):
            run_arm("treatment", output, fault, wall_seconds=40)
    assert_stopped(output)


def test_drift_after_catalog_before_worker_initial_read(tmp_path, monkeypatch):
    original = Evidence.__init__

    def fault(self, output, rows):
        original(self, output, rows)
        corrupt(Path(output), "states.jsonl", "pixel")

    monkeypatch.setattr(Evidence, "__init__", fault)

    def never(body):
        raise AssertionError("model dispatch before evidence check")

    with pytest.raises(IntegrityError, match="evidence"):
        run_arm("treatment", tmp_path / "treatment", never, wall_seconds=40)
    assert_stopped(tmp_path / "treatment")


def test_every_broker_read_rechecks(tmp_path, monkeypatch):
    # Inject after valid initial/model dispatch checks, exactly at a broker read.
    original = Evidence.read

    def fault(self, name, store=None):
        corrupt(self.output, "transitions.jsonl", "truncate")
        return original(self, name, store)

    monkeypatch.setattr(Evidence, "read", fault)
    with pytest.raises(IntegrityError, match="evidence"):
        run_arm(
            "treatment",
            tmp_path / "treatment",
            lambda body: response("print(1)"),
            wall_seconds=40,
        )
    assert_stopped(tmp_path / "treatment")


def test_host_catalog_object_tampering_stops(tmp_path, monkeypatch):
    from theory import Store

    original = Store.__init__

    def fault(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.evidence["transition:1"] = "0" * 64

    monkeypatch.setattr(Store, "__init__", fault)
    with pytest.raises(IntegrityError, match="catalog"):
        run_arm(
            "treatment",
            tmp_path / "treatment",
            lambda body: response("print(1)"),
            wall_seconds=40,
        )
    assert_stopped(tmp_path / "treatment")


@pytest.mark.parametrize("boundary", ["reconcile", "publish"])
def test_drift_at_acceptance_boundary_is_never_published(
    tmp_path, monkeypatch, boundary
):
    from theory import Store

    output = tmp_path / "treatment"
    if boundary == "publish":
        original = Store._publish

        def fault(self, theory):
            corrupt(output, "states.jsonl", "pixel")
            return original(self, theory)

        monkeypatch.setattr(Store, "_publish", fault)
    else:
        original = Store._call

        def fault(self, operation, payload):
            if operation == "reconcile":
                corrupt(output, "states.jsonl", "pixel")
            return original(self, operation, payload)

        monkeypatch.setattr(Store, "_call", fault)
    ledger = tmp_path / "ledger.json"
    initial_ledger(ledger)
    with Server([response(c, i) for i, c in enumerate(treatment_codes())]) as server:
        transport = Transport(
            "treatment",
            tmp_path / "raw",
            ledger,
            time.monotonic() + 60,
            fixture_url=server.url,
        )
        with pytest.raises(IntegrityError, match="evidence"):
            run_arm("treatment", output, transport, wall_seconds=40)
    assert_stopped(output)
    cycle = json.loads((output / "canonical/cycle.json").read_text())
    assert cycle["phase"] == ("observed" if boundary == "reconcile" else "reconciled")
