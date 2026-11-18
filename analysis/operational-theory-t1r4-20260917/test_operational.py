import json
import time

import pytest

from experiment import catalog, evidence_rows
from fixtures import Server, response, theory_one, treatment_codes
from theory import Store, FormatError, IntegrityError, replay, validate
from transport import Transport, initial_ledger, Stop


@pytest.mark.parametrize(
    "mutation", ["schema", "lineage", "evidence_hash", "claim_ref", "procedure_hash"]
)
def test_validator_rejects_mechanical_defects(mutation):
    t = theory_one()
    if mutation == "schema":
        t["claims"][0].pop("tested_scope")
    elif mutation == "lineage":
        t["version"] = 2
    elif mutation == "evidence_hash":
        t["evidence_refs"][0]["sha256"] = "0" * 64
    elif mutation == "claim_ref":
        t["claims"][0]["evidence_refs"] = ["future"]
    else:
        t["procedures"] = [
            dict(
                id="p1",
                path="helpers/p.py",
                sha256="0" * 64,
                applicability="unknown",
                claim_refs=["c1"],
            )
        ]
    with pytest.raises(FormatError):
        validate(t, None, catalog(evidence_rows()), {})


def test_cannot_skip_cycle_or_rewrite_evidence(tmp_path):
    store = Store(tmp_path, catalog(evidence_rows()))
    assert store.call("prepare", ["UP"])["status"] == "REPAIR"
    assert store.call("decide", {})["status"] == "REPAIR"
    assert store.call("revise", theory_one())["status"] == "REPAIR"
    assert store.call("declare", "explore")["status"] == "INCOMPLETE"
    assert not store.log.exists()


def test_lineage_tampering_is_hard_stop(tmp_path):
    store = Store(tmp_path, catalog(evidence_rows()))
    store.call("observe", ["transition:1"])
    store.call(
        "reconcile",
        dict(evidence_refs=["transition:1"], claim_refs=[], disposition="initial"),
    )
    assert store.call("revise", theory_one())["status"] == "OK"
    row = json.loads(store.log.read_text())
    row["theory"]["claims"][0]["status"] = "supported"
    store.log.write_text(json.dumps(row) + "\n")
    with pytest.raises(IntegrityError):
        Store(tmp_path, catalog(evidence_rows()))


def test_missing_usage_stops_pair_and_preserves_exposure(tmp_path):
    ledger = tmp_path / "ledger.json"
    initial_ledger(ledger)
    r = response("print(1)")
    r.pop("usage")
    body = dict(
        model="qwen/qwen3.8-27b",
        messages=[dict(role="user", content="fixture")],
        max_tokens=4096,
    )
    with Server([r]) as server:
        transport = Transport(
            "control",
            tmp_path / "raw",
            ledger,
            time.monotonic() + 60,
            fixture_url=server.url,
        )
        with pytest.raises(Stop):
            transport(body)
        with pytest.raises(Stop):
            transport(body)
        assert len(server.requests) == 1
    state = json.loads(ledger.read_text())
    assert state["stopped"] and float(state["unknown_exposure_usd"]) > 0
    assert state["attempts"][0]["actual_cost_usd"] is None


@pytest.mark.parametrize("arm", ["control", "treatment"])
def test_isolated_actual_actor_with_host_owned_store(tmp_path, arm):
    from supervisor import run_arm

    ledger = tmp_path / "ledger.json"
    initial_ledger(ledger)
    codes = (
        treatment_codes()
        if arm == "treatment"
        else ["self.submit_actions(['UP'], rationale='native fixture')"] * 2
    )
    with Server([response(code, i) for i, code in enumerate(codes)]) as server:
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
    assert result["metrics"]["construction_completion"]
    assert json.loads((tmp_path / arm / "SUPERVISION.json").read_text())["reaped"]
    if arm == "treatment":
        assert result["metrics"]["post_observation_same_lineage"]
        assert result["metrics"]["contradiction_driven_revisions"] == 1
        assert "submission requires exact prepared" in json.dumps(server.requests[1])
        assert "Operational cycle, version 1" in json.dumps(server.requests[1])
        restored = Store(tmp_path / arm / "canonical", catalog(evidence_rows()))
        assert restored.current["version"] == 5
        assert (
            len(json.loads((tmp_path / arm / "REPLAY.json").read_text())["decisions"])
            == 2
        )
    else:
        assert not (tmp_path / arm / "canonical").exists()
        assert "workflow_skill" not in json.dumps(server.requests)


def test_pair_budget_reconciliation_and_matching_wire(tmp_path):
    from run_pair import reconcile
    from supervisor import run_arm

    ledger = tmp_path / "ledger.json"
    initial_ledger(ledger)
    first = {}
    for arm in ("control", "treatment"):
        codes = (
            treatment_codes()
            if arm == "treatment"
            else [
                "self.submit_actions(['NOT_LEGAL'], rationale='ordinary format error')",
                "self.submit_actions(['UP'], rationale='native fixture')",
                "self.submit_actions(['UP'], rationale='native fixture')",
            ]
        )
        with Server([response(code, i) for i, code in enumerate(codes)]) as server:
            result = run_arm(
                arm,
                tmp_path / arm,
                Transport(
                    arm,
                    tmp_path / (arm + "-raw"),
                    ledger,
                    time.monotonic() + 60,
                    fixture_url=server.url,
                ),
                wall_seconds=40,
            )
        assert result["metrics"]["construction_completion"]
        first[arm] = server.requests[0]
        if arm == "control":
            assert "REJECTED" in json.dumps(server.requests[1])
    account = reconcile(ledger)
    from theory import write

    write(tmp_path / "ACCOUNTING.json", account)
    assert account["requests"] == 6 and account["total_cost_usd"] == "0.0008934"
    assert (
        account["unresolved_requests"] == 0 and account["extra_consolidator_calls"] == 0
    )
    for key in ("model", "provider", "reasoning", "max_tokens", "tool_choice", "tools"):
        assert first["control"][key] == first["treatment"][key], key
    assert first["treatment"]["messages"][0]["content"].startswith(
        first["control"]["messages"][0]["content"]
    ), "shared system prefix drift"


def test_isolated_worker_cannot_read_host_or_future_evidence(tmp_path):
    from supervisor import command
    import subprocess

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "chassis").mkdir()
    secret = tmp_path / "sentinel"
    secret.write_text("host-only")
    script = """import json,os,socket
from pathlib import Path
host = Path(%r)
assert not host.exists()
assert not Path('/package/evidence').exists()
assert 'OPENROUTER_API_KEY' not in os.environ
try:
 socket.create_connection(('1.1.1.1',443),timeout=.1)
except OSError: pass
else: raise AssertionError('network escape')
try:
 Path('/package/injected.py').write_text('escape')
except OSError: pass
else: raise AssertionError('read-only package escape')
print('isolated')
""" % str(secret)
    (stage / "worker.py").write_text(script)
    result = subprocess.run(command(stage, scratch), capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == b"isolated"


@pytest.mark.parametrize("http_status", [200, 429])
def test_absolute_deadline_covers_slow_success_and_error_body(tmp_path, http_status):
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    ledger = tmp_path / "ledger.json"
    initial_ledger(ledger)

    class Slow(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(http_status)
            self.send_header("Content-Length", "100000")
            self.end_headers()
            try:
                for _ in range(10):
                    self.wfile.write(b" ")
                    self.wfile.flush()
                    time.sleep(0.04)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = HTTPServer(("127.0.0.1", 0), Slow)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        t = Transport(
            "control",
            tmp_path / "raw",
            ledger,
            time.monotonic() + 60,
            fixture_url=f"http://127.0.0.1:{server.server_port}/chat",
            request_seconds=0.1,
        )
        started = time.monotonic()
        with pytest.raises(Stop):
            t(
                dict(
                    model="qwen/qwen3.8-27b",
                    messages=[dict(role="user", content="fixture")],
                    max_tokens=4096,
                )
            )
        assert time.monotonic() - started < 0.5
        assert json.loads(ledger.read_text())["stopped"]
        assert float(json.loads(ledger.read_text())["unknown_exposure_usd"]) > 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_replay_detects_changed_executable_artifact(tmp_path):
    store = Store(tmp_path, catalog(evidence_rows()))
    artifact = store.register("model.py", "def predict(x): return x\n")
    t = theory_one()
    t["procedures"] = [
        dict(id="p1", applicability="fixture only", claim_refs=["c1"], **artifact)
    ]
    store.call("observe", ["transition:1"])
    store.call(
        "reconcile",
        dict(evidence_refs=["transition:1"], claim_refs=[], disposition="initial"),
    )
    assert store.call("revise", t)["status"] == "OK"
    (tmp_path / "artifacts" / artifact["sha256"]).write_text("changed")
    with pytest.raises(IntegrityError, match="artifact hash drift"):
        replay(tmp_path, catalog(evidence_rows()))


def test_six_request_cap_stops_without_seventh_attempt(tmp_path):
    from transport import BudgetEnd
    from run_pair import reconcile

    ledger = tmp_path / "ledger.json"
    initial_ledger(ledger)
    body = dict(
        model="qwen/qwen3.8-27b",
        messages=[dict(role="user", content="fixture")],
        max_tokens=4096,
    )
    with Server([response("print(1)", i) for i in range(6)]) as server:
        transport = Transport(
            "control",
            tmp_path / "raw",
            ledger,
            time.monotonic() + 60,
            fixture_url=server.url,
        )
        for _ in range(6):
            transport(body)
        with pytest.raises(BudgetEnd):
            transport(body)
        assert len(server.requests) == 6
    assert reconcile(ledger)["requests"] == 6
    assert reconcile(ledger)["unresolved_requests"] == 0


def test_metadata_pin_refuses_substitution_without_network(tmp_path):
    from bootstrap import HERE
    from run_pair import metadata_check

    original = HERE / "metadata/fixed-endpoints.raw"
    assert metadata_check(original)
    changed = json.loads(original.read_text())
    changed["data"]["id"] = "another-model"
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(changed))
    with pytest.raises(Stop, match="no substitution"):
        metadata_check(path)


@pytest.mark.parametrize(
    "defect",
    [
        "http_error",
        "wrong_model",
        "wrong_provider",
        "routing_retry",
        "token_overrun",
        "nonfinite_cost",
        "reasoning_on",
    ],
)
def test_transport_failures_stop_pair_without_retry(tmp_path, defect):
    ledger = tmp_path / "ledger.json"
    initial_ledger(ledger)
    reply = response("print(1)")
    if defect == "http_error":
        reply = (429, {"error": "fixture"})
    elif defect == "wrong_model":
        reply["model"] = "other"
    elif defect == "wrong_provider":
        reply["provider"] = "other"
    elif defect == "routing_retry":
        reply["openrouter_metadata"]["attempt"] = 2
    elif defect == "token_overrun":
        reply["usage"]["completion_tokens"] = 4097
        reply["usage"]["total_tokens"] = 4197
    elif defect == "nonfinite_cost":
        reply["usage"]["cost"] = "NaN"
    elif defect == "reasoning_on":
        reply["usage"]["completion_tokens_details"]["reasoning_tokens"] = 1
    with Server([reply]) as server:
        transport = Transport(
            "control",
            tmp_path / "raw",
            ledger,
            time.monotonic() + 60,
            fixture_url=server.url,
        )
        body = dict(
            model="qwen/qwen3.8-27b",
            messages=[dict(role="user", content="fixture")],
            max_tokens=4096,
        )
        with pytest.raises(Stop):
            transport(body)
        with pytest.raises(Stop):
            transport(body)
        assert len(server.requests) == 1
    state = json.loads(ledger.read_text())
    assert state["stopped"]
    if defect in ("http_error", "nonfinite_cost"):
        assert float(state["unknown_exposure_usd"]) > 0
    else:
        assert state["known_cost_usd"] == "0.0001489"
        assert state["unknown_exposure_usd"] == "0"


def test_provider_id_resembling_framework_id_remains_opaque():
    from client import ActorClient

    seen = []

    def fixture(body):
        seen.append(body)
        r = response("print(1)", len(seen))
        r["choices"][0]["message"]["tool_calls"][0]["id"] = (
            "prefill_deadbeef" if len(seen) == 1 else "prefill_cafebabe"
        )
        return r

    client = ActorClient(fixture)
    first = client.call([{"role": "user", "content": "fixture"}])
    client.call(
        [
            {"role": "user", "content": "fixture"},
            first.assistant_message,
            {"role": "tool", "tool_call_id": "prefill_deadbeef", "content": "ok"},
        ]
    )
    assert seen[1]["messages"][1]["tool_calls"][0]["id"] == "prefill_deadbeef"
    assert seen[1]["messages"][2]["tool_call_id"] == "prefill_deadbeef"
