"""A timed-out real HTTP constructor must not publish a late freeze.

Run inside the same mapped namespace as the integration tests. Server responses
are scripted and local; no external model is used.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_ROOT = ROOT / "analysis/actor-critic-construction-integrated"


def test_timed_out_constructor_cannot_publish_after_failure(monkeypatch, tmp_path):
    sys.path.insert(0, str(CANDIDATE_ROOT))
    import construction_service as service
    spec = importlib.util.spec_from_file_location("coordinator_timeout_fixture", CANDIDATE_ROOT / "tests/test_host.py")
    assert spec is not None and spec.loader is not None
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    response = fixture.response([fixture.call("submit_candidate", {"work_item": fixture.WORK, "candidate_source": fixture.CANDIDATE})])
    entered = threading.Event()
    release = threading.Event()
    responded = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            entered.set()
            release.wait(timeout=3)
            raw = json.dumps(response).encode()
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                pass  # expected when a timed-out client is actually terminated
            finally:
                responded.set()

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    ipc = tmp_path / "ipc"
    ipc.mkdir()
    (ipc / "transitions.jsonl").write_text(json.dumps({
        "step": 1, "turn": 0, "action": "UP",
        "before": {"grid": [[0]], "levels_completed": 0},
        "after": {"grid": [[1]], "levels_completed": 0},
    }) + "\n")
    monkeypatch.setenv("ARC_CONSTRUCTOR_TRANSPORT", "http")
    monkeypatch.setenv("ARC_CONSTRUCTOR_URL", f"http://127.0.0.1:{server.server_port}/constructor")
    monkeypatch.setenv("ARC_CONSTRUCTOR_HARD_SECONDS", "0.5")
    try:
        result = service.construct_from_run(tmp_path, ipc)
        assert entered.is_set(), "constructor never reached the controlled slow HTTP request"
        assert result["construction_status"] == "FAILED"
        assert "HARD_DEADLINE" in result["error"]
        assert result["playable"] is True
        assert not (tmp_path / "FROZEN.json").exists()
        release.set()
        assert responded.wait(timeout=3)
        # Let the formerly blocked operation finish if it was abandoned, not killed.
        time.sleep(2)
        assert not (tmp_path / "FROZEN.json").exists(), "timed-out constructor published a late freeze"
        assert not (tmp_path / "RESULT.json").exists(), "timed-out constructor published a late success result"
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
