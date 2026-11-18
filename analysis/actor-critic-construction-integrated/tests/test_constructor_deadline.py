from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from construction_service import construct_from_run  # noqa: E402  # identity path
from tests.test_host import CANDIDATE, WORK, call, response  # noqa: E402


def _prefix(tmp_path: Path) -> Path:
    ipc = tmp_path / "ipc"
    ipc.mkdir()
    (ipc / "transitions.jsonl").write_text(
        '{"step":1,"turn":0,"action":"UP","before":{"grid":[[0]]},"after":{"grid":[[1]]}}\n'
    )
    return ipc


def test_hard_deadline_kills_job_no_late_artifacts(tmp_path: Path, monkeypatch) -> None:
    ipc = _prefix(tmp_path)
    payload = response([call("submit_candidate", {"work_item": WORK, "candidate_source": CANDIDATE})])
    started = {"t": None}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0") or 0))
            time.sleep(2)
            raw = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("ARC_CONSTRUCTOR_TRANSPORT", "http")
    monkeypatch.setenv("ARC_CONSTRUCTOR_URL", f"http://127.0.0.1:{server.server_port}/constructor")
    monkeypatch.setenv("ARC_CONSTRUCTOR_HARD_SECONDS", "0.2")
    try:
        started["t"] = time.monotonic()
        result = construct_from_run(tmp_path, ipc)
        elapsed = time.monotonic() - started["t"]
        assert elapsed < 1.0
        assert result["construction_status"] == "FAILED"
        assert result["playable"] is True
        assert not result.get("capped")
        assert str(result.get("error", "")).startswith("HARD_DEADLINE:")
        assert result.get("worker_alive") is False
        pid = result.get("worker_pid")
        time.sleep(2.5)
        assert not (tmp_path / "FROZEN.json").exists()
        assert not (tmp_path / "RESULT.json").exists()
        if isinstance(pid, int):
            try:
                os.kill(pid, 0)
                alive = True
            except OSError:
                alive = False
            assert not alive
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_promote_frozen_is_exclusive(tmp_path: Path) -> None:
    from construction_service import _promote

    work = tmp_path / "work"
    run = tmp_path / "run"
    work.mkdir()
    run.mkdir()
    (work / "FROZEN.json").write_text('{"ok":true}\n')
    _promote(work, run)
    try:
        _promote(work, run)
        raise AssertionError("second promote must not overwrite FROZEN.json")
    except FileExistsError:
        pass
    assert (run / "FROZEN.json").read_text() == '{"ok":true}\n'
