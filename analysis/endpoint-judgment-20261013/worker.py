"""One isolated start, native incumbent harness; invoked by screen.py."""
from __future__ import annotations
import asyncio
import json
from pathlib import Path
import sys

import screen


def frame_hash(fr):
    return screen.sha(json.dumps([[list(map(int, row)) for row in f] for f in fr.frame[-1:]]).encode())[:16]


def worker(config_path, key):
    cfg = screen.load(config_path)
    root = config_path.parent / key
    start = screen.starts()[key]
    # Imports happen after parent pins env, paths, temp/cache destinations and RESET.
    import requests
    import arcengine
    from taaf.benchmark import Benchmark
    import overlay
    from inference.framework.solver import HarnessSolver

    game_ref = []
    transport_ref = []
    halt_ref = []
    active_solver = []
    original_post = requests.post

    def snapshot():
        g = game_ref[0]
        return {"level": g.current_state.levels_completed, "state": g.current_state.raw.state.name,
                "frame_hash": frame_hash(g.current_state.raw), "actions": len(g.game_run.history),
                "action_hash": screen.digest([{"id": h.action.id.name, "data": dict(h.action.data or {})} for h in g.game_run.history])}

    def post(url, **kwargs):
        if not url.startswith("http://endpoint-screen.invalid/v1/"):
            raise screen.Halt("unexpected_network", "harness attempted a non-screen HTTP request")
        try:
            reply = transport_ref[0].receive(kwargs["json"])
        except screen.Halt as exc:
            halt_ref.append(exc)
            active_solver[0]._stop_event.set()
            raise RuntimeError(f"screen halt: {exc.status}: {exc.detail}") from None
        resp = requests.Response()
        resp.status_code = 200
        resp._content = json.dumps(reply).encode()
        resp.headers["Content-Type"] = "application/json"
        return resp

    def make_analyzer(self, g, index, local_server=None):
        active_solver[:] = [self]
        game_ref[:] = [g]
        got = snapshot()
        if got["level"] != start["level"] or got["state"] == "GAME_OVER" or got["frame_hash"] not in start["entry_hashes"].values():
            halt_ref.append(screen.Halt("prefix_failed", f"entry failed verification: {got}"))
            raise RuntimeError(f"entry failed verification: {got}")
        analyzer = original_make(self, g, index, local_server)
        screen.atomic(root / "entry.json", {**got, "expected_hashes": start["entry_hashes"], "prefix_actions_excluded": len(start["prefix"])})
        return analyzer

    solver = HarnessSolver(model=cfg["model"], concurrency=1, analyzer_timeout=cfg["request_timeout"],
                           max_runtime_s_per_game=None, max_actions_per_game=cfg["action_budget"] or None,
                           save_request_logs=False)  # full wire requests/responses are compressed receipts
    bm = Benchmark(games=[], job_dir=root / "harness", solver=solver)
    overlay.install(bm, [dict(name=key, gid=start["gid"], level=start["level"], prefix=start["prefix"], hint="")], cfg["env_dir"])
    original_make = HarnessSolver._make_analyzer
    # Class overrides survive Benchmark's deepcopy and bind to the played solver.
    HarnessSolver._make_analyzer = make_analyzer
    async def run_sequential(self, games):
        # One start per subprocess: use the same native play method directly.
        # No thread-pool completion/wakeup or priority scheduler is needed.
        self._stop_event.clear()
        for index, g in enumerate(games):
            self._play_one(g, index, 0, None)
    HarnessSolver._run_games = run_sequential
    tr = screen.Transport(root, cfg, snapshot, screen.dry_script(start) if cfg["dry"] else None)
    transport_ref.append(tr)
    requests.post = post
    status, detail = "gave_up", ""
    try:
        asyncio.run(bm.run(minimal_diagnostics=True))
    except screen.Halt as exc:
        status, detail = exc.status, exc.detail
    except Exception as exc:
        status, detail = "worker_error", f"{type(exc).__name__}: {exc}"
    finally:
        requests.post = original_post
        tr.progress()
    if halt_ref:
        status, detail = halt_ref[0].status, halt_ref[0].detail
    row = game_ref[0].game_run if game_ref else None
    snap = snapshot() if row else {"actions": 0, "level": start["level"]}
    solved = bool(row and (row.state == "won" or snap["level"] > start["level"]))
    if solved:
        status = "solved"
    elif row and row.state == "crashed" and status == "gave_up":
        status, detail = "harness_error", row.solver_note
    elif row and row.state == "lost" and status == "gave_up":
        status = "game_over"
    elif status == "gave_up" and cfg["action_budget"] and snap["actions"] >= cfg["action_budget"]:
        status = "action_budget"
    elif status == "gave_up":
        status, detail = "harness_error", "harness ended without solve or screen limit; inspect worker.log"
    previous_checkpoint = screen.load(root / "checkpoint.json", {})
    checkpoint = {"next_index": tr.cursor, "snapshot": snap}
    if previous_checkpoint.get("snapshot") == snap and previous_checkpoint.get("next_index") == tr.cursor:
        checkpoint["context_hash"] = previous_checkpoint.get("context_hash")
    screen.atomic(root / "checkpoint.json", checkpoint)
    result = {"key": key, "gid": start["gid"], "start_level": start["level"], "klass": start["klass"],
              "status": status, "detail": detail, "solved": solved, "actions": snap["actions"],
              "end_level": snap["level"], "elapsed_seconds": tr.elapsed(), **tr.totals(),
              "dry": cfg["dry"], "evidence": {"receipts": str(root / "receipts"), "entry": str(root / "entry.json"),
                                               "transcript": str(root / "harness/transcripts"), "checkpoint": str(root / "checkpoint.json")}}
    import importlib.metadata
    result["runtime_packages"] = {name: importlib.metadata.version(name) for name in ("arcengine", "arc-agi", "requests")}
    screen.atomic(root / "result.json", result)
    print(json.dumps(result, indent=2))
    return 0 if status in screen.TERMINAL else 2


if __name__ == "__main__":
    raise SystemExit(worker(Path(sys.argv[1]), sys.argv[2]))
