"""Insert the M2 KEEPALIVE cell into a dfranzen-lineage notebook, right before the run cell.

usage: python3 -B build_keepalive.py PARENT.ipynb OUT.ipynb

Every other cell stays byte-identical. The cell targets the hidden-run zero that Kaggle discussion 751598 documents:
the Flash-Next weight load now takes 35-63 min, the stock notebook releases the benchmark at 12 min with one
warmup RESET, and the gateway closes a scorecard that has seen no action for 15 min. After that, every action
gets HTTP 400 and the run ends COMPLETE 0.00 (3,411 B) about 27 min in.

The cell holds each game that holds a stream slot before its first analyzer turn until /health answers 200.
While games are held, it sends one engine RESET every ARC3_KEEPALIVE_S (default 360 s) through the same path the
stock warmup uses, so the scorecard never idles for 15 min. The 12-min release and the per-game clock are
unchanged, so a slow load cannot push the run past the 9 h limit. It only consumes the load time, as the stock
notebook already does. If the server is already healthy at release, the hold is a no-op and no extra RESET is sent.
"""
import copy
import json
import sys
from pathlib import Path

CELL = r'''# M2 KEEPALIVE: hold started games until the model answers /health; keep the scorecard alive meanwhile.
# The gateway closes a scorecard idle for 15 min; a 35-60 min weight load otherwise ends every game with
# HTTP 400 ~27 min in (COMPLETE 0.00). Kaggle discussion 751598.
import logging as _ka_logging, os as _ka_os, sys as _ka_sys, threading as _ka_threading, time as _ka_time
import urllib.request as _ka_url

_ka_solver = _ka_sys.modules[type(bm.solver).__module__]
_KA_HEALTH = f"http://127.0.0.1:{SERVED_MODEL_PORT}/health"
_KA_EVERY_S = float(_ka_os.environ.get("ARC3_KEEPALIVE_S", "360"))
_KA_HOLD_MAX_S = float(_ka_os.environ.get("ARC3_HOLD_MAX_S", str(150 * 60)))
_ka = {"lock": _ka_threading.Lock(), "last": float("-inf"), "resets": 0, "ready": False, "held": set()}
_ka_log = _ka_logging.getLogger("m2.keepalive")


def _ka_healthy():
    if not _ka["ready"]:
        try:
            with _ka_url.urlopen(_KA_HEALTH, timeout=5) as response:
                _ka["ready"] = response.status == 200
        except Exception:
            pass
        if _ka["ready"]:
            _ka_log.warning("M2 KEEPALIVE ready at t=%.0fs after %d resets, %d games held",
                            _ka_time.time() - NOTEBOOK_START_TIME, _ka["resets"], len(_ka["held"]))
    return _ka["ready"]


_ka_play_inner_stock = _ka_solver._HarnessGameSession._play_inner


def _ka_play_inner(self):
    while (not _ka_healthy() and not self.should_stop()
           and _ka_time.time() - NOTEBOOK_START_TIME < _KA_HOLD_MAX_S):
        with _ka["lock"]:
            _ka["held"].add(self.game.game_run.game_id)
            due = _ka_time.monotonic() - _ka["last"] >= _KA_EVERY_S
            if due:
                _ka["last"] = _ka_time.monotonic()
                _ka["resets"] += 1
        if due:
            _ka_log.warning("M2 KEEPALIVE RESET %d on %s at t=%.0fs", _ka["resets"],
                            self.game.game_run.game_id, _ka_time.time() - NOTEBOOK_START_TIME)
            try:
                self._execute_auto_reset()
            except Exception as exc:  # a closed scorecard cannot be reopened; let the stock loop decide
                _ka_log.warning("M2 KEEPALIVE RESET failed: %r", exc)
        _ka_time.sleep(5)
    return _ka_play_inner_stock(self)


_ka_solver._HarnessGameSession._play_inner = _ka_play_inner
print(f"M2 KEEPALIVE armed: every {_KA_EVERY_S:.0f}s, hold max {_KA_HOLD_MAX_S:.0f}s from notebook start")
'''


def build(parent: dict) -> dict:
    nb = copy.deepcopy(parent)
    cells = nb["cells"]
    run = [i for i, c in enumerate(cells)
           if c["cell_type"] == "code" and "print('Starting benchmark...')" in "".join(c["source"])]
    if len(run) != 1:
        raise SystemExit(f"expected exactly one run cell, found {len(run)}")
    if any("# M2 KEEPALIVE" in "".join(c["source"]) for c in cells):
        raise SystemExit("parent already carries the keepalive cell")
    cells.insert(run[0], {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                          "source": CELL.splitlines(keepends=True)})
    return nb


if __name__ == "__main__":
    parent_path, out_path = map(Path, sys.argv[1:3])
    out_path.write_text(json.dumps(build(json.loads(parent_path.read_text())), indent=1) + "\n")
    print(out_path)
