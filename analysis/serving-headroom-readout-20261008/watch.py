"""Read-only watcher/collector. Never pushes, retries a run or changes its source."""
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi

HERE = Path(__file__).resolve().parent
PROBE = HERE.parent / "serving-headroom-20261008"
SLUG = "pranavagrawai/serving-headroom-20261008"
send = requests.Session.send
def bounded(session, request, **kwargs):
    kwargs["timeout"] = (15, 45)
    return send(session, request, **kwargs)
requests.Session.send = bounded


def now(): return dt.datetime.now(dt.timezone.utc).isoformat()


def collect(api, status):
    raw = HERE / "raw"
    files, token = api.kernels_output(SLUG, str(raw), quiet=True, page_size=200)
    if token: raise RuntimeError("uncollected output page")
    receipt = {"utc": now(), "slug": SLUG, "status": status, "output_files": files,
               "launch_response": json.loads((PROBE / "evidence/launch-response.json").read_text()),
               "file_hashes": {str(p.relative_to(raw)): hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.rglob('*') if p.is_file()}}
    (HERE / "collection.json").write_text(json.dumps(receipt, indent=1) + "\n")
    subprocess.run([sys.executable, str(HERE / "audit.py"), str(raw)], check=True)


if __name__ == "__main__":
    deadline = time.time() + 48 * 3600
    collected = (HERE / "collection.json").exists() and (HERE / "audit.json").exists()
    if collected: raise SystemExit("Already collected and audited; no mutation.")
    sys.path.insert(0, str(HERE.parent / "mtp-depth-20261006"))
    import launch_watch
    launch_watch.EVID = HERE
    errors = 0
    while time.time() < deadline:
        try:
            launch_watch.refresh_token()
            api = KaggleApi(); api.authenticate()
            status = json.loads(str(api.kernels_status(SLUG)))
            row = {"utc": now(), **status}
            with (HERE / "status-history.jsonl").open("a") as stream: stream.write(json.dumps(row)+"\n")
            print(json.dumps(row), flush=True)
            if status["status"] in ["COMPLETE", "ERROR", "CANCELED", "CANCELLED"]:
                collect(api, status)
                break
            errors = 0
        except Exception as error:
            errors += 1
            with (HERE / "errors.jsonl").open("a") as stream: stream.write(json.dumps({"utc":now(),"error":repr(error)})+"\n")
            print("read-only watch error:", repr(error), flush=True)
            if errors >= 5: raise
        time.sleep(120)
    else: raise SystemExit("48-hour read-only watch deadline; no launch/retry.")
