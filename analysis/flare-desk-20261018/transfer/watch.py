"""Drive the FLARE transfer to a verified state: push part 5 when a CPU slot frees (Kaggle caps
batch CPU sessions at 5), then push the verifier once all six parts are COMPLETE. Idempotent:
the state file records each push, so a restart never pushes a kernel twice.

~/.local/share/uv/tools/kaggle/bin/python -B watch.py   (run in the background; logs to watch.log)
"""
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "mtp-depth-20261006"))
import launch_watch as lw  # refresh_token, field-tested against the 12 h OAuth expiry

KAGGLE = str(Path.home() / ".local/share/uv/tools/kaggle/bin/kaggle")
OWNER = "pranavagrawai"
STATE = HERE / "watch-state.json"
LOG = HERE / "watch.log"
PARTS = [f"flare-gb202-part{i}-20261018" for i in range(5)] + ["flare-gb202-part5b-20261018"]
VERIFY = "flare-gb202-verify-20261018"
lw.EVID = HERE


def log(msg):
    line = f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def status(slug):
    r = subprocess.run([KAGGLE, "kernels", "status", f"{OWNER}/{slug}"], capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip().splitlines()
    text = out[-1] if out else ""
    for s in ("COMPLETE", "ERROR", "CANCEL", "RUNNING", "QUEUED"):
        if s in text.upper():
            return s
    return f"unknown: {text[-120:]}"


def push(slug):
    r = subprocess.run([KAGGLE, "kernels", "push", "-p", str(HERE / "kernels" / slug)],
                       capture_output=True, text=True)
    text = (r.stdout + r.stderr).strip()
    log(f"push {slug}: {text[-200:]}")
    return "successfully pushed" in text


def main():
    state = json.loads(STATE.read_text()) if STATE.exists() else {"pushed": PARTS[:5]}
    while True:
        lw.refresh_token()
        seen = {slug: status(slug) for slug in state["pushed"]}
        log(" ".join(f"{s.split('-')[2]}={v}" for s, v in seen.items()))
        if any(v in ("ERROR", "CANCEL") for v in seen.values()):
            log("a transfer kernel failed; stopping for diagnosis")
            return 1
        running = sum(v in ("RUNNING", "QUEUED") or v.startswith("unknown") for v in seen.values())
        if PARTS[5] not in state["pushed"] and running < 5 and push(PARTS[5]):
            state["pushed"].append(PARTS[5])
        if VERIFY not in state["pushed"] and all(seen.get(p) == "COMPLETE" for p in PARTS):
            if push(VERIFY):
                state["pushed"].append(VERIFY)
        if VERIFY in state["pushed"] and seen.get(VERIFY) == "COMPLETE":
            log("verifier COMPLETE; read its output for sha256_ok")
            return 0
        STATE.write_text(json.dumps(state, indent=1) + "\n")
        time.sleep(120)


if __name__ == "__main__":
    sys.exit(main())
