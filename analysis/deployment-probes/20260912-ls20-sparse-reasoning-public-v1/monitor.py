"""Read-only monitor for one launched kernel; never retries."""
from pathlib import Path
import datetime
import json
import subprocess
import time
from typing import Any

HERE = Path(__file__).resolve().parent
KERNEL = "pranavagrawai/arc3-ls20-sparse-reasoning-v1-20260912"
KAGGLE = "/home/pranav/.local/bin/kaggle"


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([KAGGLE, *args], text=True, capture_output=True)


while True:
    status = run("kernels", "status", KERNEL)
    record: dict[str, Any] = {
        "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "returncode": status.returncode,
        "stdout": status.stdout.strip(),
        "stderr": status.stderr.strip(),
    }
    with (HERE / "status.jsonl").open("a") as output:
        output.write(json.dumps(record) + "\n")
    log = run("kernels", "files", KERNEL)
    (HERE / "live-log.txt").write_text(log.stdout + log.stderr)
    text = (status.stdout + status.stderr).upper()
    if any(word in text for word in ("COMPLETE", "ERROR", "CANCELLED")):
        break
    time.sleep(30)
