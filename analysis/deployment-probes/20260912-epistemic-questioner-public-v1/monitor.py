"""Read-only monitor for the one launched kernel; never retries."""
from pathlib import Path
import datetime
import json
import subprocess
import time

HERE = Path(__file__).resolve().parent
KERNEL = "pranavagrawai/arc3-epistemic-questioner-20260912"
started = time.monotonic()
while time.monotonic() - started < 3900:
    record = {"at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    for label, args in (("status", ["status", KERNEL]), ("logs", ["logs", KERNEL])):
        try:
            result = subprocess.run(
                ["/home/pranav/.local/bin/kaggle", "kernels", *args],
                capture_output=True, text=True, timeout=30,
            )
            record[label + "_returncode"] = result.returncode
            if label == "status":
                record["status"] = result.stdout.strip() or result.stderr.strip()
            else:
                (HERE / "live-log.txt").write_text(result.stdout + result.stderr)
                record["log_lines"] = len(result.stdout.splitlines())
                record["tail"] = result.stdout.splitlines()[-5:]
        except subprocess.TimeoutExpired:
            record[label + "_error"] = "read timed out"
    with (HERE / "monitor.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)
    if any(token in record.get("status", "") for token in ("COMPLETE", "ERROR", "CANCELLED")):
        break
    time.sleep(45)
else:
    print("Monitoring deadline reached; inspect status without launching again.", flush=True)
