"""Run the generated upload-kernel code locally against a real 5.5 MiB Kaggle blob session.

Six uneven fake parts (so 1 MiB chunks cross part boundaries), CHUNK shrunk to 1 MiB, paths moved
to a temp dir. Pass 1 is stopped after two chunks; pass 2 must resume from the session's committed
offset, rehash the prefix, finish, and match sha256 and the GCS object size. No dataset is created.

  ~/.local/share/uv/tools/kaggle/bin/python -B test_upload_local.py
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import build_upload
import session

SIZES = [1_300_000, 700_000, 1_048_576, 900_001, 1_200_000, 618_111]


def main():
    tmp = Path(tempfile.mkdtemp(prefix="flare-upload-test-"))
    inp, work = tmp / "input", tmp / "working"
    work.mkdir()
    blob, start = os.urandom(sum(SIZES)), 0
    for i, n in enumerate(SIZES):
        d = inp / f"part{i}"
        d.mkdir(parents=True)
        (d / f"flare-part-{i}").write_bytes(blob[start:start + n])
        (d / f"flare-part-{i}.json").write_text(json.dumps({"index": i, "start": start, "end": start + n}))
        start += n
    rec = session.start("local-test", "local-test.bin", len(blob))
    code = build_upload.CODE.format(url=rec["create_url"], total=len(blob), sha=hashlib.sha256(blob).hexdigest())
    code = (code.replace("CHUNK, AHEAD = 256 << 20, 4", "CHUNK, AHEAD = 1 << 20, 4")
                .replace("/kaggle/input", str(inp)).replace("/kaggle/working", str(work)))
    stop = code.replace("while offset < TOTAL:\n", "while offset < TOTAL:\n    if offset >= 2 * CHUNK: raise SystemExit('stopped for test')\n", 1)
    assert stop != code
    p1 = subprocess.run([sys.executable, "-c", stop], capture_output=True, text=True)
    print("pass 1 rc", p1.returncode, p1.stdout.strip().splitlines()[-1:], p1.stderr.strip().splitlines()[-1:])
    mid = session.status(rec)
    p2 = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    print(p2.stdout[-1500:], p2.stderr[-800:])
    res = json.loads((work / "upload.json").read_text())
    ok = (p1.returncode != 0 and mid == ("INCOMPLETE", 2 << 20) and p2.returncode == 0 and res["sha256_ok"]
          and "session committed 2097152" in p2.stdout and session.status(rec)[0] == "COMPLETE")
    print("mid status", mid, "LOCAL UPLOAD TEST", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
