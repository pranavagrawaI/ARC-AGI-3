"""Kaggle blob-upload sessions for the FLARE gb202 package, driven from the laptop.

The laptop asks Kaggle for a resumable upload session (a GCS session URI plus a blob token). An
internet-on Kaggle CPU kernel then PUTs the bytes into that URI from the six verified part outputs,
so no account credential leaves the laptop and no byte passes through it. The laptop polls progress
with a zero-byte PUT and creates the private dataset from the token once the blob is complete.

Session files hold a write capability, so they live outside the repo (~/.cache/flare-upload, 0600).

Run with the kaggle tool's interpreter:
  PY=~/.local/share/uv/tools/kaggle/bin/python
  $PY -B session.py test                   # 2 MiB two-chunk protocol check, no dataset created
  $PY -B session.py start gb202            # opens the 107 GB session
  $PY -B session.py status gb202           # committed bytes, or COMPLETE
"""
import json
import os
import sys
import time
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.blobs.types.blob_api_service import ApiBlobType, ApiStartBlobUploadRequest

CACHE = Path.home() / ".cache" / "flare-upload"
NAME = "qwen3.8-flash-next-tp1-c16-mtp5.safetensors"
TOTAL = 107011338717


def start(label, name, size):
    api = KaggleApi()
    api.authenticate()
    req = ApiStartBlobUploadRequest()
    req.type = ApiBlobType.DATASET
    req.name = name
    req.content_length = size
    req.last_modified_epoch_seconds = int(time.time())
    with api.build_kaggle_client() as kaggle:
        resp = kaggle.blobs.blob_api_client.start_blob_upload(req)
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{label}.json"
    rec = {"label": label, "name": name, "size": size, "token": resp.token, "create_url": resp.create_url,
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"session {label}: {size} bytes, url host {requests.utils.urlparse(resp.create_url).netloc}, "
          f"saved {path}")
    return rec


def load(label):
    return json.loads((CACHE / f"{label}.json").read_text())


def status(rec):
    """Return ("COMPLETE", size) or ("INCOMPLETE", committed_bytes) or (http_code, body)."""
    r = requests.put(rec["create_url"], headers={"Content-Length": "0",
                                                 "Content-Range": f"bytes */{rec['size']}"}, timeout=60)
    if r.status_code in (200, 201):
        return "COMPLETE", rec["size"]
    if r.status_code == 308:
        rng = r.headers.get("Range")
        return "INCOMPLETE", 0 if rng is None else int(rng.split("-")[-1]) + 1
    return r.status_code, r.text[:300]


def put_chunk(rec, offset, data):
    end = offset + len(data) - 1
    return requests.put(rec["create_url"], data=data, timeout=300,
                        headers={"Content-Length": str(len(data)),
                                 "Content-Range": f"bytes {offset}-{end}/{rec['size']}"})


def protocol_test():
    """Two 1 MiB chunks: the first must answer 308 with Range 0-1048575, the second 200/201."""
    half = 1 << 20
    rec = start("protocol-test", "protocol-test.bin", 2 * half)
    blob = os.urandom(2 * half)
    rows = [("status before", status(rec))]
    r = put_chunk(rec, 0, blob[:half])
    rows.append(("chunk 1", r.status_code, r.headers.get("Range")))
    rows.append(("status mid", status(rec)))
    r = put_chunk(rec, half, blob[half:])
    rows.append(("chunk 2", r.status_code, r.text[:200]))
    rows.append(("status after", status(rec)))
    for row in rows:
        print(*row)
    ok = (rows[1][1] == 308 and rows[1][2] == f"bytes=0-{half - 1}" and rows[2][1] == ("INCOMPLETE", half)
          and rows[3][1] in (200, 201) and rows[4][1][0] == "COMPLETE")
    print("PROTOCOL", "PASS" if ok else "FAIL")
    return ok


def main():
    cmd = sys.argv[1]
    if cmd == "test":
        sys.exit(0 if protocol_test() else 1)
    elif cmd == "start":
        label = sys.argv[2]
        if (CACHE / f"{label}.json").exists():
            sys.exit(f"session {label} already exists; refusing to open a second one")
        start(label, NAME, TOTAL)
    elif cmd == "status":
        print(*status(load(sys.argv[2])))
    else:
        sys.exit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
