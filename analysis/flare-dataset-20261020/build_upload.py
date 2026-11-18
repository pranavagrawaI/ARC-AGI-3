"""Generate the internet-on CPU kernel that PUTs the FLARE package into an open Kaggle blob session.

The kernel attaches the six verified part outputs (flare-gb202-part{0..4,5b}-20261018, read in place),
prefetches 256 MiB chunks with four reader threads, and PUTs them in order to the session URI with
Content-Range. GCS answers 308 with the committed range for every chunk but the last. On any error the
kernel asks the session how much is committed and resumes there, so a rerun of the same kernel also
resumes. sha256 runs over the committed bytes and must equal the LFS hash.

The session URI is a write capability, so the generated kernel lives in ~/.cache/flare-upload/kernels,
not in the repo. Only this template is tracked.

  PY=~/.local/share/uv/tools/kaggle/bin/python
  $PY -B session.py start gb202 && python3 -B build_upload.py gb202
"""
import json
import sys
from pathlib import Path

CACHE = Path.home() / ".cache" / "flare-upload"
OWNER = "pranavagrawai"
SLUG = "flare-gb202-upload-20261020"
TOTAL = 107011338717
SHA256 = "c3db5e1ac63b05efa67d6525826a8a792c0ce92b810e91ddd5ec0f42c73a175c"
PART_SLUGS = [f"flare-gb202-part{i}{'b' if i == 5 else ''}-20261018" for i in range(6)]

CODE = r'''import glob, hashlib, json, os, time
from concurrent.futures import ThreadPoolExecutor
import requests
URL, TOTAL, SHA256 = "{url}", {total}, "{sha}"
CHUNK, AHEAD = 256 << 20, 4
t0 = time.time()

def find(name):
    hits = glob.glob(f"/kaggle/input/**/{{name}}", recursive=True)
    assert len(hits) == 1, (name, hits)
    return hits[0]

parts = []
for i in range(6):
    meta = json.load(open(find(f"flare-part-{{i}}.json")))
    parts.append((meta["start"], meta["end"], os.open(find(f"flare-part-{{i}}"), os.O_RDONLY)))
assert parts[0][0] == 0 and parts[-1][1] == TOTAL
assert all(a[1] == b[0] for a, b in zip(parts, parts[1:]))

def read_range(off, n):
    out = bytearray()
    for start, end, fd in parts:
        while start <= off < end and len(out) < n:
            b = os.pread(fd, min(n - len(out), end - off, 64 << 20), off - start)
            assert b, (off, start)
            out += b; off += len(b)
    assert len(out) == n, (off, n, len(out))
    return bytes(out)

def committed():
    for attempt in range(20):
        try:
            r = requests.put(URL, headers={{"Content-Length": "0", "Content-Range": f"bytes */{{TOTAL}}"}}, timeout=60)
            if r.status_code in (200, 201):
                return TOTAL, r
            if r.status_code == 308:
                rng = r.headers.get("Range")
                return (0 if rng is None else int(rng.split("-")[-1]) + 1), r
            print(f"status query {{r.status_code}}: {{r.text[:200]}}", flush=True)
        except Exception as e:
            print(f"status query error {{e!r}}", flush=True)
        time.sleep(min(60, 2 ** attempt))
    raise SystemExit("session status unreachable")

offset, last = committed()
print(f"session committed {{offset}} of {{TOTAL}} at start", flush=True)
h = hashlib.sha256()
for a in range(0, offset, CHUNK):  # a rerun resumes: hash the already-committed prefix first
    h.update(read_range(a, min(CHUNK, offset - a)))

pool, ahead = ThreadPoolExecutor(AHEAD), {{}}
def chunk_at(off):
    if off % CHUNK:  # resuming after a partial commit: read up to the next boundary directly
        return read_range(off, min(TOTAL, (off // CHUNK + 1) * CHUNK) - off)
    for k in range(off, min(TOTAL, off + AHEAD * CHUNK), CHUNK):
        if k not in ahead:
            ahead[k] = pool.submit(read_range, k, min(CHUNK, TOTAL - k))
    return ahead.pop(off).result()

errors, final, next_report = 0, None, 0
while offset < TOTAL:
    data = chunk_at(offset)
    hdr = {{"Content-Length": str(len(data)),
           "Content-Range": f"bytes {{offset}}-{{offset + len(data) - 1}}/{{TOTAL}}"}}
    try:
        r = requests.put(URL, data=data, headers=hdr, timeout=600)
        code = r.status_code
    except Exception as e:
        r, code = None, repr(e)
    if code in (200, 201):
        new, final = TOTAL, r
    elif code == 308:
        rng = r.headers.get("Range")
        new = 0 if rng is None else int(rng.split("-")[-1]) + 1
    else:
        errors += 1
        print(f"PUT at {{offset}} failed: {{code}} {{getattr(r, 'text', '')[:200]}}", flush=True)
        if errors > 40:
            raise SystemExit("too many errors")
        time.sleep(min(60, 2 ** min(errors, 6)))
        new, last = committed()
        if new == TOTAL:
            final = last
    assert offset <= new <= offset + len(data), (offset, new, len(data))
    h.update(data[:new - offset])
    offset = new
    if offset >= next_report or offset == TOTAL:
        el = time.time() - t0
        print(f"{{offset / 1e9:.2f}} GB committed, {{el:.0f}} s, {{offset / el / 1e6:.0f}} MB/s, errors {{errors}}", flush=True)
        next_report = offset + (5 << 30)
pool.shutdown()
obj = {{}}
try:
    obj = final.json()
except Exception:
    pass
result = {{"bytes": offset, "sha256": h.hexdigest(), "sha256_ok": h.hexdigest() == SHA256, "errors": errors,
          "seconds": round(time.time() - t0, 1), "object_size": obj.get("size"), "object_md5": obj.get("md5Hash"),
          "object_crc32c": obj.get("crc32c")}}
json.dump(result, open("/kaggle/working/upload.json", "w"), indent=1)
print(json.dumps(result, indent=1), flush=True)
assert result["sha256_ok"] and str(result["object_size"]) == str(TOTAL), result
'''


def notebook(code):
    return {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                       "source": code.splitlines(keepends=True)}],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def main():
    rec = json.loads((CACHE / f"{sys.argv[1]}.json").read_text())
    assert rec["size"] == TOTAL, rec["size"]
    d = CACHE / "kernels" / SLUG
    d.mkdir(parents=True, exist_ok=True)
    code = CODE.format(url=rec["create_url"], total=TOTAL, sha=SHA256)
    compile(code, SLUG, "exec")
    (d / f"{SLUG}.ipynb").write_text(json.dumps(notebook(code), indent=1))
    meta = {"id": f"{OWNER}/{SLUG}", "title": SLUG.replace("-", " "), "code_file": f"{SLUG}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False,
            "enable_tpu": False, "enable_internet": True, "keywords": [], "dataset_sources": [],
            "kernel_sources": [f"{OWNER}/{s}" for s in PART_SLUGS], "competition_sources": [],
            "model_sources": []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(d)


if __name__ == "__main__":
    main()
