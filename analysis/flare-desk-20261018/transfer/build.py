"""Generate the Kaggle-side transfer of the FLARE gb202 package (107 GB) without the laptop.

Six internet-on CPU notebooks each fetch one byte range from Hugging Face into /kaggle/working
(a 20 GiB volume), so each private notebook output holds one part. A seventh CPU notebook attaches
all six outputs and stream-hashes them in order against the LFS sha256. A GPU session attaches the
same six outputs and concatenates them on its local disk. Nothing passes through the laptop.

python3 -B build.py   # writes kernels/<slug>/{notebook, kernel-metadata.json}
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OWNER = "pranavagrawai"
URL = ("https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-FLARE/resolve/gb202/"
       "qwen3.8-flash-next-tp1-c16-mtp5.safetensors")
TOTAL = 107011338717
SHA256 = "c3db5e1ac63b05efa67d6525826a8a792c0ce92b810e91ddd5ec0f42c73a175c"
PARTS = 6
SPAN = -(-TOTAL // PARTS)  # 17,835,223,120 bytes = 16.6 GiB < the 20 GiB /kaggle/working volume

PART_CODE = '''import hashlib, json, os, time, urllib.request
I, SPAN, TOTAL, URL = {i}, {span}, {total}, "{url}"
start, end = I * SPAN, min(TOTAL, (I + 1) * SPAN)  # [start, end)
path = f"/kaggle/working/flare-part-{{I}}"
h, done, t0, tries = hashlib.sha256(), 0, time.time(), 0
with open(path, "wb") as out:
    while done < end - start:
        tries += 1
        req = urllib.request.Request(URL, headers={{"Range": f"bytes={{start + done}}-{{end - 1}}"}})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                assert r.status == 206, r.status
                while True:
                    b = r.read(16 << 20)
                    if not b:
                        break
                    out.write(b); h.update(b); done += len(b)
        except Exception as e:
            print(f"try {{tries}} at {{done}}: {{e!r}}", flush=True)
            if tries > 20:
                raise
            time.sleep(min(60, 2 ** tries))
        print(f"{{done / 2**30:.2f}} GiB in {{time.time() - t0:.0f}} s", flush=True)
assert done == end - start, (done, end - start)
meta = {{"index": I, "start": start, "end": end, "bytes": done, "sha256": h.hexdigest(),
        "seconds": round(time.time() - t0, 1), "tries": tries}}
json.dump(meta, open(f"/kaggle/working/flare-part-{{I}}.json", "w"))
print(meta)
'''

VERIFY_CODE = '''import glob, hashlib, json, os, time
SHA256, PARTS, TOTAL = "{sha}", {parts}, {total}
def find(name):
    hits = glob.glob(f"/kaggle/input/**/{{name}}", recursive=True)
    assert len(hits) == 1, (name, hits)
    return hits[0]
h, n, t0, rows = hashlib.sha256(), 0, time.time(), []
for i in range(PARTS):
    meta = json.load(open(find(f"flare-part-{{i}}.json")))
    assert meta["start"] == n, (i, meta, n)
    ph = hashlib.sha256()
    with open(find(f"flare-part-{{i}}"), "rb", buffering=0) as f:
        while True:
            b = f.read(64 << 20)
            if not b:
                break
            h.update(b); ph.update(b); n += len(b)
    rows.append({{**meta, "rehash_ok": ph.hexdigest() == meta["sha256"]}})
    print(rows[-1], f"{{time.time() - t0:.0f}} s", flush=True)
result = {{"bytes": n, "bytes_ok": n == TOTAL, "sha256": h.hexdigest(), "sha256_ok": h.hexdigest() == SHA256,
          "parts": rows, "seconds": round(time.time() - t0, 1)}}
json.dump(result, open("/kaggle/working/verify.json", "w"), indent=1)
print(json.dumps(result, indent=1))
assert result["bytes_ok"] and result["sha256_ok"] and all(r["rehash_ok"] for r in rows)
'''


def notebook(code):
    return {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                       "source": code.splitlines(keepends=True)}],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def write(slug, code, internet, kernel_sources=()):
    d = HERE / "kernels" / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{slug}.ipynb").write_text(json.dumps(notebook(code), indent=1))
    meta = {"id": f"{OWNER}/{slug}", "title": slug.replace("-", " "), "code_file": f"{slug}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False,
            "enable_tpu": False, "enable_internet": internet, "keywords": [], "dataset_sources": [],
            "kernel_sources": list(kernel_sources), "competition_sources": [], "model_sources": []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


def part_slug(i):
    # part 5's first push was refused at the 5-session CPU cap; a refused push can poison an id
    return f"flare-gb202-part{i}{'b' if i == 5 else ''}-20261018"


def main():
    for i in range(PARTS):
        write(part_slug(i), PART_CODE.format(i=i, span=SPAN, total=TOTAL, url=URL), internet=True)
    write("flare-gb202-verify-20261018", VERIFY_CODE.format(sha=SHA256, parts=PARTS, total=TOTAL),
          internet=False, kernel_sources=[f"{OWNER}/{part_slug(i)}" for i in range(PARTS)])
    print(f"{PARTS} parts of {SPAN} bytes ({SPAN / 2**30:.2f} GiB), last {TOTAL - SPAN * (PARTS - 1)} bytes")


if __name__ == "__main__":
    main()
