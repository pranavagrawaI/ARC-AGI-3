"""Generate the CPU kernel that verifies the created dataset end to end and times its read paths.

The upload kernel hashed what it sent; this one hashes what Kaggle serves from /kaggle/input. Pass 1
reads 8 GiB with one sequential stream, the shape of a plain single-threaded loader. Pass 2 hashes
the six original part ranges in six threads and compares each with its 10-18 part sha256. The
whole-file LFS hash is implied by the six part hashes and their order.

  python3 -B build_verify.py   # writes kernels/flare-gb202-dataset-verify-20261020/
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLUG = "flare-gb202-dataset-verify-20261020"
VERIFY = json.loads((HERE.parent / "flare-desk-20261018" / "transfer" / "verify-output" / "verify.json").read_text())
PARTS = [(p["start"], p["end"], p["sha256"]) for p in VERIFY["parts"]]

CODE = r'''import glob, hashlib, json, os, threading, time
PARTS = {parts}
hits = glob.glob("/kaggle/input/**/qwen3.8-flash-next-tp1-c16-mtp5.safetensors", recursive=True)
assert len(hits) == 1, hits
PATH = hits[0]
size = os.path.getsize(PATH)
mounts = [l for l in open("/proc/mounts").read().splitlines() if "/kaggle/input" in l]
def mem():
    try:
        return int(open("/sys/fs/cgroup/memory.current").read()) / 2**30
    except Exception:
        return -1
out = {{"path": PATH, "size": size, "size_ok": size == PARTS[-1][1], "mounts": mounts[:6], "mem_gib_start": mem()}}
print(out, flush=True)

fd = os.open(PATH, os.O_RDONLY)
t, n, off = time.time(), 0, PARTS[2][0]  # mid-file, so pass 2 is not reading a just-read region first
while n < 8 << 30:
    b = os.pread(fd, 64 << 20, off + n)
    assert b
    n += len(b)
out["single_stream_MBps"] = round(n / 1e6 / (time.time() - t), 1)
print("single stream", out["single_stream_MBps"], "MB/s", flush=True)

res = {{}}
def part(i, start, end, sha):
    h, off, t = hashlib.sha256(), start, time.time()
    while off < end:
        b = os.pread(fd, min(64 << 20, end - off), off)
        assert b
        h.update(b); off += len(b)
    res[i] = {{"ok": h.hexdigest() == sha, "seconds": round(time.time() - t, 1)}}
    print("part", i, res[i], flush=True)
t = time.time()
th = [threading.Thread(target=part, args=(i, *p)) for i, p in enumerate(PARTS)]
[x.start() for x in th]; [x.join() for x in th]
out.update(parts=res, all_ok=out["size_ok"] and len(res) == 6 and all(r["ok"] for r in res.values()),
           parallel_MBps=round(size / 1e6 / (time.time() - t), 1), mem_gib_end=mem())
json.dump(out, open("/kaggle/working/dataset-verify.json", "w"), indent=1)
print(json.dumps(out, indent=1), flush=True)
assert out["all_ok"], out
'''


def main():
    code = CODE.format(parts=json.dumps(PARTS))
    compile(code, SLUG, "exec")
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                     "source": code.splitlines(keepends=True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    d = HERE / "kernels" / SLUG
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{SLUG}.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {"id": f"pranavagrawai/{SLUG}", "title": SLUG.replace("-", " "), "code_file": f"{SLUG}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False,
            "enable_tpu": False, "enable_internet": False, "keywords": [],
            "dataset_sources": ["pranavagrawai/flare-gb202-package-20261020"], "kernel_sources": [],
            "competition_sources": [], "model_sources": []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(d)


if __name__ == "__main__":
    main()
