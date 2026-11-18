"""Recover REAP-k448's kept-expert indices per MoE layer by matching its router rows to an unpruned
checkpoint's router rows (the card says kept router rows are bit-exact). Only router tensors are
fetched, via HTTP range reads on the safetensors shards.

  python3 prune/recover_mask.py            # writes prune/k448_mask.json
"""
import json, struct, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPOS = {"k448": ("lee-chang-93/Qwen3.8-Flash-Next-NVFP4-REAP-k448", "main", "/tmp/k448_index.json"),
         "rx": ("RadixArk/Qwen3.8-Flash-Next-NVFP4", "7b719225242aacd3dbd3f9407468c2ee9a9d2594", "/tmp/rx_index.json"),
         "nv": ("nvidia/Qwen3.8-Flash-Next-NVFP4", "fc694b54fb0174e0913e6adf86691ef85a4ead47", "/tmp/nv_index.json")}


def rng(url, a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


_headers = {}
def header(url):
    if url not in _headers:
        n = struct.unpack("<Q", rng(url, 0, 7))[0]
        _headers[url] = (8 + n, json.loads(rng(url, 8, 8 + n - 1)))
    return _headers[url]


def tensor(repo, rev, shard, name):
    url = f"https://huggingface.co/{repo}/resolve/{rev}/{shard}"
    base, h = header(url)
    meta = h[name]
    a, b = meta["data_offsets"]
    raw = rng(url, base + a, base + b - 1)
    assert meta["dtype"] in ("BF16", "F32", "F16"), meta
    if meta["dtype"] == "BF16":
        arr = np.frombuffer(raw, dtype=np.uint16).reshape(meta["shape"])
    else:
        arr = np.frombuffer(raw, dtype=np.uint16 if meta["dtype"] == "F16" else np.uint32).reshape(meta["shape"])
    return meta["dtype"], arr


def gates(key):
    repo, rev, idx = REPOS[key]
    wm = json.load(open(idx))["weight_map"]
    names = sorted((k for k in wm if k.endswith("mlp.gate.weight")), key=lambda k: int(k.split(".layers.")[1].split(".")[0]) if ".layers." in k else 999)
    with ThreadPoolExecutor(8) as ex:
        out = list(ex.map(lambda n: (n, tensor(repo, rev, wm[n], n)), names))
    return dict(out)


def main():
    base_key = sys.argv[1] if len(sys.argv) > 1 else "rx"
    k, b = gates("k448"), gates(base_key)
    mask, report = {}, {}
    for name, (dt, kw) in k.items():
        bdt, bw = b[name]
        rows = {bw[i].tobytes(): i for i in range(bw.shape[0])}
        kept = [rows.get(kw[j].tobytes()) for j in range(kw.shape[0])]
        report[name] = {"k_shape": list(kw.shape), "base_shape": list(bw.shape), "dtype": [dt, bdt],
                        "exact_matches": sum(x is not None for x in kept)}
        if all(x is not None for x in kept):
            mask[name] = kept
    ok = len(mask) == len(k)
    print(json.dumps({"base": base_key, "layers": len(k), "fully_matched": len(mask),
                      "shapes": sorted({(tuple(v["k_shape"]), tuple(v["base_shape"])) for v in report.values()}),
                      "partial": {n: v["exact_matches"] for n, v in report.items() if n not in mask}}, default=str))
    (HERE / f"k448_mask_vs_{base_key}.json").write_text(json.dumps({"base": REPOS[base_key][:2], "mask": mask, "report": report}, indent=0))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
