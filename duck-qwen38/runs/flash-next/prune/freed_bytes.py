"""Bytes the k448 mask removes from KF's RadixArk checkpoint (main-model routed experts only), from shard headers."""
import json, re, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from recover_mask import REPOS, header

repo, rev, idx = REPOS["rx"]
wm = json.load(open(idx))["weight_map"]
mask = json.load(open(Path(__file__).parent / "k448_mask_vs_rx.json"))["mask"]
kept = {int(k.split(".layers.")[1].split(".")[0]): set(v) for k, v in mask.items() if ".layers." in k and len(v) == 448}
shards = sorted(set(wm.values()))
with ThreadPoolExecutor(16) as ex:
    heads = dict(zip(shards, ex.map(lambda s: header(f"https://huggingface.co/{repo}/resolve/{rev}/{s}")[1], shards)))
tot = defaultdict(int)
rx = re.compile(r"\.layers\.(\d+)\.mlp\.experts\.(\d+)\.")
for name, shard in wm.items():
    m = heads[shard][name]; n = m["data_offsets"][1] - m["data_offsets"][0]
    tot["all"] += n
    e = rx.search(name)
    if e and "mtp" not in name:
        layer, ex_id = int(e[1]), int(e[2])
        tot["routed_experts"] += n
        if layer in kept and ex_id not in kept[layer]:
            tot["pruned"] += n
    elif "mtp" in name:
        tot["mtp"] += n
    elif "ple" in name:
        tot["ple"] += n
print({k: round(v / 2**30, 2) for k, v in tot.items()}, "GiB; pruned layers:", len(kept))
