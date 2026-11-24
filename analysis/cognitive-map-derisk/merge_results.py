"""Merge per-part JSONs into the four deliverable files."""
import glob
import json
import os

from cmlib.common import RES

P = os.path.join(RES, "parts")


def load(name):
    with open(os.path.join(P, name)) as f:
        return json.load(f)


exp2 = {"layers_note": "lists are indexed by layer -1 (embeddings), 0..15 (after each block)",
        "protocols": {"rand400": "per sequence, random 400 train / 1600 test of t in [2000,4000)",
                      "rand1000": "per sequence, random 1000/1000", "rand20": "random 20/80 within window",
                      "contig": "per sequence, first 40% of window train / last 60% test (t [2000,2800) vs [2800,4000))",
                      "xseq": "train on all of sequence i, test on sequence j != i, averaged over 20 ordered pairs"},
        "fields": {"ols_r2": "min-norm least squares (lambda=0), variance-weighted R^2 on test rows",
                   "ridge_r2": "ridge with lambda chosen on an inner validation split of the training rows",
                   "r2_by_lambda": "test R^2 for each lambda (oracle view)", "r2u": "uniform average over belief coordinates"}}
for f in sorted(glob.glob(os.path.join(P, "exp2_*.json"))):
    d = load(os.path.basename(f))
    exp2.setdefault(d["hmm"], {}).update({k: v for k, v in d.items() if k not in ("hmm", "part")})
with open(os.path.join(RES, "exp2_probes_controls.json"), "w") as f:
    json.dump(exp2, f)
exp3 = {}
for n in ("exp3_transfer.json", "exp3_switch.json"):
    if os.path.exists(os.path.join(P, n)):
        exp3[n.replace(".json", "").replace("exp3_", "")] = load(n)
with open(os.path.join(RES, "exp3_cross_symbol_icl.json"), "w") as f:
    json.dump(exp3, f)
exp4 = {"train": {}, "ckpt_probe": {}, "eval": {}}
for f in sorted(glob.glob(os.path.join(P, "exp4_train_*.json"))):
    exp4["train"][os.path.basename(f)[11:-5]] = load(os.path.basename(f))
for f in sorted(glob.glob(os.path.join(P, "exp4_ckpt_probe_*.json"))):
    exp4["ckpt_probe"][os.path.basename(f)[16:-5]] = load(os.path.basename(f))
for f in sorted(glob.glob(os.path.join(P, "exp4_eval_*.json"))):
    exp4["eval"][os.path.basename(f)[10:-5]] = load(os.path.basename(f))
with open(os.path.join(RES, "exp4_finetune_reuse.json"), "w") as f:
    json.dump(exp4, f)
print("merged")
