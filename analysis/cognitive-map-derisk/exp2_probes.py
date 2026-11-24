"""Experiment 2: probe validity under three split protocols, six controls, early-to-late trajectory.
usage: exp2_probes.py {main,ctrl01,traj} HMM"""
import json
import os
import sys

import numpy as np
import torch

from cmlib import hmm as H
from cmlib.common import CACHE, RES, Timer
from cmlib.features import cum_freq, win_freq, ewma_lastk
from cmlib.protocols import within_seq, cross_seq

part, name = sys.argv[1], sys.argv[2]
h = H.HMMS[name]()
ns, nst = h.n_sym, h.n_state
T = Timer()
W0, W1 = 2000, 4000
KSUF = [1, 2, 5, 10, 20]


def load_acts(key, a, b):
    z = np.load(os.path.join(CACHE, key + ".meta.npz"))
    keep = z["keep"]
    acts = torch.load(os.path.join(CACHE, key + ".acts.pt"))
    return z, acts[:, a - keep[0]:b - keep[0]].float().numpy()


def targets_for(xs, a, b, seed, with_controls=True):
    bel = h.beliefs(xs)[a:b]
    tg = {"belief": bel}
    for k in KSUF:
        tg[f"eta_k{k}"] = h.k_beliefs(xs, k)[a:b]
    if with_controls:
        rng = np.random.default_rng(seed)
        tg["shuffled_belief"] = bel[rng.permutation(len(bel))]
        tg["dirichlet_random"] = rng.dirichlet(np.ones(nst), size=len(bel))
        tg["cum_freq"] = cum_freq(xs, ns)[a:b]
        tg["win_freq_50"] = win_freq(xs, ns, 50)[a:b]
        tg["win_freq_200"] = win_freq(xs, ns, 200)[a:b]
    return tg


def stack(tgs):
    return {k: [t[k] for t in tgs] for k in tgs[0]}


def run_protocols(Xs, tg, protocols):
    out = {}
    for p in protocols:
        out[p] = cross_seq(Xs, tg) if p == "xseq" else within_seq(Xs, tg, p)
    return out


def flip(per_layer):
    """[layer][protocol][target] -> [protocol][target][field] = list over layers (or per-layer dicts)."""
    out = {}
    for p in per_layer[0]:
        out[p] = {}
        for k in per_layer[0][p]:
            out[p][k] = {f: [pl[p][k][f] for pl in per_layer] for f in per_layer[0][p][k]}
    return out


res = {"hmm": name, "part": part}
if part == "main":
    seqs = [f"{name}_A_s{s}" for s in range(5)]
    zs, A = zip(*[load_acts(k, W0, W1) for k in seqs])
    xss = [z["xs"] for z in zs]
    tgs = [targets_for(xs, W0, W1, seed=s) for s, xs in enumerate(xss)]
    tg = stack(tgs)
    res["target_total_variance"] = {k: float(np.mean([v.var(0).sum() for v in tg[k]])) for k in tg}
    T("loaded")
    protocols = ["rand400", "rand1000", "contig", "xseq"]
    per_layer = []
    for l in range(A[0].shape[0]):
        Xs = [a[l].astype(np.float64) for a in A]
        per_layer.append(run_protocols(Xs, tg, protocols))
        T(f"layer {l - 1}: " + " ".join(f"{p}={per_layer[-1][p]['belief']['ridge_r2']:.3f}/{per_layer[-1][p]['belief']['ols_r2']:.3f}" for p in protocols))
    res["acts_probe"] = flip(per_layer)
    # NTP / log-NTP / zero-LLM baselines with the same protocols
    base = {}
    feats = {
        "ntp": [h.next_probs(h.beliefs(xs))[W0:W1] for xs in xss],
        "log_ntp": [np.log(h.next_probs(h.beliefs(xs))[W0:W1]) for xs in xss],
        "ewma_last4": [ewma_lastk(xs, ns)[W0:W1] for xs in xss],
        "last4_onehot_only": [ewma_lastk(xs, ns, gammas=())[W0:W1] for xs in xss],
        "ewma_only": [ewma_lastk(xs, ns, lastk=0)[W0:W1] for xs in xss],
    }
    for fname, F in feats.items():
        base[fname] = run_protocols(F, {"belief": tg["belief"]}, protocols)
    res["feature_baselines"] = base
    T("baselines done")
elif part == "ctrl01":
    out = {}
    for law in ("0", "1"):
        seqs = [f"{name}_{law}HMMseq_A_s{s}" for s in range(3)]
        zs, A = zip(*[load_acts(k, W0, W1) for k in seqs])
        tg = stack([targets_for(z["xs"], W0, W1, seed=s, with_controls=False) for s, z in enumerate(zs)])
        per_layer = []
        for l in range(A[0].shape[0]):
            Xs = [a[l].astype(np.float64) for a in A]
            per_layer.append(run_protocols(Xs, tg, ["rand400", "contig"]))
        T(f"{law}-HMM seqs done; L8 rand400 belief={per_layer[9]['rand400']['belief']['ridge_r2']:.3f}")
        out[f"{law}HMM_sequences"] = flip(per_layer)
        out[f"{law}HMM_sequences_target_var"] = {k: float(np.mean([v.var(0).sum() for v in tg[k]])) for k in tg}
    res["ctrl01"] = out
elif part == "traj":
    windows = [(0, 500), (500, 1000), (1000, 2000), (2000, 3000), (3000, 4000)]
    out = {}
    for a, b in windows:
        zs, A = zip(*[load_acts(f"{name}_A_s{s}", a, b) for s in range(5)])
        tg = stack([targets_for(z["xs"], a, b, seed=s, with_controls=False) for s, z in enumerate(zs)])
        tg = {"belief": tg["belief"]}
        per_layer = []
        for l in range(A[0].shape[0]):
            Xs = [x[l].astype(np.float64) for x in A]
            per_layer.append(run_protocols(Xs, tg, ["rand20", "contig", "xseq"]))
        out[f"[{a},{b})"] = flip(per_layer)
        T(f"window [{a},{b}) best-layer ridge: " + " ".join(f"{p}={max(out[f'[{a},{b})'][p]['belief']['ridge_r2']):.3f}" for p in ["rand20", "contig", "xseq"]))
    res["trajectory"] = out
with open(os.path.join(RES, "parts", f"exp2_{part}_{name}.json"), "w") as f:
    json.dump(res, f)
T("saved")
