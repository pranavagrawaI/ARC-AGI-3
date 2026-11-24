import json
import os
import time

import numpy as np
import torch

from . import hmm as H
from .lm import GATHER_IDS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
CACHE = os.path.join(ROOT, "cache")
LAYERS = list(range(-1, 16))  # -1 = embeddings; act row index = layer + 1


def alpha_ids(alpha, n_sym):
    return H.ALPHABETS[(alpha, n_sym)]


def alpha_cols(alpha, n_sym):
    ids = alpha_ids(alpha, n_sym)
    return [GATHER_IDS.index(i) for i in ids]


def make_input(xs, alpha, n_sym):
    """[BOS, x_1..x_{N-1}] -> logits at index t predict x_{t+1}; beliefs[t] pairs with index t."""
    ids = np.asarray(alpha_ids(alpha, n_sym))
    return torch.tensor(np.r_[H.BOS_ID, ids[xs[:-1]]], dtype=torch.long)


def lm_metrics(logp, p_true, alpha, n_sym, other=None):
    """logp: (T, 6) full-vocab log-probs at gathered ids. p_true: (T, n_sym)."""
    lp = logp.double().numpy() if torch.is_tensor(logp) else logp
    la = lp[:, alpha_cols(alpha, n_sym)]
    mass = np.exp(la).sum(1)
    kl_full = H.kl(p_true, np.exp(la))
    kl_restr = H.kl(p_true, np.exp(la) / mass[:, None])
    out = {"kl_full": kl_full, "kl_restr": kl_restr, "off_alpha_mass": 1 - mass}
    if other is not None:
        out["mass_other_alpha"] = np.exp(lp[:, alpha_cols(other, n_sym)]).sum(1)
    return out


BINS = [(0, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 3000), (3000, 4000)]


def binned(arr, bins=BINS):
    """arr: (S, T) -> {bin: {mean, sem}} averaging over positions then sequences."""
    arr = np.atleast_2d(arr)
    out = {}
    for a, b in bins:
        if a >= arr.shape[1]:
            continue
        per_seq = arr[:, a:b].mean(1)
        out[f"[{a},{b})"] = {"mean": float(per_seq.mean()),
                             "sem": float(per_seq.std(ddof=1) / np.sqrt(len(per_seq))) if len(per_seq) > 1 else None}
    return out


def save_json(name, obj):
    path = os.path.join(RES, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=_default)
    return path


def _default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self, msg):
        mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
        print(f"[{time.time() - self.t0:7.1f}s | peak VRAM {mem:.2f} GiB] {msg}", flush=True)


def gen_seqs(hmm, S, N, seed0=0, law="hmm"):
    out = []
    for s in range(S):
        rng = np.random.default_rng(seed0 + s)
        if law == "hmm":
            out.append(hmm.sample(N, rng))
        else:
            out.append(hmm.sample_k_markov(N, int(law[0]), rng))
    return out
