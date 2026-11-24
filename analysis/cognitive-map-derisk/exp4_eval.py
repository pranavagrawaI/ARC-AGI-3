"""Experiment 4 evaluation of one model state on Wing: 5 A and 5 B sequences (N=4000, seeds 0..4,
identical symbol sequences across alphabets). usage: exp4_eval.py {base | REGIME STEP}"""
import json
import os
import sys

import numpy as np
import torch

from cmlib import hmm as H, lm
from cmlib.common import CACHE, RES, Timer, make_input, lm_metrics, binned
from cmlib.lora import apply_lora, load_lora_state
from cmlib.pooled import SeqStats, fit_pooled, predict, best_perm_r2, subspace
from cmlib.probe import RidgeSVD, r2_scores, LAMBDAS

T = Timer()
tag = "base" if sys.argv[1] == "base" else f"{sys.argv[1]}_step{sys.argv[2]}"
h = H.wing()
S, N = 5, 4000
model = lm.load(torch.bfloat16)
if tag != "base":
    apply_lora(model)
    load_lora_state(model, torch.load(os.path.join(CACHE, f"lora_{sys.argv[1]}", f"step{sys.argv[2]}.pt")))
model.eval()
XS = [h.sample(N, np.random.default_rng(s)) for s in range(S)]
BEL = [h.beliefs(x)[:N] for x in XS]
BINS = [(0, 100), (100, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 3000), (3000, 4000)]
res = {"tag": tag}
acts = {}
for alpha, other in (("A", "B"), ("B", "A")):
    ms, keep = [], []
    for s in range(S):
        logp, a = lm.run(model, make_input(XS[s], alpha, 2))
        ms.append(lm_metrics(logp, h.next_probs(BEL[s]), alpha, 2, other=other))
        keep.append(a[:, 2000:] if alpha == "A" else a)
    acts[alpha] = keep
    beh = {k: binned(np.array([m[k] for m in ms]), BINS) for k in ms[0]}
    beh["mass_non_alphabet_other_tokens"] = binned(np.array([m["off_alpha_mass"] - m["mass_other_alpha"] for m in ms]), BINS)
    res[f"behaviour_{alpha}"] = beh
del model
torch.cuda.empty_cache()
T("forward done")
YA = [b[2000:] for b in BEL]
layers = []
for l in range(17):
    XA = [a[l].float().numpy().astype(np.float64) for a in acts["A"]]
    st = [SeqStats(x, y) for x, y in zip(XA, YA)]
    fits = [fit_pooled([q for j, q in enumerate(st) if j != s], LAMBDAS) for s in range(S)]
    cv = {lam: float(np.mean([r2_scores(YA[s], predict(fits[s][lam], XA[s]))[0] for s in range(S)])) for lam in LAMBDAS}
    lam = max(cv, key=cv.get)
    QA = subspace(fit_pooled(st, [lam])[lam][0])
    rec = {"lam_A": lam, "A_heldout_seq_r2": cv[lam], "WA_on_B": {}, "WA_on_B_bestperm": {}, "fresh_WB": {}, "affine_refit_A_subspace": {}}
    XB = [a[l].float().numpy().astype(np.float64) for a in acts["B"]]
    for a, b in BINS:
        w = f"[{a},{b})"
        # Measure 2 literal: W_A^(-s) (fit on the other four A sequences) applied to B sequence s
        rec["WA_on_B"][w] = float(np.mean([r2_scores(BEL[s][a:b], predict(fits[s][lam], XB[s][a:b]))[0] for s in range(S)]))
        rec["WA_on_B_bestperm"][w] = float(np.mean([best_perm_r2(BEL[s][a:b], predict(fits[s][lam], XB[s][a:b]))[0] for s in range(S)]))
        step = max(1, (b - a) // 250)
        idx = np.arange(a, b, step)
        fr = {str(L_): [] for L_ in LAMBDAS}
        ref = []
        for s in range(S):
            tr = [j for j in range(S) if j != s]
            Xtr = np.vstack([XB[j][idx] for j in tr]); Ytr = np.vstack([BEL[j][idx] for j in tr])
            m = RidgeSVD(Xtr); P = m.project(XB[s][a:b])
            for L_ in LAMBDAS:
                f = m.predictor(Ytr)
                fr[str(L_)].append(r2_scores(BEL[s][a:b], f(P, L_))[0])
            Z = np.vstack([XB[j][idx] @ QA for j in tr])
            mz = RidgeSVD(Z)
            ref.append(r2_scores(BEL[s][a:b], mz.predictor(Ytr)(mz.project(XB[s][a:b] @ QA), 0.0))[0])
        fr = {k: float(np.mean(v)) for k, v in fr.items()}
        rec["fresh_WB"][w] = max(fr.values())
        rec["fresh_WB_by_lambda_" + w] = fr
        rec["affine_refit_A_subspace"][w] = float(np.mean(ref))
    layers.append(rec)
    T(f"layer {l - 1}: A {cv[lam]:.3f} | WA->B " + " ".join(f"{v:.2f}" for v in rec["WA_on_B"].values())
      + " | fresh " + " ".join(f"{v:.2f}" for v in rec["fresh_WB"].values()))
res["by_layer"] = layers
with open(os.path.join(RES, "parts", f"exp4_eval_{tag}.json"), "w") as f:
    json.dump(res, f)
T("done")
