"""Experiment 3: zero-training cross-alphabet transfer of belief probes, subspace geometry,
and the in-context alphabet switch on Wing."""
import json
import os
import sys

import numpy as np
import torch

from cmlib import hmm as H
from cmlib.common import CACHE, RES, Timer, lm_metrics, binned
from cmlib.pooled import (SeqStats, fit_pooled, predict, best_perm_r2, subspace, principal_cosines,
                          linear_cka, procrustes_r2)
from cmlib.probe import r2_scores, LAMBDAS

T = Timer()
W0, W1 = 2000, 4000
S = 5
part = sys.argv[1]


def load(key, a=None, b=None):
    z = np.load(os.path.join(CACHE, key + ".meta.npz"))
    acts = torch.load(os.path.join(CACHE, key + ".acts.pt"))
    k0 = z["keep"][0]
    a = k0 if a is None else a
    b = z["keep"][1] if b is None else b
    return z, acts[:, a - k0:b - k0].float().numpy()


def loo_eval(stats, Xs, Ys, lambdas=LAMBDAS):
    """Leave-one-sequence-out pooled fits. Returns (fits_by_heldout, cv_r2_by_lambda)."""
    fits = [fit_pooled([st for j, st in enumerate(stats) if j != s], lambdas) for s in range(len(stats))]
    cv = {l: float(np.mean([r2_scores(Ys[s], predict(fits[s][l], Xs[s]))[0] for s in range(len(stats))])) for l in lambdas}
    return fits, cv


def transfer_block(XA, YA, XB, YB):
    """XA/XB: lists over sequences of (n, D) float64 (same layer). Matched sequences share beliefs."""
    stA = [SeqStats(x, y) for x, y in zip(XA, YA)]
    stB = [SeqStats(x, y) for x, y in zip(XB, YB)]
    fitsA, cvA = loo_eval(stA, XA, YA)
    fitsB, cvB = loo_eval(stB, XB, YB)
    lamA = max(cvA, key=cvA.get); lamB = max(cvB, key=cvB.get)
    r = {"lam_A": lamA, "lam_B": lamB,
         "A_to_A_heldout_seq_r2": cvA[lamA], "B_to_B_fresh_heldout_seq_r2": cvB[lamB],
         "A_to_A_cv_by_lambda": cvA, "B_to_B_cv_by_lambda": cvB}
    # W_A^(-s) (pooled over the other 4 A sequences) applied to B sequence s
    ident, perm, perms, ident_r2u = [], [], [], []
    for s in range(S):
        Yh = predict(fitsA[s][lamA], XB[s])
        ident.append(r2_scores(YB[s], Yh)[0]); ident_r2u.append(r2_scores(YB[s], Yh)[1])
        bp, p = best_perm_r2(YB[s], Yh)
        perm.append(bp); perms.append(p)
    r["WA_pooled_to_B_r2_identity"] = float(np.mean(ident))
    r["WA_pooled_to_B_r2u_identity"] = float(np.mean(ident_r2u))
    r["WA_pooled_to_B_r2_bestperm"] = float(np.mean(perm))
    r["WA_pooled_to_B_bestperm_per_seq"] = [list(map(int, p)) for p in perms]
    # single-sequence W_A (A seq 0) applied to B sequences 1..4
    single = fit_pooled([stA[0]], [lamA])[lamA]
    r["WA_single_seq0_to_B_r2_identity"] = float(np.mean([r2_scores(YB[s], predict(single, XB[s]))[0] for s in range(1, S)]))
    r["WA_single_seq0_to_A_r2"] = float(np.mean([r2_scores(YA[s], predict(single, XA[s]))[0] for s in range(1, S)]))
    # geometry with all-5 pooled probes
    WA = fit_pooled(stA, [lamA])[lamA]; WB = fit_pooled(stB, [lamB])[lamB]
    WA_half1 = fit_pooled(stA[:2], [lamA])[lamA]; WA_half2 = fit_pooled(stA[2:], [lamA])[lamA]
    QA, QB = subspace(WA[0]), subspace(WB[0])
    QA1, QA2 = subspace(WA_half1[0]), subspace(WA_half2[0])
    r["principal_cosines_WA_WB"] = principal_cosines(QA, QB).tolist()
    r["principal_cosines_WA_half1_half2"] = principal_cosines(QA1, QA2).tolist()
    D = XA[0].shape[1]
    rng = np.random.default_rng(0)
    rnd = [principal_cosines(QA, np.linalg.qr(rng.normal(size=(D, QB.shape[1])))[0]).max() for _ in range(20)]
    r["principal_cosine_random_max_mean"] = float(np.mean(rnd))
    # affine refit of B beliefs from B activations projected on the A-probe subspace (LOO over sequences)
    ZA = [x @ QA for x in XB]; ZB = [x @ QB for x in XB]
    stZ = [SeqStats(z, y) for z, y in zip(ZA, YB)]
    _, cvZ = loo_eval(stZ, ZA, YB, lambdas=(0.0,))
    r["affine_refit_B_from_A_subspace_r2"] = cvZ[0.0]
    stZB = [SeqStats(z, y) for z, y in zip(ZB, YB)]
    _, cvZB = loo_eval(stZB, ZB, YB, lambdas=(0.0,))
    r["affine_refit_B_from_B_subspace_r2"] = cvZB[0.0]
    # random-subspace reference for the affine refit
    Qr = np.linalg.qr(rng.normal(size=(D, QA.shape[1])))[0]
    Zr = [x @ Qr for x in XB]
    _, cvR = loo_eval([SeqStats(z, y) for z, y in zip(Zr, YB)], Zr, YB, lambdas=(0.0,))
    r["affine_refit_B_from_random_subspace_r2"] = cvR[0.0]
    # Procrustes: B activations through W_A (fitted output coords) vs true beliefs
    PA = np.vstack([predict(WA, x) for x in XB]); Yall = np.vstack(YB)
    r["procrustes_r2_WA_outputs_on_B"] = procrustes_r2(PA, Yall)
    # CKA on matched positions (same belief trajectory in both alphabets)
    XAall, XBall = np.vstack(XA), np.vstack(XB)
    r["cka_XA_QA_vs_XB_QB_matched"] = linear_cka(XAall @ QA, XBall @ QB)
    r["cka_XA_QA_vs_XB_QA_matched"] = linear_cka(XAall @ QA, XBall @ QA)
    r["cka_full_acts_A_vs_B_matched"] = linear_cka(XAall[::4], XBall[::4])
    return r, WA, lamA


if part == "transfer":
    out = {}
    for name, ctor in H.HMMS.items():
        h = ctor()
        A = [load(f"{name}_A_s{s}", W0, W1) for s in range(S)]
        B = [load(f"{name}_B_s{s}", W0, W1) for s in range(S)]
        for s in range(S):
            assert np.array_equal(A[s][0]["xs"], B[s][0]["xs"])
        Y = [h.beliefs(z["xs"])[W0:W1] for z, _ in A]
        layers = []
        for l in range(17):
            XA = [a[l].astype(np.float64) for _, a in A]
            XB = [b[l].astype(np.float64) for _, b in B]
            r, _, _ = transfer_block(XA, Y, XB, Y)
            layers.append(r)
            T(f"{name} layer {l - 1}: A->A {r['A_to_A_heldout_seq_r2']:.3f} B->B {r['B_to_B_fresh_heldout_seq_r2']:.3f} "
              f"WA->B id {r['WA_pooled_to_B_r2_identity']:.3f} perm {r['WA_pooled_to_B_r2_bestperm']:.3f} "
              f"cos {np.round(r['principal_cosines_WA_WB'], 2)} refit {r['affine_refit_B_from_A_subspace_r2']:.3f}")
        out[name] = {"by_layer": layers}
        # behaviour in B on the same sequences (KL and leakage onto A tokens)
        beh = {}
        for alpha, other in (("A", "B"), ("B", "A")):
            ms = []
            for s in range(S):
                z = np.load(os.path.join(CACHE, f"{name}_{alpha}_s{s}.meta.npz"))
                p = h.next_probs(h.beliefs(z["xs"])[:4000])
                ms.append(lm_metrics(z["logp"], p, alpha, h.n_sym, other=other))
            beh[alpha] = {k: binned(np.array([m[k] for m in ms])) for k in ms[0]}
        out[name]["behaviour"] = beh
        del A, B
    with open(os.path.join(RES, "parts", "exp3_transfer.json"), "w") as f:
        json.dump(out, f)
elif part == "switch":
    h = H.wing()
    # W_A pooled over the five Wing A sequences, window [2000, 4000), lambda by sequence CV
    A = [load(f"Wing_A_s{s}", W0, W1) for s in range(S)]
    YA = [h.beliefs(z["xs"])[W0:W1] for z, _ in A]
    WA_by_layer = []
    for l in range(17):
        XA = [a[l].astype(np.float64) for _, a in A]
        st = [SeqStats(x, y) for x, y in zip(XA, YA)]
        _, cv = loo_eval(st, XA, YA)
        lam = max(cv, key=cv.get)
        WA_by_layer.append((fit_pooled(st, [lam])[lam], lam))
    del A
    T("W_A fitted")
    windows = [(0, 100), (100, 250), (250, 500), (500, 1000), (1000, 2000)]
    out = {"windows_B_t": [f"[{a},{b})" for a, b in windows]}
    for variant in ("scratch", "nosep", "sep"):
        runs = [load(f"WingSwitch_{variant}_s{s}") for s in range(S)]
        ms, YB = [], []
        for z, _ in runs:
            off = int(z["b_offset"])
            xb = z["xb"]
            p = h.next_probs(h.beliefs(xb)[:2000])
            ms.append(lm_metrics(z["logp"][off:off + 2000], p, "B", 2, other="A"))
            YB.append(h.beliefs(xb)[:2000])
        bins = [(0, 100), (100, 250), (250, 500), (500, 1000), (1000, 2000)]
        rec = {"behaviour_B_segment": {k: binned(np.array([m[k] for m in ms]), bins) for k in ms[0]}}
        fixed, fresh = [], []
        for l in range(17):
            XB = [a[l].astype(np.float64) for _, a in runs]
            wa, lam = WA_by_layer[l]
            fixed.append({f"[{a},{b})": float(np.mean([r2_scores(YB[s][a:b], predict(wa, XB[s][a:b]))[0] for s in range(S)])) for a, b in windows})
            fr = {}
            for a, b in windows:
                Xw = [x[a:b] for x in XB]; Yw = [y[a:b] for y in YB]
                st = [SeqStats(x, y) for x, y in zip(Xw, Yw)]
                _, cv = loo_eval(st, Xw, Yw)
                fr[f"[{a},{b})"] = max(cv.values())
            fresh.append(fr)
        rec["WA_fixed_probe_r2_by_layer"] = fixed
        rec["fresh_WB_loo_seq_r2_by_layer_best_lambda"] = fresh
        out[variant] = rec
        T(f"{variant}: KL_restr " + str({k: round(v['mean'], 3) for k, v in rec['behaviour_B_segment']['kl_restr'].items()})
          + " massA " + str({k: round(v['mean'], 3) for k, v in rec['behaviour_B_segment']['mass_other_alpha'].items()})
          + f" | WA->B best layer by window " + str({w: round(max(f[w] for f in fixed), 3) for w in fixed[0]})
          + " | fresh " + str({w: round(max(f[w] for f in fresh), 3) for w in fresh[0]}))
    with open(os.path.join(RES, "parts", "exp3_switch.json"), "w") as f:
        json.dump(out, f)
T("done")
