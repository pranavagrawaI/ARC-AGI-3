"""Split protocols shared by Experiments 2-4. Each protocol fits one SVD per training matrix and
evaluates every target with it."""
from collections import defaultdict

import numpy as np

from .probe import fit_eval, LAMBDAS


def _agg(recs):
    """Mean over records of each scalar field; per-lambda curves averaged; lam_sel as list."""
    out = {}
    for k in ("ols_r2", "ols_r2u", "ridge_r2", "ridge_r2u"):
        v = [r[k] for r in recs if k in r]
        if v:
            out[k] = float(np.mean(v))
            out[k + "_sd"] = float(np.std(v))
    out["r2_by_lambda"] = {l: float(np.mean([r["test_r2_by_lambda"][l] for r in recs])) for l in recs[0]["test_r2_by_lambda"]}
    if "lam_sel" in recs[0]:
        out["lam_sel"] = [r["lam_sel"] for r in recs]
    out["n"] = len(recs)
    return out


def split_indices(n, protocol, seed):
    if protocol.startswith("rand"):
        ntr = int(0.2 * n) if protocol == "rand20" else int(protocol[4:])
        perm = np.random.default_rng(seed).permutation(n)
        tr, te = perm[:ntr], perm[ntr:]
        nv = max(1, int(0.2 * len(tr)))
        return tr, te, (np.arange(len(tr) - nv), np.arange(len(tr) - nv, len(tr)))
    if protocol.startswith("contig"):
        frac = 0.4
        ntr = int(frac * n)
        tr, te = np.arange(ntr), np.arange(ntr, n)
        nv = max(1, int(0.2 * ntr))
        return tr, te, (np.arange(ntr - nv), np.arange(ntr - nv, ntr))
    raise ValueError(protocol)


def within_seq(Xs, targets, protocol, lambdas=LAMBDAS):
    """Xs: list over sequences of (n, D). targets: name -> list over sequences of (n, d)."""
    recs = defaultdict(list)
    names = list(targets)
    for s, X in enumerate(Xs):
        tr, te, val = split_indices(len(X), protocol, seed=1000 + s)
        rs = fit_eval(X[tr], [targets[k][s][tr] for k in names], X[te], [targets[k][s][te] for k in names],
                      lambdas=lambdas, Xval_split=val)
        for k, r in zip(names, rs):
            recs[k].append(r)
    return {k: _agg(v) for k, v in recs.items()}


def cross_seq(Xs, targets, lambdas=LAMBDAS):
    """Train on sequence i (all rows), test on each j != i. Inner validation = last 20% of i."""
    recs = defaultdict(list)
    names = list(targets)
    n = len(Xs[0])
    nv = int(0.2 * n)
    val = (np.arange(n - nv), np.arange(n - nv, n))
    from .probe import RidgeSVD, r2_scores
    for i, Xi in enumerate(Xs):
        model = RidgeSVD(Xi)
        inner = RidgeSVD(Xi[val[0]])
        Pva = inner.project(Xi[val[1]])
        Pj = {j: model.project(Xj) for j, Xj in enumerate(Xs) if j != i}
        for k in names:
            Yi = targets[k][i]
            f = model.predictor(Yi)
            g = inner.predictor(Yi[val[0]])
            vals = {l: r2_scores(Yi[val[1]], g(Pva, l))[0] for l in lambdas}
            best = max(lambdas, key=lambda l: vals[l] if np.isfinite(vals[l]) else -np.inf)
            for j in Pj:
                per = {l: r2_scores(targets[k][j], f(Pj[j], l)) for l in lambdas}
                recs[k].append({"test_r2_by_lambda": {str(l): per[l][0] for l in lambdas},
                                "ols_r2": per[0.0][0], "ols_r2u": per[0.0][1], "lam_sel": best,
                                "ridge_r2": per[best][0], "ridge_r2u": per[best][1]})
    return {k: _agg(v) for k, v in recs.items()}
