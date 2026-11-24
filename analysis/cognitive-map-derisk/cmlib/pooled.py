"""Pooled multi-sequence ridge via sufficient statistics: fits on any union of sequences cost
one eigendecomposition of the D x D centered Gram matrix (same objective as RidgeSVD)."""
import itertools

import numpy as np

from .probe import r2_scores


class SeqStats:
    def __init__(self, X, Y):
        X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64)
        self.n = len(X); self.sx = X.sum(0); self.sy = Y.sum(0)
        self.Sxx = X.T @ X; self.Sxy = X.T @ Y


def fit_pooled(stats, lambdas):
    n = sum(s.n for s in stats)
    sx = sum(s.sx for s in stats); sy = sum(s.sy for s in stats)
    mx, my = sx / n, sy / n
    Cxx = sum(s.Sxx for s in stats) - n * np.outer(mx, mx)
    Cxy = sum(s.Sxy for s in stats) - n * np.outer(mx, my)
    e, Q = np.linalg.eigh(Cxx)
    e = np.maximum(e, 0)
    cut = e.max() * len(e) * np.finfo(np.float64).eps * 10
    QtC = Q.T @ Cxy
    out = {}
    for lam in lambdas:
        if lam == 0:
            f = np.where(e > cut, 1.0 / np.where(e > cut, e, 1.0), 0.0)
        else:
            f = 1.0 / (e + lam)
        W = Q @ (f[:, None] * QtC)
        out[lam] = (W, my - mx @ W)
    return out


def predict(Wb, X):
    W, b = Wb
    return np.asarray(X, np.float64) @ W + b


def best_perm_r2(Y, Yhat):
    best = (-np.inf, None)
    for p in itertools.permutations(range(Y.shape[1])):
        r = r2_scores(Y, Yhat[:, list(p)])[0]
        if r > best[0]:
            best = (r, p)
    return best


def subspace(W, tol=1e-6):
    """Orthonormal basis of the column space of W (drops numerically null directions)."""
    U, s, _ = np.linalg.svd(W, full_matrices=False)
    return U[:, s > tol * s.max()]


def principal_cosines(Qa, Qb):
    return np.linalg.svd(Qa.T @ Qb, compute_uv=False)


def linear_cka(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    num = np.linalg.norm(A.T @ B, "fro") ** 2
    return float(num / (np.linalg.norm(A.T @ A, "fro") * np.linalg.norm(B.T @ B, "fro")))


def procrustes_r2(Z, Y):
    """Best scaled orthogonal map of centered Z onto centered Y (Z, Y same width); test on same rows."""
    Zc, Yc = Z - Z.mean(0), Y - Y.mean(0)
    U, s, Vt = np.linalg.svd(Zc.T @ Yc)
    R = U @ Vt
    scale = s.sum() / (Zc ** 2).sum()
    return r2_scores(Y, scale * Zc @ R + Y.mean(0))[0]
