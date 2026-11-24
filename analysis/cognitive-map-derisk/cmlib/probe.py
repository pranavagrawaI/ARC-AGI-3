"""Exact linear probes in float64: affine ridge regression via SVD of the centered design.
lam = 0 is the minimum-norm least-squares solution (pinv with lstsq-style cutoff)."""
import numpy as np

LAMBDAS = (0.0, 1e-2, 1.0, 10.0, 100.0, 1000.0)


def r2_scores(Y, Yhat):
    """(variance-weighted R^2, uniform-average R^2); means taken over the evaluated rows."""
    Y = np.asarray(Y, np.float64)
    Yhat = np.asarray(Yhat, np.float64)
    res = ((Y - Yhat) ** 2).sum(0)
    tot = ((Y - Y.mean(0)) ** 2).sum(0)
    vw = 1.0 - res.sum() / tot.sum() if tot.sum() > 0 else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        per = np.where(tot > 0, 1.0 - res / tot, np.nan)
    return float(vw), float(np.nanmean(per)) if np.any(tot > 0) else np.nan


class RidgeSVD:
    """Fit once (one SVD), predict for any lambda and any number of target matrices."""

    def __init__(self, X):
        X = np.asarray(X, np.float64)
        self.mu = X.mean(0)
        U, s, Vt = np.linalg.svd(X - self.mu, full_matrices=False)
        self.U, self.s, self.Vt = U, s, Vt
        self.cut = s.max() * max(X.shape) * np.finfo(np.float64).eps

    def _filt(self, lam):
        if lam == 0:
            return np.where(self.s > self.cut, 1.0 / np.where(self.s > self.cut, self.s, 1.0), 0.0)
        return self.s / (self.s ** 2 + lam)

    def coef(self, Y, lam):
        Y = np.asarray(Y, np.float64)
        my = Y.mean(0)
        W = self.Vt.T @ (self._filt(lam)[:, None] * (self.U.T @ (Y - my)))
        b = my - self.mu @ W
        return W, b

    def project(self, X):
        return (np.asarray(X, np.float64) - self.mu) @ self.Vt.T

    def predictor(self, Y):
        """Returns f(P, lam) where P = self.project(Xtest), computed once per test matrix."""
        Y = np.asarray(Y, np.float64)
        my = Y.mean(0)
        UtY = self.U.T @ (Y - my)

        def f(P, lam):
            return P @ (self._filt(lam)[:, None] * UtY) + my

        return f


def fit_eval(Xtr, Ytr_list, Xte, Yte_list, lambdas=LAMBDAS, Xval_split=None):
    """For each target pair, test R^2 at every lambda, plus lambda chosen on an inner
    validation split of the training rows (Xval_split = (train_idx, val_idx) into Xtr).
    Returns list of dicts."""
    model = RidgeSVD(Xtr)
    Pte = model.project(Xte)
    if Xval_split is not None:
        inner = RidgeSVD(Xtr[Xval_split[0]])
        Pva = inner.project(Xtr[Xval_split[1]])
    out = []
    for Ytr, Yte in zip(Ytr_list, Yte_list):
        f = model.predictor(Ytr)
        per = {}
        for lam in lambdas:
            vw, ua = r2_scores(Yte, f(Pte, lam))
            per[lam] = (vw, ua)
        rec = {"test_r2_by_lambda": {str(l): per[l][0] for l in lambdas},
               "test_r2u_by_lambda": {str(l): per[l][1] for l in lambdas},
               "ols_r2": per[0.0][0], "ols_r2u": per[0.0][1]}
        if Xval_split is not None:
            g = inner.predictor(Ytr[Xval_split[0]])
            vals = {l: r2_scores(Ytr[Xval_split[1]], g(Pva, l))[0] for l in lambdas}
            best = max(lambdas, key=lambda l: vals[l] if np.isfinite(vals[l]) else -np.inf)
            rec.update({"lam_sel": best, "ridge_r2": per[best][0], "ridge_r2u": per[best][1]})
        out.append(rec)
    return out
