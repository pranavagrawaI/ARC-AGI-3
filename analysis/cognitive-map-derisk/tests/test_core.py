import itertools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cmlib import hmm as H
from cmlib.probe import RidgeSVD, r2_scores, fit_eval

ALL = [H.mess3(), H.wing(), H.arch()]


@pytest.mark.parametrize("h", ALL, ids=lambda h: h.name)
def test_row_stochastic_and_stationary(h):
    assert np.all(h.T >= 0)
    np.testing.assert_allclose(h.Tsum.sum(1), 1.0, atol=1e-12)
    np.testing.assert_allclose(h.pi @ h.Tsum, h.pi, atol=1e-12)
    assert abs(h.pi.sum() - 1) < 1e-12 and np.all(h.pi > 0)
    np.testing.assert_allclose(h.M.sum(1), 1.0, atol=1e-12)


@pytest.mark.parametrize("h", ALL, ids=lambda h: h.name)
def test_filter_matches_brute_force_forward(h):
    rng = np.random.default_rng(0)
    for _ in range(5):
        xs = h.sample(6, rng)
        b = h.beliefs(xs)
        bf, px = H.brute_force_filter(h, xs)
        np.testing.assert_allclose(b[-1], bf, atol=1e-12)
        np.testing.assert_allclose(h.seq_prob(xs), px, rtol=1e-10)
        # chain rule: P(x_1:n) = prod_t p(x_t | x_<t) with p = eta_{t-1} M
        p = h.next_probs(b[:-1])[np.arange(len(xs)), xs]
        np.testing.assert_allclose(np.prod(p), px, rtol=1e-10)


@pytest.mark.parametrize("h", ALL, ids=lambda h: h.name)
def test_sequence_probs_sum_to_one(h):
    L = 5
    tot = sum(h.seq_prob(xs) for xs in itertools.product(range(h.n_sym), repeat=L))
    assert abs(tot - 1) < 1e-12


@pytest.mark.parametrize("h", ALL, ids=lambda h: h.name)
def test_k_hmm(h):
    rng = np.random.default_rng(1)
    xs = h.sample(3000, rng)
    b = h.beliefs(xs)
    p = h.next_probs(b)
    # k=0 is the stationary marginal; k=1 is filter from pi over one symbol
    np.testing.assert_allclose(h.k_next_probs(xs, 0), np.tile(h.pi @ h.M, (len(xs) + 1, 1)))
    k1 = h.k_beliefs(xs, 1)
    for t in [1, 7, 100]:
        e = h.pi @ h.T[xs[t - 1]]
        np.testing.assert_allclose(k1[t], e / e.sum(), atol=1e-12)
    # k >= t equals full filter
    kb = h.k_beliefs(xs, 8)
    np.testing.assert_allclose(kb[:9], b[:9], atol=1e-12)
    # mean KL decreasing in k, and large k approaches 0
    kls = [H.kl(p, h.k_next_probs(xs, k))[500:].mean() for k in [0, 1, 2, 4, 8, 64]]
    assert all(a >= b_ - 1e-9 for a, b_ in zip(kls, kls[1:])), kls
    assert kls[-1] < 1e-2 * max(kls[0], 1e-12) + 1e-9, kls


@pytest.mark.parametrize("h", ALL, ids=lambda h: h.name)
def test_k_markov_samplers(h):
    rng = np.random.default_rng(2)
    xs0 = h.sample_k_markov(200000, 0, rng)
    np.testing.assert_allclose(np.bincount(xs0, minlength=h.n_sym) / len(xs0), h.pi @ h.M, atol=5e-3)
    xs1 = h.sample_k_markov(200000, 1, rng)
    p1 = h.k_next_probs(xs1, 1)  # row t predicts x_{t+1}
    for a in range(h.n_sym):
        m = xs1[:-1] == a
        emp = np.bincount(xs1[1:][m], minlength=h.n_sym) / m.sum()
        np.testing.assert_allclose(emp, p1[1:-1][m][0], atol=1e-2)
    # full-HMM filter must stay defined on these sequences
    h.beliefs(xs1[:5000]); h.beliefs(xs0[:5000])


def test_sampler_marginals():
    rng = np.random.default_rng(3)
    for h in ALL:
        xs = h.sample(200000, rng)
        np.testing.assert_allclose(np.bincount(xs, minlength=h.n_sym) / len(xs), h.pi @ h.M, atol=1e-2)


def test_r2_and_ridge_exact():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 20))
    W = rng.normal(size=(20, 3))
    Y = X @ W + 0.5 + 0.01 * rng.normal(size=(300, 3))
    m = RidgeSVD(X)
    W0, b0 = m.coef(Y, 0.0)
    A = np.c_[X, np.ones(len(X))]
    ref = np.linalg.lstsq(A, Y, rcond=None)[0]
    np.testing.assert_allclose(W0, ref[:-1], atol=1e-10)
    np.testing.assert_allclose(b0, ref[-1], atol=1e-10)
    lam = 3.0
    Wl, bl = m.coef(Y, lam)
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    np.testing.assert_allclose(Wl, np.linalg.solve(Xc.T @ Xc + lam * np.eye(20), Xc.T @ Yc), atol=1e-10)
    # R^2 definitions
    Yhat = Y + rng.normal(size=Y.shape) * np.array([0.1, 1.0, 3.0])
    vw, ua = r2_scores(Y, Yhat)
    res = ((Y - Yhat) ** 2).sum(0); tot = ((Y - Y.mean(0)) ** 2).sum(0)
    assert abs(vw - (1 - res.sum() / tot.sum())) < 1e-12
    assert abs(ua - np.mean(1 - res / tot)) < 1e-12
    # min-norm solution when n < D
    Xs = rng.normal(size=(30, 100)); Ys = rng.normal(size=(30, 2))
    Wm, bm = RidgeSVD(Xs).coef(Ys, 0.0)
    np.testing.assert_allclose(Xs @ Wm + bm, Ys, atol=1e-9)
    Xsc = Xs - Xs.mean(0)
    np.testing.assert_allclose(Wm, np.linalg.pinv(Xsc, rcond=1e-10) @ (Ys - Ys.mean(0)), atol=1e-9)
    r = fit_eval(X[:200], [Y[:200]], X[200:], [Y[200:]], Xval_split=(np.arange(150), np.arange(150, 200)))
    assert r[0]["ridge_r2"] > 0.99


def test_tokenization_alignment():
    from cmlib import lm
    tok = lm.load_tokenizer()
    for key, strs in H.ALPHABET_STR.items():
        ids = H.ALPHABETS[key]
        assert [tok(s, add_special_tokens=False).input_ids for s in strs] == [[i] for i in ids]
        rng = np.random.default_rng(0)
        xs = rng.integers(0, len(ids), 500)
        text = "".join(strs[x] for x in xs)
        enc = tok(text, add_special_tokens=False).input_ids
        assert enc == [ids[x] for x in xs], key  # 1:1, no merges across symbols
        assert tok.decode(enc) == text
    assert tok("\n\n", add_special_tokens=False).input_ids == [H.SEP_ID]
    assert tok.convert_ids_to_tokens(H.BOS_ID) == "<|endoftext|>"
    assert tok("hi").input_ids == tok("hi", add_special_tokens=False).input_ids  # no implicit BOS


def test_index_alignment_convention():
    """Input = [BOS, x_1..x_{N-1}]; input index t holds x_t (t>=1) and BOS at t=0.
    Hidden/logits at index t have seen x_1..x_t, predict x_{t+1}, pair with eta(x_{1:t}) = beliefs[t]."""
    h = H.wing()
    xs = h.sample(10, np.random.default_rng(0))
    b = h.beliefs(xs)                       # b[t] = eta(x_{1:t}), t=0..10
    inp = np.r_[H.BOS_ID, np.array(H.ALPHABETS[("A", 2)])[xs[:-1]]]
    assert len(inp) == len(xs)
    for t in range(1, len(xs)):
        assert inp[t] == H.ALPHABETS[("A", 2)][xs[t - 1]]       # token at index t is x_t
    targets = xs                              # logits at index t predict x_{t+1} = xs[t]
    probs = h.next_probs(b[:len(xs)])
    np.testing.assert_allclose(probs[0], h.pi @ h.M)
    np.testing.assert_allclose(h.seq_prob(xs), np.prod(probs[np.arange(len(xs)), targets]), rtol=1e-10)


def test_features_causal():
    from cmlib.features import cum_freq, win_freq, ewma_lastk
    xs = np.array([0, 1, 1, 2, 0, 0])
    cf = cum_freq(xs, 3)
    np.testing.assert_allclose(cf[3], [1 / 3, 2 / 3, 0])
    wf = win_freq(xs, 3, 2)
    np.testing.assert_allclose(wf[4], [0, 0.5, 0.5])
    np.testing.assert_allclose(wf[1], [1, 0, 0])
    f = ewma_lastk(xs, 3, gammas=(0.5,), lastk=2)
    np.testing.assert_allclose(f[2], [0.25, 0.5, 0, 0, 1, 0, 1, 0, 0])
    # changing a future symbol never changes earlier rows
    xs2 = xs.copy(); xs2[5] = 1
    np.testing.assert_allclose(ewma_lastk(xs2, 3)[:6], ewma_lastk(xs, 3)[:6])


def test_pooled_matches_svd_ridge():
    from cmlib.pooled import SeqStats, fit_pooled
    rng = np.random.default_rng(5)
    Xs = [rng.normal(size=(80, 10)) for _ in range(3)]
    Ys = [x @ rng.normal(size=(10, 2)) + 1 for x in Xs]
    X, Y = np.vstack(Xs), np.vstack(Ys)
    fits = fit_pooled([SeqStats(x, y) for x, y in zip(Xs, Ys)], [0.0, 2.0])
    m = RidgeSVD(X)
    for lam in [0.0, 2.0]:
        W, b = m.coef(Y, lam)
        np.testing.assert_allclose(fits[lam][0], W, atol=1e-8)
        np.testing.assert_allclose(fits[lam][1], b, atol=1e-8)
