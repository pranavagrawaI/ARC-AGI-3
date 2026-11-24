import numpy as np

GAMMAS = (0.5, 0.7, 0.85, 0.92, 0.96, 0.985)


def onehot(xs, n_sym):
    return np.eye(n_sym)[xs]


def ewma_lastk(xs, n_sym, gammas=GAMMAS, lastk=4):
    """Row t (t=0..n) uses x_1..x_t only: EWMA bank (zero init) + one-hots of x_t..x_{t-lastk+1}."""
    n = len(xs)
    oh = np.vstack([np.zeros((1, n_sym)), onehot(xs, n_sym)])  # oh[t] = onehot(x_t), oh[0] = 0
    feats = []
    for g in gammas:
        e = np.zeros((n + 1, n_sym))
        for t in range(1, n + 1):
            e[t] = g * e[t - 1] + (1 - g) * oh[t]
        feats.append(e)
    for j in range(lastk):
        f = np.zeros((n + 1, n_sym))
        f[j:] = oh[: n + 1 - j]
        feats.append(f)
    return np.hstack(feats)


def cum_freq(xs, n_sym):
    oh = np.vstack([np.zeros((1, n_sym)), onehot(xs, n_sym)])
    c = np.cumsum(oh, 0)
    t = np.arange(len(oh))[:, None]
    out = c / np.maximum(t, 1)
    out[0] = 1.0 / n_sym
    return out


def win_freq(xs, n_sym, w):
    oh = np.vstack([np.zeros((1, n_sym)), onehot(xs, n_sym)])
    c = np.cumsum(oh, 0)
    lag = np.vstack([np.zeros((w, n_sym)), c[:-w]])
    t = np.arange(len(oh))[:, None]
    return (c - lag) / np.minimum(np.maximum(t, 1), w)
