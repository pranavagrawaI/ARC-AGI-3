"""Exact HMM processes from Balcells et al. (2026), Appendix D.1, in the edge-emitting
convention T^(x)_{ij} = P(x, s_{t+1}=j | s_t=i). Belief eta_t = P(s | x_{1:t}), eta_0 = pi."""
import itertools
import numpy as np


class HMM:
    def __init__(self, name, T):
        self.name = name
        self.T = np.asarray(T, dtype=np.float64)  # (n_sym, n_state, n_state)
        self.n_sym, self.n_state, _ = self.T.shape
        self.Tsum = self.T.sum(0)
        self.M = self.T.sum(2).T  # M[i, c] = sum_j T^(c)_{ij}
        w, v = np.linalg.eig(self.Tsum.T)
        k = np.argmin(np.abs(w - 1.0))
        pi = np.real(v[:, k])
        self.pi = pi / pi.sum()

    def sample(self, n, rng):
        """Returns symbols x_1..x_n (ints)."""
        flat = self.T.transpose(1, 0, 2).reshape(self.n_state, -1)  # [i, c*S+j]
        cdf = np.cumsum(flat, 1)
        cdf /= cdf[:, -1:]
        s = rng.choice(self.n_state, p=self.pi)
        u = rng.random(n)
        xs = np.empty(n, dtype=np.int64)
        for t in range(n):
            idx = int(np.searchsorted(cdf[s], u[t], side="right"))
            idx = min(idx, flat.shape[1] - 1)
            xs[t], s = divmod(idx, self.n_state)
        return xs

    def beliefs(self, xs):
        """eta_t for t = 0..n (row t = belief after t symbols). Shape (n+1, n_state)."""
        n = len(xs)
        out = np.empty((n + 1, self.n_state))
        eta = self.pi.copy()
        out[0] = eta
        for t, x in enumerate(xs):
            eta = eta @ self.T[x]
            z = eta.sum()
            if z <= 0:
                raise ValueError(f"{self.name}: sequence has zero probability at t={t + 1}")
            eta = eta / z
            out[t + 1] = eta
        return out

    def next_probs(self, beliefs):
        return beliefs @ self.M

    def k_beliefs(self, xs, k):
        """eta(x_{max(1,t-k+1):t}) for t = 0..n: filter from pi over the last k symbols."""
        n = len(xs)
        out = np.tile(self.pi, (n + 1, 1))
        if k == 0:
            return out
        # vectorized: for each t, apply symbols x_{t-k+1..t}; positions with t<k start later.
        xs_pad = np.concatenate([np.full(k, -1), xs])  # xs_pad[k + t - 1] = x_t
        eta = np.tile(self.pi, (n + 1, 1))
        for j in range(k):  # apply symbol x_{t-k+1+j}
            sym = xs_pad[np.arange(n + 1) + j]  # x_{t-k+1+j} lives at pad index t+j
            valid = sym >= 0
            nxt = np.einsum("ti,tij->tj", eta[valid], self.T[sym[valid]])
            nxt /= nxt.sum(1, keepdims=True)
            eta[valid] = nxt
        return eta

    def k_next_probs(self, xs, k):
        return self.k_beliefs(xs, k) @ self.M

    def seq_prob(self, xs):
        v = self.pi.copy()
        for x in xs:
            v = v @ self.T[x]
        return v.sum()

    def sample_k_markov(self, n, k, rng):
        """Sample from the k-HMM law p_k(x_{t+1}|x_{1:t}) = eta(x_{t-k+1:t}) M (k in {0, 1})."""
        if k == 0:
            p = self.pi @ self.M
            return rng.choice(self.n_sym, size=n, p=p / p.sum())
        if k == 1:
            P1 = np.stack([(self.pi @ self.T[c]) / (self.pi @ self.T[c]).sum() for c in range(self.n_sym)]) @ self.M
            xs = np.empty(n, dtype=np.int64)
            p0 = self.pi @ self.M
            xs[0] = rng.choice(self.n_sym, p=p0 / p0.sum())
            for t in range(1, n):
                xs[t] = rng.choice(self.n_sym, p=P1[xs[t - 1]] / P1[xs[t - 1]].sum())
            return xs
        raise ValueError(k)


def mess3(x=0.02, alpha=0.01):
    b = (1 - alpha) / 2
    y = 1 - 2 * x
    T0 = [[alpha * y, b * x, b * x], [alpha * x, b * y, b * x], [alpha * x, b * x, b * y]]
    T1 = [[b * y, alpha * x, b * x], [b * x, alpha * y, b * x], [b * x, alpha * x, b * y]]
    T2 = [[b * y, b * x, alpha * x], [b * x, b * y, alpha * x], [b * x, b * x, alpha * y]]
    return HMM("Mess3", [T0, T1, T2])


def wing(alpha=0.98, x=0.4):
    b = (1 - alpha) / 2
    T0 = [[0, b, 0], [0, x * alpha, b / 2], [b, 0, 0]]
    T1 = [[alpha, 0, b], [b, (1 - x) * alpha, b / 2], [0, b, alpha]]
    return HMM("Wing", [T0, T1])


def arch(alpha=0.99):
    b = (1 - alpha) / 3
    T0 = np.diag([0.8 * alpha, 0.2 * alpha, 0.4 * alpha, 0.6 * alpha])
    T1 = [[0, 0, 0, 0], [0, 0.4 * alpha, 0, 0.4 * b], [0, 0, 0.3 * alpha, 0], [0, 0, 0, 0.16 * alpha]]
    T2 = [[0.2 * alpha, b, b, b], [b, 0.4 * alpha, b, 0.6 * b], [b, b, 0.3 * alpha, b], [b, b, b, 0.24 * alpha]]
    return HMM("Arch", [T0, T1, T2])


HMMS = {"Mess3": mess3, "Wing": wing, "Arch": arch}

ALPHABETS = {
    ("A", 3): [435, 1229, 650],   # " F", " Q", " V"
    ("A", 2): [435, 1229],        # " F", " Q"
    ("B", 3): [735, 386, 1901],   # " K", " M", " Z"
    ("B", 2): [735, 386],         # " K", " M"
}
ALPHABET_STR = {("A", 3): [" F", " Q", " V"], ("A", 2): [" F", " Q"],
                ("B", 3): [" K", " M", " Z"], ("B", 2): [" K", " M"]}
BOS_ID = 100257   # <|endoftext|>, OLMo-2's document separator
SEP_ID = 271      # "\n\n"


def kl(p, q, eps=0.0):
    """Row-wise KL(p||q) in nats with 0 log 0 = 0."""
    p = np.asarray(p, np.float64)
    q = np.asarray(q, np.float64)
    m = p > 0
    out = np.zeros(p.shape[0])
    lp = np.where(m, np.log(np.where(m, p, 1.0)), 0.0)
    lq = np.log(np.maximum(q, 1e-300))
    out = np.sum(np.where(m, p * (lp - lq), 0.0), axis=1)
    return out


def brute_force_filter(hmm, xs):
    """P(s_t | x_{1:t}) by explicit enumeration over state paths (test oracle)."""
    n = len(xs)
    joint = np.zeros(hmm.n_state)
    for path in itertools.product(range(hmm.n_state), repeat=n + 1):
        p = hmm.pi[path[0]]
        for t in range(n):
            p *= hmm.T[xs[t], path[t], path[t + 1]]
        joint[path[-1]] += p
    return joint / joint.sum(), joint.sum()
