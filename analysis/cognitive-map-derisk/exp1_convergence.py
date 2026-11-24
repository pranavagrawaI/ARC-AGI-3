"""Experiment 1: in-context convergence of OLMo predictions toward the Bayes-optimal HMM predictor."""
import os

import numpy as np

from cmlib import hmm as H
from cmlib.common import CACHE, BINS, binned, lm_metrics, save_json

KS = [0, 1, 2, 3, 4, 6, 8]
KS_EXTRA = [12, 16, 32, 64]
N = 4000


def k_eff(val, ks, kls):
    """Interpolate (linearly in k, on log KL) where val crosses the decreasing k-HMM KL curve."""
    if val > kls[ks.index(1)]:
        return "<1" if val <= kls[0] else "<0"
    for i in range(len(ks) - 1):
        hi, lo = kls[i], kls[i + 1]
        if lo <= val <= hi:
            f = (np.log(hi) - np.log(val)) / (np.log(hi) - np.log(lo)) if hi > lo else 0.0
            return float(ks[i] + f * (ks[i + 1] - ks[i]))
    return f">{ks[-1]}"


out = {"definition": {
    "position_t": "number of symbols observed; prediction of x_{t+1}; t=0 uses BOS (<|endoftext|>) only",
    "kl_full": "KL(p_HMM || p_OLMo) over the full vocabulary, nats",
    "kl_restr": "KL(p_HMM || p_OLMo renormalized on the alphabet), nats",
    "off_alpha_mass": "1 - sum over alphabet of p_OLMo", "k_eff": "Markov order k at which the k-HMM KL equals OLMo's mean KL on t in [3000,4000)"}}
for name, ctor in H.HMMS.items():
    h = ctor()
    M = {k: [] for k in ["kl_full", "kl_restr", "off_alpha_mass"]}
    khmm = {k: [] for k in KS + KS_EXTRA}
    for s in range(5):
        z = np.load(os.path.join(CACHE, f"{name}_A_s{s}.meta.npz"))
        xs = z["xs"]
        p = h.next_probs(h.beliefs(xs)[:N])
        m = lm_metrics(z["logp"], p, "A", h.n_sym)
        for k in M:
            M[k].append(m[k])
        for k in khmm:
            khmm[k].append(H.kl(p, h.k_next_probs(xs, k)[:N]))
    M = {k: np.array(v) for k, v in M.items()}
    khmm = {k: np.array(v) for k, v in khmm.items()}
    late = {k: float(v[:, 3000:].mean()) for k, v in M.items()}
    late_k = {str(k): float(v[:, 3000:].mean()) for k, v in khmm.items()}
    allks = KS + KS_EXTRA
    rec = {
        "binned": {k: binned(v) for k, v in M.items()},
        "khmm_binned": {str(k): binned(v) for k, v in khmm.items()},
        "late_t3000_4000": late,
        "late_per_seq_kl_restr": M["kl_restr"][:, 3000:].mean(1).tolist(),
        "late_per_seq_kl_full": M["kl_full"][:, 3000:].mean(1).tolist(),
        "khmm_late_t3000_4000": late_k,
        "k_eff_full": k_eff(late["kl_full"], allks, [late_k[str(k)] for k in allks]),
        "k_eff_restr": k_eff(late["kl_restr"], allks, [late_k[str(k)] for k in allks]),
        "k_eff_restr_per_seq": [k_eff(v, allks, [float(khmm[k][s, 3000:].mean()) for k in allks])
                                for s, v in enumerate(M["kl_restr"][:, 3000:].mean(1))],
        "decomposition_check": "kl_full = kl_restr - E_pHMM[log alphabet mass] = kl_restr - log(1 - off_alpha_mass)",
        "mean_neg_log_mass_late": float(-np.log(1 - M["off_alpha_mass"][:, 3000:]).mean()),
    }
    out[name] = rec
    print(name, "late", {k: round(v, 4) for k, v in late.items()},
          "\n  kHMM", {k: round(v, 4) for k, v in late_k.items()},
          "\n  k_eff full", rec["k_eff_full"], "restr", rec["k_eff_restr"], rec["k_eff_restr_per_seq"])
    for k in ["kl_restr", "off_alpha_mass"]:
        print("  ", k, {b: round(v["mean"], 4) for b, v in rec["binned"][k].items()})
save_json("exp1_convergence.json", out)
