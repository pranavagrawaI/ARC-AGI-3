"""Diagnostic: (1) next-token alignment by shift test, (2) run() vs HF logits, (3) BOS vs no-BOS."""
import numpy as np
import torch

from cmlib import hmm as H, lm
from cmlib.common import make_input, lm_metrics, alpha_cols, save_json

model = lm.load()
res = {}
for name, ctor in H.HMMS.items():
    h = ctor(); ns = h.n_sym
    xs = h.sample(4000, np.random.default_rng(0))
    inp = make_input(xs, "A", ns)
    logp, _ = lm.run(model, inp, want_acts=False)
    with torch.no_grad():
        ref = model(inp[None].cuda()).logits[0].float().log_softmax(-1)[:, lm.GATHER_IDS].cpu()
    la = logp.numpy()[:, alpha_cols("A", ns)]
    # log p of observed next symbol under shifts: row t predicts xs[t] (= x_{t+1})
    shifts = {}
    for sh in (-1, 0, 1):
        idx = np.arange(5, 3995)
        shifts[sh] = float(la[idx, xs[idx + sh]].mean())
    p = h.next_probs(h.beliefs(xs)[:4000])
    m_bos = lm_metrics(logp, p, "A", ns)
    nob = torch.tensor(np.array(H.ALPHABETS[("A", ns)])[xs[:-1]])  # no BOS: row t predicts x_{t+2}
    lpn, _ = lm.run(model, nob, want_acts=False)
    m_nob = lm_metrics(lpn, p[1:], "A", ns)
    res[name] = {"run_vs_hf_max_abs_logp_diff": float((ref - logp).abs().max()),
                 "mean_logp_observed_by_shift": shifts,
                 "true_mean_logp_observed": float(np.log(p[np.arange(5, 3995), xs[5:3995]]).mean()),
                 "bos_late_kl_restr": float(m_bos["kl_restr"][3000:].mean()), "nobos_late_kl_restr": float(m_nob["kl_restr"][2999:].mean()),
                 "bos_late_off_mass": float(m_bos["off_alpha_mass"][3000:].mean()), "nobos_late_off_mass": float(m_nob["off_alpha_mass"][2999:].mean())}
    print(name, res[name])
save_json("diag_alignment.json", res)
