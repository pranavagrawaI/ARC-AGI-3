"""Step 0: bf16 vs fp32 forward agreement on one Wing sequence (N=2000)."""
import gc
import sys

import numpy as np
import torch

from cmlib import hmm as H, lm
from cmlib.common import make_input, lm_metrics, save_json, Timer
from cmlib.probe import fit_eval

T = Timer()
h = H.wing()
N = 2000
xs = h.sample(N, np.random.default_rng(0))
b = h.beliefs(xs)[:N]
p = h.next_probs(b)
inp = make_input(xs, "A", 2)
res = {}
store = {}
for name, dt in [("bf16", torch.bfloat16), ("fp32", torch.float32)]:
    torch.cuda.reset_peak_memory_stats()
    model = lm.load(dt)
    logp, acts = lm.run(model, inp)
    T(f"{name} forward done")
    m = lm_metrics(logp, p, "A", 2)
    rng = np.random.default_rng(0)
    idx = np.arange(1000, 2000)
    perm = rng.permutation(len(idx))
    tr, te = idx[perm[:200]], idx[perm[200:]]
    r2 = []
    for l in range(acts.shape[0]):
        X = acts[l].float().numpy().astype(np.float64)
        r = fit_eval(X[tr], [b[tr]], X[te], [b[te]], lambdas=(0.0, 1.0, 100.0))[0]
        r2.append(r["test_r2_by_lambda"])
    store[name] = dict(logp=logp.double().numpy(), kl=m["kl_full"], acts=acts.float())
    res[name] = {"peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
                 "kl_full_mean_t1000_2000": float(m["kl_full"][1000:].mean()),
                 "kl_restr_mean_t1000_2000": float(m["kl_restr"][1000:].mean()),
                 "off_alpha_mass_mean_t1000_2000": float(m["off_alpha_mass"][1000:].mean()),
                 "probe_r2_by_layer": r2,
                 "peak_probe_r2_ols": max(x["0.0"] for x in r2),
                 "peak_probe_r2_ridge1": max(x["1.0"] for x in r2)}
    del model
    gc.collect(); torch.cuda.empty_cache()
a, c = store["bf16"], store["fp32"]
d = np.abs(a["kl"][1000:] - c["kl"][1000:])
rel_act = [float(((a["acts"][l] - c["acts"][l]).norm() / c["acts"][l].norm())) for l in range(17)]
res["comparison"] = {
    "max_abs_diff_kl_full_t1000_2000": float(d.max()),
    "mean_abs_diff_kl_full_t1000_2000": float(d.mean()),
    "max_abs_diff_logp_alphabet": float(np.abs(a["logp"][1000:, :2] - c["logp"][1000:, :2]).max()),
    "kl_between_bf16_fp32_restricted_mean": float(H.kl(np.exp(c["logp"][1000:, :2]) / np.exp(c["logp"][1000:, :2]).sum(1, keepdims=True),
                                                       np.exp(a["logp"][1000:, :2]) / np.exp(a["logp"][1000:, :2]).sum(1, keepdims=True)).mean()),
    "relative_act_diff_by_layer": rel_act,
    "peak_probe_r2_diff_ols": res["bf16"]["peak_probe_r2_ols"] - res["fp32"]["peak_probe_r2_ols"],
}
print(res["comparison"], {k: (res[k]["peak_vram_gib"], res[k]["kl_full_mean_t1000_2000"], res[k]["peak_probe_r2_ols"], res[k]["peak_probe_r2_ridge1"]) for k in ["bf16", "fp32"]})
save_json("exp0_bf16_vs_fp32.json", res)
