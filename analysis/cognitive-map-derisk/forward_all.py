"""Base-model forward passes for Experiments 1-3. Caches per-sequence log-probs (full-vocab
normalized, at F Q V K M Z) and bf16 residual activations for a kept position range."""
import os

import numpy as np
import torch

from cmlib import hmm as H, lm
from cmlib.common import CACHE, Timer, make_input

T = Timer()
N = 4000
jobs = []  # (key, input_ids, xs_dict, keep_range)
for name, ctor in H.HMMS.items():
    h = ctor()
    ns = h.n_sym
    for s in range(5):
        xs = h.sample(N, np.random.default_rng(s))
        for alpha in "AB":
            jobs.append((f"{name}_{alpha}_s{s}", make_input(xs, alpha, ns), {"xs": xs}, (0, N)))
    if name in ("Wing", "Arch"):
        for law in ("0", "1"):
            for s in range(3):
                xs = h.sample_k_markov(N, int(law), np.random.default_rng(100 + 10 * int(law) + s))
                jobs.append((f"{name}_{law}HMMseq_A_s{s}", make_input(xs, "A", ns), {"xs": xs}, (2000, N)))
# In-context alphabet switch on Wing: 2000 tokens in A, then (optional "\n\n"), then 2000 tokens in B.
h = H.wing()
A2, B2 = np.array(H.ALPHABETS[("A", 2)]), np.array(H.ALPHABETS[("B", 2)])
for s in range(5):
    xa = h.sample(2000, np.random.default_rng(200 + s))
    xb = h.sample(2000, np.random.default_rng(300 + s))
    scratch = torch.tensor(np.r_[H.BOS_ID, B2[xb[:-1]]])
    nosep = torch.tensor(np.r_[H.BOS_ID, A2[xa], B2[xb[:-1]]])
    sep = torch.tensor(np.r_[H.BOS_ID, A2[xa], H.SEP_ID, B2[xb[:-1]]])
    d = {"xa": xa, "xb": xb}
    jobs.append((f"WingSwitch_scratch_s{s}", scratch, dict(d, b_offset=0), (0, 2000)))
    jobs.append((f"WingSwitch_nosep_s{s}", nosep, dict(d, b_offset=2000), (2000, 4000)))
    jobs.append((f"WingSwitch_sep_s{s}", sep, dict(d, b_offset=2001), (2001, 4001)))

model = lm.load(torch.bfloat16)
T(f"model loaded, {len(jobs)} jobs")
for key, ids, meta, (a, b) in jobs:
    out = os.path.join(CACHE, key)
    if os.path.exists(out + ".acts.pt"):
        continue
    logp, acts = lm.run(model, ids)
    np.savez(out + ".meta.npz", logp=logp.numpy(), ids=ids.numpy(), keep=np.array([a, b]), **meta)
    torch.save(acts[:, a:b].clone(), out + ".acts.pt")
    T(f"{key} len={len(ids)} kept [{a},{b})")
T("done")
