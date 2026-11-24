"""Experiment 4 training: LoRA on Wing in alphabet A (F, Q). usage: exp4_train.py REGIME LR [STEPS]"""
import json
import os
import sys

import numpy as np
import torch

from cmlib import hmm as H, lm
from cmlib.common import CACHE, RES, Timer, make_input, lm_metrics
from cmlib.lora import apply_lora, lora_state
from cmlib.pooled import SeqStats, fit_pooled, predict
from cmlib.probe import r2_scores, LAMBDAS

regime, lr = sys.argv[1], float(sys.argv[2])
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 100
CKPTS = {0, 10, 25, 50, 100, steps}
MB, ACC, L = 2, 2, 1024
T = Timer()
torch.set_num_threads(4)
h = H.wing()
torch.manual_seed(0)
model = lm.load(torch.bfloat16)
nmod = apply_lora(model)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
params = [p for p in model.parameters() if p.requires_grad]
T(f"LoRA on {nmod} modules, {sum(p.numel() for p in params) / 1e6:.2f}M trainable params")
opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 5))
A2 = np.array(H.ALPHABETS[("A", 2)])
data_rng = np.random.default_rng(12345)
# held-out evaluation sequences (seeds disjoint from training stream and from the final-eval seeds 0..4)
EV = [h.sample(4000, np.random.default_rng(500 + s)) for s in range(3)]
KL3 = float(np.mean([H.kl(h.next_probs(h.beliefs(x)[:4000]), h.k_next_probs(x, 3)[:4000])[2000:].mean() for x in EV]))


@torch.no_grad()
def quick_eval():
    model.eval()
    out = {}
    for alpha, other in (("A", "B"), ("B", "A")):
        ms, X, Y = [], [], []
        for xs in EV:
            logp, acts = lm.run(model, make_input(xs, alpha, 2), want_acts=False)
            b = h.beliefs(xs)[:4000]
            ms.append(lm_metrics(logp, h.next_probs(b), alpha, 2, other=other))
            if acts is not None:
                X.append(acts[:, 2000:4000].float().numpy()); Y.append(b[2000:4000])
        for k in ("kl_full", "kl_restr", "off_alpha_mass", "mass_other_alpha"):
            arr = np.array([m[k] for m in ms])
            out[f"{alpha}_{k}_late"] = float(arr[:, 2000:].mean())
            out[f"{alpha}_{k}_early_0_250"] = float(arr[:, :250].mean())
    model.train()
    return out


log = {"regime": regime, "lr": lr, "micro_batch": MB, "grad_accum": ACC, "seq_len": L, "r": 8, "alpha": 16,
       "kl_3hmm_late_on_eval_seqs": KL3, "train_loss": [], "checkpoints": {}}
ckdir = os.path.join(CACHE, f"lora_{regime}")
os.makedirs(ckdir, exist_ok=True)
model.train()
for step in range(steps + 1):
    if step in CKPTS:
        ev = quick_eval()
        log["checkpoints"][step] = ev
        torch.save(lora_state(model), os.path.join(ckdir, f"step{step}.pt"))
        T(f"step {step}: A KLfull {ev['A_kl_full_late']:.4f} KLr {ev['A_kl_restr_late']:.4f} off {ev['A_off_alpha_mass_late']:.3f} "
          f"| B KLfull {ev['B_kl_full_late']:.3f} KLr {ev['B_kl_restr_late']:.4f} "
          f"massA {ev['B_mass_other_alpha_late']:.3f} off {ev['B_off_alpha_mass_late']:.3f} (3-HMM KL {KL3:.4f})")
    if step == steps:
        break
    opt.zero_grad(set_to_none=True)
    tot = 0.0
    for _ in range(ACC):
        xs = np.stack([h.sample(L, data_rng) for _ in range(MB)])
        ids = torch.tensor(np.c_[np.full(MB, H.BOS_ID), A2[xs]]).cuda()
        out = model(input_ids=ids, labels=ids)
        (out.loss / ACC).backward()
        tot += out.loss.item() / ACC
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step(); sched.step()
    log["train_loss"].append(tot)
    if step % 5 == 0:
        T(f"step {step} loss {tot:.4f}")
with open(os.path.join(RES, "parts", f"exp4_train_{regime}.json"), "w") as f:
    json.dump(log, f)
T("done")
