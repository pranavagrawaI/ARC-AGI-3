"""Light probe trajectory over LoRA checkpoints: held-out A probe R^2 and fixed W_A -> B transfer.
usage: exp4_ckpt_probe.py REGIME STEP [STEP ...]"""
import json
import os
import sys

import numpy as np
import torch

from cmlib import hmm as H, lm
from cmlib.common import CACHE, RES, Timer, make_input
from cmlib.lora import apply_lora, load_lora_state
from cmlib.pooled import SeqStats, fit_pooled, predict
from cmlib.probe import r2_scores, LAMBDAS

T = Timer()
regime, steps = sys.argv[1], [int(s) for s in sys.argv[2:]]
h = H.wing()
EV = [h.sample(4000, np.random.default_rng(500 + s)) for s in range(3)]
BEL = [h.beliefs(x)[:4000] for x in EV]
model = lm.load(torch.bfloat16)
apply_lora(model)
out_path = os.path.join(RES, "parts", f"exp4_ckpt_probe_{regime}.json")
out = json.load(open(out_path)) if os.path.exists(out_path) else {}
for step in steps:
    load_lora_state(model, torch.load(os.path.join(CACHE, f"lora_{regime}", f"step{step}.pt")))
    model.eval()
    A = [lm.run(model, make_input(x, "A", 2))[1][:, 2000:] for x in EV]
    B = [lm.run(model, make_input(x, "B", 2))[1] for x in EV]
    rec = {"A_heldout_seq_r2": [], "WA_to_B_late_2000_4000": [], "WA_to_B_early_0_500": []}
    for l in range(17):
        XA = [a[l].float().numpy().astype(np.float64) for a in A]
        YA = [b[2000:] for b in BEL]
        st = [SeqStats(x, y) for x, y in zip(XA, YA)]
        fits = [fit_pooled([q for j, q in enumerate(st) if j != i], LAMBDAS) for i in range(3)]
        cv = {lam: np.mean([r2_scores(YA[i], predict(fits[i][lam], XA[i]))[0] for i in range(3)]) for lam in LAMBDAS}
        lam = max(cv, key=cv.get)
        rec["A_heldout_seq_r2"].append(float(cv[lam]))
        late = [r2_scores(BEL[i][2000:], predict(fits[i][lam], B[i][l, 2000:].float().numpy()))[0] for i in range(3)]
        early = [r2_scores(BEL[i][:500], predict(fits[i][lam], B[i][l, :500].float().numpy()))[0] for i in range(3)]
        rec["WA_to_B_late_2000_4000"].append(float(np.mean(late)))
        rec["WA_to_B_early_0_500"].append(float(np.mean(early)))
    out[str(step)] = rec
    T(f"{regime} step {step}: best A {max(rec['A_heldout_seq_r2']):.3f} (layer {int(np.argmax(rec['A_heldout_seq_r2'])) - 1}) "
      f"best WA->B late {max(rec['WA_to_B_late_2000_4000']):.3f} early {max(rec['WA_to_B_early_0_500']):.3f}")
    with open(out_path, "w") as f:
        json.dump(out, f)
