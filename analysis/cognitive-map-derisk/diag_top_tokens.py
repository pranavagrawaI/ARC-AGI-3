"""Which tokens absorb the off-alphabet mass late in a sequence (base model and LoRA checkpoints)."""
import json
import os
import sys

import numpy as np
import torch

from cmlib import hmm as H, lm
from cmlib.common import CACHE, RES, make_input
from cmlib.lora import apply_lora, load_lora_state

tok = lm.load_tokenizer()
model = lm.load()
out = {}


@torch.no_grad()
def top(name, h, ns, alpha, k=10):
    xs = h.sample(4000, np.random.default_rng(0))
    inp = make_input(xs, alpha, ns).cuda()
    hs = model.model(input_ids=inp[None]).last_hidden_state[0]
    acc = {}
    for a, b in ((0, 250), (3000, 4000)):
        p = torch.cat([model.lm_head(hs[s:s + 250]).float().softmax(-1) for s in range(a, b, 250)]).mean(0)
        t = p.topk(k)
        acc[f"[{a},{b})"] = [(tok.convert_ids_to_tokens(i.item()), round(v.item(), 4)) for v, i in zip(t.values, t.indices)]
    return acc


for name, ctor in H.HMMS.items():
    h = ctor()
    for alpha in "AB":
        out[f"base_{name}_{alpha}"] = top(name, h, h.n_sym, alpha)
        print(name, alpha, out[f"base_{name}_{alpha}"])
apply_lora(model)
for reg, step in (("4A", 100), ("4B", 100)):
    load_lora_state(model, torch.load(os.path.join(CACHE, f"lora_{reg}", f"step{step}.pt")))
    out[f"{reg}_step{step}_Wing_B"] = top("Wing", H.wing(), 2, "B")
    print(reg, out[f"{reg}_step{step}_Wing_B"])
json.dump(out, open(os.path.join(RES, "diag_top_tokens.json"), "w"), indent=1)
