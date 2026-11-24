import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "allenai/OLMo-2-0425-1B"
GATHER_IDS = [435, 1229, 650, 735, 386, 1901]  # F Q V K M Z


def _local_only():
    return os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes")


def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL, local_files_only=_local_only())


def load(dtype=torch.bfloat16, device="cuda"):
    m = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=_local_only(), dtype=dtype,
                                             attn_implementation="sdpa").to(device)
    m.eval()
    return m


@torch.no_grad()
def run(model, ids, want_acts=True, chunk=512, gather=GATHER_IDS):
    """ids: 1-D LongTensor. Returns
    logp: (T, len(gather)) float32 full-vocab-normalized log-probs at every input position
          (row i predicts the token after ids[i]);
    acts: (17, T, D) residual stream (row 0 = embeddings, row l+1 = after block l), CPU, model dtype."""
    dev = next(model.parameters()).device
    x = ids.to(dev)[None]
    acts = []
    hooks = []
    if want_acts:
        hooks.append(model.model.embed_tokens.register_forward_hook(lambda m, i, o: acts.append(o[0].detach().cpu())))
        for layer in model.model.layers:
            hooks.append(layer.register_forward_hook(
                lambda m, i, o: acts.append((o[0] if isinstance(o, tuple) else o)[0].detach().cpu())))
    try:
        h = model.model(input_ids=x, use_cache=False).last_hidden_state[0]
    finally:
        for hk in hooks:
            hk.remove()
    g = torch.tensor(gather, device=dev)
    outs = []
    for s in range(0, h.shape[0], chunk):
        lg = model.lm_head(h[s:s + chunk]).float()
        outs.append((lg[:, g] - torch.logsumexp(lg, -1, keepdim=True)).cpu())
    logp = torch.cat(outs)
    return logp, (torch.stack(acts) if want_acts else None)
