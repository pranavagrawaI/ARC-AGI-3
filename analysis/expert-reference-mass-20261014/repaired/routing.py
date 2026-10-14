"""Five expert outputs, with weights normalized over the trained top ten."""
import os

import torch


def route_reference_mass(*, hidden_states, gating_output, topk, renormalize):
    if topk != 5 or not renormalize or gating_output.shape[-1] != 512:
        raise ValueError("reference routing requires top5/renormalize/512 experts")
    logits, ids = torch.topk(gating_output.float(), 10, dim=-1, sorted=True)
    # Global-softmax normalization cancels in p_i / sum(top10 p).
    weights = torch.softmax(logits, dim=-1)
    return weights[:, :5].contiguous(), ids[:, :5].to(torch.int32).contiguous()


def reference_routing_for(config, is_nextn, layer_id):
    reference = int(os.environ.get("ARC3_REFERENCE_TOPK", "0"))
    if is_nextn:
        print(f"REFERENCE_ROUTING_DRAFT layer={layer_id} active={config.num_experts_per_tok} custom=False", flush=True)
        return None
    if reference != 10 or config.num_experts_per_tok != 5 or config.num_experts != 512 or not config.norm_topk_prob:
        raise ValueError("candidate must use active5/reference10/native512/norm_topk_prob")
    print(f"REFERENCE_ROUTING_TARGET layer={layer_id} active=5 reference=10", flush=True)
    return route_reference_mass
