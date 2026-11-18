"""Exercise actual stochastic renormalization before any 27B model is loaded."""

import argparse
import json
from pathlib import Path
import time
import os
from install_sampling_patch import install


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    patched = install()
    (out / "sampling-patch.json").write_text(json.dumps(patched, indent=2))
    os.environ["SGLANG_SPEC_NATIVE_RENORM"] = "1"
    os.environ["FLASHINFER_DISABLE_JIT"] = "1"  # This preflight must need no compiler.
    import torch
    import flashinfer

    assert torch.__version__ == "2.13.0+cu130"
    assert flashinfer.__version__ == "0.6.17"
    assert torch.cuda.get_device_capability() == (12, 0)
    torch.manual_seed(2718)
    probs = torch.softmax(
        torch.randn(4, 248320, device="cuda", dtype=torch.float32), dim=-1
    )
    top_k = torch.full((4,), 20, device="cuda", dtype=torch.int32)
    top_p = torch.full((4,), 0.95, device="cuda", dtype=torch.float32)
    start = time.monotonic()
    from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob
    import sgl_kernel.sampling as sampling

    assert not sampling._has_flashinfer, "Native route not selected"
    actual_k = top_k_renorm_prob(probs, top_k)
    cutoff = torch.topk(probs, 20, dim=-1).values[:, -1:]
    reference_k = torch.where(probs >= cutoff, probs, 0)
    reference_k /= reference_k.sum(-1, keepdim=True)
    torch.testing.assert_close(actual_k, reference_k, atol=2e-6, rtol=2e-5)
    actual_p = top_p_renorm_prob(actual_k, top_p)
    values, indices = torch.sort(reference_k, descending=True, dim=-1)
    keep = values.cumsum(-1) - values < 0.95
    filtered = torch.where(keep, values, 0)
    filtered /= filtered.sum(-1, keepdim=True)
    reference_p = torch.zeros_like(probs).scatter(-1, indices, filtered)
    torch.testing.assert_close(actual_p, reference_p, atol=2e-6, rtol=2e-5)
    torch.cuda.synchronize()
    result = {
        "status": "PASS",
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "flashinfer": flashinfer.__version__,
        "renormalization_backend": "sgl_kernel precompiled native",
        "seconds": time.monotonic() - start,
        "shape": list(probs.shape),
        "sampling": {"temperature": 1.0, "top_k": 20, "top_p": 0.95},
        "top_k_max_absolute_error": (actual_k - reference_k).abs().max().item(),
        "top_p_max_absolute_error": (actual_p - reference_p).abs().max().item(),
        "flashinfer_jit_disabled": True,
        "scope": "Real top-k and top-p kernels against numerical references; full EAGLE tested by subsequent server preflight",
    }
    (out / "sampling.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.out)
