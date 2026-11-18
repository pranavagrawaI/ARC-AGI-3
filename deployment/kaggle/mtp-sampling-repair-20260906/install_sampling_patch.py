"""Opt-in native renormalization in the exact pinned wrapper; no wheel replacement."""

import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

ORIGINAL_SHA = "80f01a3812ff7be617c169ab82828a09fa4f22969c42cdcaedf3a01c628bfdf8"
OLD = "try:\n    import flashinfer.sampling as _flashinfer_sampling\n\n    _has_flashinfer = True\nexcept ImportError:\n    _has_flashinfer = False"
NEW = """# Qualification candidate: use existing precompiled kernels when requested.
import os
if os.environ.get("SGLANG_SPEC_NATIVE_RENORM") == "1":
    _has_flashinfer = False
else:
    try:
        import flashinfer.sampling as _flashinfer_sampling
        _has_flashinfer = True
    except ImportError:
        _has_flashinfer = False"""


def install():
    distribution = metadata.distribution("sglang-kernel")
    assert distribution.version == "0.4.6.post1", "sglang-kernel version drift"
    path = Path(distribution.locate_file("sgl_kernel/sampling.py"))
    current = path.read_text()
    if NEW in current:
        original = current.replace(NEW, OLD, 1)
    else:
        original = current
    assert hashlib.sha256(original.encode()).hexdigest() == ORIGINAL_SHA, (
        "Sampling source drift"
    )
    assert original.count(OLD) == 1
    candidate = original.replace(OLD, NEW, 1)
    compile(candidate, str(path), "exec")
    if current != candidate:
        path.write_text(candidate)
    return {
        "path": str(path),
        "version": distribution.version,
        "original_sha256": ORIGINAL_SHA,
        "patched_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
        "enable_environment": {"SGLANG_SPEC_NATIVE_RENORM": "1"},
        "scope": "Only choose existing sgl_kernel native top-k/top-p implementations; tree verification unchanged",
    }


if __name__ == "__main__":
    print(json.dumps(install(), indent=2))
