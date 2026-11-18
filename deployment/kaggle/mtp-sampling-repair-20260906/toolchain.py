"""Select a real SM120-capable toolkit before importing FlashInfer. No bypasses."""

import importlib.metadata as metadata
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def inspect_nvcc(path):
    path = Path(path).resolve()
    version = subprocess.check_output([str(path), "--version"], text=True, timeout=10)
    match = re.search(r"release (\d+)\.(\d+)", version)
    if not match:
        raise RuntimeError("Cannot determine nvcc version")
    codes = subprocess.check_output(
        [str(path), "--list-gpu-code"], text=True, timeout=10
    )
    root = path.parent.parent
    return {
        "nvcc": str(path),
        "root": str(root),
        "version": list(map(int, match.groups())),
        "version_output": version,
        "gpu_codes": codes.split(),
        "headers_present": (root / "include/cuda_runtime.h").is_file(),
    }


def select():
    candidates = []
    for key in ("CUDA_HOME", "CUDA_PATH"):
        if os.getenv(key):
            candidates.append(Path(os.environ[key]) / "bin/nvcc")
    if shutil.which("nvcc"):
        candidates.append(Path(shutil.which("nvcc")))
    for name in ("nvidia-cuda-nvcc", "nvidia-cuda-nvcc-cu12"):
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        candidates.extend(
            Path(distribution.locate_file(f))
            for f in distribution.files or []
            if str(f).endswith("/bin/nvcc")
        )
    records, eligible = [], []
    for path in dict.fromkeys(candidates):
        try:
            record = inspect_nvcc(path)
            records.append(record)
            # nvcc lists base SM codes, not the architecture-feature suffixes.
            # FlashInfer selects 120f through its normal >=12.9 version gate.
            if (
                tuple(record["version"]) >= (12, 9)
                and "sm_120" in record["gpu_codes"]
                and record["headers_present"]
            ):
                eligible.append(record)
        except Exception as error:
            records.append({"nvcc": str(path), "error": str(error)})
    if not eligible:
        raise RuntimeError(f"No complete SM120f toolkit found: {records}")
    chosen = max(eligible, key=lambda r: r["version"])
    env = {
        "CUDA_HOME": chosen["root"],
        "CUDA_PATH": chosen["root"],
        "FLASHINFER_NVCC": chosen["nvcc"],
    }
    env["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    )
    os.environ.update(env)
    # Do not force FLASHINFER_CUDA_ARCH_LIST: real device discovery and its
    # compiler-version gate must succeed naturally in the fresh process.
    if os.environ.get("FLASHINFER_CUDA_ARCH_LIST"):
        raise RuntimeError(
            "Unexpected architecture override; refuse ambiguous qualification"
        )
    return {"candidates": records, "selected": chosen, "environment": env}
