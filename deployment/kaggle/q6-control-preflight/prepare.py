"""Build or verify the single control preflight; never launches a kernel."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CANDIDATE = HERE.parent / "q6-nooa-candidate"
OUTPUT = ROOT / "analysis/deployment-probes/20260905-q6-control-preflight"
CARD = OUTPUT / "EXPERIMENT-CARD.md"
KERNEL = "pranavagrawai/q6-control-preflight-20260905"
CONTRACT = "1d65e3669ff7ace45eb271aaf943dff35fc279a6ebeba95912e89884d3a05459"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected():
    sys.path.insert(0, str(CANDIDATE))
    spec = importlib.util.spec_from_file_location(
        "q6_preflight_builder", CANDIDATE / "build_notebook.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    from freeze_candidate import verify

    manifest = verify()
    if manifest["contract_sha256"] != CONTRACT:
        raise RuntimeError(
            "preflight requires its qualified contract; use a new preflight for changes"
        )
    control = CANDIDATE / "control.ipynb"
    # Rebuilding compressed payloads across Python/zlib versions can change bytes.
    if (
        digest(control)
        != "49bbacca7841a9605b73bd56ddbd0b347357ad91b15e9d2ce925b3d52304d67e"
    ):
        raise RuntimeError("qualified control notebook drift")
    notebook = json.loads(control.read_text())
    builder.validate(notebook)
    metadata = {
        "id": KERNEL,
        "title": "Q6 control preflight 20260905",
        "code_file": "preflight.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "machine_shape": "NvidiaRtxPro6000",
        "enable_internet": False,
        "enable_tpu": False,
        "dataset_sources": [
            "jakobbrggen/qwen3-8-27b-fp8-hf-snapshot",
            "mahdiseddigh/kaggle-t4-stack-2026-08-28",
            "pranavagrawai/sglang-r8l3b-orderly-shutdown-runtime",
            "pranavagrawai/tycho-nooa-solver-wheelhouse-cp312-v11",
        ],
        "kernel_sources": [],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "model_sources": [],
    }
    return {
        "preflight.ipynb": json.dumps(notebook, indent=1) + "\n",
        "kernel-metadata.json": json.dumps(metadata, indent=2) + "\n",
    }


def binding():
    return {
        "schema": "q6-control-preflight-v1",
        "kernel": KERNEL,
        "contract_sha256": CONTRACT,
        "kaggle_cli_version": "2.2.4",
        "session_timeout_seconds": 14400,
        "maximum_gpu_hours": 4,
        "maximum_pushes": 1,
        "automatic_retries": 0,
        "argv": [
            "kaggle",
            "kernels",
            "push",
            "--path",
            str(HERE.relative_to(ROOT)),
            "--timeout",
            "14400",
            "--accelerator",
            "NvidiaRtxPro6000",
        ],
        "files": {
            str(p.relative_to(ROOT)): digest(p)
            for p in (
                HERE / "preflight.ipynb",
                HERE / "kernel-metadata.json",
                HERE / "prepare.py",
                CARD,
            )
        },
    }


def prepare(check=False):
    artifacts = expected()
    if check:
        for name, content in artifacts.items():
            if (HERE / name).read_text() != content:
                raise RuntimeError(f"preflight drift: {name}")
        if json.loads((HERE / "LAUNCH.json").read_text()) != binding():
            raise RuntimeError("preflight launch/card drift")
    else:
        for name, content in artifacts.items():
            (HERE / name).write_text(content)
        (HERE / "LAUNCH.json").write_text(json.dumps(binding(), indent=2) + "\n")
    return binding()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.check), indent=2))
