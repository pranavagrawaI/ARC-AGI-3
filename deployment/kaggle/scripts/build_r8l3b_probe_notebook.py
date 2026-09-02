#!/usr/bin/env python3
"""Build the frozen Kaggle R8-L3B lifecycle qualification notebook."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "PROTOTYPE-qwen3_8_r8_l3b_probe.py"
MANIFEST = ROOT / "PROTOTYPE-qwen3_8_fp8-manifest.json"
COMPETITION = "arc-prize-2026-arc-agi-3"
MACHINE_SHAPE = "NvidiaRtxPro6000"
MODEL_DATASET_SOURCE = "jakobbrggen/qwen3-8-27b-fp8-hf-snapshot"
BASE_RUNTIME_DATASET_SOURCE = "mahdiseddigh/kaggle-t4-stack-2026-08-28"
PATCH_RUNTIME_DATASET_SOURCE = "pranavagrawai/sglang-r8l3b-orderly-shutdown-runtime"
SGLANG_REPOSITORY = "https://github.com/sgl-project/sglang.git"
SGLANG_COMMIT = "8e04f66a7046fa083cda3f5b8e8325632a5af8fd"
SGLANG_WHEEL = "sglang-0.5.19.dev591+g8e04f66a7-cp312-cp312-linux_x86_64.whl"
REMOTE_SGLANG_WHEEL = "sglang-0.5.19.dev591g8e04f66a7-cp312-cp312-linux_x86_64.whl"
BASE_SGLANG_WHEEL_SHA256 = "9d07f8bf3b718dc4cd970e72a03415b0b0597da89625acbee7a655d6387e7a8d"
SGLANG_WHEEL_SHA256 = "06cc48d1e067c0eb08a1548c5c244bd0f541b114947c29b87042ce3156bb5cc0"
REMOTE_DATASET_READBACK_SHA256 = (
    "cee470f08217da591b8a89d3a80f1f9b4f03aecffced29f69d1b2a5c510fda9d"
)
RUNTIME_LOCK_SHA256 = "d473b6e092e01f66313e12069ac8873f2ff1c6c3b84b51c1b37312229de5475c"
RUNTIME_DATASET_MANIFEST = "runtime-dataset-manifest.json"
RUNTIME_DATASET_MANIFEST_SHA256 = (
    "f1a50bb224dc26ac48c0fd219c28e4fadc92ede6d2a80a26fd18d9a004a79b94"
)


def code_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "metadata": {"trusted": True},
        "outputs": [],
        "execution_count": None,
        "source": source,
    }


def markdown_cell(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def validate_source(value: str, parts: int, name: str) -> str:
    if len(value.split("/")) != parts:
        raise SystemExit(f"{name} must have {parts} slash-separated fields, got {value!r}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-id", required=True, help="Kaggle username/kernel-slug")
    parser.add_argument(
        "--model-dataset-source",
        required=True,
        help="username/dataset containing the exact model snapshot",
    )
    parser.add_argument(
        "--base-runtime-dataset-source",
        required=True,
        help="username/dataset containing the inherited offline wheelhouse",
    )
    parser.add_argument(
        "--patch-runtime-dataset-source",
        required=True,
        help="username/dataset containing the exact R8-L3B wheel and lock",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "r8l3b-notebook",
        help="generated Kaggle kernel directory",
    )
    parser.add_argument(
        "--gate",
        choices=("r8-l3b",),
        default="r8-l3b",
        help="R8-L3B complete lifecycle qualification",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_source(args.kernel_id, 2, "--kernel-id")
    validate_source(args.model_dataset_source, 2, "--model-dataset-source")
    validate_source(
        args.base_runtime_dataset_source, 2, "--base-runtime-dataset-source"
    )
    validate_source(
        args.patch_runtime_dataset_source, 2, "--patch-runtime-dataset-source"
    )
    if args.model_dataset_source != MODEL_DATASET_SOURCE:
        raise SystemExit(f"probe is bound to model dataset {MODEL_DATASET_SOURCE!r}")
    if args.base_runtime_dataset_source != BASE_RUNTIME_DATASET_SOURCE:
        raise SystemExit(
            f"probe is bound to base runtime dataset {BASE_RUNTIME_DATASET_SOURCE!r}"
        )
    if args.patch_runtime_dataset_source != PATCH_RUNTIME_DATASET_SOURCE:
        raise SystemExit(
            f"probe is bound to patch runtime dataset {PATCH_RUNTIME_DATASET_SOURCE!r}"
        )
    output_root = args.output_dir.resolve()
    notebook_name = (
        "r8l3b_probe.ipynb"
        if args.gate == "r8-l3b"
        else "r8a_probe.ipynb"
        if args.gate == "r8-a"
        else "r8b2_probe.ipynb"
        if args.gate == "r8-b2"
        else "r8b_probe.ipynb"
        if args.gate == "r8-b"
        else "phase_a_probe.ipynb"
    )
    notebook_path = output_root / notebook_name
    metadata_path = output_root / "kernel-metadata.json"
    suffix = (
        "r8l3b"
        if args.gate == "r8-l3b"
        else "r8a"
        if args.gate == "r8-a"
        else "r8b2"
        if args.gate == "r8-b2"
        else "r8b"
        if args.gate == "r8-b"
        else "phase-a"
    )
    bootstrap_name = f"qwen38-fp8-{suffix}-bootstrap"
    probe_output_name = f"qwen38-fp8-{suffix}-probe"
    gate_arguments = (
        [
            "--attention-backend",
            "triton",
            "--sampling-backend",
            "pytorch",
            "--profile-imports",
            "--r8-l3b",
        ]
        if args.gate == "r8-l3b"
        else [
            "--attention-backend",
            "triton",
            "--profile-imports",
            "--readiness-only",
        ]
        if args.gate == "r8-a"
        else [
            "--attention-backend",
            "triton",
            "--sampling-backend",
            "pytorch",
            "--profile-imports",
            "--r8-b2",
        ]
        if args.gate == "r8-b2"
        else [
            "--attention-backend",
            "triton",
            "--sampling-backend",
            "pytorch",
            "--profile-imports",
            "--r8-b",
        ]
        if args.gate == "r8-b"
        else []
    )

    probe_b64 = base64.b64encode(PROBE.read_bytes()).decode("ascii")
    manifest_b64 = base64.b64encode(MANIFEST.read_bytes()).decode("ascii")

    write_assets = code_cell(
        dedent(
            f"""\
            import base64
            from pathlib import Path

            Path('/tmp/PROTOTYPE-qwen3_8_r8_l3b_probe.py').write_bytes(
                base64.b64decode({probe_b64!r})
            )
            Path('/tmp/PROTOTYPE-qwen3_8_fp8-manifest.json').write_bytes(
                base64.b64decode({manifest_b64!r})
            )
            """
        )
    )

    install_runtime = code_cell(
        dedent(
            f"""\
            import hashlib
            import json
            import shutil
            import subprocess
            import sys
            from pathlib import Path

            def sha256_file(path):
                digest = hashlib.sha256()
                with path.open('rb') as handle:
                    while chunk := handle.read(8 * 1024 * 1024):
                        digest.update(chunk)
                return digest.hexdigest()

            def require_regular(path, label):
                if not path.is_file() or path.is_symlink():
                    raise RuntimeError(f'{{label}} is not one mounted regular file: {{path}}')
                return path

            input_root = Path('/kaggle/input')

            def unique_exact_file(name, expected_sha256, label):
                matches = [
                    path
                    for path in input_root.rglob(name)
                    if path.is_file()
                    and not path.is_symlink()
                    and sha256_file(path) == expected_sha256
                ]
                if len(matches) != 1:
                    raise RuntimeError(
                        f'expected one exact {{label}} under {{input_root}}, found {{matches}}'
                    )
                return matches[0]

            mounted_wheel_path = unique_exact_file(
                {REMOTE_SGLANG_WHEEL!r},
                {SGLANG_WHEEL_SHA256!r},
                'provider-normalized patched wheel',
            )
            dataset_manifest_path = unique_exact_file(
                {RUNTIME_DATASET_MANIFEST!r},
                {RUNTIME_DATASET_MANIFEST_SHA256!r},
                'patch dataset manifest',
            )
            if mounted_wheel_path.parent != dataset_manifest_path.parent:
                raise RuntimeError(
                    'exact patched wheel and dataset manifest are not co-located: '
                    f'wheel={{mounted_wheel_path}}, manifest={{dataset_manifest_path}}'
                )
            patch_root = mounted_wheel_path.parent
            lock_path = require_regular(patch_root / 'requirements.lock', 'patch lock')
            if sha256_file(lock_path) != {RUNTIME_LOCK_SHA256!r}:
                raise RuntimeError('mounted patch lock does not match the inherited bytes')
            dataset_manifest = json.loads(dataset_manifest_path.read_text())
            if (
                dataset_manifest.get('schema') != 'sglang-r8l3b-thin-runtime-dataset-v1'
                or dataset_manifest.get('dataset', {{}}).get('source')
                != {PATCH_RUNTIME_DATASET_SOURCE!r}
            ):
                raise RuntimeError('mounted patch dataset manifest has the wrong semantic identity')
            manifest_files = {{record['path']: record for record in dataset_manifest.get('files', [])}}
            local_wheel_record = manifest_files.get({SGLANG_WHEEL!r})
            if (
                local_wheel_record is None
                or local_wheel_record.get('sha256') != {SGLANG_WHEEL_SHA256!r}
            ):
                raise RuntimeError('patch dataset manifest does not bind the canonical wheel bytes')

            base_wheel_path = unique_exact_file(
                {SGLANG_WHEEL!r},
                {BASE_SGLANG_WHEEL_SHA256!r},
                'base SGLang wheel',
            )
            base_lock_paths = [
                path
                for path in input_root.rglob('requirements.lock')
                if path.is_file()
                and not path.is_symlink()
                and path != lock_path
                and sha256_file(path) == {RUNTIME_LOCK_SHA256!r}
            ]
            if len(base_lock_paths) != 1:
                raise RuntimeError(
                    'expected one exact non-patch base lock under the input root, '
                    f'found locks={{base_lock_paths}}'
                )
            base_lock_path = base_lock_paths[0]
            if lock_path.read_bytes() != base_lock_path.read_bytes():
                raise RuntimeError('patch lock bytes differ from the inherited base lock')
            wheelhouse = base_wheel_path.parent
            print(json.dumps({{
                'resolved_patch_root': str(patch_root),
                'resolved_patch_wheel': str(mounted_wheel_path),
                'resolved_base_wheel': str(base_wheel_path),
                'resolved_base_lock': str(base_lock_path),
                'resolved_wheelhouse': str(wheelhouse),
            }}, sort_keys=True))
            bootstrap_dir = Path('/kaggle/working/{bootstrap_name}')
            bootstrap_dir.mkdir(parents=True, exist_ok=False)
            canonical_wheel_path = bootstrap_dir / {SGLANG_WHEEL!r}
            shutil.copyfile(mounted_wheel_path, canonical_wheel_path)
            require_regular(canonical_wheel_path, 'canonical copied patched wheel')
            if sha256_file(canonical_wheel_path) != {SGLANG_WHEEL_SHA256!r}:
                raise RuntimeError('canonical copied SGLang wheel does not match the pinned bytes')
            runtime_provenance = Path('/tmp/sglang-runtime-provenance.json')
            runtime_provenance.write_text(json.dumps({{
                'dataset_source': {PATCH_RUNTIME_DATASET_SOURCE!r},
                'dataset_version': 1,
                'base_dataset_source': {BASE_RUNTIME_DATASET_SOURCE!r},
                'base_dataset_version': 1,
                'repository': {SGLANG_REPOSITORY!r},
                'commit': {SGLANG_COMMIT!r},
                'artifact_path': str(canonical_wheel_path),
                'artifact_sha256': {SGLANG_WHEEL_SHA256!r},
                'mounted_artifact_path': str(mounted_wheel_path),
                'mounted_artifact_name': {REMOTE_SGLANG_WHEEL!r},
                'canonical_artifact_name': {SGLANG_WHEEL!r},
                'remote_dataset_readback_sha256': {REMOTE_DATASET_READBACK_SHA256!r},
                'lock_path': str(lock_path),
                'lock_sha256': {RUNTIME_LOCK_SHA256!r},
                'dataset_manifest_path': str(dataset_manifest_path),
                'dataset_manifest_sha256': {RUNTIME_DATASET_MANIFEST_SHA256!r},
                'base_artifact_path': str(base_wheel_path),
                'base_artifact_sha256': {BASE_SGLANG_WHEEL_SHA256!r},
                'base_lock_path': str(base_lock_path),
                'base_lock_sha256': {RUNTIME_LOCK_SHA256!r},
            }}, indent=2) + '\\n')

            overrides_path = Path('/tmp/sglang-runtime-overrides.txt')
            overrides_path.write_text(
                '\\n'.join([
                    'flashinfer-python==0.6.17',
                    'numba==0.65.1',
                    'compressed-tensors==0.18.0',
                    'outlines==1.3.3',
                    'opencv-python-headless==5.0.0.93',
                    'openai==3.5.0',
                    '',
                ])
            )
            uv_install_command = [
                sys.executable,
                '-m',
                'pip',
                'install',
                '--no-index',
                '--find-links',
                str(wheelhouse),
                'uv==0.12.7',
            ]
            with (bootstrap_dir / 'runtime-install.log').open('w') as install_log:
                subprocess.run(
                    uv_install_command,
                    check=True,
                    stdout=install_log,
                    stderr=subprocess.STDOUT,
                )
            uv_executable = shutil.which('uv')
            if not uv_executable:
                raise RuntimeError('uv executable missing after offline installation')
            install_command = [
                uv_executable,
                'pip',
                'install',
                '--system',
                '--no-index',
                '--find-links',
                str(wheelhouse),
                '--override',
                str(overrides_path),
                '--upgrade',
                str(canonical_wheel_path),
                'torch==2.13.0+cu130',
                'transformers==5.12.1',
                'flashinfer-python==0.6.17',
                'sglang-kernel==0.4.6.post1',
                'triton==3.7.1',
                'cuda-tile==1.6.0rc5',
                'flash-attn-4==4.0.0b19',
                'opentelemetry-semantic-conventions==0.65b0',
            ]
            (bootstrap_dir / 'runtime-install-command.json').write_text(
                json.dumps({{
                    'uv_install': uv_install_command,
                    'runtime_install': install_command,
                    'overrides': overrides_path.read_text().splitlines(),
                }}, indent=2) + '\\n'
            )
            with (bootstrap_dir / 'runtime-install.log').open('a') as install_log:
                subprocess.run(
                    install_command,
                    check=True,
                    stdout=install_log,
                    stderr=subprocess.STDOUT,
                )
            (bootstrap_dir / 'runtime-provenance.json').write_text(
                runtime_provenance.read_text()
            )
            with (bootstrap_dir / 'runtime-packages.json').open('w') as package_log:
                subprocess.run(
                    [sys.executable, '-m', 'pip', 'list', '--format', 'json'],
                    check=True,
                    stdout=package_log,
                )
            """
        )
    )

    run_probe = code_cell(
        dedent(
            f"""\
            import subprocess
            import sys
            from pathlib import Path

            output_dir = Path('/kaggle/working/{probe_output_name}')
            probe_command = [
                    sys.executable,
                    '/tmp/PROTOTYPE-qwen3_8_r8_l3b_probe.py',
                    '--output-dir',
                    str(output_dir),
                    '--manifest',
                    '/tmp/PROTOTYPE-qwen3_8_fp8-manifest.json',
                    '--input-root',
                    '/kaggle/input',
                    '--model-source',
                    {args.model_dataset_source!r},
                    '--runtime-provenance',
                    str(runtime_provenance),
                    '--startup-timeout-s',
                    '1200',
                    '--request-timeout-s',
                    '180',
                ]
            probe_command.extend({gate_arguments!r})
            subprocess.run(
                probe_command,
                check=True,
            )
            print((output_dir / 'result.json').read_text())
            """
        )
    )

    notebook = {
        "metadata": {
            "kernelspec": {
                "language": "python",
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {"name": "python", "file_extension": ".py"},
            "kaggle": {
                "accelerator": "nvidiaRtx6000",
                "isInternetEnabled": False,
                "isGpuEnabled": True,
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            markdown_cell(
                (
                    "# PROTOTYPE — Qwen3.8 FP8 R8-L3B root-first lifecycle qualification\n\n"
                    "**Question:** after the unchanged R8-B2 warmup and frozen request, does "
                    "installing the exact R8-L3B patched SGLang wheel and sending TERM only to "
                    "the root let coordinated shutdown close the "
                    "complete externally observed process tree, port listener, GPU applications, "
                    "and teardown log by the frozen deadline?\n\n"
                    "It preserves the accepted R8-L2 request, observer, timeout, ownership, "
                    "containment, and strict log-classification bytes. The only runtime treatment "
                    "is the separately reviewed R8-L3B wheel; dependencies still come from the "
                    "unchanged base runtime dataset. "
                    "It runs no ARC game, submits nothing, and tests no concurrency."
                    if args.gate == "r8-l3b"
                    else "# PROTOTYPE — Qwen3.8 FP8 R8-A Blackwell initialization\n\n"
                    "**Question:** can the immutable Qwen3.8-27B-FP8 checkpoint and "
                    "pinned SGLang runtime become request-ready on SM120 after changing "
                    "only the attention backend from FlashInfer to Triton? The resolved "
                    "sampling backend is observed but not changed.\n\n"
                    "This notebook stops at readiness. It sends no model request, runs no "
                    "ARC game, submits nothing, and tests no concurrency."
                    if args.gate == "r8-a"
                    else "# PROTOTYPE — Qwen3.8 FP8 R8-B2 warmup-complete request qualification\n\n"
                    "**Question:** after the pinned SGLang runtime completes its one automatic "
                    "VLM warmup, can the R8-A2-qualified Qwen route execute exactly one frozen "
                    "probe-owned visual request and return one parser-accepted native tool call?\n\n"
                    "This notebook preserves the production warmup and all lifecycle behavior. "
                    "It records one runtime-owned warmup request separately from one probe-owned "
                    "request, runs no ARC game, submits nothing, and tests no concurrency."
                    if args.gate == "r8-b2"
                    else "# PROTOTYPE — Qwen3.8 FP8 R8-B request qualification\n\n"
                    "**Question:** can the R8-A2-qualified Qwen/SGLang substrate, with "
                    "sampling explicitly resolved to PyTorch, execute one frozen visual "
                    "request and return one parser-accepted native tool call?\n\n"
                    "This notebook changes no lifecycle behavior. It records shutdown separately, "
                    "runs no ARC game, submits nothing, and tests no concurrency."
                    if args.gate == "r8-b"
                    else "# PROTOTYPE — Qwen3.8 FP8 offline Phase A qualification\n\n"
                    "**Question:** can the immutable Qwen3.8-27B-FP8 checkpoint and "
                    "pinned SGLang route cold-load without network access and turn one "
                    "ARC-like image into one parsed native tool call?\n\n"
                    "This notebook stops after one request. It does not run an ARC game, "
                    "submit to the leaderboard, or test concurrency."
                )
            ),
            write_assets,
            install_runtime,
            run_probe,
        ],
    }
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile(str(cell["source"]), f"{notebook_name}:cell-{index}", "exec")
    metadata = {
        "id": args.kernel_id,
        "title": (
            "PROTOTYPE Qwen3.8 FP8 R8-L3B root-first lifecycle qualification"
            if args.gate == "r8-l3b"
            else "PROTOTYPE Qwen3.8 FP8 R8-A Blackwell initialization"
            if args.gate == "r8-a"
            else "PROTOTYPE Qwen3.8 FP8 R8-B request path"
            if args.gate in {"r8-b", "r8-b2"}
            else "PROTOTYPE Qwen3.8 FP8 Phase A qualification"
        ),
        "code_file": notebook_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "machine_shape": MACHINE_SHAPE,
        "enable_internet": False,
        "enable_tpu": False,
        "dataset_sources": [
            args.model_dataset_source,
            args.base_runtime_dataset_source,
            args.patch_runtime_dataset_source,
        ],
        "kernel_sources": [],
        "competition_sources": [COMPETITION],
        "model_sources": [],
    }

    output_root.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(json.dumps(notebook, indent=1) + "\n")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {notebook_path}")
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
