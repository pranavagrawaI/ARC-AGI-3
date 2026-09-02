#!/usr/bin/env python3
"""Build the throwaway Kaggle F0 fallback contract notebook."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "PROTOTYPE-qwen3_vl_f0_probe.py"
COMPETITION = "arc-prize-2026-arc-agi-3"
MACHINE_SHAPE = "NvidiaRtxPro6000"
MODEL_SOURCE = "qwen-lm/qwen-3-vl/transformers/8b-instruct-fp8/1"
RUNTIME_DATASET_SOURCE = "mahdiseddigh/kaggle-t4-stack-2026-08-28"
VLLM_WHEEL = "vllm-0.28.0-cp38-abi3-manylinux_2_28_x86_64.whl"
VLLM_WHEEL_SHA256 = "addb0ffdaafd8155d75e9b3f5ddb3da28fdee9e8a7097ede91f7db2e9e1a3889"
RUNTIME_LOCK_SHA256 = "d473b6e092e01f66313e12069ac8873f2ff1c6c3b84b51c1b37312229de5475c"
DERIVED_RUNTIME_LOCK_SHA256 = "d07866e8a6961193a1285d1514560c46610413f9a05fc19738c286320a98c481"
EXCLUDED_SGLANG_REQUIREMENT = (
    "sglang @ git+https://github.com/sgl-project/sglang.git@"
    "8e04f66a7046fa083cda3f5b8e8325632a5af8fd#subdirectory=python"
)
MAX_NOTEBOOK_RUNTIME_S = 1_200
FINALIZATION_MARGIN_S = 90


def main() -> None:
    args = parse_args()
    validate_source(args.kernel_id, 2, "--kernel-id")
    validate_source(args.runtime_dataset_source, 2, "--runtime-dataset-source")
    validate_source(args.model_source, 5, "--model-source")
    if args.runtime_dataset_source != RUNTIME_DATASET_SOURCE:
        raise SystemExit(f"builder is bound to runtime dataset {RUNTIME_DATASET_SOURCE!r}")
    if args.model_source != MODEL_SOURCE:
        raise SystemExit(f"builder is bound to model source {MODEL_SOURCE!r}")

    output_root = args.output_dir.resolve()
    notebook_path = output_root / "f0_probe.ipynb"
    metadata_path = output_root / "kernel-metadata.json"
    probe_b64 = base64.b64encode(PROBE.read_bytes()).decode("ascii")

    write_probe = code_cell(
        dedent(
            f"""\
            import base64
            import json
            import os
            import signal
            import threading
            import time
            from pathlib import Path

            F0_RUN_STARTED = time.monotonic()
            F0_HARD_DEADLINE = F0_RUN_STARTED + {MAX_NOTEBOOK_RUNTIME_S}
            F0_OPERATION_DEADLINE = F0_HARD_DEADLINE - {FINALIZATION_MARGIN_S}

            def f0_hard_watchdog():
                time.sleep({MAX_NOTEBOOK_RUNTIME_S})
                record = {{
                    'status': 'hard_deadline_elapsed',
                    'hard_deadline_s': {MAX_NOTEBOOK_RUNTIME_S},
                    'notebook_pid': os.getpid(),
                }}
                Path('/kaggle/working/qwen3-vl-f0-r4-hard-timeout.json').write_text(
                    json.dumps(record, indent=2) + '\\n'
                )
                process_files = [
                    Path('/kaggle/working/qwen3-vl-f0-r4-outer-process.json'),
                    Path('/kaggle/working/qwen3-vl-f0-r4-probe/server-process.json'),
                ]
                for process_file in process_files:
                    if process_file.is_file():
                        try:
                            process = json.loads(process_file.read_text())
                            os.killpg(int(process['pgid']), signal.SIGKILL)
                        except (KeyError, ProcessLookupError, ValueError, json.JSONDecodeError):
                            pass
                os.kill(os.getpid(), signal.SIGKILL)

            threading.Thread(target=f0_hard_watchdog, daemon=True).start()
            Path('/tmp/PROTOTYPE-qwen3_vl_f0_probe.py').write_bytes(
                base64.b64decode({probe_b64!r})
            )
            """
        )
    )

    install_runtime = code_cell(
        dedent(
            f"""\
            import hashlib
            import json
            import os
            import shutil
            import signal
            import subprocess
            import sys
            import time
            from datetime import UTC, datetime
            from pathlib import Path

            def sha256_file(path):
                digest = hashlib.sha256()
                with path.open('rb') as handle:
                    while chunk := handle.read(8 * 1024 * 1024):
                        digest.update(chunk)
                return digest.hexdigest()

            def run_bounded(command, stdout, merge_stderr=True):
                remaining = F0_OPERATION_DEADLINE - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('F0 operation deadline elapsed before subprocess launch')
                process = subprocess.Popen(
                    command,
                    stdout=stdout,
                    stderr=subprocess.STDOUT if merge_stderr else None,
                    start_new_session=True,
                )
                try:
                    returncode = process.wait(timeout=remaining)
                except subprocess.TimeoutExpired as error:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
                    raise TimeoutError(
                        f'F0 operation deadline elapsed during {{command[0]}}'
                    ) from error
                if returncode != 0:
                    raise subprocess.CalledProcessError(returncode, command)

            bootstrap_started = time.monotonic()
            bootstrap_dir = Path('/kaggle/working/qwen3-vl-f0-r4-bootstrap')
            bootstrap_dir.mkdir(parents=True, exist_ok=False)
            bootstrap_result = {{
                'started_at': datetime.now(UTC).isoformat(),
                'status': 'started',
            }}
            try:
                input_root = Path('/kaggle/input')
                wheel_paths = list(input_root.rglob({VLLM_WHEEL!r}))
                lock_paths = [
                    path
                    for path in input_root.rglob('requirements.lock')
                    if sha256_file(path) == {RUNTIME_LOCK_SHA256!r}
                ]
                if len(wheel_paths) != 1 or len(lock_paths) != 1:
                    raise RuntimeError(
                        'expected one pinned vLLM wheel and lockfile in attached inputs, '
                        f'found wheels={{wheel_paths}}, locks={{lock_paths}}'
                    )
                wheel_path = wheel_paths[0]
                lock_path = lock_paths[0]
                actual_wheel_sha256 = sha256_file(wheel_path)
                if actual_wheel_sha256 != {VLLM_WHEEL_SHA256!r}:
                    raise RuntimeError('mounted vLLM wheel does not match the pinned bytes')
                source_lock_lines = lock_path.read_text().splitlines()
                excluded_requirements = [
                    line
                    for line in source_lock_lines
                    if line.startswith('sglang @ git+')
                ]
                if excluded_requirements != [{EXCLUDED_SGLANG_REQUIREMENT!r}]:
                    raise RuntimeError(
                        'shared lock SGLang exclusion changed: '
                        f'{{excluded_requirements}}'
                    )
                derived_lock_path = Path('/tmp/vllm-requirements.lock')
                derived_lock_path.write_text(
                    '\\n'.join(
                        line
                        for line in source_lock_lines
                        if line != {EXCLUDED_SGLANG_REQUIREMENT!r}
                    )
                    + '\\n'
                )
                derived_lock_sha256 = sha256_file(derived_lock_path)
                if derived_lock_sha256 != {DERIVED_RUNTIME_LOCK_SHA256!r}:
                    raise RuntimeError(
                        'derived vLLM lock differs from the pinned transformation'
                    )
                wheelhouse = wheel_path.parent
                runtime_provenance = Path('/tmp/vllm-runtime-provenance.json')
                runtime_provenance.write_text(json.dumps({{
                    'dataset_source': {RUNTIME_DATASET_SOURCE!r},
                    'dataset_version': 1,
                    'artifact_path': str(wheel_path),
                    'artifact_sha256': actual_wheel_sha256,
                    'lock_path': str(lock_path),
                    'lock_sha256': {RUNTIME_LOCK_SHA256!r},
                    'derived_lock_path': str(derived_lock_path),
                    'derived_lock_sha256': derived_lock_sha256,
                    'excluded_requirements': excluded_requirements,
                }}, indent=2) + '\\n')

                overrides_path = Path('/tmp/vllm-runtime-overrides.txt')
                overrides_path.write_text('flashinfer-python==0.6.17\\n')
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
                    run_bounded(uv_install_command, install_log)
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
                    '--requirement',
                    str(derived_lock_path),
                ]
                (bootstrap_dir / 'runtime-install-command.json').write_text(
                    json.dumps({{
                        'uv_install': uv_install_command,
                        'runtime_install': install_command,
                        'overrides': overrides_path.read_text().splitlines(),
                    }}, indent=2) + '\\n'
                )
                with (bootstrap_dir / 'runtime-install.log').open('a') as install_log:
                    run_bounded(install_command, install_log)
                (bootstrap_dir / 'runtime-provenance.json').write_text(
                    runtime_provenance.read_text()
                )
                with (bootstrap_dir / 'runtime-packages.json').open('w') as package_log:
                    run_bounded(
                        [sys.executable, '-m', 'pip', 'list', '--format', 'json'],
                        package_log,
                        merge_stderr=False,
                    )
                bootstrap_result['status'] = 'passed'
            except Exception as error:
                bootstrap_result |= {{
                    'status': 'failed',
                    'error_type': type(error).__name__,
                    'error': str(error),
                }}
                raise
            finally:
                bootstrap_result |= {{
                    'ended_at': datetime.now(UTC).isoformat(),
                    'elapsed_s': round(time.monotonic() - bootstrap_started, 3),
                }}
                (bootstrap_dir / 'bootstrap-result.json').write_text(
                    json.dumps(bootstrap_result, indent=2) + '\\n'
                )
            """
        )
    )

    run_probe = code_cell(
        dedent(
            f"""\
            import json
            import os
            import signal
            import subprocess
            import sys
            import time
            from pathlib import Path

            output_dir = Path('/kaggle/working/qwen3-vl-f0-r4-probe')
            probe_command = [
                sys.executable,
                '/tmp/PROTOTYPE-qwen3_vl_f0_probe.py',
                '--output-dir',
                str(output_dir),
                '--input-root',
                '/kaggle/input',
                '--runtime-provenance',
                '/tmp/vllm-runtime-provenance.json',
                '--startup-timeout-s',
                '1200',
                '--request-timeout-s',
                '180',
            ]
            remaining = F0_OPERATION_DEADLINE - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('F0 operation deadline elapsed before probe launch')
            probe_process = subprocess.Popen(probe_command, start_new_session=True)
            Path('/kaggle/working/qwen3-vl-f0-r4-outer-process.json').write_text(
                json.dumps({{'pid': probe_process.pid, 'pgid': probe_process.pid}}, indent=2) + '\\n'
            )
            try:
                returncode = probe_process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                outer_timeout = {{
                    'status': 'operation_deadline_elapsed',
                    'soft_deadline_s': {MAX_NOTEBOOK_RUNTIME_S - FINALIZATION_MARGIN_S},
                    'hard_deadline_s': {MAX_NOTEBOOK_RUNTIME_S},
                    'probe_pid': probe_process.pid,
                }}
                os.kill(probe_process.pid, signal.SIGINT)
                finalization_remaining = max(1, F0_HARD_DEADLINE - time.monotonic())
                try:
                    returncode = probe_process.wait(timeout=finalization_remaining)
                except subprocess.TimeoutExpired:
                    outer_timeout['hard_deadline_elapsed'] = True
                    server_process_path = output_dir / 'server-process.json'
                    if server_process_path.is_file():
                        server_process = json.loads(server_process_path.read_text())
                        try:
                            os.killpg(int(server_process['pgid']), signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    try:
                        os.killpg(probe_process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    returncode = probe_process.wait(timeout=10)
                Path('/kaggle/working/qwen3-vl-f0-r4-outer-timeout.json').write_text(
                    json.dumps(outer_timeout, indent=2) + '\\n'
                )
            if returncode != 0:
                raise subprocess.CalledProcessError(returncode, probe_command)
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
                "# PROTOTYPE — Qwen3-VL F0-R4 fallback contract qualification\n\n"
                "**Question:** can the exact Qwen3-VL-8B-Instruct-FP8 Kaggle mount and "
                "pinned vLLM runtime cold-install without network access, become ready, "
                "accept the primary route's exact visual request under its served-model "
                "alias, emit one parsed native `take_action(LEFT)` call, and release the "
                "process group and GPU memory on shutdown?\n\n"
                "This notebook performs one model request. It grants the fallback deployment "
                "qualification only—no Tycho behavioral authority, ARC action, score, or optimization."
            ),
            write_probe,
            install_runtime,
            run_probe,
        ],
    }
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile(str(cell["source"]), f"f0_probe.ipynb:cell-{index}", "exec")

    metadata = {
        "id": args.kernel_id,
        "title": "PROTOTYPE Qwen3 VL F0 R4",
        "code_file": notebook_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "machine_shape": MACHINE_SHAPE,
        "enable_internet": False,
        "enable_tpu": False,
        "dataset_sources": [args.runtime_dataset_source],
        "kernel_sources": [],
        "competition_sources": [COMPETITION],
        "model_sources": [args.model_source],
    }

    output_root.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(json.dumps(notebook, indent=1) + "\n")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {notebook_path}")
    print(f"wrote {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-id", required=True, help="Kaggle username/kernel-slug")
    parser.add_argument("--model-source", required=True, help="owner/model/framework/variation/version")
    parser.add_argument("--runtime-dataset-source", required=True, help="owner/dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "f0-r4-notebook",
        help="generated Kaggle kernel directory",
    )
    return parser.parse_args()


def validate_source(value: str, parts: int, name: str) -> str:
    if len(value.split("/")) != parts:
        raise SystemExit(f"{name} must have {parts} slash-separated fields, got {value!r}")
    return value


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


if __name__ == "__main__":
    main()
