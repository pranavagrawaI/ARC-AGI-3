#!/usr/bin/env python3
"""One-request Qwen3-VL fallback contract probe for Kaggle."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import signal
import struct
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct-FP8"
MODEL_REVISION = "9cdc6310a8cb770ce18efaf4e9935334512aee45"
MODEL_SOURCE = "qwen-lm/qwen-3-vl/transformers/8b-instruct-fp8/1"
SERVED_MODEL_ALIAS = "Qwen/Qwen3.8-27B-FP8"
RUNTIME_DATASET_SOURCE = "mahdiseddigh/kaggle-t4-stack-2026-08-28"
RUNTIME_DATASET_VERSION = 1
RUNTIME_WHEEL_SHA256 = "addb0ffdaafd8155d75e9b3f5ddb3da28fdee9e8a7097ede91f7db2e9e1a3889"
RUNTIME_LOCK_SHA256 = "d473b6e092e01f66313e12069ac8873f2ff1c6c3b84b51c1b37312229de5475c"
RUNTIME_DERIVED_LOCK_SHA256 = "d07866e8a6961193a1285d1514560c46610413f9a05fc19738c286320a98c481"
EXPECTED_GPU_FRAGMENT = "RTX PRO 6000"
EXPECTED_GPU_MEMORY_MIB = 90_000
EXPECTED_COMPUTE_CAPABILITY = "12.0"
EXPECTED_SAFETENSORS_BYTES = 10_590_299_512
ATTEMPT_ID = "F0-R4"
PROBE_SCHEMA = "qwen3-vl-f0-contract-probe-v2"


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    events_path = output_dir / "events.log"
    started_at = utc_now()
    started = time.monotonic()
    process: subprocess.Popen[str] | None = None
    server_log_handle: Any = None
    system_before: dict[str, Any] | None = None
    stage = "preflight"
    contract_passed = False
    failure: dict[str, Any] | None = None

    try:
        write_json(output_dir / "probe-config.json", vars(args) | probe_identity())
        system_before = system_snapshot()
        write_json(output_dir / "system-before.json", system_before)
        accelerator_verification = verify_accelerator(system_before)
        write_json(output_dir / "accelerator-verification.json", accelerator_verification)

        stage = "model_identity"
        model_path = discover_model(args.input_root)
        append_event(events_path, f"verifying exact model mount at {model_path}")
        model_verification = verify_model(model_path)
        write_json(output_dir / "model-verification.json", model_verification)

        stage = "runtime_identity"
        runtime_verification = verify_runtime(args.runtime_provenance)
        write_json(output_dir / "runtime-verification.json", runtime_verification)

        stage = "request_freeze"
        image_bytes = arc_like_png()
        request = build_request(SERVED_MODEL_ALIAS, image_bytes)
        request_bytes = json.dumps(request).encode("utf-8")
        (output_dir / "request-image.png").write_bytes(image_bytes)
        (output_dir / "request.bin").write_bytes(request_bytes)
        write_json(
            output_dir / "request-identity.json",
            {
                "image_sha256": sha256_bytes(image_bytes),
                "request_bytes": len(request_bytes),
                "request_sha256": sha256_bytes(request_bytes),
                "served_model_alias": SERVED_MODEL_ALIAS,
                "expected_action": "LEFT",
                "tool_name": "take_action",
            },
        )

        stage = "server_startup"
        command = vllm_command(model_path, args.host, args.port)
        environment = server_environment()
        write_json(output_dir / "server-command.json", command)
        write_json(output_dir / "server-environment.json", environment)
        append_event(events_path, "launching pinned vLLM fallback route")
        server_log_handle = (output_dir / "server.log").open("w")
        process = subprocess.Popen(
            command,
            stdout=server_log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ | environment,
            start_new_session=True,
        )
        write_json(output_dir / "server-process.json", {"pid": process.pid, "pgid": process.pid})

        readiness = wait_for_server(args.base_url, process, args.startup_timeout_s)
        write_json(output_dir / "readiness.json", readiness)
        append_event(events_path, f"server ready after {readiness['elapsed_s']}s")

        stage = "visual_native_tool_request"
        transport = post_bytes(
            args.base_url.rstrip("/") + "/v1/chat/completions",
            request_bytes,
            args.request_timeout_s,
        )
        response_bytes = transport.pop("body")
        (output_dir / "response.bin").write_bytes(response_bytes)
        write_json(output_dir / "response-transport.json", transport)
        if transport["status"] != 200:
            raise RuntimeError(
                f"completion endpoint returned HTTP {transport['status']}; "
                "raw response bytes were preserved"
            )
        response = json.loads(response_bytes)
        write_json(output_dir / "response.json", response)

        stage = "parser_acceptance"
        validation = validate_response(response)
        write_json(output_dir / "validation.json", validation)
        append_event(events_path, "native take_action(LEFT) response accepted")
        contract_passed = True
    except Exception as error:
        failure = {
            "stage": stage,
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(output_dir / "failure.json", failure)
        append_event(events_path, f"failed at {stage}: {type(error).__name__}: {error}")
    finally:
        finalization_failures: list[dict[str, str]] = []
        try:
            shutdown = shutdown_server(process, system_before)
        except Exception as error:
            shutdown = {
                "clean": False,
                "process_started": process is not None,
                "method": "shutdown_exception",
                "error_type": type(error).__name__,
                "error": str(error),
            }
            finalization_failures.append(
                {"stage": "shutdown", "type": type(error).__name__, "message": str(error)}
            )
        write_json(output_dir / "shutdown.json", shutdown)
        if server_log_handle is not None:
            try:
                server_log_handle.close()
            except Exception as error:
                finalization_failures.append(
                    {"stage": "server_log_close", "type": type(error).__name__, "message": str(error)}
                )
        try:
            system_after = system_snapshot()
        except Exception as error:
            system_after = {"captured_at": utc_now(), "error": f"{type(error).__name__}: {error}"}
            finalization_failures.append(
                {"stage": "system_after", "type": type(error).__name__, "message": str(error)}
            )
        write_json(output_dir / "system-after.json", system_after)
        status = (
            "passed"
            if contract_passed and shutdown["clean"] and not finalization_failures
            else "failed"
        )
        if contract_passed and status == "failed" and failure is None:
            failure = {
                "stage": finalization_failures[0]["stage"] if finalization_failures else "shutdown",
                "type": (
                    finalization_failures[0]["type"]
                    if finalization_failures
                    else "ShutdownContractError"
                ),
                "message": (
                    finalization_failures[0]["message"]
                    if finalization_failures
                    else "visual/native-tool contract passed, but process exit or GPU-memory release did not"
                ),
            }
        result = {
            "schema": PROBE_SCHEMA,
            "status": status,
            "started_at": started_at,
            "ended_at": utc_now(),
            "elapsed_s": round(time.monotonic() - started, 3),
            "identity": probe_identity(),
            "contract_passed_before_shutdown": contract_passed,
            "shutdown_clean": shutdown["clean"],
            "failure": failure,
            "finalization_failures": finalization_failures,
            "claim_ceiling": (
                "A pass establishes only deployment qualification of the exact fallback model/runtime "
                "for one frozen visual native-tool request and clean shutdown on the observed Kaggle "
                "substrate. It establishes no Tycho behavioral authority, ARC score, throughput, or "
                "authority over the Qwen incumbent."
            ),
        }
        write_json(output_dir / "result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))

    if status != "passed":
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("qwen3-vl-f0-probe"))
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--runtime-provenance", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--startup-timeout-s", type=int, default=1200)
    parser.add_argument("--request-timeout-s", type=int, default=180)
    args = parser.parse_args()
    args.base_url = f"http://{args.host}:{args.port}"
    return args


def probe_identity() -> dict[str, Any]:
    return {
        "attempt_id": ATTEMPT_ID,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_source": MODEL_SOURCE,
        "served_model_alias": SERVED_MODEL_ALIAS,
        "runtime_dataset_source": RUNTIME_DATASET_SOURCE,
        "runtime_dataset_version": RUNTIME_DATASET_VERSION,
        "runtime_wheel_sha256": RUNTIME_WHEEL_SHA256,
        "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
        "runtime_derived_lock_sha256": RUNTIME_DERIVED_LOCK_SHA256,
        "tool_parser": "hermes",
        "reasoning_parser": "qwen3",
    }


def utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def append_event(path: Path, message: str) -> None:
    with path.open("a") as handle:
        handle.write(f"{utc_now()} {message}\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run_capture(command: list[str], timeout_s: int = 30) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "elapsed_s": round(time.monotonic() - started, 3),
        }
    except Exception as error:
        return {
            "command": command,
            "error": f"{type(error).__name__}: {error}",
            "elapsed_s": round(time.monotonic() - started, 3),
        }


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def system_snapshot() -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    gpu = (
        run_capture(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ]
        )
        if nvidia_smi
        else {"error": "nvidia-smi not found"}
    )
    return {
        "captured_at": utc_now(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
        "kaggle": {
            "competition_rerun": bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN")),
            "kernel_run_type": os.getenv("KAGGLE_KERNEL_RUN_TYPE"),
            "working_directory_present": Path("/kaggle/working").is_dir(),
            "input_directory_present": Path("/kaggle/input").is_dir(),
        },
        "packages": {
            name: package_version(name)
            for name in (
                "vllm",
                "torch",
                "transformers",
                "flashinfer-python",
                "flash-attn-4",
                "triton",
            )
        },
        "commands": {
            "vllm": shutil.which("vllm"),
            "git": shutil.which("git"),
            "nvidia_smi": nvidia_smi,
        },
        "gpu": gpu,
    }


def verify_accelerator(snapshot: dict[str, Any]) -> dict[str, Any]:
    gpu = snapshot["gpu"]
    if gpu.get("returncode") != 0:
        raise RuntimeError(f"cannot inspect GPU: {gpu}")
    lines = gpu.get("stdout", "").strip().splitlines()
    if len(lines) != 1:
        raise RuntimeError(f"expected exactly one GPU, got {lines}")
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 4:
        raise RuntimeError(f"unexpected nvidia-smi row: {lines[0]!r}")
    name, total_text, free_text, driver = fields
    total_mib = int(total_text)
    free_mib = int(free_text)
    capability_probe = run_capture(
        [
            snapshot["commands"]["nvidia_smi"],
            "--query-gpu=compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    if capability_probe.get("returncode") != 0:
        raise RuntimeError(f"cannot inspect GPU compute capability: {capability_probe}")
    capability_lines = capability_probe.get("stdout", "").strip().splitlines()
    if capability_lines != [EXPECTED_COMPUTE_CAPABILITY]:
        raise RuntimeError(
            f"wrong compute capability: expected {EXPECTED_COMPUTE_CAPABILITY}, got {capability_lines}"
        )
    if EXPECTED_GPU_FRAGMENT not in name:
        raise RuntimeError(f"wrong accelerator: expected {EXPECTED_GPU_FRAGMENT!r}, got {name!r}")
    if total_mib < EXPECTED_GPU_MEMORY_MIB:
        raise RuntimeError(
            f"insufficient GPU memory: expected at least {EXPECTED_GPU_MEMORY_MIB} MiB, got {total_mib}"
        )
    return {
        "status": "matched",
        "name": name,
        "memory_total_mib": total_mib,
        "memory_free_mib": free_mib,
        "driver_version": driver,
        "compute_capability": capability_lines[0],
        "compute_capability_probe": capability_probe,
    }


def gpu_free_mib(snapshot: dict[str, Any]) -> int | None:
    gpu = snapshot.get("gpu", {})
    if gpu.get("returncode") != 0:
        return None
    fields = gpu.get("stdout", "").strip().splitlines()[0].split(",")
    return int(fields[2].strip()) if len(fields) >= 3 else None


def discover_model(input_root: Path) -> Path:
    candidates: list[Path] = []
    for config_path in input_root.rglob("config.json"):
        root = config_path.parent
        tensors = sorted(root.glob("*.safetensors"))
        if len(tensors) == 2 and sum(path.stat().st_size for path in tensors) == EXPECTED_SAFETENSORS_BYTES:
            candidates.append(root)
    if len(candidates) != 1:
        raise RuntimeError(f"expected one exact Qwen3-VL FP8 mount, found {candidates}")
    return candidates[0]


def git_bytes(git: str, model_path: Path, relative: str) -> bytes:
    completed = subprocess.run(
        [git, "-C", str(model_path), "show", f"HEAD:{relative}"],
        check=True,
        capture_output=True,
        timeout=30,
    )
    return completed.stdout


def parse_lfs_pointer(value: bytes, name: str) -> tuple[str, int]:
    text = value.decode("ascii")
    digest_match = re.search(r"^oid sha256:([0-9a-f]{64})$", text, re.MULTILINE)
    size_match = re.search(r"^size ([0-9]+)$", text, re.MULTILINE)
    if not digest_match or not size_match:
        raise RuntimeError(f"HEAD:{name} is not a complete Git LFS pointer")
    return digest_match.group(1), int(size_match.group(1))


def verify_model(model_path: Path) -> dict[str, Any]:
    git = shutil.which("git")
    if not git:
        raise RuntimeError("git is required for mounted model identity verification")
    revision = subprocess.run(
        [git, "-C", str(model_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    if revision != MODEL_REVISION:
        raise RuntimeError(f"model revision mismatch: expected {MODEL_REVISION}, got {revision}")

    regular_files = [
        ".gitattributes",
        "README.md",
        "chat_template.json",
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
        "vocab.json",
    ]
    regular_checks: list[dict[str, Any]] = []
    for relative in regular_files:
        path = model_path / relative
        if not path.is_file():
            raise RuntimeError(f"required model file missing: {relative}")
        committed = git_bytes(git, model_path, relative)
        actual = path.read_bytes()
        check = {
            "path": relative,
            "committed_sha256": sha256_bytes(committed),
            "actual_sha256": sha256_bytes(actual),
            "size": len(actual),
            "matched": actual == committed,
        }
        regular_checks.append(check)
        if not check["matched"]:
            raise RuntimeError(f"mounted file differs from HEAD:{relative}")

    shard_checks: list[dict[str, Any]] = []
    shard_paths = sorted(model_path.glob("*.safetensors"))
    if [path.name for path in shard_paths] != [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]:
        raise RuntimeError(f"unexpected safetensors set: {shard_paths}")
    for path in shard_paths:
        expected_sha256, expected_size = parse_lfs_pointer(
            git_bytes(git, model_path, path.name), path.name
        )
        actual_sha256 = sha256_file(path)
        check = {
            "path": path.name,
            "expected_sha256_from_head_lfs_pointer": expected_sha256,
            "actual_sha256": actual_sha256,
            "expected_size": expected_size,
            "actual_size": path.stat().st_size,
            "matched": expected_sha256 == actual_sha256 and expected_size == path.stat().st_size,
        }
        shard_checks.append(check)
        if not check["matched"]:
            raise RuntimeError(f"mounted LFS blob differs from HEAD:{path.name}")

    config = json.loads((model_path / "config.json").read_text())
    quantization = config.get("quantization_config", {})
    structure = {
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "quant_method": quantization.get("quant_method"),
        "fmt": quantization.get("fmt"),
        "activation_scheme": quantization.get("activation_scheme"),
    }
    structure_matched = (
        structure["model_type"] == "qwen3_vl"
        and structure["architectures"] == ["Qwen3VLForConditionalGeneration"]
        and structure["quant_method"] == "fp8"
        and structure["fmt"] == "e4m3"
        and structure["activation_scheme"] == "dynamic"
    )
    if not structure_matched:
        raise RuntimeError(f"unexpected fallback model structure: {structure}")

    remote = run_capture([git, "-C", str(model_path), "remote", "get-url", "origin"])
    return {
        "status": "matched",
        "model_id": MODEL_ID,
        "model_source": MODEL_SOURCE,
        "model_path": str(model_path),
        "revision": revision,
        "remote": remote,
        "structure": structure,
        "regular_files": regular_checks,
        "lfs_shards": shard_checks,
        "safetensors_bytes": sum(path.stat().st_size for path in shard_paths),
    }


def verify_runtime(provenance_path: Path) -> dict[str, Any]:
    provenance = json.loads(provenance_path.read_text())
    expected = {
        "dataset_source": RUNTIME_DATASET_SOURCE,
        "dataset_version": RUNTIME_DATASET_VERSION,
        "artifact_sha256": RUNTIME_WHEEL_SHA256,
        "lock_sha256": RUNTIME_LOCK_SHA256,
        "derived_lock_sha256": RUNTIME_DERIVED_LOCK_SHA256,
    }
    for field, value in expected.items():
        if provenance.get(field) != value:
            raise RuntimeError(f"runtime provenance mismatch for {field}: {provenance.get(field)!r}")
    packages = {
        name: package_version(name)
        for name in (
            "vllm",
            "torch",
            "transformers",
            "flashinfer-python",
            "flash-attn-4",
            "triton",
        )
    }
    required_packages = {
        "vllm": "0.28.0",
        "torch": "2.13.0+cu130",
        "transformers": "5.12.1",
        "flashinfer-python": "0.6.17",
        "flash-attn-4": "4.0.0b19",
        "triton": "3.7.1",
    }
    if packages != required_packages:
        raise RuntimeError(f"installed runtime mismatch: expected {required_packages}, got {packages}")
    executable = shutil.which("vllm")
    if not executable:
        raise RuntimeError("vllm executable missing after offline install")
    return provenance | {"installed_packages": packages, "vllm_executable": executable, "status": "matched"}


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def arc_like_png() -> bytes:
    """Return the primary probe's 256x256 ARC-like board."""
    size = 256
    pixels: list[bytes] = []
    for y in range(size):
        row = bytearray([0])
        for x in range(size):
            color = (22, 24, 29)
            if x % 32 in (0, 1) or y % 32 in (0, 1):
                color = (58, 62, 70)
            if 48 <= x < 112 and 96 <= y < 160:
                color = (220, 48, 48)
            if 176 <= x < 240 and 96 <= y < 160:
                color = (245, 205, 48)
            row.extend(color)
        pixels.append(bytes(row))
    raw = b"".join(pixels)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(raw, 9))
        + png_chunk(b"IEND", b"")
    )


def build_request(model_name: str, image_bytes: bytes) -> dict[str, Any]:
    image_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
    return {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the Game Judgment component for an ARC-like visual game. "
                    "Inspect the image and commit exactly one available action with the "
                    "take_action tool."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "The image contains one red square and one yellow square. "
                            "Choose LEFT if the red square is left of the yellow square; "
                            "otherwise choose RIGHT. Available actions: LEFT, RIGHT."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "take_action",
                    "description": "Commit exactly one available environment action.",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "action": {"type": "string", "enum": ["LEFT", "RIGHT"]}
                        },
                        "required": ["action"],
                    },
                },
            }
        ],
        "tool_choice": "required",
        "temperature": 0,
        "max_tokens": 1024,
        "chat_template_kwargs": {
            "enable_thinking": True,
            "preserve_thinking": True,
        },
    }


def server_environment() -> dict[str, str]:
    return {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER": "0",
    }


def vllm_command(model_path: Path, host: str, port: int) -> list[str]:
    executable = shutil.which("vllm")
    if not executable:
        raise RuntimeError("vllm executable not found")
    return [
        executable,
        "serve",
        str(model_path),
        "--served-model-name",
        SERVED_MODEL_ALIAS,
        "--attention-backend",
        "FLASH_ATTN",
        "--mm-encoder-attn-backend",
        "FLASH_ATTN",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        "--reasoning-parser",
        "qwen3",
        "--host",
        host,
        "--port",
        str(port),
    ]


def wait_for_server(base_url: str, process: subprocess.Popen[str], timeout_s: int) -> dict[str, Any]:
    started = time.monotonic()
    last_error = "not attempted"
    while time.monotonic() - started < timeout_s:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited before readiness with code {process.returncode}")
        try:
            with urllib.request.urlopen(base_url.rstrip("/") + "/v1/models", timeout=5) as response:
                raw = response.read()
            models = json.loads(raw)
            ids = [item.get("id") for item in models.get("data", [])]
            if ids != [SERVED_MODEL_ALIAS]:
                raise RuntimeError(f"served-model alias mismatch: {ids}")
            return {
                "elapsed_s": round(time.monotonic() - started, 3),
                "response_sha256": sha256_bytes(raw),
                "models": models,
            }
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
            time.sleep(2)
    raise TimeoutError(f"server was not ready in {timeout_s}s; last error: {last_error}")


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


def post_bytes(url: str, payload: bytes, timeout_s: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(NoRedirectHandler())
    try:
        with opener.open(request, timeout=timeout_s) as response:
            return {
                "status": response.status,
                "reason": response.reason,
                "headers": dict(response.headers.items()),
                "body": response.read(),
            }
    except urllib.error.HTTPError as error:
        return {
            "status": error.code,
            "reason": error.reason,
            "headers": dict(error.headers.items()),
            "body": error.read(),
        }


def validate_response(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError("expected exactly one completion choice")
    choice = choices[0]
    if choice.get("finish_reason") != "tool_calls":
        raise RuntimeError(f"expected native tool_calls finish reason, got {choice.get('finish_reason')!r}")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("completion choice has no message object")
    reasoning = message.get("reasoning_content")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise RuntimeError("expected non-empty parsed reasoning_content")
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise RuntimeError("expected exactly one native tool call")
    call = calls[0]
    if call.get("type") != "function":
        raise RuntimeError("tool call is not a native function call")
    function = call.get("function")
    if not isinstance(function, dict) or function.get("name") != "take_action":
        raise RuntimeError("native tool call is not take_action")
    arguments_text = function.get("arguments")
    if not isinstance(arguments_text, str):
        raise RuntimeError("native tool-call arguments are not a JSON string")
    arguments = json.loads(arguments_text)
    if arguments != {"action": "LEFT"}:
        raise RuntimeError(f"unexpected action arguments: {arguments}")
    return {
        "accepted": True,
        "finish_reason": choice["finish_reason"],
        "reasoning_content": reasoning,
        "tool_call": call,
        "parsed_arguments": arguments,
    }


def process_group_members(pgid: int) -> list[dict[str, Any]]:
    completed = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,pgid=,stat=,comm="],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    members: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = line.split(None, 4)
        if len(fields) == 5 and fields[2].isdigit() and int(fields[2]) == pgid:
            members.append(
                {
                    "pid": int(fields[0]),
                    "ppid": int(fields[1]),
                    "pgid": int(fields[2]),
                    "stat": fields[3],
                    "command": fields[4],
                }
            )
    return members


def shutdown_server(
    process: subprocess.Popen[str] | None,
    system_before: dict[str, Any] | None,
) -> dict[str, Any]:
    if process is None:
        return {
            "clean": True,
            "process_started": False,
            "method": "not_started",
            "returncode": None,
            "group_members_after": [],
            "memory_released": True,
        }
    started = time.monotonic()
    pgid = process.pid
    method = "sigint"
    escalated = False
    try:
        os.killpg(pgid, signal.SIGINT)
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        method = "sigterm_after_sigint_timeout"
        escalated = True
        os.killpg(pgid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            method = "sigkill_after_sigterm_timeout"
            os.killpg(pgid, signal.SIGKILL)
            process.wait(timeout=30)
    except ProcessLookupError:
        process.wait(timeout=5)

    before_free = gpu_free_mib(system_before) if system_before else None
    threshold = before_free - 1024 if before_free is not None else None
    memory_samples: list[dict[str, Any]] = []
    memory_released = threshold is None
    members = process_group_members(pgid)
    release_started = time.monotonic()
    while time.monotonic() - release_started < 60:
        snapshot = system_snapshot()
        free = gpu_free_mib(snapshot)
        members = process_group_members(pgid)
        memory_samples.append(
            {
                "captured_at": snapshot["captured_at"],
                "free_mib": free,
                "group_members": members,
            }
        )
        memory_released = threshold is None or (free is not None and free >= threshold)
        if memory_released and not members:
            memory_released = True
            break
        time.sleep(2)
    clean = not escalated and not members and memory_released
    return {
        "clean": clean,
        "process_started": True,
        "pid": process.pid,
        "pgid": pgid,
        "method": method,
        "escalated": escalated,
        "returncode": process.returncode,
        "elapsed_s": round(time.monotonic() - started, 3),
        "group_members_after": members,
        "free_memory_threshold_mib": threshold,
        "memory_samples": memory_samples,
        "memory_released": memory_released,
    }


if __name__ == "__main__":
    main()
