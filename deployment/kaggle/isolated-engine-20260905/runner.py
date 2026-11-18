"""Bounded, offline single-stream engine comparison; never starts the Actor."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

OUT = Path("/kaggle/working/isolated-engine-20260905")
BASE = "http://127.0.0.1:30000"
ARMS = ["baseline", "tuned-triton", "shallow-mtp"]


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deadline(_signum, _frame):
    raise TimeoutError("3200-second runner budget exhausted")


def get(route, *, as_json=True):
    with urllib.request.urlopen(BASE + route, timeout=20) as response:
        return json.load(response) if as_json else response.read().decode()


def generate(directory, name, payload):
    save(directory / f"{name}-request.json", payload)
    request = urllib.request.Request(
        BASE + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first = last = None
    with (directory / f"{name}-stream.jsonl").open("w") as raw:
        with urllib.request.urlopen(request, timeout=600) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].strip()
                if data == b"[DONE]":
                    break
                value = json.loads(data)
                raw.write(
                    json.dumps(
                        {"elapsed_s": time.perf_counter() - start, "data": value}
                    )
                    + "\n"
                )
                raw.flush()
                if "meta_info" not in value:
                    raise RuntimeError(value)
                first = value if first is None else first
                last = value
    elapsed = time.perf_counter() - start
    if last is None or not last["meta_info"].get("finish_reason"):
        raise RuntimeError("No attributable completed response")
    meta = last["meta_info"]
    n = meta["completion_tokens"]
    initial = first["meta_info"]["completion_tokens"]
    reported = meta.get("decode_throughput")
    # Pinned TokenizerReqTimeStats uses (N-1)/(finished-first_output_batch).
    # Adjust numerator when speculation emits multiple tokens in the first batch.
    span = (n - 1) / reported if reported and n > 1 else None
    result = {
        "name": name,
        "complete_request_s": elapsed,
        "completion_tokens": n,
        "first_batch_tokens": initial,
        "tokenizer_decode_span_s": span,
        "steady_decode_tok_s": (n - initial) / span if span and n > initial else None,
        "timing_boundary": "server tokenizer first output batch to finished; not SSE chunk timing",
        "meta_info": meta,
        "output_ids": last.get("output_ids"),
        "text": last.get("text"),
    }
    if payload["sampling_params"].get("ignore_eos"):
        assert n == payload["sampling_params"]["max_new_tokens"], result
        if name != "warmup":
            assert result["steady_decode_tok_s"] and initial == 1, (
                "First batch boundary requires review"
            )
    save(directory / f"{name}-result.json", result)
    return result


def install_tuning(model, config):
    import sglang
    import torch

    q = config.get("quantization_config", {})
    text_config = config.get("text_config", config)
    assert q.get(
        "weight_block_size",
        text_config.get("quantization_config", {}).get("weight_block_size"),
    ) == [128, 128]
    assert (
        text_config["hidden_size"] == 5120 and text_config["intermediate_size"] == 17408
    )
    assert (
        torch.cuda.get_device_name() == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    )
    root = Path(sglang.__file__).parent / "kernels/ops/quantization"
    source = root / "fp8_kernel.py"
    assert source.exists() and "get_w8a8_block_fp8_configs" in source.read_text()
    bundle = json.loads(Path("/tmp/engine-tuning.json").read_text())
    records = []
    originals = {}
    for entry in bundle["records"]:
        data = entry["text"].encode()
        assert (
            hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            == entry["git_blob_sha1"]
        )
        path = root / "configs" / entry["filename"]
        originals[path] = path.read_bytes() if path.exists() else None
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)
        records.append(
            {
                "path": str(path),
                "sha256": sha(path),
                "git_blob_sha1": entry["git_blob_sha1"],
            }
        )
    save(
        OUT / "installed-tuning.json",
        {
            "lookup_source_sha256": sha(source),
            "records": records,
            "shape_gate": "dense hidden=5120 intermediate=17408; five PR candidates; actual use checked in server log",
        },
    )
    return originals


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(3200)
    import torch

    assert torch.cuda.device_count() == 1
    assert "RTX PRO 6000" in torch.cuda.get_device_name()
    save(
        OUT / "hardware.json",
        {
            "device": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
        },
    )
    spec = importlib.util.spec_from_file_location(
        "frozen_runtime", "/tmp/official_launcher_runtime.py"
    )
    runtime = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runtime
    spec.loader.exec_module(runtime)
    model = runtime.discover_model(
        Path("/kaggle/input"), Path("/tmp/PROTOTYPE-qwen3_8_fp8-manifest.json")
    )
    save(
        OUT / "model-verification.json",
        runtime.verify_model(model, Path("/tmp/PROTOTYPE-qwen3_8_fp8-manifest.json")),
    )
    config = json.loads((model / "config.json").read_text())
    save(OUT / "config.json", config)
    save(
        OUT / "generation_config.json",
        json.loads((model / "generation_config.json").read_text()),
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model, trust_remote_code=True, local_files_only=True
    )
    messages = [
        {
            "role": "user",
            "content": "Find the smallest positive integer divisible by every integer from 1 through 20. Explain your reasoning and give the integer as your final answer.",
        }
    ]
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=True
    )
    save(
        OUT / "frozen-input.json",
        {
            "messages": messages,
            "input_ids": ids,
            "sampling": {"temperature": 0, "top_p": 1, "top_k": -1},
            "scope": "synthetic engine diagnostic, not gameplay replay; natural reasoning ceiling 16000 unchanged across arms",
        },
    )
    results = []
    for arm in ARMS:
        directory = OUT / arm
        directory.mkdir()
        originals = {}
        process = target = None
        ready = False
        try:
            if arm == "tuned-triton":
                originals = install_tuning(model, config)
            command = runtime.sglang_command(
                model,
                "127.0.0.1",
                30000,
                attention_backend="triton",
                sampling_backend="pytorch",
            )
            command += ["--enable-metrics", "--stream-interval", "1"]
            if arm == "tuned-triton":
                command += ["--fp8-gemm-backend", "triton"]
            if arm == "shallow-mtp":
                command += [
                    "--speculative-algorithm",
                    "EAGLE",
                    "--speculative-num-steps",
                    "1",
                    "--speculative-eagle-topk",
                    "1",
                    "--speculative-num-draft-tokens",
                    "2",
                ]
            env = dict(os.environ)
            env.update(
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
                R8_L3B_OWNER=runtime.R8_L3B_OWNER,
            )
            save(
                directory / "launch.json",
                {
                    "command": command,
                    "frozen_runtime_sha256": sha(
                        Path("/tmp/official_launcher_runtime.py")
                    ),
                    "environment": {
                        k: v
                        for k, v in env.items()
                        if k.startswith(
                            (
                                "SGLANG_",
                                "HF_HUB_OFFLINE",
                                "TRANSFORMERS_OFFLINE",
                                "R8_L3B_OWNER",
                            )
                        )
                    },
                },
            )
            with (directory / "server.log").open("w") as log:
                try:
                    start = time.perf_counter()
                    process = subprocess.Popen(
                        command,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        text=True,
                    )
                    target = runtime.process_identity(process.pid)
                    runtime.wait_for_server(BASE, process, 900)
                    ready = True
                    runtime.wait_for_sglang_warmup(
                        directory / "server.log", process, 120
                    )
                    save(directory / "server-info.json", get("/get_server_info"))
                    arm_results = []
                    for name, limit, ignore in [
                        ("warmup", 64, True),
                        ("fixed-1", 1024, True),
                        ("fixed-2", 1024, True),
                        ("natural", 16000, False),
                    ]:
                        # Flush reusable request state, not model/graph warmup, before each measurement.
                        save(
                            directory / f"{name}-cache-flush.json",
                            get("/flush_cache", as_json=False),
                        )
                        subprocess.run(
                            [
                                "nvidia-smi",
                                "-q",
                                "-d",
                                "PERFORMANCE,CLOCK,POWER,TEMPERATURE,MEMORY",
                            ],
                            stdout=(directory / f"{name}-gpu.txt").open("w"),
                            check=True,
                        )
                        payload = {
                            "input_ids": ids,
                            "stream": True,
                            "sampling_params": {
                                "temperature": 0,
                                "top_p": 1,
                                "top_k": -1,
                                "max_new_tokens": limit,
                                "ignore_eos": ignore,
                            },
                        }
                        arm_results.append(generate(directory, name, payload))
                    result = {
                        "arm": arm,
                        "status": "measured",
                        "startup_and_requests_s": time.perf_counter() - start,
                        "requests": arm_results,
                    }
                    if arm == "tuned-triton":
                        log.flush()
                        names = [path.name for path in originals]
                        used = [
                            line
                            for line in (directory / "server.log")
                            .read_text()
                            .splitlines()
                            if "Using configuration from" in line
                            and any(name in line for name in names)
                        ]
                        save(directory / "selected-config-log.json", used)
                        assert used, "No evidence tuned config was selected"
                    results.append(result)
                finally:
                    if process is not None:
                        if ready and process.poll() is None:
                            try:
                                lifecycle = runtime.qualify_lifecycle(
                                    process, log, directory, port=30000
                                )
                                save(directory / "lifecycle.json", lifecycle)
                                assert lifecycle["status"] == "passed", (
                                    "Do not admit next arm after unresolved cleanup"
                                )
                            except BaseException:
                                save(
                                    directory / "containment.json",
                                    runtime.emergency_contain_process_group(
                                        process, target
                                    ),
                                )
                                raise
                        else:
                            save(
                                directory / "containment.json",
                                runtime.emergency_contain_process_group(
                                    process, target
                                ),
                            )
            save(OUT / "results.json", results)
        finally:
            for path, data in originals.items():
                if data is None:
                    path.unlink()
                else:
                    path.write_bytes(data)
    save(
        OUT / "VERDICT.json",
        {
            "status": "MEASUREMENTS_RETAINED_REVIEW_REQUIRED",
            "arms": ARMS,
            "limits": "single allocation, synthetic prompt, no gameplay/tool correctness or lifecycle promotion",
        },
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        OUT.mkdir(parents=True, exist_ok=True)
        save(OUT / "FAILURE.json", {"type": type(error).__name__, "error": str(error)})
        raise
