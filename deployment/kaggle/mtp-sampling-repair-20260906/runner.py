"""Matched Actor requests, isolated servers, no tools executed and no gameplay."""

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request
from install_sampling_patch import install
from watchdog import guarded_call

OUT = Path("/kaggle/working/mtp-sampling-repair-20260906")
BASE = "http://127.0.0.1:30000"


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def get(route):
    with urllib.request.urlopen(BASE + route, timeout=30) as response:
        return response.read().decode()


def post(payload):
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as response:
        result = json.load(response)
    return result, time.perf_counter() - start


def inspect_response(response, request):
    choice = response["choices"][0]
    message = choice["message"]
    schemas = {
        t["function"]["name"]: t["function"]["parameters"] for t in request["tools"]
    }
    errors = []
    calls = message.get("tool_calls") or []
    import jsonschema

    for call in calls:
        try:
            name = call["function"]["name"]
            if name not in schemas:
                raise ValueError("unknown tool " + name)
            jsonschema.validate(
                json.loads(call["function"]["arguments"]), schemas[name]
            )
        except Exception as error:
            errors.append(str(error))
    forced = request.get("tool_choice")
    if forced == "required" and not calls:
        errors.append("required tool call absent")
    if isinstance(forced, dict):
        expected = forced["function"]["name"]
        if not calls or any(c["function"]["name"] != expected for c in calls):
            errors.append("forced tool restriction not met")
    return {
        "tool_schema_errors": errors,
        "finish_reason": choice["finish_reason"],
        "limited": choice["finish_reason"] == "length",
        "semantic_correctness": "UNREVIEWED; tool code is never executed",
    }


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    save(OUT / 'sampling-patch.json', install())
    os.environ['SGLANG_SPEC_NATIVE_RENORM'] = '1'

    def expired(_s, _f):
        raise TimeoutError("6400-second runner budget exhausted")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(6400)
    import torch

    save(
        OUT / "hardware.json",
        {
            "device": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
        },
    )
    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (
        12,
        0,
    )
    assert "RTX PRO 6000" in torch.cuda.get_device_name()
    corpus = json.loads(Path("/tmp/actor-corpus.json").read_text())
    contract = load_module("corpus_contract", "/tmp/actor-corpus-contract.py")
    contract.validate(corpus)
    runtime = load_module("frozen_runtime", "/tmp/official_launcher_runtime.py")
    metric = load_module("frozen_engine_metric", "/tmp/engine-runner.py")
    model = runtime.discover_model(
        Path("/kaggle/input"), Path("/tmp/PROTOTYPE-qwen3_8_fp8-manifest.json")
    )
    save(
        OUT / "model-verification.json",
        runtime.verify_model(model, Path("/tmp/PROTOTYPE-qwen3_8_fp8-manifest.json")),
    )
    for filename in ["config.json", "generation_config.json"]:
        save(OUT / filename, json.loads((model / filename).read_text()))
    assert (
        json.loads((model / "generation_config.json").read_text())
        == corpus["generation_config"]
    ), "Captured generation defaults drifted"
    results = []
    baseline_prompt_ids = {}
    for arm in ["baseline", "shallow-mtp"]:
        directory = OUT / arm
        directory.mkdir()
        command = runtime.sglang_command(
            model,
            "127.0.0.1",
            30000,
            attention_backend="triton",
            sampling_backend="pytorch",
        )
        command += [
            "--enable-metrics",
            "--stream-interval",
            "1",
            "--sampling-defaults",
            "model",
        ]
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
        env = dict(
            os.environ,
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            R8_L3B_OWNER=runtime.R8_L3B_OWNER,
        )
        save(directory / "launch.json", {"command": command})
        process = target = None
        ready = False
        with (directory / "server.log").open("w") as log:
            try:
                process = subprocess.Popen(
                    command,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    text=True,
                )
                target = runtime.process_identity(process.pid)
                runtime.wait_for_server(BASE, process, 1000)
                ready = True
                runtime.wait_for_sglang_warmup(directory / "server.log", process, 120)
                info = json.loads(get("/server_info"))
                save(directory / "server-info.json", info)
                assert info["sampling_defaults"] == "model"
                assert info["speculative_algorithm"] == (
                    "EAGLE" if arm == "shallow-mtp" else None
                )
                # Separate bounded warmup; never included in corpus latency.
                warm = dict(
                    corpus["records"][0]["request"],
                    model=str(model),
                    max_tokens=64,
                    return_meta_info=True,
                    return_token_ids=True,
                )
                save(directory / 'stochastic-preflight-request.json', warm)
                warm_response, warm_elapsed = guarded_call(lambda: post(warm), process)
                save(directory / 'stochastic-preflight-response.json', warm_response)
                if arm == 'shallow-mtp':
                    assert warm_response['choices'][0]['meta_info']['spec_verify_ct'] > 0, 'MTP verification not exercised'
                for index, record in enumerate(corpus["records"]):
                    name = f"{index}-{record['kind']}"
                    get("/flush_cache")
                    request = dict(
                        record["request"],
                        model=str(model),
                        return_meta_info=True,
                        return_token_ids=True,
                        return_prompt_token_ids=True,
                    )
                    save(directory / f"{name}-request.json", request)
                    response, elapsed = guarded_call(lambda: post(request), process)
                    save(directory / f"{name}-response.json", response)
                    ids = response["choices"][0].get("prompt_token_ids")
                    if (
                        not isinstance(ids, list)
                        or not ids
                        or not all(type(t) is int for t in ids)
                    ):
                        raise RuntimeError("Missing actual rendered prompt token IDs")
                    if (
                        response["usage"]["prompt_tokens"]
                        != record["captured_usage"]["prompt_tokens"]
                    ):
                        raise RuntimeError(
                            "Rendered prompt count differs from capture; inspect before comparing speed"
                        )
                    if arm == "baseline":
                        baseline_prompt_ids[record["id"]] = ids
                    elif ids != baseline_prompt_ids[record["id"]]:
                        raise RuntimeError("Rendered prompt IDs differ between arms")
                    quality = inspect_response(response, request)
                    result = {
                        "arm": arm,
                        "case": record["id"],
                        "e2e_seconds": elapsed,
                        "usage": response.get("usage"),
                        "quality": quality,
                        "captured_quality": record.get("captured_quality"),
                        "request_provenance": record["provenance"],
                        "meta_info": response["choices"][0].get("meta_info"),
                        "decode_timing": "Not inferred from non-streaming response timing; separate native diagnostic below",
                    }
                    results.append(result)
                    save(OUT / "results.json", results)
                    # Invalid tool output is a measured outcome, not a reason to
                    # drop failure-prone inputs or prevent the other arm running.
                    if record["id"] == corpus["fixed_diagnostic_case"]:
                        # Controlled execution diagnostic only: no chat grammar and fixed length.
                        payload = {
                            "input_ids": ids,
                            "stream": True,
                            "sampling_params": {
                                "temperature": 0,
                                "top_p": 1,
                                "top_k": -1,
                                "max_new_tokens": 1024,
                                "ignore_eos": True,
                            },
                        }
                        for repeat in [1, 2]:
                            get("/flush_cache")
                            metric.generate(directory, f"fixed-{repeat}", payload)
                save(
                    directory / "server-info-after.json",
                    json.loads(get("/server_info")),
                )
            finally:
                original_error = sys.exc_info()[1]
                if process is not None:
                    try:
                        if ready and process.poll() is None:
                            lifecycle = runtime.qualify_lifecycle(
                                process, log, directory, port=30000
                            )
                            save(directory / "lifecycle.json", lifecycle)
                            assert lifecycle["status"] == "passed"
                        else:
                            raise RuntimeError(
                                "Server did not reach an orderly cleanup boundary"
                            )
                    except BaseException as cleanup_error:
                        save(directory / 'cleanup-error.json', {'type': type(cleanup_error).__name__, 'error': str(cleanup_error)})
                        save(
                            directory / "containment.json",
                            runtime.emergency_contain_process_group(process, target),
                        )
                        if original_error is not None:
                            raise original_error from cleanup_error
                        raise
    save(
        OUT / "VERDICT.json",
        {
            "status": "MEASURED_REVIEW_REQUIRED",
            "requests": len(results),
            "limits": "No gameplay; no semantic or hybrid-state correctness certification; cold request cache; one sequence; no run-order control",
        },
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        OUT.mkdir(exist_ok=True, parents=True)
        save(OUT / "FAILURE.json", {"type": type(error).__name__, "error": str(error)})
        raise
