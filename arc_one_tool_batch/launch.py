"""Launch one vLLM server and register a one-tool agent in a writable Swarm copy."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import httpx

from one_tool_runtime import Config, Deadline, Journal, SYSTEM, VLLM, compact, decode_reply


def stop(process: subprocess.Popen | None, grace: float = 15) -> None:
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def preflight(config: Config) -> None:
    """Two fake observations, zero environment actions, checks the tool round trip."""
    backend = VLLM(replace(config, max_new_tokens=min(config.max_new_tokens, 2048)))
    journal = Journal(config.log_dir, "__protocol_preflight__")
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": 'Protocol check, not a scored game. Call do_actions '
         'with ACTION1 exactly once. Available actions: ACTION1, ACTION2. Grid: 00\\n01.'},
    ]
    try:
        for i in range(2):
            output = backend.complete(messages, f"preflight-{i}", journal)
            assistant, call_id, actions = decode_reply(output, config.max_tool_actions)
            if len(actions) != 1 or actions[0].name != "ACTION1":
                raise RuntimeError("Preflight did not follow the one-action protocol")
            messages.append(assistant)
            messages.append({"role": "tool", "tool_call_id": call_id, "content": compact({
                "executed": 1, "state": "NOT_FINISHED", "available_actions": ["ACTION1", "ACTION2"],
                "frames": [["00", "01"]],
                "instruction": "Continue this protocol check by calling ACTION1 exactly once again."
            })})
        journal.write("preflight_passed", generations=2, environment_actions=0)
    finally:
        backend.close_thread()
        journal.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="Mounted local checkpoint directory")
    parser.add_argument("--agents-dir", type=Path, required=True, help="Writable copy of ARC-AGI-3-Agents")
    parser.add_argument("--game", help="Optional game filter for development only")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()
    if "ARC_RUN_STARTED_MONOTONIC" not in os.environ:
        parser.error("Set ARC_RUN_STARTED_MONOTONIC in the notebook's first cell, before setup")
    if not args.model.is_dir():
        parser.error("--model must be an existing local checkpoint directory; no downloads")
    args.agents_dir = args.agents_dir.resolve()
    if not (args.agents_dir / "main.py").exists():
        parser.error("--agents-dir must contain the official main.py")
    executable = shutil.which("vllm")
    if not executable:
        parser.error("vLLM is not installed; install your tested offline wheels first")

    # One run-specific journal directory, shared by all game threads.
    os.environ.setdefault("ONE_TOOL_LOG_DIR", str(Path.cwd() / f"one_tool_run_{time.time_ns()}"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    config = Config.from_env()
    config.remaining()
    logs = Path(config.log_dir)
    logs.mkdir(parents=True, exist_ok=True)

    # Keep the provided competition gateway, scorecard lifecycle and game discovery.
    destination = args.agents_dir / "agents" / "templates"
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("one_tool_runtime.py", "one_tool_agent.py"):
        shutil.copyfile(Path(__file__).parent / name, destination / name)
    init = args.agents_dir / "agents" / "__init__.py"
    marker = "# one-tool-runtime registration"
    text = init.read_text()
    if marker not in text:
        init.write_text(text + '\n' + marker + '\nfrom .templates.one_tool_agent import OneTool\n'
                        'AVAILABLE_AGENTS["onetool"] = OneTool\n')

    host = httpx.URL(config.base_url)
    if host.host not in {"127.0.0.1", "localhost"} or host.scheme != "http":
        parser.error("This launcher only binds a local HTTP model endpoint")
    model_len = int(os.getenv("MAX_MODEL_LEN", "32768"))
    if config.max_new_tokens >= model_len:
        parser.error("MAX_NEW_TOKENS must leave room for the prompt inside MAX_MODEL_LEN")
    command = [
        executable, "serve", str(args.model.resolve()),
        "--host", str(host.host), "--port", str(host.port or 8000),
        "--served-model-name", config.model,
        "--dtype", "auto", "--tensor-parallel-size", "1",
        "--max-model-len", str(model_len),
        "--max-num-seqs", os.getenv("MAX_NUM_SEQS", str(config.concurrency)),
        "--max-num-batched-tokens", os.getenv("MAX_BATCHED_TOKENS", "16384"),
        "--gpu-memory-utilization", os.getenv("GPU_MEMORY_UTILIZATION", "0.90"),
        "--enable-prefix-caching", "--enable-chunked-prefill",
        "--enable-auto-tool-choice",
        "--tool-call-parser", os.getenv("TOOL_CALL_PARSER", "hermes"),
        "--reasoning-parser", os.getenv("REASONING_PARSER", "qwen3"),
    ]
    # A JSON array avoids shell splitting/quoting ambiguities for extra engine flags.
    extra_args = json.loads(os.getenv("EXTRA_VLLM_ARGS_JSON", "[]"))
    if not isinstance(extra_args, list) or any(not isinstance(x, str) for x in extra_args):
        parser.error("EXTRA_VLLM_ARGS_JSON must be an array of strings")
    command.extend(extra_args)
    versions = {}
    for package in ("vllm", "torch", "transformers", "httpx", "arc-agi", "arcengine"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    hashes = {name: hashlib.sha256((args.model / name).read_bytes()).hexdigest()
              for name in ("config.json", "tokenizer_config.json", "generation_config.json")
              if (args.model / name).exists()}
    (logs / "manifest.json").write_text(json.dumps({
        "config": asdict(config), "server_command": command, "versions": versions,
        "checkpoint_config_hashes": hashes,
        "checkpoint_path": str(args.model.resolve()),
        "python": sys.version, "started_monotonic": float(os.environ["ARC_RUN_STARTED_MONOTONIC"]),
        "system_prompt": SYSTEM,
        "note": "Checkpoint weight hashes and GPU throughput are not inferred from config hashes."
    }, indent=2))
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv"], capture_output=True, text=True, timeout=10)
        (logs / "gpu.txt").write_text(gpu.stdout + gpu.stderr)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    server = agent = None
    with (logs / "server.log").open("w") as server_log, (logs / "agent.log").open("w") as agent_log:
        try:
            print("Starting:", " ".join(command), flush=True)
            server = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
            with httpx.Client(base_url=config.base_url, trust_env=False, timeout=3) as http:
                while True:
                    config.remaining()
                    if server.poll() is not None:
                        raise RuntimeError(f"vLLM exited; inspect {logs / 'server.log'}")
                    try:
                        if http.get("/health").status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(0.5)  # health checks during startup only; no generation polling
                if not args.skip_preflight:
                    preflight(config)
                (logs / "metrics_before.txt").write_text(http.get("/metrics").text)
                runner = [sys.executable, "main.py", "--agent", "onetool"]
                if args.game:
                    runner.extend(["--game", args.game])
                agent = subprocess.Popen(runner, cwd=args.agents_dir, stdout=agent_log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                print(f"Playing through official Swarm; logs: {logs}", flush=True)
                try:
                    agent.wait(timeout=config.remaining())
                except (subprocess.TimeoutExpired, Deadline):
                    # Official main.py handles SIGINT by closing its scorecard.
                    os.killpg(agent.pid, signal.SIGINT)
                    agent.wait(timeout=60)
                try:
                    (logs / "metrics_after.txt").write_text(http.get("/metrics").text)
                except httpx.HTTPError:
                    pass
                if agent.returncode not in (0, -signal.SIGINT):
                    raise RuntimeError(f"Agent exited {agent.returncode}; inspect agent.log")
        finally:
            stop(agent)
            stop(server)


if __name__ == "__main__":
    main()
