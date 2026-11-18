"""Bounded exact-request branches with real workspace tools, never engine actions.

Initial messages/schema/sampling are checkpoint values plus the selected treatment.
Subsequent tool history uses the native bounded-context functions and projection.
Only `run` contacts a provider. `self-test` uses scripted responses and real tools.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ARMS = ("baseline", "terminal", "time", "both")
FINAL_TOKENS = 2048
FINAL_ATTEMPTS = 2


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def append(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


def tree(root):
    rows = [{"path": str(p.relative_to(root)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(root.rglob("*")) if p.is_file() and "__pycache__" not in p.parts]
    return {"sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(), "files": rows}


def validate_packet(packet, request):
    manifest = json.loads((packet / "workspace.json").read_text())
    def commitment(value):
        raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        return "sha256:" + hashlib.sha256(raw).hexdigest()
    if commitment(manifest) != request["workspace_commitment"]:
        raise ValueError("Workspace manifest commitment drift")
    if commitment(request["request_body"]) != request["request_payload_commitment"]:
        raise ValueError("Saved request commitment drift")
    for entry in manifest["files"]:
        path = packet / "workspace" / entry["path"]
        raw = path.read_bytes()
        if len(raw) != entry["size"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("Saved workspace bytes drift: " + entry["path"])


def inject(messages, index, suffix):
    result = copy.deepcopy(messages)
    if suffix:
        content = result[index]["content"]
        if isinstance(content, str):
            result[index]["content"] = content + suffix
        else:
            first = next(p for p in content if p.get("type") == "text")
            first["text"] += suffix
    return result


class ObservationEvents:
    """The clock wrapper needs only the current observation's mutable prompt."""
    def __init__(self, suffix, observation_ref):
        self.event = SimpleNamespace(prompt=suffix, metadata={observation_ref: True})

    def items(self):
        return [("branch-observation", self.event)]

    def update(self, tag, **fields):
        if tag != "branch-observation":
            raise KeyError(tag)
        for key, value in fields.items():
            setattr(self.event, key, value)


class ScriptedClient:
    """Network-free test adapter with ordinary NOOA responses."""
    def __init__(self, script):
        self.script = iter(script)
        self.calls = []

    async def acall(self, messages, tools, **kwargs):
        from nooa.unifiedllm import LLMResponse, ToolCall
        item = next(self.script)
        self.calls.append({"messages": copy.deepcopy(messages), "tools": [t.name for t in tools], **kwargs})
        if item.get("delay"):
            await asyncio.sleep(item["delay"])
        tc = ToolCall(id=f"scripted-{len(self.calls)}", name=item["name"], arguments=json.dumps(item["arguments"]))
        return LLMResponse(raw_response={"scripted": item}, content="", tool_calls=[tc],
                           finish_reason="tool_calls", assistant_message={}, reasoning=item.get("reasoning", ""),
                           usage={"prompt_tokens": 1000, "completion_tokens": item.get("tokens", 20)})

    async def aclose(self):
        pass


class RequestRuntime:
    def __init__(self, *, client, request, messages, observation_index, surface, destination,
                 suffix, seconds, cap, clock_config, config):
        from tycho_nooa.deadlines import DecisionDeadlineConfig
        from tycho_nooa.history import OBSERVATION_REF
        from hybrid.policy import ContextProjection
        self.client, self.request = client, request
        self.messages, self.observation_index = messages, observation_index
        self.destination = destination
        self.event_manager = ObservationEvents(suffix, OBSERVATION_REF)
        self.agent = SimpleNamespace(_tool_surface=surface, _native_tool_config=config,
                                     _active_deadline=DecisionDeadlineConfig().start(seconds))
        self.projection = ContextProjection(window=65536)
        self.started = time.monotonic()
        self.completion_tokens = 0
        self.unknown_reserved_tokens = 0
        self.cap = cap
        self.final_tokens_remaining = FINAL_TOKENS * FINAL_ATTEMPTS
        if cap < self.final_tokens_remaining:
            raise ValueError("Completion cap must cover both final action attempts")
        self.calls = 0
        self.clock_config = clock_config

    async def generate(self, **kwargs):
        tools = kwargs.pop("tools")
        params = {**self.request["request_body"]["parameters"], **kwargs}
        reserved = int(params["max_tokens"])
        final = bool(tools) and all(t.name == "take_actions" for t in tools)
        protected = self.final_tokens_remaining - reserved if final else self.final_tokens_remaining
        if protected < 0 or self.completion_tokens + self.unknown_reserved_tokens + reserved + protected > self.cap:
            append(self.destination / "admission.jsonl", {"request_sent": False, "action_demand": final,
                   "completed_tokens": self.completion_tokens, "unknown_reserved_tokens": self.unknown_reserved_tokens,
                   "requested_tokens": reserved, "protected_final_tokens": self.final_tokens_remaining,
                   "completion_token_cap": self.cap})
            raise TokenBudgetExhausted("Next request would consume protected final-action tokens or exceed cap")
        messages = inject(self.messages, self.observation_index, self.event_manager.event.prompt)
        # The initial request has already passed projection; preserve it verbatim.
        if self.calls:
            messages = self.projection.project(messages, tools, reserved)
        self.calls += 1
        if final:
            self.final_tokens_remaining = protected
        record = {"index": self.calls, "elapsed_seconds": time.monotonic() - self.started,
                  "model": self.request["request_body"]["model"], "messages": messages,
                  "tools": [{"name": t.name, "description": t.description, "parameters": t.get_parameter_schema()} for t in tools],
                  "parameters": params}
        write(self.destination / "requests" / f"{self.calls:03d}.json", record)
        try:
            response = await self.client.acall(messages=messages, tools=tools, **params)
        except BaseException as error:
            self.unknown_reserved_tokens += reserved
            append(self.destination / "provider-usage.jsonl", {"call": self.calls, "usage": "unknown",
                   "reserved_max_tokens": reserved, "error_type": type(error).__name__, "provider_cancellation_confirmed": False})
            raise
        raw = response.raw_response
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump()
        usage = response.usage or {}
        completed = usage.get("completion_tokens")
        if completed is None:
            self.unknown_reserved_tokens += reserved
        else:
            self.completion_tokens += int(completed)
        write(self.destination / "responses" / f"{self.calls:03d}.json",
              {"raw": raw, "content": str(response.content), "reasoning": response.reasoning,
               "usage": usage, "tool_calls": [asdict(t) for t in response.tool_calls],
               "elapsed_seconds": time.monotonic() - self.started})
        self.projection.calibrate(usage.get("prompt_tokens", 0))
        return response


from hybrid.decision_time import RequestNotSent


class TokenBudgetExhausted(RequestNotSent):
    pass


def exact_tools(surface, records):
    from nooa.unifiedllm import Tool
    native = {t.name: t for t in surface.native_tools()}

    class SavedTool(Tool):
        def get_parameter_schema(self, *, strict=False):
            if strict:
                raise ValueError("Saved schema is non-strict; original completion route required")
            return copy.deepcopy(self.saved_schema)

    result = []
    for record in records:
        actual = native[record["name"]]
        if actual.get_parameter_schema() != record["parameters"]:
            raise ValueError("Native validation schema drift: " + actual.name)
        tool = SavedTool(name=actual.name, description=record["description"], callable=actual.callable,
                         parameters_model=actual.parameters_model)
        tool.saved_schema = record["parameters"]
        result.append(tool)
    return result


async def case(packet, destination, arm, runtime_path, *, endpoint=None, seconds=180, cap=25000,
               client=None, max_calls=25):
    if arm not in ARMS:
        raise ValueError(arm)
    if destination.exists():
        raise FileExistsError(destination)
    # Each case is a separate process in the suite; no shared environment or imports.
    sys.path.insert(0, str(runtime_path.resolve()))
    os.environ.update(TYCHO_MODE="no_world_model", TYCHO_NOOA_TREATMENT="lean_prune",
                      TYCHO_TIME_AWARE="1" if arm in ("time", "both") else "0",
                      TYCHO_TERMINAL_FRAMES="1" if arm in ("terminal", "both") else "0",
                      TYCHO_SANDBOX_RUNTIME="host")
    from tycho.workspace.workspace import GameWorkspace
    from tycho_nooa.models import GameTurn
    from tycho_nooa.tools import WorkspaceToolSurface, actor_tool_profile
    from tycho_nooa.modes import mode_from_env
    from tycho_nooa.wm_probe import GatedToolExecutor, TrustedRuntimePythonSandbox
    from tycho_nooa.strategy import NativeToolConfig, ACTION_DEMAND_REQUEST
    from tycho_nooa.history import bounded_context_text, bounded_tool_arguments, replayed_reasoning
    from hybrid import decision_time, terminal_frames

    request = json.loads((packet / "request.json").read_text())
    validate_packet(packet, request)
    manifest = json.loads((packet / "manifest.json").read_text())
    turn = GameTurn.model_validate_json((packet / "turn.json").read_text())
    destination.mkdir(parents=True)
    game_id = manifest["game_id"]
    workspace_path = destination / "workspace" / game_id.split("-")[0]
    shutil.copytree(packet / "workspace", workspace_path)
    if os.environ.get("TYCHO_KAGGLE_HARDENED_HOST") == "1":
        from hybrid_local_solver import _prepare_hardened_session, _require_hardened_host_supervisor
        _require_hardened_host_supervisor(dict(os.environ))
        # Suite assigns a distinct identity before starting each process. Never
        # use a per-process allocator, whose first UID would collide across arms.
        if not os.environ.get("TYCHO_KAGGLE_GAME_UID") or not os.environ.get("TYCHO_KAGGLE_GAME_GID"):
            raise ValueError("Hardened case requires suite-assigned unique UID/GID")
        _prepare_hardened_session(destination, workspace_path.parent, dict(os.environ))
    before = tree(workspace_path)
    write(destination / "workspace-before.json", before)
    ws = GameWorkspace(game_id, root=workspace_path.parent, resume=True, render=False, seed_world_model=False,
                       available_actions=turn.available_actions)
    executor = GatedToolExecutor(ws, wm_feedback_enabled=False, python_sandbox=TrustedRuntimePythonSandbox(runtime="host"))
    surface = WorkspaceToolSurface(executor, profile=actor_tool_profile(mode_from_env()), mode_policy=mode_from_env(),
                                   trace_path=destination / "tool-trace.jsonl")
    surface.begin_turn(turn)
    tools = exact_tools(surface, request["request_body"]["tools"])
    by_name = {t.name: t for t in tools}
    config = NativeToolConfig(max_tokens=int(request["request_body"]["parameters"]["max_tokens"]))
    if before != tree(workspace_path):
        raise ValueError("Native tool construction mutated the exact restored workspace")
    if client is None:
        from nooa.unifiedllm import RetryConfig
        from nooa.unifiedllm.registry import get_llm_client
        if not endpoint:
            raise ValueError("Explicit matching model endpoint required")
        cc = copy.deepcopy(request["replay_context"]["client_config"])
        cc.update(api_base=endpoint, api_key=os.environ.get("ARC_LLM_API_KEY", "local"),
                  num_retries=0, retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0))
        client = get_llm_client(request["request_body"]["model"], client_type="completion", **cc)
        client.cache_control_injection_points = request["replay_context"]["cache_control_injection_points"]
    messages = copy.deepcopy(request["request_body"]["messages"])
    obs_index = max(i for i, m in enumerate(messages) if m["role"] == "user")
    extra = terminal_frames.context(ws, turn)
    raw = RequestRuntime(client=client, request=request, messages=messages, observation_index=obs_index,
                         surface=surface, destination=destination, suffix=extra, seconds=seconds, cap=cap,
                         clock_config=None, config=config)
    runtime = decision_time.decision_runtime(raw)
    final = False
    final_attempts = 0
    stop, action = "unstarted", None
    write(destination / "case.json", {"arm": arm, "packet": manifest, "seconds": seconds, "completion_token_cap": cap,
          "claim_ceiling": "Exact initial saved request plus selected treatment; real restored tools; first valid action commitment only, not a game score.",
          "terminal_context_bytes": len(extra.encode()), "workspace_unchanged_on_construct": before == tree(workspace_path)})
    try:
        async with asyncio.timeout(seconds):
            while raw.calls < max_calls:
                final_at_start = final
                if final:
                    final_attempts += 1
                selected = [by_name["take_actions"]] if final else tools
                options = {"tools": selected, "max_tokens": FINAL_TOKENS if final else config.max_tokens,
                           "temperature": request["request_body"]["parameters"].get("temperature", 0.7),
                           "tool_choice": request["request_body"]["parameters"].get("tool_choice", "auto")}
                if final:
                    options["extra_body"] = {"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}
                try:
                    response = await runtime.generate(**options)
                except decision_time.ExplorationExpired:
                    if final:
                        raise
                    final = True
                    raw.messages.append({"role": "user", "content": ACTION_DEMAND_REQUEST})
                    continue
                except TokenBudgetExhausted:
                    if not final:
                        final = True
                        raw.messages.append({"role": "user", "content": ACTION_DEMAND_REQUEST})
                        continue
                    stop = "completion_token_guard"
                    break
                if response.content:
                    message = {"role": "assistant", "content": bounded_context_text(str(response.content), limit=config.max_context_value_bytes, kind="assistant prose")}
                    if response.reasoning:
                        message["reasoning_content"] = bounded_context_text(str(response.reasoning), limit=config.max_context_value_bytes, kind="assistant reasoning")
                    raw.messages.append(message)
                if not response.tool_calls:
                    raw.messages.append({"role": "user", "content": "A workspace or take_actions tool call is required. Do not finish the turn with prose alone."})
                for index, tc in enumerate(response.tool_calls):
                    try:
                        args = json.loads(tc.arguments)
                        tool = by_name[tc.name]
                        if final and tc.name != "take_actions":
                            raise ValueError("Only take_actions is available during final demand")
                        projected = bounded_tool_arguments(args, limit=config.max_context_value_bytes, tool_name=tc.name)
                        am = {"role": "assistant", "content": None, "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(projected)}}]}
                        if index == 0 and (reasoning := replayed_reasoning(response, max_bytes=config.max_context_value_bytes)):
                            am["reasoning_content"] = reasoning
                        raw.messages.append(am)
                        if tc.name == "run_python":
                            # Common external bound also covers blocking subprocesses in baseline.
                            remaining = int(seconds - (time.monotonic() - raw.started))
                            if remaining < 1:
                                raise TimeoutError("External decision deadline exhausted before Python")
                            args = {**args, "timeout_s": min(args.get("timeout_s", 15), remaining)}
                        result = decision_time.dispatch(runtime, surface, tool, args, tool_call_id=tc.id, raw_arguments=tc.arguments)
                        if tc.name == "take_actions":
                            action = result.model_dump() if hasattr(result, "model_dump") else asdict(result)
                            stop = "action_committed"
                            break
                        visible = bounded_context_text(str(result), limit=config.max_context_value_bytes, kind=f"{tc.name} result")
                    except decision_time.ExplorationExpired:
                        final = True
                        visible = "Investigation time exhausted; commit an action now."
                    except Exception as error:
                        visible = f"Tool error: {type(error).__name__}: {error}"
                        # Invalid JSON or unknown name still needs a structurally paired result.
                        if not raw.messages or not raw.messages[-1].get("tool_calls"):
                            raw.messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}]})
                    raw.messages.append({"role": "tool", "tool_call_id": tc.id, "content": visible})
                if action is not None:
                    break
                if final and not final_at_start:
                    raw.messages.append({"role": "user", "content": ACTION_DEMAND_REQUEST})
                    continue
                if final_at_start:
                    if final_attempts >= FINAL_ATTEMPTS:
                        stop = "final_demand_no_action"
                        break
                    raw.messages.append({"role": "user", "content": ACTION_DEMAND_REQUEST})
                    continue
                if raw.calls >= max_calls - FINAL_ATTEMPTS:
                    final = True
                    raw.messages.append({"role": "user", "content": ACTION_DEMAND_REQUEST})
            else:
                stop = "call_limit"
    except TimeoutError:
        stop = "decision_deadline"
    except Exception as error:
        stop = "runtime_error"
        write(destination / "error.json", {"type": type(error).__name__, "message": str(error)})
    finally:
        await client.aclose()
    result = {"arm": arm, "packet": manifest["label"], "stop_reason": stop, "action": action,
              "elapsed_seconds": time.monotonic() - raw.started, "completed_completion_tokens": raw.completion_tokens,
              "unknown_usage_reserved_tokens": raw.unknown_reserved_tokens, "provider_usage_complete": raw.unknown_reserved_tokens == 0,
              "calls_started": raw.calls, "tool_calls_completed": len(surface.turn_tool_calls()), "environment_actions_executed": 0}
    result["unspent_final_token_reservation"] = raw.final_tokens_remaining
    write(destination / "workspace-after.json", tree(workspace_path))
    write(destination / "result.json", result)
    return result


def suite(args):
    from concurrent.futures import ThreadPoolExecutor
    args.output.mkdir(parents=True, exist_ok=False)
    blocks = [(p, replica) for replica in range(args.replicas) for p in sorted(args.packets.iterdir()) if (p / "manifest.json").exists()]
    rng = random.Random(20261007)
    rng.shuffle(blocks)
    manifest = []
    write(args.output / "suite-status.json", {"status": "running", "planned_blocks": len(blocks), "completed_blocks": 0})
    for block_index, (packet, replica) in enumerate(blocks):
        arms = list(ARMS)
        rng.shuffle(arms)
        def invoke(arm):
            dest = args.output / f"block-{block_index:02d}-{packet.name}-r{replica}" / arm
            dest.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(Path(__file__).resolve()), "case", "--packet", str(packet.resolve()),
                       "--output", str(dest.resolve()), "--arm", arm, "--runtime", str(args.runtime.resolve()),
                       "--endpoint", args.endpoint, "--seconds", str(args.seconds), "--tokens", str(args.tokens)]
            environment = dict(os.environ)
            if environment.get("TYCHO_KAGGLE_HARDENED_HOST") == "1":
                uid = 60001 + block_index * 4 + ARMS.index(arm)
                if uid > 60999:
                    raise ValueError("Per-case hardened identity range exhausted")
                environment.update(TYCHO_KAGGLE_GAME_UID=str(uid), TYCHO_KAGGLE_GAME_GID=str(uid))
            completed = subprocess.run(command, capture_output=True, text=True, env=environment)
            (dest.parent / f"{arm}-process.log").write_text(completed.stdout + completed.stderr)
            return {"block": block_index, "packet": packet.name, "replica": replica, "arm": arm,
                    "returncode": completed.returncode, "result": json.loads((dest / "result.json").read_text()) if (dest / "result.json").exists() else None}
        with ThreadPoolExecutor(max_workers=4) as pool:
            block_results = list(pool.map(invoke, arms))
            manifest.extend(block_results)
        write(args.output / "suite.json", manifest)
        failed = [r for r in block_results if r["returncode"] != 0 or r["result"] is None
                  or r["result"]["stop_reason"] == "runtime_error"]
        if failed:
            write(args.output / "suite-status.json", {"status": "aborted", "planned_blocks": len(blocks),
                  "completed_blocks": block_index + 1, "reason": "Case process/setup/provider failure; subsequent blocks not launched",
                  "failed_cases": [{k: r[k] for k in ("packet", "arm", "replica", "returncode")} for r in failed]})
            raise RuntimeError("Block qualification failed; retained receipts and aborted remaining blocks")
        write(args.output / "suite-status.json", {"status": "complete" if block_index + 1 == len(blocks) else "running",
              "planned_blocks": len(blocks), "completed_blocks": block_index + 1})
        print(json.dumps({"block_complete": block_index, "cases": len(manifest)}), flush=True)
    return manifest


async def self_test(args):
    # Exercise real restored run_python, real invalid-action validation, then valid commitment.
    script = [{"name": "run_python", "arguments": {"code": "import wmlib; print(grid.shape); print(len(wmlib.frames()))"}},
              {"name": "take_actions", "arguments": {"actions": [{"action": "FLY"}], "rationale": "invalid test"}},
              {"name": "take_actions", "arguments": {"actions": [{"action": "LEFT"}], "rationale": "test commitment"}}]
    fake = ScriptedClient(script)
    result = await case(args.packet, args.output, "baseline", args.runtime, client=fake, seconds=30)
    assert result["stop_reason"] == "action_committed", result
    assert result["calls_started"] == 3, result
    original = json.loads((args.packet / "request.json").read_text())["request_body"]
    assert fake.calls[0]["messages"] == original["messages"]
    trace = [json.loads(s) for s in (args.output / "tool-trace.jsonl").read_text().splitlines()]
    assert trace[0]["name"] == "run_python" and "(64, 64)" in trace[0]["result"], trace[0]
    assert trace[1]["error_stage"] == "validation", trace[1]
    # Force a timed-out exploratory request, then let the actual clock wrapper
    # reserve and obtain a final action. Unknown provider usage remains charged.
    delayed = [{"name": "run_python", "arguments": {"code": "print(1)"}, "delay": 3},
               {"name": "take_actions", "arguments": {"actions": [{"action": "LEFT"}], "rationale": "clock fallback"}}]
    clock_client = ScriptedClient(delayed)
    clock_result = await case(args.packet, args.output / "clock", "both", args.runtime,
                              client=clock_client, seconds=2)
    assert clock_result["stop_reason"] == "action_committed", clock_result
    assert clock_result["unknown_usage_reserved_tokens"] == 12288, clock_result
    assert clock_result["completed_completion_tokens"] == 20, clock_result
    assert clock_client.calls[1]["tools"] == ["take_actions"]
    assert clock_client.calls[1]["max_tokens"] == 2048
    assert clock_client.calls[1]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    first_text = json.dumps(clock_client.calls[0]["messages"])
    assert "OBSERVED WINNING TRANSITIONS" in first_text
    assert "HOST DECISION CLOCK" in first_text
    cap_client = ScriptedClient(delayed)
    cap_result = await case(args.packet, args.output / "token-cap", "time", args.runtime,
                            client=cap_client, seconds=2, cap=16384)
    assert cap_result["stop_reason"] == "action_committed", cap_result
    assert len(cap_client.calls) == 2
    assert cap_result["unknown_usage_reserved_tokens"] == 12288
    # The observed ledgers must force an unsent exploration rejection while
    # preserving both final attempts, even if the first final consumes its max.
    ledger_results = []
    for known in (10966, 11412, 11154, 10945):
        ledger_client = ScriptedClient([{**script[0], "tokens": known},
                                       {**script[1], "tokens": 2048}, {**script[2], "tokens": 2048}])
        row = await case(args.packet, args.output / f"ledger-{known}", "time", args.runtime,
                         client=ledger_client, seconds=30)
        assert row["stop_reason"] == "action_committed", row
        assert row["calls_started"] == 3 and row["unknown_usage_reserved_tokens"] == 0, row
        assert row["completed_completion_tokens"] == known + 4096, row
        assert [c["tools"] for c in ledger_client.calls[1:]] == [["take_actions"], ["take_actions"]]
        lifecycle = [json.loads(s) for s in (args.output / f"ledger-{known}" / "actor-lifecycle.jsonl").read_text().splitlines()]
        assert sum(r["phase"] == "request_not_sent" for r in lifecycle) == 1, lifecycle
        assert not any(r["phase"] == "provider_interrupted" for r in lifecycle), lifecycle
        ledger_results.append(row)
    # Exact cap: known8616 + canceled12288 + two final2048 =25000.
    boundary_client = ScriptedClient([{**script[0], "tokens": 8616}, {**delayed[0], "delay": 5},
                                     {**script[1], "tokens": 2048}, {**script[2], "tokens": 2048}])
    boundary = await case(args.packet, args.output / "cancel-boundary", "both", args.runtime,
                          client=boundary_client, seconds=3)
    assert boundary["stop_reason"] == "action_committed", boundary
    assert boundary["completed_completion_tokens"] + boundary["unknown_usage_reserved_tokens"] == 25000, boundary
    assert boundary["calls_started"] == 4 and boundary["unspent_final_token_reservation"] == 0, boundary
    refusal = ScriptedClient([script[0], script[1], script[1]])
    refusal_result = await case(args.packet, args.output / "final-refusal", "baseline", args.runtime,
                                client=refusal, seconds=30, max_calls=3)
    assert refusal_result["stop_reason"] == "final_demand_no_action", refusal_result
    assert refusal_result["action"] is None
    assert len(refusal.calls) == 3
    assert [c["tools"] for c in refusal.calls[1:]] == [["take_actions"], ["take_actions"]]
    assert all(c["tool_choice"] == "auto" for c in refusal.calls)
    # Qualify the real saved CompletionClient and its wire conversion against a
    # local HTTP fixture. This never calls an external model/provider.
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    wire = []
    fixture = [script[0], script[2]]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *unused):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            wire.append(body)
            index = len(wire) - 1
            item = fixture[index]
            response = {"id": f"loopback-{index}", "object": "chat.completion", "created": 1,
                        "model": body["model"], "choices": [{"index": 0, "finish_reason": "tool_calls",
                        "message": {"role": "assistant", "content": None, "reasoning_content": "Fixture reasoning.",
                                    "tool_calls": [{"id": f"wire-{index}", "type": "function", "function": {
                                        "name": item["name"], "arguments": json.dumps(item["arguments"])}}]}}],
                        "usage": {"prompt_tokens": 1000, "completion_tokens": 20, "total_tokens": 1020}}
            raw_body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw_body)))
            self.end_headers()
            self.wfile.write(raw_body)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        loopback_result = await case(args.packet, args.output / "loopback", "baseline", args.runtime,
                                     endpoint=f"http://127.0.0.1:{server.server_port}/v1", seconds=30)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    write(args.output / "loopback-wire.json", wire)
    assert loopback_result["stop_reason"] == "action_committed", loopback_result
    assert len(wire) == 2
    for body in wire:
        assert body["model"] == "flashnext", body["model"]
        assert body["temperature"] == 0.7
        assert body["max_tokens"] == 12288
        assert body["top_p"] == 0.95
        assert body["top_k"] == 20
        assert body["chat_template_kwargs"]["enable_thinking"] is True
        assert body["tool_choice"] == "auto"
    assert any(m.get("role") == "tool" and m.get("tool_call_id") == "wire-0" and "(64, 64)" in json.dumps(m["content"]) for m in wire[1]["messages"])
    assert [t["function"]["name"] for t in wire[0]["tools"]] == [t["name"] for t in original["tools"]]
    # A corrupt packet must stop the suite before its second block and before
    # constructing any provider client. Four processes in the first block fail.
    bad_packets = args.output / "bad-packets"
    for name in ("bad-a", "bad-b"):
        write(bad_packets / name / "manifest.json", {"schema": "intentionally-invalid-fixture"})
    failed_suite = args.output / "failed-suite"
    failed = await asyncio.to_thread(subprocess.run, [sys.executable, str(Path(__file__).resolve()), "run",
        "--runtime", str(args.runtime.resolve()), "--packets", str(bad_packets.resolve()),
        "--output", str(failed_suite.resolve()), "--replicas", "1", "--endpoint", "http://127.0.0.1:1/v1"],
        capture_output=True, text=True)
    assert failed.returncode != 0
    failure_status = json.loads((failed_suite / "suite-status.json").read_text())
    assert failure_status["status"] == "aborted" and failure_status["completed_blocks"] == 1, failure_status
    assert len(json.loads((failed_suite / "suite.json").read_text())) == 4
    print(json.dumps({"self_test": "PASS", "baseline": result, "clock": clock_result, "token_cap": cap_result,
                      "recorded_ledgers": ledger_results, "cancel_boundary": boundary,
                      "final_refusal": refusal_result, "completion_client_loopback": loopback_result, "suite_failfast": failure_status}))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("case", "self-test", "run"):
        p = sub.add_parser(name)
        p.add_argument("--runtime", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        if name == "run":
            p.add_argument("--packets", type=Path, required=True)
            p.add_argument("--replicas", type=int, default=2)
        else:
            p.add_argument("--packet", type=Path, required=True)
        if name != "self-test":
            p.add_argument("--endpoint", required=True)
            p.add_argument("--seconds", type=float, default=180)
            p.add_argument("--tokens", type=int, default=25000)
        if name == "case":
            p.add_argument("--arm", choices=ARMS, required=True)
    args = parser.parse_args()
    if args.command == "run":
        suite(args)
    elif args.command == "self-test":
        asyncio.run(self_test(args))
    else:
        print(json.dumps(asyncio.run(case(args.packet, args.output, args.arm, args.runtime,
                                        endpoint=args.endpoint, seconds=args.seconds, cap=args.tokens))))


if __name__ == "__main__":
    main()
