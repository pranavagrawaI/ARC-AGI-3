"""Exercise one real Tycho-NOOA worker decision through an OpenAI endpoint.

With no ``--base-url`` this starts a deterministic local fake endpoint. That
mode proves the Solver/NOOA/LiteLLM/HTTP/protocol join without claiming model
quality. A later frozen Kaggle candidate points the same probe at SGLang.
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import threading
from pathlib import Path
from typing import Any, cast

from runtime.local_solver import create_solver
from tycho_nooa.models import GameTurn
from tycho_nooa.protocol import (
    PROTOCOL_VERSION,
    ActionResultMessage,
    project_arc_observation,
)

SCHEMA = "tycho-nooa-s0-solver-join-v1"


class FakeCompletionHandler(http.server.BaseHTTPRequestHandler):
    requests: list[dict[str, Any]] = []

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        size = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(size))
        type(self).requests.append(request)
        response = {
            "id": "solver-join-fake",
            "object": "chat.completion",
            "model": request.get("model"),
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "UP is one of the available actions.",
                        "tool_calls": [
                            {
                                "id": "solver-join-call-0",
                                "type": "function",
                                "function": {
                                    "name": "take_action",
                                    "arguments": '{"action":"UP"}',
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }
        encoded = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def start_fake_server() -> tuple[http.server.ThreadingHTTPServer, threading.Thread, str]:
    FakeCompletionHandler.requests = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeCompletionHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = cast(tuple[str, int], server.server_address)
    return server, thread, f"http://{host}:{port}/v1"


def run(
    *, base_url: str, model: str, workspace_root: Path, fake: bool
) -> dict[str, Any]:
    before = project_arc_observation(
        frames=[[0, 1], [0, 0]],
        available_action_ids=[1, 2],
        state="NOT_FINISHED",
        levels_completed=0,
        full_reset=False,
    )
    after = project_arc_observation(
        frames=[[0, 0], [0, 1]],
        available_action_ids=[1, 2],
        state="NOT_FINISHED",
        levels_completed=0,
        full_reset=False,
    )
    os.environ.update(
        {
            "ARC_LLM_API_KEY": "local-solver-join",
            "ARC_LLM_BASE_URL": base_url,
            "ARC_LLM_CONTEXT_WINDOW": "32768",
            "ARC_LLM_FORCE_ACTION_TOOL": "1",
            "ARC_LLM_MAX_ACTOR_CALLS": "26",
            "ARC_LLM_MAX_TOKENS": "16000",
            "ARC_LLM_MODEL": model,
            "ARC_LLM_RETRIES": "0",
            "ARC_LLM_TIMEOUT_SECONDS": "600",
            "ARC_LLM_USE_RESPONSES": "0",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "TYCHO_NOOA_TREATMENT": "decision_discipline",
            "TYCHO_NOOA_WORKER_TIMEOUT_SECONDS": "7200",
            "TYCHO_MODE": "trigger",
        }
    )
    solver = create_solver(
        game_id="s0-solver-join",
        workspace_root=workspace_root,
        initial_observation=before,
    )
    try:
        decision = solver.decide(
            GameTurn(
                grid=before.grid,
                available_actions=before.available_actions,
                sequence=0,
                level=0,
                turn_in_level=0,
                levels_completed=0,
                state=before.state,
            )
        )
        solver.observe(
            ActionResultMessage(
                protocol_version=PROTOCOL_VERSION,
                type="action_result",
                sequence=0,
                level=0,
                turn_in_level=0,
                decision=decision,
                before=before,
                after=after,
            )
        )
        solver.finish(after, actions=1)
        worker = {
            "session": str(solver.session_root),
            "stdout": solver.stdout_log.read_text(),
            "stderr": solver.stderr_log.read_text(),
        }
    finally:
        solver.close()

    requests = FakeCompletionHandler.requests if fake else []
    request_facts = None
    if fake:
        request = requests[0] if len(requests) == 1 else {}
        request_facts = {
            "count": len(requests),
            "model": request.get("model"),
            "tool_names": [
                tool.get("function", {}).get("name")
                for tool in request.get("tools", [])
            ],
            "message_count": len(request.get("messages", [])),
            "max_tokens": request.get("max_tokens"),
            "tool_choice": request.get("tool_choice"),
        }
    return {
        "schema": SCHEMA,
        "endpoint_kind": "fake" if fake else "external",
        "decision": decision.model_dump(),
        "request_facts": request_facts,
        "worker_metadata": solver.worker_metadata,
        "worker": worker,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url")
    parser.add_argument("--model", default="s0-solver-join-model")
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    server = None
    thread = None
    try:
        if args.base_url:
            base_url = args.base_url
            fake = False
        else:
            server, thread, base_url = start_fake_server()
            fake = True
        result = run(
            base_url=base_url,
            model=args.model,
            workspace_root=args.workspace_root,
            fake=fake,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, sort_keys=True))
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=5)


if __name__ == "__main__":
    main()
