#!/usr/bin/env python3
"""Run the actual notebook worker/headless/CodeAct against barrier-controlled HTTP.

Source extraction is data-only; never execute notebook installation/launch cells.
An independent process watchdog bounds each trial. No model or GPU is used.
"""

from __future__ import annotations

import ast
import argparse
import asyncio
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import shutil
import sys
import tempfile
import time
import zipfile

from reproduce import NOOA, ROOT, digest

sys.path.insert(0, str(ROOT / "solvers/tycho_nooa/src"))
from tycho_nooa.protocol import project_arc_observation  # noqa: E402

HERE = Path(__file__).resolve().parent
NOTEBOOK = (
    ROOT
    / "deployment/kaggle/s0-nooa-join-public3-reasoning-notebook/s0_nooa_join_public3_reasoning.ipynb"
)
CURRENT = ROOT / "deployment/kaggle/s0-nooa-join-candidate"
LITELLM = Path("/home/pranav/.cache/uv/archive-v0/_HySxVjE4sG4S8gi")
PYTHON = ROOT / "deployment/kaggle/.venv/bin/python"


def extract(destination, notebook_path=NOTEBOOK):
    notebook = json.loads(notebook_path.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.walk(ast.parse("".join(cell["source"]))):
            if (
                not isinstance(node, ast.Constant)
                or not isinstance(node.value, str)
                or not node.value.startswith("UEsDB")
            ):
                continue
            with zipfile.ZipFile(io.BytesIO(base64.b64decode(node.value))) as archive:
                if "nooa_worker.py" not in archive.namelist():
                    continue
                for member in archive.namelist():
                    target = destination / member
                    assert target.resolve().is_relative_to(destination.resolve())
                archive.extractall(destination)
                return {
                    name: hashlib.sha256(archive.read(name)).hexdigest()
                    for name in archive.namelist()
                }
    raise RuntimeError("notebook has no worker archive")


def observation(value=0):
    return project_arc_observation(
        frames=[[value, 1], [0, 0]],
        available_action_ids=[1, 2, 3, 4],
        state="NOT_FINISHED",
        levels_completed=0,
        full_reset=False,
    ).model_dump()


class Trial:
    def __init__(self, path, source, name):
        self.path, self.source, self.name = path, source, name
        self.events = []
        self.admissions = []
        self.release = [asyncio.Event() for _ in range(8)]
        self.handlers = set()
        self.process = None
        self.server = None
        self.watchdog = None
        self.response_codes = {}

    def record(self, event, **fields):
        self.events.append(
            {"event": event, "monotonic_ns": time.monotonic_ns(), **fields}
        )

    def traces(self):
        result = []
        for path in self.path.glob("trace-*.jsonl"):
            for line in path.read_text().splitlines():
                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # A writer may be between write and reader visibility.
        return sorted(result, key=lambda e: e["monotonic_ns"])

    async def until(self, predicate, label, limit=25):
        deadline = time.monotonic() + limit
        while not predicate():
            if self.process and self.process.returncode is not None:
                raise RuntimeError(f"worker exited {self.process.returncode}: {label}")
            if time.monotonic() > deadline:
                raise TimeoutError(label)
            await asyncio.sleep(0.01)  # Observe barriers; does not choose race timing.

    async def endpoint(self, reader, writer):
        task = asyncio.current_task()
        self.handlers.add(task)
        try:
            header = await reader.readuntil(b"\r\n\r\n")
            headers = {
                k.lower(): v.strip()
                for k, v in (
                    line.split(":", 1)
                    for line in header.decode().split("\r\n")[1:]
                    if ":" in line
                )
            }
            payload = json.loads(
                await reader.readexactly(int(headers["content-length"]))
            )
            index = len(self.admissions)
            self.admissions.append(payload)
            self.record(
                "http_admitted",
                attempt=index,
                stream=payload.get("stream", False),
                prompt_hash=hashlib.sha256(
                    json.dumps(payload["messages"], sort_keys=True).encode()
                ).hexdigest(),
                tool_names=[t["function"]["name"] for t in payload.get("tools", [])],
            )
            await self.release[index].wait()
            code = self.response_codes.get(index, 'self.submit_actions(["UP"], rationale="predict: move upward")')
            response = {
                "id": f"probe-{self.name}-{index}",
                "object": "chat.completion",
                "created": 0,
                "model": "ownership-probe",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"call-{index}",
                                    "type": "function",
                                    "function": {
                                        "name": "execute_python",
                                        "arguments": json.dumps({"code": code}),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
            body = json.dumps(response).encode()
            self.record("response_released", attempt=index)
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: "
                + str(len(body)).encode()
                + b"\r\n\r\n"
                + body
            )
            try:
                await writer.drain()
                self.record("response_write_delivery_unknown", attempt=index)
            except ConnectionError:
                self.record("response_write_failed", attempt=index)
        except (asyncio.IncompleteReadError, ConnectionError):
            self.record("http_disconnected")
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass
            self.handlers.discard(task)

    async def send(self, kind, **fields):
        self.record("host_sent", kind=kind, sequence=fields.get("sequence"))
        self.process.stdin.write(
            (
                "ARC_HOST\t"
                + json.dumps({"protocol_version": 3, "type": kind, **fields})
                + "\n"
            ).encode()
        )
        await self.process.stdin.drain()

    async def receive(self, expected):
        while True:
            raw = await asyncio.wait_for(self.process.stdout.readline(), 30)
            if not raw:
                raise RuntimeError("worker stdout closed")
            if raw.startswith(b"ARC_AGENT\t"):
                obj = json.loads(raw.split(b"\t", 1)[1])
                self.record("host_received", payload=obj)
                if obj["type"] != expected:
                    raise RuntimeError(obj)
                return obj

    async def start(self):
        self.path.mkdir()
        self.server = await asyncio.start_server(self.endpoint, "127.0.0.1", 0)
        port = self.server.sockets[0].getsockname()[1]
        environment = {
            k: v
            for k, v in os.environ.items()
            if k in {"PATH", "HOME", "LANG", "LD_LIBRARY_PATH"}
        }
        environment.update(
            {
                "PYTHONPATH": os.pathsep.join(
                    [
                        str(HERE / "worker_observer"),
                        str(NOOA),
                        str(LITELLM),
                        str(ROOT / "solvers/tycho_nooa/src"),
                    ]
                ),
                "OWNERSHIP_TRACE_DIR": str(self.path),
                "PYTHONUNBUFFERED": "1",
                "LITELLM_LOCAL_MODEL_COST_MAP": "True",
                "DO_NOT_TRACK": "1",
                "ARC_LLM_MODEL": "ownership-probe",
                "ARC_LLM_API_KEY": "local-probe-only",
                "ARC_LLM_BASE_URL": f"http://127.0.0.1:{port}/v1",
                "ARC_LLM_USE_RESPONSES": "0",
                "NOOA_AGENT_READY_TIMEOUT_SECONDS": "30",
            }
        )
        stderr = (self.path / "worker-stderr.log").open("wb")
        try:
            self.process = await asyncio.create_subprocess_exec(
                str(PYTHON),
                str(self.source / "nooa_worker.py"),
                cwd=self.path,
                env=environment,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=stderr,
                start_new_session=True,
            )
        finally:
            stderr.close()
        # Independent OS process watchdog, targeting only this new process group.
        self.watchdog = await asyncio.create_subprocess_exec(
            str(PYTHON),
            "-c",
            "import os,signal,sys,time; time.sleep(65); os.killpg(int(sys.argv[1]),signal.SIGKILL)",
            str(self.process.pid),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await self.send(
            "start_game",
            game_id="ownership-probe",
            seed=0,
            workspace_root=str(self.path / "workspace"),
            initial_observation=observation(),
        )
        await self.receive("ready")

    async def turn(self, sequence=0, value=0):
        await self.send(
            "turn",
            sequence=sequence,
            level=0,
            turn_in_level=sequence,
            observation=observation(value),
            previous_decision=None,
            decision_timeout_seconds=40,
        )

    async def stop(self):
        if self.process and self.process.returncode is None:
            try:
                await self.send("close")
                await asyncio.wait_for(self.process.wait(), 6)
            except (TimeoutError, ConnectionError):
                os.killpg(self.process.pid, signal.SIGKILL)
                await self.process.wait()
        if self.watchdog and self.watchdog.returncode is None:
            self.watchdog.terminate()
            await self.watchdog.wait()
        for barrier in self.release:
            barrier.set()
        if self.handlers:
            await asyncio.gather(*list(self.handlers), return_exceptions=True)
        if self.server:
            self.server.close()
            await self.server.wait_closed()

    def save(self, outcome):
        (self.path / "result.json").write_text(
            json.dumps(
                {
                    "case": self.name,
                    "outcome": outcome,
                    "http_admissions": len(self.admissions),
                    "events": self.events,
                    "trace": self.traces(),
                },
                indent=2,
            )
            + "\n"
        )


async def repeated_polls(trial, candidate=False, defer_receive=False, successor_value=1):
    await trial.turn()
    await trial.until(lambda: len(trial.admissions) >= 1, "first HTTP admission")
    at_admission = time.monotonic_ns()
    await trial.until(
        lambda: len(
            [
                e
                for e in trial.traces()
                if e["event"] == "host_poll" and e["monotonic_ns"] > at_admission
            ]
        )
        >= 5,
        "five actual host polls after admission",
    )
    polls = [
        e
        for e in trial.traces()
        if e["event"] == "host_poll" and e["monotonic_ns"] > at_admission
    ]
    assert len({e["deadline"] for e in polls}) == 1
    assert len({e["operation_id"] for e in polls}) == 1
    assert len(trial.admissions) == 1
    assert not [e for e in trial.traces() if e["event"] == "handle_returned"]
    trial.release[0].set()
    if defer_receive:
        await trial.until(
            lambda: any(e["event"] == "handle_returned" for e in trial.traces()),
            "completion before host receive",
        )
        trial.record("host_receive_attached_after_completion")
    answer = await trial.receive("decision")
    assert answer["decision"]["action"] == "UP"
    await trial.until(
        lambda: any(e["event"] == "handle_returned" for e in trial.traces()),
        "WAIT returned after action",
    )
    assert len(trial.admissions) == 1
    actions = (trial.path / "ipc/actions.jsonl").read_text().splitlines()
    assert len(actions) == 1
    # Reconcile what the dispatcher does after WAIT, before any successor state.
    if candidate:
        await trial.until(
            lambda: any(
                e["event"] == "stale_state_notification_skipped" for e in trial.traces()
            ),
            "stale notification consumed without generation",
        )
        assert len(trial.admissions) == 1
        # Ensure the filter does not suppress the next legitimate decision.
        await trial.send(
            "action_result",
            sequence=0,
            level=0,
            turn_in_level=0,
            decision=answer["decision"],
            before=observation(),
            after=observation(successor_value),
        )
        await trial.receive("observed")
        await trial.turn(1, successor_value)
        await trial.until(
            lambda: len(trial.admissions) >= 2, "successor request admitted"
        )
        assert len(trial.admissions) == 2
        trial.release[1].set()
        second = await trial.receive("decision")
        assert second["sequence"] == 1 and second["decision"]["action"] == "UP"
        actions = [
            json.loads(line)
            for line in (trial.path / "ipc/actions.jsonl").read_text().splitlines()
        ]
        assert [a["turn"] for a in actions] == [0, 1]
        await trial.send(
            "finish",
            stop_reason="probe_complete",
            actions=2,
            levels_completed=0,
            final_observation=observation(successor_value),
        )
        await trial.receive("finished")
        return "pass: stale state skipped; each of two legitimate turns uses one request and commits once"
    await trial.until(lambda: len(trial.admissions) >= 2, "post-WAIT HTTP admission")
    assert len((trial.path / "ipc/states.jsonl").read_text().splitlines()) == 1
    trial.record("redundant_generation_after_committed_turn", state_lines=1, actions=1)
    return "reproduced: original result retained, but stale queued state admits a second request"


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-only", action="store_true")
    args = parser.parse_args()
    assert (
        digest(NOOA)
        == "f2217a8d95d2239ca830a0d47ec7973123f618253a6c219fa64e1e43f278cf0c"
    )
    output = Path(tempfile.mkdtemp(prefix="worker-trial-", dir=HERE))
    source = output / "frozen-source"
    source.mkdir()
    identity = extract(source)
    (output / "identity.json").write_text(
        json.dumps(
            {
                "notebook": str(NOTEBOOK),
                "notebook_sha256": digest(NOTEBOOK),
                "files": identity,
                "nooa_wheel_sha256": digest(NOOA),
            },
            indent=2,
        )
        + "\n"
    )
    candidate_source = output / "candidate-source"
    shutil.copytree(source, candidate_source)
    shutil.copyfile(HERE / "candidate/headless.py", candidate_source / "headless.py")
    (output / "candidate-identity.json").write_text(
        json.dumps(
            {
                "parent_files": identity,
                "only_override": {
                    "headless.py": digest(candidate_source / "headless.py")
                },
            },
            indent=2,
        )
        + "\n"
    )
    for name, selected, candidate, gap in [
        ("baseline-completion-gap", source, False, True),
        ("candidate-repeated-polls", candidate_source, True, False),
        ("candidate-completion-gap", candidate_source, True, True),
    ]:
        if args.candidate_only and not candidate:
            continue
        trial = Trial(output / name, selected, name)
        outcome = "unresolved"
        try:
            await trial.start()
            outcome = await repeated_polls(trial, candidate, gap)
        except Exception as exc:
            outcome = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            await trial.stop()
            trial.save(outcome)
            print(json.dumps({"path": str(trial.path), "outcome": outcome}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
