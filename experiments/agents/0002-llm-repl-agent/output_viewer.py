"""Serve a read-only browser replayer for LLM run evidence."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse


AGENT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS_ROOT = AGENT_ROOT / "artifacts" / "runs"
PAGE_TEMPLATE = (AGENT_ROOT / "output_viewer.html").read_text(encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from error
            if isinstance(value, dict):
                records.append(value)
    return records


def _run_names(runs_root: Path) -> list[str]:
    names: list[str] = []
    for path in runs_root.glob("run-*"):
        if not (path / "process.jsonl").is_file():
            continue
        try:
            resolve_run_directory(runs_root, path.name)
        except ValueError:
            continue
        names.append(path.name)
    return sorted(names, reverse=True)


def resolve_run_directory(runs_root: Path, run_name: str) -> Path:
    resolved_root = runs_root.resolve()
    run_dir = (resolved_root / run_name).resolve()
    if run_dir.parent != resolved_root or not run_dir.is_dir():
        raise ValueError(f"unknown run: {run_name}")
    return run_dir


def resolve_asset_path(
    runs_root: Path, run_name: str, relative_name: str
) -> Path:
    run_dir = resolve_run_directory(runs_root, run_name)
    requested = (run_dir / relative_name).resolve()
    if not requested.is_relative_to(run_dir) or requested.suffix.lower() != ".png":
        raise PermissionError(relative_name)
    if not requested.is_file():
        raise FileNotFoundError(relative_name)
    return requested


def _image_url(run_name: str, relative_path: str) -> str:
    return (
        "/asset?run="
        + quote(run_name, safe="")
        + "&file="
        + quote(relative_path, safe="/")
    )


def _observation_image(run_dir: Path, run_name: str, step: int) -> str | None:
    relative = f"workspace/observations/step-{step:05d}.png"
    if (run_dir / relative).is_file():
        return _image_url(run_name, relative)
    return None


def _tool_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for raw in response.get("tool_calls", []):
        if not isinstance(raw, dict):
            continue
        function = raw.get("function")
        if not isinstance(function, dict):
            continue
        arguments = function.get("arguments", "")
        parsed_arguments: Any = arguments
        if isinstance(arguments, str):
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass
        calls.append(
            {
                "name": function.get("name", "tool"),
                "arguments": parsed_arguments,
                "id": raw.get("id"),
            }
        )
    return calls


def _python_result(raw_output: Any) -> dict[str, Any]:
    if not isinstance(raw_output, str):
        return {"display": raw_output}
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        return {"display": raw_output}
    if not isinstance(parsed, dict):
        return {"display": parsed}
    channels = [
        {"label": label, "text": parsed.get(key)}
        for key, label in (
            ("stdout", "Stdout"),
            ("stderr", "Stderr"),
            ("traceback", "Traceback"),
        )
        if parsed.get(key)
    ]
    return {
        "ok": parsed.get("ok"),
        "stdout": parsed.get("stdout"),
        "stderr": parsed.get("stderr"),
        "traceback": parsed.get("traceback"),
        "timed_out": parsed.get("timed_out"),
        "truncated": parsed.get("truncated"),
        "original_chars": parsed.get("original_chars"),
        "limit": parsed.get("limit"),
        "channels": channels,
    }


def _normalize_event(
    record: dict[str, Any], *, run_dir: Path, run_name: str
) -> dict[str, Any]:
    event_type = str(record.get("type", "unknown"))
    action_step = record.get("action_step")
    if not isinstance(action_step, int):
        raw_step = record.get("step")
        action_step = raw_step if isinstance(raw_step, int) else 0
    observation_step = action_step if event_type == "transition" else max(0, action_step - 1)

    event: dict[str, Any] = {
        "type": event_type,
        "event_index": record.get("event_index"),
        "timestamp": record.get("timestamp"),
        "action_step": action_step,
        "model_round": record.get("model_round"),
        "image": _observation_image(run_dir, run_name, observation_step),
    }

    if event_type == "model_response":
        response = record.get("response")
        response = response if isinstance(response, dict) else {}
        event.update(
            reasoning=response.get("reasoning_content") or "",
            content=response.get("content") or "",
            tool_calls=_tool_calls(response),
            response_id=record.get("response_id"),
            usage=record.get("usage") or {},
        )
    elif event_type == "python_tool":
        event.update(
            code=record.get("code") or "",
            result=_python_result(record.get("output")),
            output_chars=record.get("output_chars"),
            output_truncated=record.get("output_truncated"),
            phase=record.get("phase"),
        )
    elif event_type == "decision":
        reasoning = record.get("reasoning")
        event.update(
            action=record.get("action"),
            data=record.get("data"),
            reason=reasoning.get("reason") if isinstance(reasoning, dict) else None,
        )
    elif event_type == "transition":
        event.update(
            action=record.get("action"),
            action_data=record.get("action_data"),
            changed_pixels=record.get("changed_pixels"),
            levels_before=record.get("levels_before"),
            levels_after=record.get("levels_after"),
            state_before=record.get("state_before"),
            state_after=record.get("state_after"),
            before_image=_observation_image(
                run_dir, run_name, max(0, action_step - 1)
            ),
            after_image=_observation_image(run_dir, run_name, action_step),
        )
    elif event_type == "workspace_change":
        event["changes"] = record.get("changes") or []
    elif event_type == "observation":
        event["metadata"] = record.get("metadata") or {}
        event["image"] = _observation_image(run_dir, run_name, 0)
    elif event_type in {"model_budget_exhausted", "run_stopped"}:
        event.update(
            reason=record.get("reason") or record.get("stop_reason"),
            message=record.get("message"),
            usage=record.get("usage") or {},
        )

    raw_without_messages = dict(record)
    raw_without_messages.pop("request_messages", None)
    event["raw"] = raw_without_messages
    return event


def load_run(runs_root: Path, run_name: str) -> dict[str, Any]:
    run_dir = resolve_run_directory(runs_root, run_name)

    manifest = _read_json(run_dir / "manifest.json")
    scorecard = _read_json(run_dir / "scorecard.json")
    records = _read_jsonl(run_dir / "process.jsonl")
    events = [
        _normalize_event(record, run_dir=run_dir, run_name=run_name)
        for record in records
    ]
    last_attempt_step = max(
        (
            event["action_step"]
            for event in events
            if event["type"] not in {"model_budget_exhausted", "run_stopped"}
        ),
        default=0,
    )
    for event in events:
        raw = event["raw"]
        if (
            event["type"] in {"model_budget_exhausted", "run_stopped"}
            and "action_step" not in raw
            and "step" not in raw
        ):
            event["action_step"] = last_attempt_step
    transitions = [event for event in events if event["type"] == "transition"]
    model_responses = [event for event in events if event["type"] == "model_response"]
    python_calls = [event for event in events if event["type"] == "python_tool"]
    total_levels = scorecard.get("total_levels_completed")
    if not isinstance(total_levels, int):
        total_levels = manifest.get("levels_completed")

    return {
        "run_name": run_name,
        "manifest": manifest,
        "summary": {
            "status": manifest.get("status", "unknown"),
            "stop_reason": manifest.get("stop_reason"),
            "model": manifest.get("model", "unknown"),
            "prompt": manifest.get("policy_prompt_variant", "neutral"),
            "game": manifest.get("game", "unknown"),
            "actions": manifest.get("actions", len(transitions)),
            "levels_completed": total_levels,
            "model_calls": len(model_responses),
            "python_calls": len(python_calls),
            "changed_pixels": sum(
                event.get("changed_pixels") or 0 for event in transitions
            ),
            "usage": manifest.get("usage") or {},
            "started_at": manifest.get("started_at"),
            "finished_at": manifest.get("finished_at"),
            "error": manifest.get("error"),
        },
        "events": events,
        "source": str(run_dir / "process.jsonl"),
    }


def render_page(*, data: dict[str, Any], run_names: list[str]) -> bytes:
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    run_options = json.dumps(run_names).replace("</", "<\\/")
    html = PAGE_TEMPLATE.replace("__RUN_DATA__", payload)
    html = html.replace("__RUN_OPTIONS__", run_options)
    return html.encode("utf-8")


class ViewerServer(ThreadingHTTPServer):
    runs_root: Path


class Handler(BaseHTTPRequestHandler):
    server: ViewerServer

    def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/asset":
            self._serve_asset(query)
            return
        if parsed.path not in {"/", "/llm-output"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        run_names = _run_names(self.server.runs_root)
        if not run_names:
            self.send_error(HTTPStatus.NOT_FOUND, "no process runs found")
            return
        requested_run = query.get("run", [run_names[0]])[0]
        if requested_run not in run_names:
            self.send_error(HTTPStatus.NOT_FOUND, "unknown run")
            return
        try:
            data = load_run(self.server.runs_root, requested_run)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))
            return
        body = render_page(data=data, run_names=run_names)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_asset(self, query: dict[str, list[str]]) -> None:
        run_name = query.get("run", [""])[0]
        relative_name = query.get("file", [""])[0]
        if run_name not in _run_names(self.server.runs_root):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            requested = resolve_asset_path(
                self.server.runs_root, run_name, relative_name
            )
        except PermissionError:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        except (FileNotFoundError, ValueError):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = requested.read_bytes()
        content_type = mimetypes.guess_type(requested.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        print(f"viewer: {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve a read-only replayer for LLM run evidence."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    args = parser.parse_args()

    runs_root = args.runs_root.resolve()
    run_names = _run_names(runs_root)
    if not run_names:
        parser.error(f"no process runs found under {runs_root}")

    server = ViewerServer((args.host, args.port), Handler)
    server.runs_root = runs_root
    print(f"LLM run replayer: http://{args.host}:{args.port}/")
    print(f"Runs: {len(run_names)} (latest: {run_names[0]})")
    print("Viewer is read-only. Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
