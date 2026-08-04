"""Serve a read-only browser replayer for LLM run evidence."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
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


def _read_jsonl(
    path: Path,
    *,
    allow_incomplete_tail: bool = False,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            is_incomplete_tail = line_number == len(lines) and not line.endswith(("\n", "\r"))
            if allow_incomplete_tail and is_incomplete_tail:
                break
            raise ValueError(f"{path}:{line_number}: invalid JSONL") from error
        if isinstance(value, dict):
            records.append(value)
    return records


def _run_names(runs_root: Path) -> list[str]:
    names: list[str] = []
    for path in runs_root.glob("*"):
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


def resolve_evidence_path(
    runs_root: Path, run_name: str, relative_name: str
) -> Path:
    run_dir = resolve_run_directory(runs_root, run_name)
    relative = Path(relative_name)
    allowed = (
        len(relative.parts) == 2
        and relative.parts[0] in {"request-payloads", "response-bodies"}
        and (
            (
                relative.parts[0] == "request-payloads"
                and re.fullmatch(r"\d{6}\.json", relative.parts[1])
            )
            or (
                relative.parts[0] == "response-bodies"
                and re.fullmatch(r"\d{6}\.raw", relative.parts[1])
            )
        )
    )
    if not allowed:
        raise PermissionError(relative_name)
    requested = (run_dir / relative).resolve()
    if not requested.is_relative_to(run_dir):
        raise PermissionError(relative_name)
    if not requested.is_file():
        raise FileNotFoundError(relative_name)
    return requested


def resolve_process_event_bytes(
    runs_root: Path, run_name: str, event_value: str
) -> bytes:
    if not re.fullmatch(r"(?:0|[1-9][0-9]*)", event_value):
        raise PermissionError("invalid process event index")
    wanted = int(event_value)
    process_path = resolve_run_directory(runs_root, run_name) / "process.jsonl"
    if not process_path.is_file():
        raise FileNotFoundError(process_path)
    with process_path.open("rb") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event_index") == wanted:
                return line
    raise FileNotFoundError(f"process event {wanted}")


def _image_url(run_name: str, relative_path: str) -> str:
    return (
        "/asset?run="
        + quote(run_name, safe="")
        + "&file="
        + quote(relative_path, safe="/")
    )


def _evidence_url(run_name: str, relative_path: str) -> str:
    return (
        "/evidence?run="
        + quote(run_name, safe="")
        + "&file="
        + quote(relative_path, safe="/")
    )


def _process_event_url(run_name: str, event_index: int) -> str:
    return f"/process-event?run={quote(run_name, safe='')}&event={event_index}"


def _observation_image(run_dir: Path, run_name: str, step: int) -> str | None:
    for root in ("workspace/observations", "live/observations"):
        relative = f"{root}/step-{step:05d}.png"
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
    if event_type == "human_feedback" and not isinstance(action_step, int):
        action_step = record.get("after_step")
    if not isinstance(action_step, int):
        raw_step = record.get("step")
        action_step = raw_step if isinstance(raw_step, int) else 0
    observation_step = (
        action_step
        if event_type in {"transition", "human_feedback", "prediction_result"}
        else max(0, action_step - 1)
    )

    event: dict[str, Any] = {
        "type": event_type,
        "event_index": record.get("event_index"),
        "timestamp": record.get("timestamp"),
        "action_step": action_step,
        "model_round": record.get("model_round"),
        "image": _observation_image(run_dir, run_name, observation_step),
    }
    if isinstance(event["event_index"], int):
        event["process_event"] = _process_event_url(
            run_name, event["event_index"]
        )

    if event_type == "model_response":
        response = record.get("response")
        response = response if isinstance(response, dict) else {}
        event_index = record.get("event_index")
        request_relative = (
            f"request-payloads/{event_index:06d}.json"
            if isinstance(event_index, int)
            else None
        )
        response_relative = (
            f"response-bodies/{event_index:06d}.raw"
            if isinstance(event_index, int)
            else None
        )
        event.update(
            reasoning=response.get("reasoning_content") or "",
            content=response.get("content") or "",
            tool_calls=_tool_calls(response),
            response_id=record.get("response_id"),
            usage=record.get("usage") or {},
            request_payload=(
                _evidence_url(run_name, request_relative)
                if request_relative and (run_dir / request_relative).is_file()
                else None
            ),
            response_body=(
                _evidence_url(run_name, response_relative)
                if response_relative and (run_dir / response_relative).is_file()
                else None
            ),
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
            commitment=(
                reasoning.get("commitment") if isinstance(reasoning, dict) else None
            ),
        )
    elif event_type == "prediction_result":
        event.update(
            outcome=record.get("outcome"),
            checks=record.get("checks") or [],
            objective_result=record.get("objective_result"),
            schema=record.get("schema"),
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
    elif event_type == "human_feedback":
        event.update(
            text=record.get("text") or "",
            after_step=record.get("after_step"),
            next_action=action_step + 1,
            channel=record.get("channel"),
        )
    elif event_type in {"model_budget_exhausted", "run_stopped", "run_failed"}:
        event.update(
            reason=(
                record.get("reason")
                or record.get("stop_reason")
                or record.get("error")
            ),
            message=record.get("message"),
            usage=record.get("usage") or {},
        )

    raw_without_messages = dict(record)
    raw_without_messages.pop("request_messages", None)
    event["raw"] = raw_without_messages
    return event


def _steering_audit(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    terminal_types = {"run_failed", "run_stopped", "model_budget_exhausted"}
    audits: list[dict[str, Any]] = []
    for feedback in (event for event in events if event["type"] == "human_feedback"):
        source_step = feedback["action_step"]
        next_action = feedback["next_action"]
        feedback_index = feedback.get("event_index")
        if not isinstance(feedback_index, int):
            continue
        before = [
            event
            for event in events
            if event["type"] != "human_feedback"
            and event["action_step"] == source_step
            and isinstance(event.get("event_index"), int)
            and event["event_index"] < feedback_index
        ]
        after = [
            event
            for event in events
            if event["type"] != "human_feedback"
            and event["action_step"] == next_action
            and isinstance(event.get("event_index"), int)
            and event["event_index"] > feedback_index
        ]
        decision = next(
            (event for event in reversed(after) if event["type"] == "decision"),
            None,
        )
        transition = next(
            (event for event in reversed(after) if event["type"] == "transition"),
            None,
        )
        terminal = next(
            (event for event in reversed(after) if event["type"] in terminal_types),
            None,
        )
        audits.append(
            {
                "feedback_event_index": feedback_index,
                "source_step": source_step,
                "next_action": next_action,
                "before_event_indices": [event["event_index"] for event in before],
                "after_event_indices": [event["event_index"] for event in after],
                "model_rounds": sum(
                    event["type"] == "model_response" for event in after
                ),
                "python_probes": sum(event["type"] == "python_tool" for event in after),
                "decision_event_index": decision.get("event_index") if decision else None,
                "transition_event_index": (
                    transition.get("event_index") if transition else None
                ),
                "terminal_event_index": terminal.get("event_index") if terminal else None,
            }
        )
    return audits


def load_run(runs_root: Path, run_name: str) -> dict[str, Any]:
    run_dir = resolve_run_directory(runs_root, run_name)

    manifest = _read_json(run_dir / "manifest.json")
    scorecard = _read_json(run_dir / "scorecard.json")
    records = _read_jsonl(
        run_dir / "process.jsonl",
        allow_incomplete_tail=manifest.get("status") == "running",
    )
    events = [
        _normalize_event(record, run_dir=run_dir, run_name=run_name)
        for record in records
    ]
    last_attempt_step = max(
        (
            event["action_step"]
            for event in events
            if event["type"]
            not in {"model_budget_exhausted", "run_stopped", "run_failed"}
        ),
        default=0,
    )
    for event in events:
        raw = event["raw"]
        if (
            event["type"] in {"model_budget_exhausted", "run_stopped", "run_failed"}
            and "action_step" not in raw
            and "step" not in raw
        ):
            event["action_step"] = last_attempt_step
    transitions = [event for event in events if event["type"] == "transition"]
    model_responses = [event for event in events if event["type"] == "model_response"]
    python_calls = [event for event in events if event["type"] == "python_tool"]
    terminal_event = next(
        (
            event
            for event in reversed(events)
            if event["type"] in {"run_failed", "run_stopped", "model_budget_exhausted"}
        ),
        None,
    )
    effective_status = manifest.get("status", "unknown")
    if effective_status == "running" and terminal_event is not None:
        effective_status = {
            "run_failed": "failed",
            "run_stopped": "stopped",
            "model_budget_exhausted": "budget_exhausted",
        }[terminal_event["type"]]
    stop_reason = manifest.get("stop_reason")
    if stop_reason is None and terminal_event is not None:
        terminal_raw = terminal_event["raw"]
        stop_reason = (
            terminal_raw.get("error")
            or terminal_raw.get("stop_reason")
            or terminal_raw.get("reason")
        )
    total_levels = scorecard.get("total_levels_completed")
    if not isinstance(total_levels, int):
        total_levels = manifest.get("levels_completed")
    environment = manifest.get("environment")
    if not isinstance(environment, dict):
        environment = {}

    return {
        "run_name": run_name,
        "manifest": manifest,
        "summary": {
            "status": effective_status,
            "stop_reason": stop_reason,
            "model": manifest.get("model") or environment.get("model", "unknown"),
            "prompt": manifest.get("policy_prompt_variant", "neutral"),
            "game": manifest.get("game") or environment.get("game", "unknown"),
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
        "steering_audit": _steering_audit(events),
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
        if parsed.path == "/evidence":
            self._serve_evidence(query)
            return
        if parsed.path == "/process-event":
            self._serve_process_event(query)
            return
        if parsed.path == "/api/run":
            self._serve_run_data(query)
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

    def _serve_run_data(self, query: dict[str, list[str]]) -> None:
        run_name = query.get("run", [""])[0]
        if run_name not in _run_names(self.server.runs_root):
            self.send_error(HTTPStatus.NOT_FOUND, "unknown run")
            return
        try:
            data = load_run(self.server.runs_root, run_name)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))
            return
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
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

    def _serve_evidence(self, query: dict[str, list[str]]) -> None:
        run_name = query.get("run", [""])[0]
        relative_name = query.get("file", [""])[0]
        if run_name not in _run_names(self.server.runs_root):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            requested = resolve_evidence_path(
                self.server.runs_root, run_name, relative_name
            )
        except PermissionError:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        except (FileNotFoundError, ValueError):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = requested.read_bytes()
        content_type = (
            "application/json; charset=utf-8"
            if requested.suffix == ".json"
            else "text/plain; charset=utf-8"
        )
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_process_event(self, query: dict[str, list[str]]) -> None:
        run_name = query.get("run", [""])[0]
        event_value = query.get("event", [""])[0]
        if run_name not in _run_names(self.server.runs_root):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            body = resolve_process_event_bytes(
                self.server.runs_root, run_name, event_value
            )
        except PermissionError:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        except (FileNotFoundError, ValueError):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
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
