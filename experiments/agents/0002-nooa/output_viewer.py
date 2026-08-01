"""Serve a read-only browser viewer for paired 0002-NOOA evidence."""

from __future__ import annotations

import argparse
from collections import defaultdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, quote, urlparse


AGENT_ROOT = Path(__file__).resolve().parent
DEFAULT_PAIRS_ROOT = AGENT_ROOT / "artifacts" / "paired-usability-probes"
PAGE_TEMPLATE = (AGENT_ROOT / "output_viewer.html").read_text(encoding="utf-8")
ARM_NAMES = ("candidate", "control")
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
PAIR_DIRECT_ARTIFACTS = frozenset(
    {"paired-run-spec.json", "summary.json", "provider-profile-report.json"}
)
DIRECT_ARTIFACTS = frozenset(
    {
        "trace.jsonl",
        "summary.json",
        "scorecard.json",
        "provider-crossings.json",
        "process.jsonl",
        "arm-manifest.json",
    }
)
ARTIFACT_PREFIXES = (
    "worker/host-observations/",
    "worker/ipc/",
    "worker/workspace/observations/",
    "workspace/observations/",
    "request-payloads/",
    "response-bodies/",
    "raw-recordings/",
)
ARTIFACT_SUFFIXES = frozenset({".json", ".jsonl", ".png", ".raw"})


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            incomplete_tail = line_number == len(lines) and not line.endswith(("\n", "\r"))
            if incomplete_tail:
                break
            raise ValueError(f"{path}:{line_number}: invalid JSONL") from error
        if isinstance(value, dict):
            records.append(value)
    return records


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def pair_names(pairs_root: Path) -> list[str]:
    """Return completed or in-progress paired runs in newest-first name order."""

    if not pairs_root.is_dir():
        return []
    names = [
        path.name
        for path in pairs_root.iterdir()
        if path.is_dir()
        and (path / "paired-run-spec.json").is_file()
        and any((path / arm).is_dir() for arm in ARM_NAMES)
    ]
    return sorted(names, reverse=True)


def resolve_pair_directory(pairs_root: Path, pair_name: str) -> Path:
    root = pairs_root.resolve()
    pair = (root / pair_name).resolve()
    if pair.parent != root or not pair.is_dir():
        raise ValueError(f"unknown pair: {pair_name}")
    return pair


def arm_names(pair_dir: Path) -> list[str]:
    return [arm for arm in ARM_NAMES if (pair_dir / arm).is_dir()]


def resolve_arm_directory(pairs_root: Path, pair_name: str, arm: str) -> Path:
    if arm not in ARM_NAMES:
        raise ValueError(f"unknown arm: {arm}")
    pair = resolve_pair_directory(pairs_root, pair_name)
    arm_dir = (pair / arm).resolve()
    if arm_dir.parent != pair or not arm_dir.is_dir():
        raise ValueError(f"unknown arm: {arm}")
    return arm_dir


def resolve_pair_artifact_path(pairs_root: Path, pair_name: str, name: str) -> Path:
    """Resolve one frozen pair-level artifact without exposing other run files."""

    if name not in PAIR_DIRECT_ARTIFACTS:
        raise PermissionError(name)
    pair_dir = resolve_pair_directory(pairs_root, pair_name)
    path = (pair_dir / name).resolve()
    if path.parent != pair_dir or not path.is_file():
        raise FileNotFoundError(name)
    return path


def resolve_artifact_path(
    pairs_root: Path, pair_name: str, arm: str, relative_name: str
) -> Path:
    """Resolve one evidence artifact without exposing arbitrary run-local files."""

    relative = Path(relative_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise PermissionError(relative_name)
    normalized = relative.as_posix()
    is_direct = normalized in DIRECT_ARTIFACTS
    has_allowed_prefix = normalized.startswith(ARTIFACT_PREFIXES)
    if not (is_direct or has_allowed_prefix) or relative.suffix not in ARTIFACT_SUFFIXES:
        raise PermissionError(relative_name)
    arm_dir = resolve_arm_directory(pairs_root, pair_name, arm)
    requested = (arm_dir / relative).resolve()
    if not requested.is_relative_to(arm_dir) or not requested.is_file():
        raise FileNotFoundError(relative_name)
    return requested


def _asset_url(pair_name: str, arm: str, relative_name: str) -> str:
    return (
        "/asset?pair="
        + quote(pair_name, safe="")
        + "&arm="
        + quote(arm, safe="")
        + "&file="
        + quote(relative_name, safe="/")
    )


def _artifact_url(pair_name: str, arm: str, relative_name: str) -> str:
    return (
        "/artifact?pair="
        + quote(pair_name, safe="")
        + "&arm="
        + quote(arm, safe="")
        + "&file="
        + quote(relative_name, safe="/")
    )


def _pair_artifact_url(pair_name: str, name: str) -> str:
    return "/pair-artifact?pair=" + quote(pair_name, safe="") + "&file=" + quote(name, safe="")


def _existing_arm_evidence_link(
    arm_dir: Path, pair_name: str, arm: str, label: str, relative_name: str
) -> dict[str, str] | None:
    if (arm_dir / relative_name).is_file():
        return {"label": label, "href": _artifact_url(pair_name, arm, relative_name)}
    return None


def _append_evidence_link(event: dict[str, Any], link: dict[str, str] | None) -> None:
    if link is not None:
        event.setdefault("evidence_links", []).append(link)


def _loopback_host(value: str) -> str:
    if value not in LOOPBACK_HOSTS:
        raise argparse.ArgumentTypeError("the evidence viewer may bind only to loopback")
    return value


def _observation_image(
    arm_dir: Path, pair_name: str, arm: str, step: int
) -> str | None:
    roots = (
        "worker/workspace/observations" if arm == "candidate" else "workspace/observations",
    )
    for root in roots:
        relative = f"{root}/step-{step:05d}.png"
        if (arm_dir / relative).is_file():
            return _asset_url(pair_name, arm, relative)
    return None


def _trace_frames(arm_dir: Path) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for record in _read_jsonl(arm_dir / "trace.jsonl"):
        kind = record.get("type")
        if kind == "observation":
            payload = record
        elif kind == "transition":
            payload = _as_dict(record.get("observation"))
        else:
            continue
        step = record.get("step")
        if not isinstance(step, int):
            continue
        frames.append(
            {
                "step": step,
                "action": record.get("action"),
                "action_data": record.get("action_data"),
                "state": payload.get("state"),
                "levels_completed": payload.get("levels_completed"),
                "available_actions": payload.get("available_actions") or [],
                "full_reset": payload.get("full_reset"),
            }
        )
    return sorted(frames, key=lambda frame: frame["step"])


def _text_parts(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        candidates = (value.get("text"), value.get("content"), value.get("value"))
        return [item for item in candidates if isinstance(item, str)]
    if isinstance(value, list):
        return [part for item in value for part in _text_parts(item)]
    return []


def _unique_text(parts: Iterable[str]) -> list[str]:
    unique: list[str] = []
    for part in parts:
        if part and part not in unique:
            unique.append(part)
    return unique


def _provider_output(call: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    reasoning: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    outcome = _as_dict(call.get("outcome"))
    response = _as_dict(outcome.get("response"))
    reasoning.extend(_text_parts(response.get("content")))
    reasoning.extend(_text_parts(response.get("assistant_message")))
    for tool_call in _as_list(response.get("tool_calls")):
        if isinstance(tool_call, dict):
            tool_calls.append(tool_call)
    for crossing in _as_list(call.get("actual_provider_crossings")):
        body = _as_dict(_as_dict(crossing).get("response")).get("body")
        for output in _as_list(_as_dict(body).get("output")):
            output_dict = _as_dict(output)
            output_type = output_dict.get("type")
            if output_type in {"reasoning", "message"}:
                reasoning.extend(_text_parts(output_dict.get("content")))
            elif output_type == "function_call":
                tool_calls.append(
                    {
                        "name": output_dict.get("name"),
                        "arguments": output_dict.get("arguments"),
                        "call_id": output_dict.get("call_id"),
                    }
                )
    return _unique_text(reasoning), tool_calls


def _model_visible_messages(call: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    messages: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    for message in _as_list(call.get("messages")):
        message_dict = _as_dict(message)
        role = message_dict.get("role", "unknown")
        content = message_dict.get("content")
        projection = {"role": role, "content": content}
        messages.append(projection)
        if role in {"tool", "function"}:
            tool_results.append(projection)
    return messages, tool_results


def _project_nooa_model_call(call: dict[str, Any]) -> dict[str, Any]:
    reasoning, tool_calls = _provider_output(call)
    messages, tool_results = _model_visible_messages(call)
    outcome = _as_dict(call.get("outcome"))
    return {
        "outcome": outcome.get("kind"),
        "reasoning": reasoning,
        "content": "\n\n".join(reasoning),
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "messages": messages,
        "request_sha256": call.get("canonical_request_sha256"),
        "actual_crossings": len(_as_list(call.get("actual_provider_crossings"))),
    }


def _project_control_model_call(record: dict[str, Any]) -> dict[str, Any]:
    """Project a recorded 0002-R provider response through the shared call schema."""

    projected = _project_nooa_model_call(
        {
            "outcome": {"kind": "recorded", "response": record.get("response")},
            "actual_provider_crossings": [
                {"response": {"body": record.get("response_body")}}
            ],
            "messages": record.get("request_messages"),
            "canonical_request_sha256": record.get("request_payload_sha256"),
        }
    )
    projected["model_round"] = record.get("model_round")
    return projected


def _candidate_events(
    arm_dir: Path, pair_name: str, arm: str
) -> dict[int, dict[str, Any]]:
    events: dict[int, dict[str, Any]] = defaultdict(dict)
    for record in _read_jsonl(arm_dir / "worker/process.jsonl"):
        action_point = record.get("action_point_id")
        if not isinstance(action_point, int):
            continue
        event = events[action_point]
        kind = record.get("kind")
        if kind == "action_request":
            request = _as_dict(record.get("request"))
            host_observation = _as_dict(record.get("host_observation"))
            event["input"] = {
                "legal_actions": request.get("legal_actions") or [],
                "coordinate_actions": request.get("coordinate_actions") or [],
                "host_observation": host_observation,
                "request_sha256": record.get("request_sha256"),
            }
            host_path = host_observation.get("path")
            if isinstance(host_path, str):
                _append_evidence_link(
                    event,
                    _existing_arm_evidence_link(
                        arm_dir,
                        pair_name,
                        arm,
                        "Exact host observation",
                        f"worker/{host_path}",
                    ),
                )
            _append_evidence_link(
                event,
                _existing_arm_evidence_link(
                    arm_dir, pair_name, arm, "IPC requests", "worker/ipc/requests.jsonl"
                ),
            )
            _append_evidence_link(
                event,
                _existing_arm_evidence_link(
                    arm_dir, pair_name, arm, "IPC responses", "worker/ipc/responses.jsonl"
                ),
            )
        elif kind == "model_crossing":
            audit = _as_dict(record.get("audit"))
            event["model_calls"] = [
                _project_nooa_model_call(_as_dict(call))
                for call in _as_list(audit.get("model_calls"))
            ]
            event["worker_decision"] = record.get("worker_decision")
        elif kind == "decision_validated":
            event["decision"] = record.get("decision")
            event["decision_label"] = "Host validation"
            event["workspace_commitment"] = record.get("workspace_commitment")
        elif kind == "transition_recorded":
            event["transition"] = record.get("transition")
        elif kind == "host_execution_failure":
            event["execution_failure"] = record.get("error")
    return dict(events)


def _control_events(
    arm_dir: Path, pair_name: str, arm: str
) -> dict[int, dict[str, Any]]:
    """Align 0002-R's one-based action steps to their zero-based input frames."""

    events: dict[int, dict[str, Any]] = defaultdict(dict)
    for record in _read_jsonl(arm_dir / "process.jsonl"):
        kind = record.get("type")
        if kind == "transition":
            successor_step = record.get("step")
            if not isinstance(successor_step, int) or successor_step < 1:
                continue
            action_point = successor_step - 1
        else:
            action_step = record.get("action_step")
            if not isinstance(action_step, int):
                continue
            action_point = max(0, action_step - 1)
        event = events[action_point]
        if kind == "observation":
            event["input"] = {
                "public_observation": record.get("metadata") or {},
                "representations": record.get("representations") or {},
            }
        elif kind == "model_response":
            model_call = _project_control_model_call(record)
            links: list[dict[str, str]] = []
            request_archive = record.get("request_payload_archive")
            if isinstance(request_archive, str):
                link = _existing_arm_evidence_link(
                    arm_dir, pair_name, arm, "Exact request payload", request_archive
                )
                if link is not None:
                    links.append(link)
            response_archive = record.get("response_body_archive")
            if isinstance(response_archive, str):
                link = _existing_arm_evidence_link(
                    arm_dir, pair_name, arm, "Exact response body", response_archive
                )
                if link is not None:
                    links.append(link)
            model_call["evidence_links"] = links
            event.setdefault("model_calls", []).append(model_call)
            for link in links:
                _append_evidence_link(event, link)
        elif kind == "python_tool":
            event.setdefault("python_calls", []).append(
                {
                    "code": record.get("code"),
                    "output": record.get("output"),
                    "output_truncated": record.get("output_truncated"),
                    "tool_call_id": record.get("tool_call_id"),
                }
            )
        elif kind == "workspace_change":
            event.setdefault("workspace_changes", []).append(
                {
                    "changes": record.get("changes") or [],
                    "tool_call_id": record.get("tool_call_id"),
                }
            )
        if kind == "decision":
            event["decision"] = {
                "action": record.get("action"),
                "data": record.get("data"),
                "reasoning": record.get("reasoning"),
            }
            event["decision_label"] = "Recorded final action"
        elif kind == "transition":
            event["transition"] = record
    return dict(events)


def _run_summary(arm_dir: Path, arm: str) -> dict[str, Any]:
    summary = _as_dict(_read_json(arm_dir / "summary.json"))
    scorecard = _as_dict(_read_json(arm_dir / "scorecard.json"))
    return {
        "arm": summary.get("arm", "0002-NOOA" if arm == "candidate" else "0002-R"),
        "actions": summary.get("actions", scorecard.get("total_actions", 0)),
        "levels_completed": summary.get(
            "levels_completed", scorecard.get("total_levels_completed", 0)
        ),
        "state": summary.get("state"),
        "stop_reason": summary.get("stop_reason"),
        "trace_complete": summary.get("trace_complete"),
        "clean_shutdown": summary.get("clean_shutdown"),
        "capability_probe": summary.get("capability_probe"),
    }


def load_arm(pairs_root: Path, pair_name: str, arm: str) -> dict[str, Any]:
    """Project one arm into replay stages without changing its evidence."""

    pair_dir = resolve_pair_directory(pairs_root, pair_name)
    arm_dir = resolve_arm_directory(pairs_root, pair_name, arm)
    frames = _trace_frames(arm_dir)
    event_by_step = (
        _candidate_events(arm_dir, pair_name, arm)
        if arm == "candidate"
        else _control_events(arm_dir, pair_name, arm)
    )
    stages: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        step = frame["step"]
        next_step = frames[index + 1]["step"] if index + 1 < len(frames) else None
        stage = {
            "index": index,
            "step": step,
            "action_point_id": step,
            "frame": frame,
            "before_image": _observation_image(arm_dir, pair_name, arm, step),
            "after_image": (
                _observation_image(arm_dir, pair_name, arm, next_step)
                if next_step is not None
                else None
            ),
            "process_evidence": _artifact_url(
                pair_name,
                arm,
                "worker/process.jsonl" if arm == "candidate" else "process.jsonl",
            ),
            "provider_evidence": _artifact_url(pair_name, arm, "provider-crossings.json"),
            **event_by_step.get(step, {}),
        }
        stages.append(stage)
    return {
        "pair_name": pair_name,
        "pair_summary": _as_dict(_read_json(pair_dir / "summary.json")),
        "paired_run_spec": _as_dict(_read_json(pair_dir / "paired-run-spec.json")),
        "provider_profile": _as_dict(_read_json(pair_dir / "provider-profile-report.json")),
        "pair_spec_evidence": _pair_artifact_url(pair_name, "paired-run-spec.json"),
        "arm": "0002-NOOA" if arm == "candidate" else "0002-R",
        "arm_key": arm,
        "arms": arm_names(pair_dir),
        "summary": _run_summary(arm_dir, arm),
        "steps": stages,
    }


def render_page(*, data: dict[str, Any], pair_names: list[str]) -> bytes:
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    options = json.dumps(pair_names, ensure_ascii=False).replace("</", "<\\/")
    return (
        PAGE_TEMPLATE.replace("__INITIAL_DATA__", payload)
        .replace("__PAIR_OPTIONS__", options)
        .encode("utf-8")
    )


class ViewerServer(ThreadingHTTPServer):
    pairs_root: Path


class Handler(BaseHTTPRequestHandler):
    server: ViewerServer

    def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/api/arm":
            self._serve_arm(query)
            return
        if parsed.path in {"/asset", "/artifact"}:
            self._serve_artifact(query)
            return
        if parsed.path == "/pair-artifact":
            self._serve_pair_artifact(query)
            return
        if parsed.path != "/":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        names = pair_names(self.server.pairs_root)
        if not names:
            self.send_error(HTTPStatus.NOT_FOUND, "no paired 0002 runs found")
            return
        pair_name = query.get("pair", [names[0]])[0]
        try:
            pair_dir = resolve_pair_directory(self.server.pairs_root, pair_name)
            available_arms = arm_names(pair_dir)
            arm = query.get("arm", ["candidate"])[0]
            if arm not in available_arms:
                arm = available_arms[0]
            data = load_arm(self.server.pairs_root, pair_name, arm)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self.send_error(HTTPStatus.NOT_FOUND, str(error))
            return
        self._send(HTTPStatus.OK, render_page(data=data, pair_names=names), "text/html; charset=utf-8")

    def _serve_arm(self, query: dict[str, list[str]]) -> None:
        pair_name = query.get("pair", [""])[0]
        arm = query.get("arm", [""])[0]
        try:
            data = load_arm(self.server.pairs_root, pair_name, arm)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self.send_error(HTTPStatus.NOT_FOUND, str(error))
            return
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._send(HTTPStatus.OK, body, "application/json; charset=utf-8")

    def _serve_artifact(self, query: dict[str, list[str]]) -> None:
        pair_name = query.get("pair", [""])[0]
        arm = query.get("arm", [""])[0]
        relative_name = query.get("file", [""])[0]
        try:
            path = resolve_artifact_path(
                self.server.pairs_root, pair_name, arm, relative_name
            )
        except PermissionError:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        except (FileNotFoundError, ValueError):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send(HTTPStatus.OK, path.read_bytes(), content_type)

    def _serve_pair_artifact(self, query: dict[str, list[str]]) -> None:
        pair_name = query.get("pair", [""])[0]
        name = query.get("file", [""])[0]
        try:
            path = resolve_pair_artifact_path(self.server.pairs_root, pair_name, name)
        except PermissionError:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        except (FileNotFoundError, ValueError):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._send(HTTPStatus.OK, path.read_bytes(), "application/json; charset=utf-8")

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        print(f"nooa-viewer: {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve a read-only browser viewer for paired 0002-NOOA evidence."
    )
    parser.add_argument("--host", default="127.0.0.1", type=_loopback_host)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--pairs-root", type=Path, default=DEFAULT_PAIRS_ROOT)
    args = parser.parse_args()
    pairs_root = args.pairs_root.resolve()
    names = pair_names(pairs_root)
    if not names:
        parser.error(f"no paired runs found under {pairs_root}")
    server = ViewerServer((args.host, args.port), Handler)
    server.pairs_root = pairs_root
    print(f"NOOA evidence viewer: http://{args.host}:{args.port}/")
    print(f"Pairs: {len(names)} (latest: {names[0]})")
    print("Viewer is read-only. Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
