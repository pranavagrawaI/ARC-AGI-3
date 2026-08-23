"""Data-only subprocess worker for model-generated Python cells.

This file is executed directly under ``python -I``. It intentionally imports no
Tycho module and receives no live parent object.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import struct
import sys
from itertools import islice
from pathlib import Path
from typing import Any

_PROTOCOL_VERSION = 1
_MAX_PROTOCOL_BYTES = 1_048_576
_PROTOCOL_INPUT = sys.stdin.buffer
_PROTOCOL_OUTPUT = sys.stdout.buffer


class _ReturnResult(BaseException):
    def __init__(  # pragma: no cover - exercised only through the exec process seam
        self, payload: Any
    ) -> None:
        self.payload = payload


class _BoundedText(io.TextIOBase):
    def __init__(  # pragma: no cover - exec process seam
        self, limit: int
    ) -> None:
        self._limit = limit
        self._parts: list[str] = []
        self._size = 0

    def write(self, text: str) -> int:  # pragma: no cover - exec process seam
        text = str(text)
        remaining = max(0, self._limit - self._size)
        accepted = text.encode("utf-8")[:remaining].decode("utf-8", errors="ignore")
        if accepted:
            self._parts.append(accepted)
            self._size += len(accepted.encode("utf-8"))
        return len(text)

    def getvalue(self) -> str:  # pragma: no cover - exec process seam
        return "".join(self._parts)


class _CapabilityError(RuntimeError):
    pass


class _CapabilityClient:
    def __init__(  # pragma: no cover - exec process seam
        self, invocation_id: str
    ) -> None:
        self._invocation_id = invocation_id
        self._request_id = 0

    def call(  # pragma: no cover - exec process seam
        self, name: str, arguments: dict[str, Any]
    ) -> Any:
        self._request_id += 1
        request_id = self._request_id
        _write(
            {
                "protocol_version": _PROTOCOL_VERSION,
                "type": "capability_request",
                "invocation_id": self._invocation_id,
                "capability_request_id": request_id,
                "name": name,
                "arguments": _json_value(arguments),
            }
        )
        response = _read()
        if response is None or response.get("type") != "capability_result":
            raise _CapabilityError("capability response is unavailable")
        if response.get("protocol_version") != _PROTOCOL_VERSION:
            raise _CapabilityError("capability protocol version mismatch")
        if response.get("capability_request_id") != request_id:
            raise _CapabilityError("capability response did not match its request")
        if response.get("status") == "complete":
            return response.get("result")
        error = response.get("exception", {})
        raise _CapabilityError(
            f"{error.get('type', 'CapabilityError')}: {error.get('message', '')}"
        )


class _EvidenceProxy:
    def __init__(  # pragma: no cover - exec process seam
        self, client: _CapabilityClient
    ) -> None:
        self._client = client
        self._cache: dict[str, str] = {}

    def query(self, **arguments: Any) -> Any:  # pragma: no cover - exec process seam
        key = json.dumps(arguments, allow_nan=False, sort_keys=True, separators=(",", ":"))
        cached = self._cache.get(key)
        if cached is None:
            value = self._client.call("evidence.query", arguments)
            cached = json.dumps(value, allow_nan=False, separators=(",", ":"))
            self._cache[key] = cached
        return json.loads(cached)


def _read() -> dict[str, Any] | None:  # pragma: no cover - exec process seam
    header = _PROTOCOL_INPUT.read(4)
    if not header:
        return None
    if len(header) != 4:
        raise RuntimeError("truncated sterile protocol header")
    size = struct.unpack(">I", header)[0]
    if size > _MAX_PROTOCOL_BYTES:
        raise RuntimeError("sterile protocol request exceeds its byte limit")
    payload = _PROTOCOL_INPUT.read(size)
    if len(payload) != size:
        raise RuntimeError("truncated sterile protocol payload")
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise TypeError("sterile protocol request must be an object")
    return parsed


def _write(payload: dict[str, Any]) -> None:  # pragma: no cover - exec process seam
    encoded = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode(
        "utf-8"
    )
    if len(encoded) > _MAX_PROTOCOL_BYTES:
        raise RuntimeError("sterile protocol response exceeds its byte limit")
    _PROTOCOL_OUTPUT.write(struct.pack(">I", len(encoded)) + encoded)
    _PROTOCOL_OUTPUT.flush()


def _json_value(value: Any) -> Any:  # pragma: no cover - exec process seam
    encoded = json.dumps(value, allow_nan=False, separators=(",", ":"))
    return json.loads(encoded)


def _return_result(  # pragma: no cover - exec process seam
    *args: Any, **kwargs: Any
) -> None:
    if args and kwargs:
        raise ValueError("return_result accepts either one value or keyword fields")
    if len(args) > 1:
        raise ValueError("return_result accepts at most one positional value")
    payload = args[0] if args else kwargs
    raise _ReturnResult(_json_value(payload))


def _preview(value: Any) -> str:  # pragma: no cover - exec process seam
    if isinstance(value, list):
        return f"list(len={len(value)})"
    if isinstance(value, tuple):
        return f"tuple(len={len(value)})"
    if isinstance(value, dict):
        return f"dict(len={len(value)})"
    if isinstance(value, set):
        return f"set(len={len(value)})"
    return type(value).__name__


def _locals_preview(  # pragma: no cover - exec process seam
    namespace: dict[str, Any], reserved: set[str], limit: int
) -> dict[str, str]:
    visible = (
        (name, value)
        for name, value in sorted(namespace.items())
        if name not in reserved and not name.startswith("__")
    )
    return {
        name[:128]: _preview(value)[:128]
        for name, value in islice(visible, limit)
    }


def _install_guards(  # pragma: no cover - exercised only through the exec process seam
    manifest: dict[str, Any],
) -> None:
    """Irrevocably restrict the clean child before any model cell runs."""
    path = Path(__file__).with_name("_sterile_guards.py")
    spec = importlib.util.spec_from_file_location("_tycho_sterile_guards", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("sterile guard module is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.install(
        scratch=os.getcwd(),
        memory_headroom_mb=int(manifest["memory_headroom_mb"]),
        cpu_seconds=int(manifest["cpu_seconds"]),
        file_size_bytes=int(manifest["file_size_bytes"]),
        open_files=int(manifest["open_files"]),
    )
    del sys.modules[spec.name]


def main() -> None:  # pragma: no cover - exercised only through the exec process seam
    manifest, input_values = _initialization()
    capability_values = _capability_namespace(manifest)
    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "return_result": _return_result,
        **input_values,
        **capability_values,
    }
    _install_guards(manifest)
    _write({"protocol_version": _PROTOCOL_VERSION, "type": "ready"})
    _serve(
        invocation_id=str(manifest["invocation_id"]),
        namespace=namespace,
        reserved=set(namespace),
        output_bytes=int(manifest["output_bytes"]),
        locals_preview_items=int(manifest["locals_preview_items"]),
    )


def _capability_namespace(  # pragma: no cover - exec process seam
    manifest: dict[str, Any],
) -> dict[str, Any]:
    capabilities = manifest.get("capabilities", [])
    if not isinstance(capabilities, list):
        raise TypeError("sterile worker capabilities must be a list")
    unknown = set(capabilities) - {"evidence.query"}
    if unknown:
        raise RuntimeError(f"unknown sterile capabilities: {sorted(unknown)!r}")
    client = _CapabilityClient(str(manifest["invocation_id"]))
    return {"evidence": _EvidenceProxy(client)} if capabilities else {}


def _initialization(  # pragma: no cover - exec process seam
) -> tuple[dict[str, Any], dict[str, Any]]:
    initialize = _read()
    if initialize is None or initialize.get("type") != "initialize":
        raise RuntimeError("sterile worker requires initialize first")
    manifest = initialize.get("manifest")
    if not isinstance(manifest, dict):
        raise TypeError("sterile worker manifest must be an object")
    if manifest.get("protocol_version") != _PROTOCOL_VERSION:
        raise RuntimeError("sterile worker protocol version mismatch")
    input_values = _json_value(manifest.get("input_values", {}))
    if not isinstance(input_values, dict):
        raise TypeError("sterile worker input_values must be an object")
    return manifest, input_values


def _serve(  # pragma: no cover - exec process seam
    *,
    invocation_id: str,
    namespace: dict[str, Any],
    reserved: set[str],
    output_bytes: int,
    locals_preview_items: int,
) -> None:
    while True:
        request = _read()
        if request is None:
            return
        if request.get("type") == "shutdown":
            return
        _write(
            _execute_request(
                request,
                invocation_id=invocation_id,
                namespace=namespace,
                reserved=reserved,
                output_bytes=output_bytes,
                locals_preview_items=locals_preview_items,
            )
        )


def _execute_request(  # pragma: no cover - exec process seam
    request: dict[str, Any],
    *,
    invocation_id: str,
    namespace: dict[str, Any],
    reserved: set[str],
    output_bytes: int,
    locals_preview_items: int,
) -> dict[str, Any]:
    if request.get("type") != "execute":
        raise RuntimeError(f"unknown sterile worker request {request.get('type')!r}")
    if request.get("invocation_id") != invocation_id:
        raise RuntimeError("sterile worker invocation mismatch")
    stdout = _BoundedText(output_bytes)
    stderr = _BoundedText(output_bytes)
    response = _response(request.get("request_id"))
    _run_code(str(request.get("code", "")), namespace, stdout, stderr, response)
    response["stdout"] = stdout.getvalue()
    response["stderr"] = stderr.getvalue()
    response["locals_preview"] = _locals_preview(
        namespace, reserved, locals_preview_items
    )
    return response


def _response(  # pragma: no cover - exec process seam
    request_id: Any,
) -> dict[str, Any]:
    return {
        "protocol_version": _PROTOCOL_VERSION,
        "type": "execution_result",
        "request_id": request_id,
        "stdout": "",
        "stderr": "",
        "locals_preview": {},
    }


def _run_code(  # pragma: no cover - exec process seam
    code: str,
    namespace: dict[str, Any],
    stdout: _BoundedText,
    stderr: _BoundedText,
    response: dict[str, Any],
) -> None:
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(  # noqa: S102 - executing the model cell is this worker's sole job
                code, namespace, namespace
            )
        response["status"] = "complete"
    except _ReturnResult as signal:
        response["status"] = "returned"
        response["return_payload"] = signal.payload
    except BaseException as error:  # noqa: BLE001 - generated SystemExit stays in T2
        response["status"] = "error"
        response["exception"] = {
            "type": type(error).__name__,
            "message": str(error)[:2_000],
        }


if __name__ == "__main__":
    main()
