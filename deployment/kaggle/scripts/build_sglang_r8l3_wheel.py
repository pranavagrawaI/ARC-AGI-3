#!/usr/bin/env python3
"""Build the exact SGLang R8-L3B orderly-shutdown wheel successor."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import zipfile
from collections.abc import Mapping
from pathlib import Path

TOKENIZER_MANAGER_PATH = "sglang/srt/managers/tokenizer_manager.py"
SCHEDULER_PATH = "sglang/srt/managers/scheduler.py"
HTTP_SERVER_PATH = "sglang/srt/entrypoints/http_server.py"
UPSTREAM_WHEEL_SHA256 = "9d07f8bf3b718dc4cd970e72a03415b0b0597da89625acbee7a655d6387e7a8d"
UPSTREAM_SOURCE_SHA256 = {
    TOKENIZER_MANAGER_PATH: "0d194294e7d56fe71cb7f13701d6021a0f7c41f5b6480319627ee1bbe8d94110",
    SCHEDULER_PATH: "0d4190dcff3dcda16812fbe95f08ec9c04234a8f12e80a81ff1406c58d46f748",
    HTTP_SERVER_PATH: "056422af61f51573256560509d42ed4f23151364158474aff1c012cce95c880b",
}


def _replace_once(source: bytes, old: bytes, new: bytes, *, label: str) -> bytes:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"{label} expected one source match, found {count}")
    return source.replace(old, new)


def patch_runtime_sources(files: Mapping[str, bytes]) -> dict[str, bytes]:
    """Return runtime source entries after the orderly-shutdown treatment."""
    patched = dict(files)

    tokenizer = patched[TOKENIZER_MANAGER_PATH]
    tokenizer = _replace_once(
        tokenizer,
        b"""        # Subprocess liveness watchdog \xe2\x80\x94 set by Engine or http_server after construction
        self._subprocess_watchdog = None
""",
        b"""        # Subprocess liveness watchdog \xe2\x80\x94 set by Engine or http_server after construction
        self._subprocess_watchdog = None
        # Bound by the owning HTTP server before it enters its run loop.
        self._http_server_shutdown_callback = None
""",
        label="tokenizer shutdown callback state",
    )
    tokenizer = _replace_once(
        tokenizer,
        b"""        # Stop the watchdog: child exits are expected during shutdown, not crashes.
        if self._subprocess_watchdog is not None:
            self._subprocess_watchdog.stop()
        # Ask schedulers to release resources in userspace and exit (see
        # ShutdownReq), then wait for them before hard-killing the rest.
        self._dispatch_to_scheduler(ShutdownReq())
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and collect_scheduler_processes():
            time.sleep(0.1)
        kill_process_tree(os.getpid(), include_parent=False, wait_timeout=60)
        sys.exit(0)
""",
        b"""        # Stop the watchdog: child exits are expected during shutdown, not crashes.
        if self._subprocess_watchdog is not None:
            self._subprocess_watchdog.stop()
        # Ask schedulers to release resources in userspace and exit (see
        # ShutdownReq), then wait for them before hard-killing the rest.
        self._dispatch_to_scheduler(ShutdownReq())
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and collect_scheduler_processes():
            time.sleep(0.1)
        kill_process_tree(os.getpid(), include_parent=False, wait_timeout=60)
        if self._http_server_shutdown_callback is None:
            raise RuntimeError("HTTP server shutdown callback is not configured")
        self._http_server_shutdown_callback()
""",
        label="tokenizer orderly shutdown return",
    )
    patched[TOKENIZER_MANAGER_PATH] = tokenizer

    scheduler = patched[SCHEDULER_PATH]
    scheduler = _replace_once(
        scheduler,
        b"from sglang.srt.distributed.parallel_state import get_tp_group\n",
        b"""from sglang.srt.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    get_tp_group,
)
""",
        label="scheduler distributed cleanup import",
    )
    scheduler = _replace_once(
        scheduler,
        b"""    scheduler = None
    try:
""",
        b"""    scheduler = None
    scheduler_loop_returned_normally = False
    try:
""",
        label="scheduler normal-return state",
    )
    scheduler = _replace_once(
        scheduler,
        b"""        scheduler.run_event_loop()

    except Exception:
""",
        b"""        scheduler.run_event_loop()
        scheduler_loop_returned_normally = True

    except Exception:
""",
        label="scheduler normal-return witness",
    )
    scheduler = _replace_once(
        scheduler,
        b"""            if scheduler.gracefully_exit:
                scheduler.release_host_resources()
""",
        b"""            if scheduler.gracefully_exit and scheduler_loop_returned_normally:
                try:
                    scheduler.release_host_resources()
                finally:
                    cleanup_dist_env_and_memory()
""",
        label="scheduler graceful distributed cleanup",
    )
    patched[SCHEDULER_PATH] = scheduler

    http_server = patched[HTTP_SERVER_PATH]
    http_server = _replace_once(
        http_server,
        b"""            else:
                # Default case, one tokenizer process
                uvicorn.run(
                    app,
                    host=get_serving().host,
                    port=get_serving().port,
                    root_path=server_args.fastapi_root_path,
                    log_level=get_observability().log_level_http
                    or get_observability().log_level,
                    timeout_keep_alive=envs.SGLANG_TIMEOUT_KEEP_ALIVE.get(),
                    loop="uvloop",
                    ssl_keyfile=server_args.ssl_keyfile,
                    ssl_certfile=server_args.ssl_certfile,
                    ssl_ca_certs=server_args.ssl_ca_certs,
                    ssl_keyfile_password=server_args.ssl_keyfile_password,
                )
""",
        b"""            else:
                # Default case, one tokenizer process. Keep the Server object so the
                # tokenizer watchdog can request ordinary ASGI shutdown after ranks exit.
                config = uvicorn.Config(
                    app,
                    host=get_serving().host,
                    port=get_serving().port,
                    root_path=server_args.fastapi_root_path,
                    log_level=get_observability().log_level_http
                    or get_observability().log_level,
                    timeout_keep_alive=envs.SGLANG_TIMEOUT_KEEP_ALIVE.get(),
                    loop="uvloop",
                    ssl_keyfile=server_args.ssl_keyfile,
                    ssl_certfile=server_args.ssl_certfile,
                    ssl_ca_certs=server_args.ssl_ca_certs,
                    ssl_keyfile_password=server_args.ssl_keyfile_password,
                )
                server = uvicorn.Server(config)

                def request_http_server_shutdown():
                    server.should_exit = True

                tokenizer_manager._http_server_shutdown_callback = request_http_server_shutdown
                try:
                    server.run()
                finally:
                    tokenizer_manager._http_server_shutdown_callback = None
""",
        label="bound Uvicorn shutdown callback",
    )
    patched[HTTP_SERVER_PATH] = http_server
    return patched


def _record_digest(data: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"sha256={encoded.decode('ascii')}"


def rewrite_wheel(
    source_wheel: Path,
    output_wheel: Path,
    replacements: Mapping[str, bytes],
) -> None:
    """Copy one wheel deterministically and regenerate its complete RECORD."""
    with zipfile.ZipFile(source_wheel) as source:
        infos = source.infolist()
        record_infos = [info for info in infos if info.filename.endswith(".dist-info/RECORD")]
        if len(record_infos) != 1:
            raise ValueError(f"expected one wheel RECORD, found {len(record_infos)}")
        record_info = record_infos[0]
        names = {info.filename for info in infos}
        unknown = set(replacements) - names
        if unknown:
            raise ValueError(f"replacement entries are absent from wheel: {sorted(unknown)}")

        entry_data = {
            info.filename: (
                replacements[info.filename]
                if info.filename in replacements
                else source.read(info.filename)
            )
            for info in infos
            if info.filename != record_info.filename
        }

    record_buffer = io.StringIO(newline="")
    writer = csv.writer(record_buffer, lineterminator="\n")
    for info in infos:
        if info.filename == record_info.filename:
            continue
        data = entry_data[info.filename]
        writer.writerow([info.filename, _record_digest(data), str(len(data))])
    writer.writerow([record_info.filename, "", ""])
    record_data = record_buffer.getvalue().encode("utf-8")

    output_wheel.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_wheel, "w") as output:
        for info in infos:
            if info.filename == record_info.filename:
                continue
            output.writestr(info, entry_data[info.filename])
        output.writestr(record_info, record_data)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_patched_wheel(source_wheel: Path, output_wheel: Path) -> dict[str, object]:
    """Build the R8-L3 wheel only from the exact Kaggle v12 runtime bytes."""
    source_wheel = source_wheel.resolve()
    output_wheel = output_wheel.resolve()
    if source_wheel == output_wheel:
        raise ValueError("output wheel must differ from upstream wheel")
    observed_wheel_sha256 = _sha256_file(source_wheel)
    if observed_wheel_sha256 != UPSTREAM_WHEEL_SHA256:
        raise ValueError(
            "upstream wheel SHA-256 mismatch: "
            f"expected {UPSTREAM_WHEEL_SHA256}, observed {observed_wheel_sha256}"
        )

    with zipfile.ZipFile(source_wheel) as archive:
        sources = {name: archive.read(name) for name in UPSTREAM_SOURCE_SHA256}
    for name, expected in UPSTREAM_SOURCE_SHA256.items():
        observed = _sha256_bytes(sources[name])
        if observed != expected:
            raise ValueError(
                f"upstream source SHA-256 mismatch for {name}: "
                f"expected {expected}, observed {observed}"
            )

    patched = patch_runtime_sources(sources)
    rewrite_wheel(source_wheel, output_wheel, patched)
    return {
        "schema": "sglang-r8l3b-orderly-shutdown-wheel-v1",
        "upstream_wheel": str(source_wheel),
        "upstream_wheel_sha256": observed_wheel_sha256,
        "output_wheel": str(output_wheel),
        "output_wheel_sha256": _sha256_file(output_wheel),
        "patched_sources": {
            name: {
                "upstream_sha256": UPSTREAM_SOURCE_SHA256[name],
                "patched_sha256": _sha256_bytes(patched[name]),
            }
            for name in sorted(patched)
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-wheel", type=Path, required=True)
    parser.add_argument("--output-wheel", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_patched_wheel(args.input_wheel, args.output_wheel)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
