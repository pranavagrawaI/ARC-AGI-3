#!/usr/bin/env python3
"""Compare C1/C2/C4 on four frozen v6 Solver successor requests."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import statistics
import subprocess
import threading
import time
import traceback
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

SCHEMA = "tycho-nooa-successor-concurrency-probe-v1"
FIXTURE_SCHEMA = "tycho-nooa-successor-concurrency-fixtures-v1"
FIXTURE_SHA256 = "3fcabf7b410327e824b3dbc87d78adbc229ebd2ab412edeb8753e4c625f57029"
BASE_PROBE_SHA256 = "b792325c746a76440703fda793252159364ba256251e939d0039cc6c1e8ed64f"
SOLVER_WHEEL_SHA256 = "c78c3e41bf33b7007310bd6198f67786ea01f366c0b9b56b93d8e8c19a84c1e9"
SOLVER_SOURCE_MANIFEST_SHA256 = (
    "010dbb2a11b717646735d5731173baf657c8098d1a9689393015b2bc9c7ae364"
)
EXPECTED_FIXTURE_HASHES = {
    "keyboard-16": "929e529c33567de7cbb3956fcc3eceba6f24e7e04776c6d00f222848cafc9562",
    "click-32": "2f8b4debd162ad846354462320aec8b0bf11a2d3d01f2d5b189a28404245f1c4",
    "mixed-40": "87109fe33093e5faef7f84b017ca3319bad75f9ee3681e479ce136b1fe452981",
    "space-64": "55dcc894ce6eeab05a6999a7cca8713e85e72807fdbd6a802943745fcb7c2c97",
}
CONDITION_ORDER = (4, 2, 1)
COHORT_CALLS = 1181
GAMEPLAY_CAP_SECONDS = 27_000
SERVED_MODEL_PLACEHOLDER = "__SERVED_MODEL__"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_base_probe(path: Path) -> ModuleType:
    if sha256_file(path) != BASE_PROBE_SHA256:
        raise RuntimeError("R8-L3B base probe bytes differ from the frozen route")
    spec = importlib.util.spec_from_file_location("r8l3b_base_probe", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import base probe: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_fixtures(path: Path) -> list[dict[str, Any]]:
    if sha256_file(path) != FIXTURE_SHA256:
        raise RuntimeError("successor fixture file bytes differ from the frozen fixture")
    value = json.loads(path.read_text())
    if (
        value.get("schema") != FIXTURE_SCHEMA
        or value.get("fixture_count") != 4
        or value.get("served_model_placeholder") != SERVED_MODEL_PLACEHOLDER
        or value.get("solver_wheel_sha256") != SOLVER_WHEEL_SHA256
        or value.get("solver_source_manifest_sha256")
        != SOLVER_SOURCE_MANIFEST_SHA256
    ):
        raise RuntimeError("successor fixture semantic identity differs")
    fixtures = value.get("fixtures")
    if not isinstance(fixtures, list) or len(fixtures) != 4:
        raise RuntimeError("successor fixture set must contain exactly four requests")
    observed_hashes: dict[str, str] = {}
    for fixture in fixtures:
        name = fixture.get("name")
        request = fixture.get("request")
        if not isinstance(name, str) or not isinstance(request, dict):
            raise RuntimeError("fixture name/request is invalid")
        request_bytes = canonical_bytes(request)
        observed_hashes[name] = sha256_bytes(request_bytes)
        functions = [tool.get("function", {}) for tool in request.get("tools", [])]
        metadata = fixture.get("worker_metadata", {})
        envelope = metadata.get("decision_envelope", {})
        if (
            fixture.get("request_bytes") != len(request_bytes)
            or fixture.get("request_sha256") != observed_hashes[name]
            or request.get("model") != SERVED_MODEL_PLACEHOLDER
            or request.get("max_tokens") != 4096
            or request.get("tool_choice") != "required"
            or [function.get("name") for function in functions] != ["take_action"]
            or metadata.get("mode") != "single"
            or metadata.get("treatment") != "single_decision_discipline"
            or envelope.get("max_actor_calls_after_initial") != 1
            or envelope.get("max_output_tokens_per_call_after_initial") != 4096
            or envelope.get("model_retries") != 0
        ):
            raise RuntimeError(f"fixture {name!r} is not an exact v6 successor request")
    if observed_hashes != EXPECTED_FIXTURE_HASHES:
        raise RuntimeError(f"successor fixture hashes differ: {observed_hashes}")
    return fixtures


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


def post_bytes(url: str, payload: bytes, timeout_s: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(NoRedirectHandler())
    try:
        with opener.open(request, timeout=timeout_s) as response:
            return {
                "status": response.status,
                "reason": response.reason,
                "headers": dict(response.headers.items()),
                "body": response.read(),
            }
    except urllib.error.HTTPError as error:
        return {
            "status": error.code,
            "reason": error.reason,
            "headers": dict(error.headers.items()),
            "body": error.read(),
        }


def validate_response(response: Any, fixture: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise RuntimeError("response body is not one JSON object")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError("response does not contain exactly one choice")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(message, dict):
        raise RuntimeError("response choice has no message object")
    reasoning = message.get("reasoning_content")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise RuntimeError("response has no nonempty reasoning_content")
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise RuntimeError("response does not contain exactly one native tool call")
    function = calls[0].get("function") if isinstance(calls[0], dict) else None
    if not isinstance(function, dict) or function.get("name") != "take_action":
        raise RuntimeError("response tool is not take_action")
    raw_arguments = function.get("arguments")
    if not isinstance(raw_arguments, str):
        raise RuntimeError("take_action arguments are not a JSON string")
    arguments = json.loads(raw_arguments)
    action_schema = fixture["request"]["tools"][0]["function"]["parameters"]
    actions = action_schema["properties"]["action"]["enum"]
    action = arguments.get("action") if isinstance(arguments, dict) else None
    if action not in actions:
        raise RuntimeError(f"response action {action!r} is not legal for {fixture['name']}")
    expected_keys = {"action", "row", "col"} if action == "CLICK" else {"action"}
    if set(arguments) != expected_keys:
        raise RuntimeError(f"response arguments differ for {action}: {arguments}")
    if action == "CLICK" and (
        type(arguments["row"]) is not int or type(arguments["col"]) is not int
    ):
        raise RuntimeError("CLICK row/col must be integers")
    if action == "CLICK":
        click_schema = next(
            option["properties"]
            for option in action_schema["oneOf"]
            if option["properties"].get("action", {}).get("const") == "CLICK"
        )
        for coordinate in ("row", "col"):
            coordinate_schema = click_schema[coordinate]
            value = arguments[coordinate]
            if not (
                coordinate_schema["minimum"]
                <= value
                <= coordinate_schema["maximum"]
            ):
                raise RuntimeError(
                    f"CLICK {coordinate} is outside the frozen schema: {value}"
                )
    if choice.get("finish_reason") != "tool_calls":
        raise RuntimeError("response finish_reason is not tool_calls")
    usage = response.get("usage")
    if not isinstance(usage, dict) or any(
        type(usage.get(key)) is not int or usage[key] < 0
        for key in ("prompt_tokens", "completion_tokens")
    ):
        raise RuntimeError("response usage lacks nonnegative prompt/completion token counts")
    return {
        "action": action,
        "reasoning_chars": len(reasoning),
        "finish_reason": choice.get("finish_reason"),
        "usage": usage,
    }


def run_condition(
    *,
    concurrency: int,
    fixtures: list[dict[str, Any]],
    model_name: str,
    endpoint: str,
    output_dir: Path,
    timeout_s: int,
) -> dict[str, Any]:
    condition_dir = output_dir / f"c{concurrency}"
    condition_dir.mkdir()
    wall_started = time.monotonic()
    lock = threading.Lock()
    active = 0
    peak = 0
    records: list[dict[str, Any]] = []

    def one(fixture: dict[str, Any]) -> dict[str, Any]:
        nonlocal active, peak
        request = json.loads(json.dumps(fixture["request"]))
        request["model"] = model_name
        request_bytes = canonical_bytes(request)
        name = fixture["name"]
        request_path = condition_dir / f"{name}.request.bin"
        response_path = condition_dir / f"{name}.response.bin"
        request_path.write_bytes(request_bytes)
        started = time.monotonic()
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            transport = post_bytes(endpoint, request_bytes, timeout_s)
        finally:
            with lock:
                active -= 1
        ended = time.monotonic()
        response_bytes = transport.pop("body")
        response_path.write_bytes(response_bytes)
        write_json(condition_dir / f"{name}.transport.json", transport)
        if transport["status"] != 200:
            raise RuntimeError(f"C{concurrency} {name} returned HTTP {transport['status']}")
        response = json.loads(response_bytes)
        write_json(condition_dir / f"{name}.response.json", response)
        validation = validate_response(response, fixture)
        record = {
            "fixture": name,
            "request_sha256": sha256_bytes(request_bytes),
            "response_sha256": sha256_bytes(response_bytes),
            "started_offset_s": round(started - wall_started, 6),
            "ended_offset_s": round(ended - wall_started, 6),
            "elapsed_s": round(ended - started, 6),
            "validation": validation,
        }
        write_json(condition_dir / f"{name}.result.json", record)
        return record

    # Submit all four together. The executor bounds active client requests to C.
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(one, fixture) for fixture in fixtures]
        for future in futures:
            records.append(future.result())
    wall_s = time.monotonic() - wall_started
    records.sort(key=lambda record: record["fixture"])
    latencies = [record["elapsed_s"] for record in records]
    usages = [record["validation"].get("usage") or {} for record in records]
    result = {
        "concurrency": concurrency,
        "request_count": len(records),
        "client_inflight_peak": peak,
        "wall_s": round(wall_s, 6),
        "requests_per_s": round(len(records) / wall_s, 6),
        "latency_mean_s": round(statistics.mean(latencies), 6),
        "latency_max_s": round(max(latencies), 6),
        "prompt_tokens": sum(int(usage.get("prompt_tokens", 0)) for usage in usages),
        "completion_tokens": sum(
            int(usage.get("completion_tokens", 0)) for usage in usages
        ),
        "records": records,
    }
    write_json(condition_dir / "condition-result.json", result)
    if peak != concurrency:
        raise RuntimeError(f"C{concurrency} client peak was {peak}, not {concurrency}")
    return result


def verify_runtime(base: ModuleType, provenance_path: Path) -> dict[str, Any]:
    value = json.loads(provenance_path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError("runtime provenance must be one JSON object")
    expected = {
        "repository": base.SGLANG_REPOSITORY,
        "commit": base.SGLANG_COMMIT,
        "dataset_source": base.RUNTIME_DATASET_SOURCE,
        "dataset_version": base.RUNTIME_DATASET_VERSION,
        "base_dataset_source": base.BASE_RUNTIME_DATASET_SOURCE,
        "base_dataset_version": base.BASE_RUNTIME_DATASET_VERSION,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise RuntimeError("runtime provenance does not bind the accepted R8-L3B route")
    hashes = {
        "artifact_path": base.RUNTIME_WHEEL_SHA256,
        "lock_path": base.RUNTIME_LOCK_SHA256,
        "dataset_manifest_path": base.RUNTIME_DATASET_MANIFEST_SHA256,
        "base_artifact_path": base.BASE_RUNTIME_WHEEL_SHA256,
        "base_lock_path": base.RUNTIME_LOCK_SHA256,
    }
    for key, expected_hash in hashes.items():
        path = Path(str(value.get(key, "")))
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"runtime provenance member {key} differs")
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    started_at = utc_now()
    events: list[str] = []

    def event(message: str) -> None:
        events.append(f"{utc_now()} {message}")
        (output_dir / "events.log").write_text("\n".join(events) + "\n")
        print(message, flush=True)

    base = load_base_probe(args.base_probe)
    fixtures = load_fixtures(args.fixtures)
    write_json(
        output_dir / "probe-config.json",
        {
            "schema": SCHEMA,
            "condition_order": CONDITION_ORDER,
            "fixture_sha256": FIXTURE_SHA256,
            "base_probe_sha256": BASE_PROBE_SHA256,
            "cohort_calls": COHORT_CALLS,
            "gameplay_cap_seconds": GAMEPLAY_CAP_SECONDS,
            "request_timeout_s": args.request_timeout_s,
            "startup_timeout_s": args.startup_timeout_s,
            "claim_ceiling": (
                "A pass compares model-server admission C1/C2/C4 on four exact synthetic "
                "v6 successor requests. It runs no ARC environment and proves neither "
                "gameplay nor full-history all-25 runtime."
            ),
        },
    )
    write_json(output_dir / "system-before.json", base.system_snapshot())
    process: subprocess.Popen[str] | None = None
    server_log: Any = None
    launch_target: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    lifecycle: dict[str, Any] | None = None
    failure: BaseException | None = None
    try:
        if base.EXPECTED_GPU_FRAGMENT not in base.gpu_name(base.system_snapshot()):
            raise RuntimeError("expected Kaggle RTX PRO 6000 GPU")
        preflight = base.lifecycle_observer_preflight(args.port)
        write_json(output_dir / "lifecycle-observer-preflight.json", preflight)
        if preflight["status"] != "passed":
            raise RuntimeError(f"lifecycle observer preflight failed: {preflight['errors']}")
        runtime = verify_runtime(base, args.runtime_provenance)
        write_json(output_dir / "runtime-verification.json", runtime)
        model_path = base.discover_model(args.input_root, args.manifest)
        event(f"verifying immutable model files at {model_path}")
        model_verification = base.verify_model(model_path, args.manifest)
        write_json(output_dir / "model-verification.json", model_verification)
        command = base.sglang_command(
            model_path,
            "127.0.0.1",
            args.port,
            attention_backend="triton",
            sampling_backend="pytorch",
        )
        write_json(output_dir / "server-command.json", command)
        server_log = (output_dir / "server.log").open("w")
        environment = os.environ.copy()
        environment["R8_L3B_OWNER"] = base.R8_L3B_OWNER
        environment["PYTHONPROFILEIMPORTTIME"] = "1"
        write_json(
            output_dir / "server-environment.json",
            {
                "R8_L3B_OWNER": base.R8_L3B_OWNER,
                "PYTHONPROFILEIMPORTTIME": "1",
            },
        )
        event("launching accepted R8-L3B SGLang route")
        process = subprocess.Popen(
            command,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
            start_new_session=True,
        )
        launch_target = base.process_identity(process.pid)
        observer = base.observer_identity()
        launch = {
            "captured_at": utc_now(),
            "observer": observer,
            "target": launch_target,
            "outside_target_pgid": observer["pgid"] != launch_target["pgid"],
            "outside_target_sid": observer["sid"] != launch_target["sid"],
        }
        write_json(output_dir / "lifecycle-target-launch.json", launch)
        if (
            not launch["outside_target_pgid"]
            or not launch["outside_target_sid"]
            or not launch_target["owner_marker"]
            or launch_target["pid"] != launch_target["pgid"]
            or launch_target["pid"] != launch_target["sid"]
        ):
            raise RuntimeError("model server launch identity is not externally owned")
        startup_started = time.monotonic()
        readiness = base.wait_for_server(
            f"http://127.0.0.1:{args.port}", process, args.startup_timeout_s
        )
        models = readiness["models"].get("data", [])
        if len(models) != 1 or models[0].get("id") != base.EXPECTED_SERVED_MODEL_ID:
            raise RuntimeError(f"unexpected served model: {readiness['models']}")
        remaining = args.startup_timeout_s - (time.monotonic() - startup_started)
        warmup = base.wait_for_sglang_warmup(
            output_dir / "server.log", process, remaining
        )
        write_json(output_dir / "warmup-facts.json", warmup)
        if (
            warmup["chat_completion_accesses"] != 1
            or warmup["successful_chat_completion_accesses"] != 1
            or warmup["successful_freeze_gc_accesses"] != 1
            or warmup["ready_marker_count"] != 1
        ):
            raise RuntimeError(f"unexpected warmup facts: {warmup}")
        startup_s = time.monotonic() - startup_started
        server_log.flush()
        backend = base.backend_facts(
            (output_dir / "server.log").read_text(errors="replace"),
            attention_backend="triton",
            sampling_backend="pytorch",
            request_ready=True,
        )
        write_json(output_dir / "backend-facts.json", backend)
        if (
            backend["selected_attention_backend"] != "triton"
            or backend["selected_sampling_backend"] != "pytorch"
        ):
            raise RuntimeError("accepted model-server backend route drifted")
        endpoint = f"http://127.0.0.1:{args.port}/v1/chat/completions"
        conditions = []
        for concurrency in CONDITION_ORDER:
            event(f"running exact four-request condition C{concurrency}")
            conditions.append(
                run_condition(
                    concurrency=concurrency,
                    fixtures=fixtures,
                    model_name=models[0]["id"],
                    endpoint=endpoint,
                    output_dir=output_dir,
                    timeout_s=args.request_timeout_s,
                )
            )
        server_log.flush()
        request_facts = base.wait_for_chat_completion_count(
            output_dir / "server.log", process, expected_count=13
        )
        write_json(output_dir / "request-count-facts.json", request_facts)
        if (
            request_facts["chat_completion_accesses"] != 13
            or request_facts["successful_chat_completion_accesses"] != 13
            or request_facts["successful_freeze_gc_accesses"] != 1
            or request_facts["ready_marker_count"] != 1
        ):
            raise RuntimeError(f"model-server request counts differ: {request_facts}")
        by_c = {condition["concurrency"]: condition for condition in conditions}
        projections = {
            str(concurrency): {
                "successor_service_seconds": round(
                    COHORT_CALLS * by_c[concurrency]["wall_s"] / 4, 3
                ),
                "startup_plus_successor_service_seconds": round(
                    startup_s + COHORT_CALLS * by_c[concurrency]["wall_s"] / 4,
                    3,
                ),
                "inside_27000_seconds_on_this_fixture_only": (
                    startup_s + COHORT_CALLS * by_c[concurrency]["wall_s"] / 4
                    <= GAMEPLAY_CAP_SECONDS
                ),
            }
            for concurrency in (1, 2, 4)
        }
        sufficient = [
            concurrency
            for concurrency in (1, 2, 4)
            if projections[str(concurrency)]["inside_27000_seconds_on_this_fixture_only"]
        ]
        result = {
            "schema": SCHEMA,
            "status": "pending_lifecycle",
            "started_at": started_at,
            "ended_at": utc_now(),
            "elapsed_before_teardown_s": round(time.monotonic() - started, 3),
            "startup_s": round(startup_s, 3),
            "condition_order": list(CONDITION_ORDER),
            "conditions": conditions,
            "request_count_facts": request_facts,
            "projections": projections,
            "smallest_fixture_sufficient_concurrency": min(sufficient) if sufficient else None,
            "claim_ceiling": (
                "Synthetic v6 successor-request concurrency only; no ARC action, gameplay, "
                "long-history, initial-call, score, or submission claim."
            ),
        }
    except BaseException as error:
        failure = error
        result = {
            "schema": SCHEMA,
            "status": "failed",
            "started_at": started_at,
            "ended_at": utc_now(),
            "elapsed_before_teardown_s": round(time.monotonic() - started, 3),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "claim_ceiling": "The run failed and establishes no concurrency claim.",
        }
    finally:
        if process is not None:
            try:
                lifecycle = base.qualify_lifecycle(
                    process, server_log, output_dir, port=args.port
                )
            except BaseException as error:
                lifecycle = {
                    "status": "invalid",
                    "failure_classification": "observer_contract_invalid",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
                if process.poll() is None:
                    base.emergency_contain_process_group(process, launch_target)
        else:
            lifecycle = {
                "status": "invalid",
                "failure_classification": "target_not_started",
            }
        if server_log is not None:
            server_log.close()
        assert result is not None
        result["lifecycle"] = lifecycle
        if result["status"] == "pending_lifecycle":
            result["status"] = "passed" if lifecycle.get("status") == "passed" else "failed"
        result["ended_at"] = utc_now()
        result["elapsed_s"] = round(time.monotonic() - started, 3)
        write_json(output_dir / "shutdown.json", lifecycle)
        write_json(output_dir / "result.json", result)
        write_json(output_dir / "system-after.json", base.system_snapshot())
    if failure is not None:
        raise failure
    if result["status"] != "passed":
        raise RuntimeError(f"concurrency probe lifecycle failed: {lifecycle}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--base-probe", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime-provenance", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--startup-timeout-s", type=int, default=1200)
    parser.add_argument("--request-timeout-s", type=int, default=600)
    parser.add_argument("--port", type=int, default=30000)
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
