"""Run Muse in one persistent NOOA conversation against online LS20."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import select
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import arc_agi
import numpy as np
from arc_agi.base import OperationMode
from arcengine import GameAction
from PIL import Image

from experiments.harness import runner as shared_runner
from experiments.harness import types as shared_types
from experiments.harness.runner import TERMINAL_STATES, public_observation

EXPERIMENT = Path(__file__).resolve().parent
RUNTIME = EXPERIMENT / "nooa-runtime"
WORKER = RUNTIME / "worker.py"
CARDS = EXPERIMENT.parents[1] / "cards"
EXPERIMENT_CARDS = {
    "predict": CARDS / "0159-muse-glimmer-nooa-persistent-ls20-v2.md",
    "codeact": CARDS / "0159-muse-glimmer-native-encoding-ls20-v12.md",
    "codeact_notes": CARDS / "0159-muse-glimmer-human-action-labels-ls20-v15.md",
}
MODEL = "meta/muse-glimmer-30b"
NOOA_REVISION = "55a4b57872b49bfe9698cda6fc8e1f3b0bdd723e"
READY_PREFIX = "0159_READY\t"
RESPONSE_PREFIX = "0159_RESPONSE\t"
DEFAULT_ACTION_CAP = 44
ANALYSIS_MAX_ITERATIONS = 11
COMMIT_MAX_ATTEMPTS = 3
MAX_PROVIDER_CALLS_PER_TURN = ANALYSIS_MAX_ITERATIONS + COMMIT_MAX_ATTEMPTS
VARIANTS = {"predict", "codeact", "codeact_notes"}
ACTION_ID_TO_NAME = {
    0: "RESET",
    1: "UP",
    2: "DOWN",
    3: "LEFT",
    4: "RIGHT",
    5: "SPACEBAR",
    6: "CLICK",
    7: "UNDO",
}
ACTION_NAME_TO_ID = {name: action_id for action_id, name in ACTION_ID_TO_NAME.items()}
WORKER_ENVIRONMENT_KEYS = (
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
    "TZ",
)


class SandboxUnavailableError(RuntimeError):
    pass


class WorkerIdentityError(RuntimeError):
    pass


WORKER_FAILURE_STOP_REASONS = {
    "model": "model_failure",
    "provider": "provider_failure",
    "sandbox": "sandbox_failure",
    "worker": "worker_failure",
}
INVALID_STOP_REASONS = {
    "host_failure",
    "provider_failure",
    "sandbox_failure",
    "uncertain_environment_action",
    "worker_failure",
    "worker_identity_failure",
    "worker_shutdown_failure",
    "worker_timeout",
}


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()


def write_json(path: Path, record: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_commitments(experiment_card: Path) -> tuple[dict[str, str], str]:
    paths = {
        "host": Path(__file__).resolve(),
        "worker": WORKER.resolve(),
        "worker_lock": (RUNTIME / "uv.lock").resolve(),
        "experiment_card": experiment_card.resolve(),
        "shared_harness_runner": Path(
            inspect.getsourcefile(shared_runner) or ""
        ).resolve(),
        "shared_harness_types": Path(
            inspect.getsourcefile(shared_types) or ""
        ).resolve(),
    }
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    bundle = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return hashes, bundle


def record_failure(run_dir: Path, *, stage: str, error: BaseException) -> None:
    record = {
        "stage": stage,
        "error_type": type(error).__name__,
        "message": str(error),
    }
    append_jsonl(run_dir / "failures.jsonl", record)
    failure_path = run_dir / "failure.json"
    if not failure_path.exists():
        write_json(failure_path, record)


def frame_sequence(frame: Any) -> np.ndarray:
    frames = public_observation(frame).pixels
    if frames.ndim == 2:
        frames = frames[np.newaxis, ...]
    if frames.ndim != 3 or frames.shape[0] == 0:
        raise ValueError(
            f"public frame must contain one or more grids, got {frames.shape}"
        )
    return frames


def frame_grid(frame: Any) -> np.ndarray:
    """Return ARC's settled decision grid; earlier entries are animation frames."""

    return frame_sequence(frame)[-1]


def grid_values(frame: Any) -> list[list[int]]:
    return [[int(value) for value in row] for row in frame_grid(frame)]


def save_frame(frame: Any, path: Path) -> None:
    save_grid(frame_grid(frame), path)


def save_grid(grid: np.ndarray, path: Path) -> None:
    from arc_agi.rendering import frame_to_rgb_array

    rgb = frame_to_rgb_array(0, grid, scale=4)
    Image.fromarray(rgb).save(path, format="PNG")


def save_animation_event(
    frame: Any, *, turn: int, animations_dir: Path
) -> dict[str, Any] | None:
    frames = frame_sequence(frame)
    if len(frames) == 1:
        return None

    event_dir = animations_dir / f"turn-{turn:03d}"
    event_dir.mkdir(parents=True, exist_ok=False)
    image_paths: list[str] = []
    for index, grid in enumerate(frames):
        image_path = event_dir / f"frame-{index:03d}.png"
        save_grid(grid, image_path)
        image_paths.append(str(image_path.resolve()))
    event_path = event_dir / "event.json"
    write_json(
        event_path,
        {
            "turn": turn,
            "frame_count": len(frames),
            "settled_frame_index": len(frames) - 1,
            "frames": frames.astype(int).tolist(),
            "image_paths": image_paths,
        },
    )
    return {
        "type": "animation_event",
        "turn": turn,
        "frame_count": len(frames),
        "settled_frame_index": len(frames) - 1,
        "event_path": str(event_path.resolve()),
    }


def frame_record(*, turn: int, frame: Any, image_path: Path) -> dict[str, Any]:
    state = frame.state.value if hasattr(frame.state, "value") else str(frame.state)
    return {
        "type": "observation",
        "turn": turn,
        "image_path": str(image_path),
        "state": state,
        "levels_completed": int(frame.levels_completed),
        "full_reset": bool(frame.full_reset),
        "legal_actions": [
            action_name(int(action)) for action in frame.available_actions
        ],
    }


def transition_record(
    *, turn: int, action: str, x: int | None, y: int | None, before: Any, after: Any
) -> dict[str, Any]:
    state = after.state.value if hasattr(after.state, "value") else str(after.state)
    return {
        "turn": turn,
        "action": action,
        "x": x,
        "y": y,
        "before": grid_values(before),
        "after": grid_values(after),
        "levels_completed": int(after.levels_completed),
        "state": state,
        "full_reset": bool(after.full_reset),
    }


def action_name(action_id: int) -> str:
    """Translate the ARC SDK identifier into the human-facing control label."""
    try:
        return ACTION_ID_TO_NAME[action_id]
    except KeyError as error:
        raise ValueError(f"unsupported ARC action id {action_id}") from error


def game_action(action: str) -> GameAction:
    """Translate a validated human-facing control label at the SDK boundary."""
    try:
        return GameAction.from_id(ACTION_NAME_TO_ID[action])
    except KeyError as error:
        raise ValueError(f"unsupported ARC action label {action!r}") from error


def trajectory_sha256(trajectory: list[dict[str, Any]]) -> str:
    payload = json.dumps(trajectory, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def worker_environment(
    audit_log: Path | None = None,
    *,
    variant: str = "predict",
    knowledge_dir: Path | None = None,
) -> dict[str, str]:
    llm_key = os.environ.get("ARC_LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not llm_key:
        raise RuntimeError("ARC_LLM_API_KEY or OPENROUTER_API_KEY is required")
    environment = {
        key: os.environ[key] for key in WORKER_ENVIRONMENT_KEYS if key in os.environ
    }
    environment["ARC_LLM_API_KEY"] = llm_key
    if audit_log is not None:
        environment["NOOA_0159_AUDIT_LOG"] = str(audit_log.resolve())
    environment["NOOA_0159_VARIANT"] = variant
    if variant == "codeact_notes":
        if knowledge_dir is None:
            raise ValueError("codeact_notes requires a run-local knowledge directory")
        environment["NOOA_0159_KNOWLEDGE_DIR"] = str(knowledge_dir.resolve())
    return environment


class Worker:
    """A single worker process containing a single persistent NOOA Agent."""

    def __init__(
        self,
        run_dir: Path,
        *,
        variant: str = "predict",
        timeout_seconds: float = 3_600.0,
    ):
        if variant not in VARIANTS:
            raise ValueError(f"unknown worker variant {variant!r}")
        self.variant = variant
        self.timeout_seconds = timeout_seconds
        self.stdout_log = run_dir / "worker-stdout.log"
        self.stderr_handle = (run_dir / "worker-stderr.log").open("w", encoding="utf-8")
        self.process: subprocess.Popen[str] | None = None
        try:
            self.process = subprocess.Popen(
                [
                    "uv",
                    "run",
                    "--frozen",
                    "--project",
                    str(RUNTIME),
                    "python",
                    str(WORKER),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.stderr_handle,
                text=True,
                bufsize=1,
                env=worker_environment(
                    run_dir / "provider-events.jsonl",
                    variant=variant,
                    knowledge_dir=(
                        run_dir / "knowledge" if variant == "codeact_notes" else None
                    ),
                ),
            )
            self.ready = self._read(READY_PREFIX)
            if self.ready.get("variant") != variant:
                raise WorkerIdentityError(
                    "NOOA worker started with the wrong capability variant"
                )
            if self.ready.get("client_kind") != "live":
                raise WorkerIdentityError(
                    "NOOA live worker did not start with the live model client"
                )
            if self.ready.get("model") != MODEL:
                raise WorkerIdentityError("NOOA worker started with the wrong model")
            if self.ready.get("provider_route") != "deepinfra/bf16":
                raise WorkerIdentityError(
                    "NOOA worker started with the wrong provider route"
                )
            if (
                variant != "predict"
                and self.ready.get("sandbox_enforceable") is not True
            ):
                missing = self.ready.get("missing_sandbox_guards", [])
                raise SandboxUnavailableError(
                    f"NOOA sandbox guardrails are unavailable: {missing}"
                )
        except BaseException:
            try:
                self._stop_process()
            finally:
                self.stderr_handle.close()
            raise

    def _read(self, prefix: str) -> dict[str, Any]:
        assert self.process is not None
        assert self.process.stdout is not None
        while True:
            readable, _, _ = select.select(
                [self.process.stdout], [], [], self.timeout_seconds
            )
            if not readable:
                raise TimeoutError("NOOA worker response timed out")
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(
                    f"NOOA worker exited with code {self.process.poll()}"
                )
            if line.startswith(prefix):
                return json.loads(line[len(prefix) :])
            with self.stdout_log.open("a", encoding="utf-8") as handle:
                handle.write(line)

    def decide(self, request: dict[str, Any]) -> dict[str, Any]:
        assert self.process is not None
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        response = self._read(RESPONSE_PREFIX)
        if response.get("sequence") != request["sequence"]:
            raise RuntimeError("NOOA worker returned a stale or mismatched response")
        return response

    def _stop_process(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)

    def close(self) -> None:
        if (
            self.process is not None
            and self.process.poll() is None
            and self.process.stdin is not None
        ):
            try:
                self.process.stdin.write('{"type":"close"}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=10)
            except BrokenPipeError, OSError, subprocess.TimeoutExpired:
                self._stop_process()
        if not self.stderr_handle.closed:
            self.stderr_handle.close()


def default_run_dir(variant: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return EXPERIMENT / "artifacts" / f"{stamp}-{variant}-nooa-ls20"


def run(
    run_dir: Path,
    *,
    action_cap: int = DEFAULT_ACTION_CAP,
    variant: str = "predict",
) -> dict[str, Any]:
    arc_key = os.environ.get("ARC_API_KEY")
    if not arc_key:
        raise RuntimeError("ARC_API_KEY is required")
    if action_cap < 1:
        raise ValueError("action cap must be positive")
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")

    run_dir.mkdir(parents=True, exist_ok=False)
    frames_dir = run_dir / "frames"
    frames_dir.mkdir()
    animations_dir = run_dir / "animations"
    animations_dir.mkdir()
    transcript = run_dir / "transcript.jsonl"
    trajectory_trace = run_dir / "trajectory.jsonl"
    experiment_card = EXPERIMENT_CARDS[variant]
    source_sha256, source_bundle_sha256 = source_commitments(experiment_card)
    write_json(
        run_dir / "run-config.json",
        {
            "schema": "0159-nooa-ls20-run-v15",
            "game": "ls20",
            "seed": 0,
            "operation_mode": "online",
            "fresh_reset": True,
            "model": MODEL,
            "provider_route": "deepinfra/bf16",
            "action_cap": action_cap,
            "variant": variant,
            "capability": {
                "predict": "persistent conversation only",
                "codeact": "persistent conversation plus read-only trajectory CodeAct",
                "codeact_notes": (
                    "persistent conversation plus read-only trajectory CodeAct plus "
                    "model-authored durable notes"
                ),
            }[variant],
            "max_provider_calls_per_turn": (
                MAX_PROVIDER_CALLS_PER_TURN if variant != "predict" else None
            ),
            "analysis_max_iterations": (
                ANALYSIS_MAX_ITERATIONS if variant != "predict" else None
            ),
            "commitment_max_attempts": (
                COMMIT_MAX_ATTEMPTS if variant != "predict" else None
            ),
            "commitment_interface": (
                "return_result(result=Decision) or exact JSON Decision"
                if variant != "predict"
                else None
            ),
            "commitment_tool_choice": (
                "named_return_result_requested_with_strict_json_fallback"
                if variant != "predict"
                else None
            ),
            "durable_knowledge": (
                {
                    "interface": "NOOA-style list/read/append/write notes",
                    "location": "knowledge/*.md",
                    "host_authored_semantics": False,
                }
                if variant == "codeact_notes"
                else None
            ),
            "ipc_stdout": "duplicated_os_file_descriptor",
            "arc_frame_sequence": "archive_all_use_final_as_decision_grid",
            "host_action_fallback": False,
            "action_interface": {
                "model_visible": [
                    "RESET",
                    "UP",
                    "DOWN",
                    "LEFT",
                    "RIGHT",
                    "SPACEBAR",
                    "CLICK x y",
                    "UNDO",
                ],
                "sdk_ids_model_visible": False,
                "click_coordinates": "x,y integers in [0,63]",
                "sdk_translation_owner": "host",
            },
            "environment_actions_per_turn": 1,
            "nooa_revision": NOOA_REVISION,
            "source_sha256": source_sha256,
            "source_bundle_sha256": source_bundle_sha256,
            "experiment_card": str(experiment_card.resolve()),
            "agent_lifetime": "one process and one Agent for the full run",
            "success": "levels_completed >= 1",
        },
    )

    worker: Worker | None = None
    environment: Any = None
    actions = 0
    action_attempts = 0
    levels_completed = 0
    stop_reason = "host_failure"
    previous_action: int | None = None
    trajectory: list[dict[str, Any]] = []
    animation_events = 0
    try:
        arcade = arc_agi.Arcade(
            arc_api_key=arc_key,
            operation_mode=OperationMode.ONLINE,
            recordings_dir=str(run_dir / "raw-recordings"),
        )
        environment = arcade.make("ls20", seed=0, save_recording=True)
        if environment is None:
            raise RuntimeError("online ARC service did not create LS20")
        current = environment.reset()
        if current is None:
            raise RuntimeError("online ARC reset returned no public frame")
        levels_completed = int(current.levels_completed)
        if levels_completed != 0:
            raise RuntimeError(
                "fresh LS20 reset did not start at zero completed levels"
            )

        worker = Worker(run_dir, variant=variant)
        append_jsonl(transcript, {"type": "worker_ready", **worker.ready})
        print(json.dumps({"event": "worker_ready", **worker.ready}), flush=True)

        for turn in range(action_cap):
            image_path = frames_dir / f"turn-{turn:03d}.png"
            save_frame(current, image_path)
            observation = frame_record(turn=turn, frame=current, image_path=image_path)
            observation["trajectory_length"] = len(trajectory)
            observation["trajectory_sha256"] = trajectory_sha256(trajectory)
            append_jsonl(transcript, observation)
            if observation["state"].upper() in TERMINAL_STATES:
                stop_reason = "terminal"
                break
            if not observation["legal_actions"]:
                stop_reason = "no_available_actions"
                break
            request = {
                "type": "turn",
                "sequence": turn,
                "image_path": str(image_path.resolve()),
                "legal_actions": observation["legal_actions"],
                "previous_action": previous_action,
                "levels_completed": levels_completed,
            }
            if variant != "predict":
                request["trajectory"] = trajectory
            decision = worker.decide(request)
            append_jsonl(transcript, decision)
            if decision.get("type") != "decision":
                failure_stage = str(decision.get("failure_stage", "worker"))
                stop_reason = WORKER_FAILURE_STOP_REASONS.get(
                    failure_stage, "worker_failure"
                )
                record_failure(
                    run_dir,
                    stage=failure_stage,
                    error=RuntimeError(
                        str(decision.get("message", "worker turn failed"))
                    ),
                )
                break
            action = decision.get("action")
            x = decision.get("x")
            y = decision.get("y")
            if (
                not isinstance(action, str)
                or action not in observation["legal_actions"]
            ):
                stop_reason = "illegal_model_action"
                break
            if action == "CLICK":
                if (
                    type(x) is not int
                    or type(y) is not int
                    or not 0 <= x <= 63
                    or not 0 <= y <= 63
                ):
                    stop_reason = "illegal_model_action"
                    break
            elif x is not None or y is not None:
                stop_reason = "illegal_model_action"
                break

            action_attempts += 1
            append_jsonl(
                transcript,
                {
                    "type": "action_attempt",
                    "turn": turn,
                    "action": action,
                    "x": x,
                    "y": y,
                },
            )
            try:
                if action == "CLICK":
                    successor = environment.step(
                        game_action(action), data={"x": x, "y": y}
                    )
                else:
                    successor = environment.step(game_action(action))
            except Exception as error:  # noqa: BLE001 - acceptance may precede any failure
                stop_reason = "uncertain_environment_action"
                record_failure(run_dir, stage="environment_step", error=error)
                break
            if successor is None:
                stop_reason = "uncertain_environment_action"
                record_failure(
                    run_dir,
                    stage="environment_step",
                    error=RuntimeError("environment step returned no successor frame"),
                )
                break
            transition = transition_record(
                turn=turn,
                action=action,
                x=x,
                y=y,
                before=current,
                after=successor,
            )
            trajectory.append(transition)
            append_jsonl(trajectory_trace, transition)
            actions += 1
            previous_action = action
            levels_completed = int(successor.levels_completed)
            state = (
                successor.state.value
                if hasattr(successor.state, "value")
                else str(successor.state)
            )
            append_jsonl(
                transcript,
                {
                    "type": "action_result",
                    "turn": turn,
                    "action": action,
                    "levels_completed": levels_completed,
                    "state": state,
                    "full_reset": bool(successor.full_reset),
                    "transition_sha256": hashlib.sha256(
                        json.dumps(
                            transition, sort_keys=True, separators=(",", ":")
                        ).encode()
                    ).hexdigest(),
                },
            )
            animation_event = save_animation_event(
                successor,
                turn=turn,
                animations_dir=animations_dir,
            )
            if animation_event is not None:
                animation_events += 1
                append_jsonl(transcript, animation_event)
            print(
                json.dumps(
                    {
                        "event": "action_result",
                        "turn": turn,
                        "action": action,
                        "levels_completed": levels_completed,
                        "state": state,
                    }
                ),
                flush=True,
            )
            current = successor
            if levels_completed >= 1:
                stop_reason = "level_completed"
                break
            if state.upper() in TERMINAL_STATES:
                stop_reason = "terminal"
                break
            if bool(successor.full_reset):
                stop_reason = "full_reset"
                break
        else:
            stop_reason = "action_cap"
    except Exception as error:  # noqa: BLE001 - preserve a bounded terminal artifact
        if isinstance(error, SandboxUnavailableError):
            stop_reason = "sandbox_failure"
            failure_stage = "sandbox"
        elif isinstance(error, WorkerIdentityError):
            stop_reason = "worker_identity_failure"
            failure_stage = "worker_identity"
        elif isinstance(error, TimeoutError):
            stop_reason = "worker_timeout"
            failure_stage = "worker"
        else:
            stop_reason = "host_failure"
            failure_stage = "host"
        record_failure(run_dir, stage=failure_stage, error=error)
    finally:
        if worker is not None:
            try:
                worker.close()
            except Exception as error:  # noqa: BLE001 - teardown must not erase evidence
                record_failure(run_dir, stage="worker_shutdown", error=error)
                if stop_reason not in {"level_completed", "action_cap", "terminal"}:
                    stop_reason = "worker_shutdown_failure"
        close = getattr(environment, "close", None)
        if callable(close):
            try:
                close()
            except Exception as error:  # noqa: BLE001 - teardown must not erase evidence
                record_failure(run_dir, stage="environment_shutdown", error=error)

    summary = {
        "schema": "0159-nooa-ls20-summary-v3",
        "run_dir": str(run_dir),
        "variant": variant,
        "stop_reason": stop_reason,
        "actions": actions,
        "action_attempts": action_attempts,
        "levels_completed": levels_completed,
        "success": levels_completed >= 1,
        "valid": stop_reason not in INVALID_STOP_REASONS,
        "invalid_reason": stop_reason if stop_reason in INVALID_STOP_REASONS else None,
        "trajectory_length": len(trajectory),
        "trajectory_sha256": trajectory_sha256(trajectory),
        "animation_events": animation_events,
        "source_bundle_sha256": source_bundle_sha256,
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--action-cap", type=int, default=DEFAULT_ACTION_CAP)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="predict")
    args = parser.parse_args()
    summary = run(
        (args.run_dir or default_run_dir(args.variant)).resolve(),
        action_cap=args.action_cap,
        variant=args.variant,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
