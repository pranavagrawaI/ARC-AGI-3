"""Run Muse in one persistent NOOA conversation against online LS20."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import select
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any
from uuid import uuid4

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
E0_EVIDENCE_PATH = EXPERIMENT / "e0_evidence.py"
CARDS = EXPERIMENT.parents[1] / "cards"
EXPERIMENT_CARDS = {
    "predict": CARDS / "0159-muse-glimmer-nooa-persistent-ls20-v2.md",
    "codeact": CARDS / "0159-muse-glimmer-native-encoding-ls20-v12.md",
    "codeact_notes": CARDS / "0159-muse-glimmer-human-action-labels-ls20-v15.md",
    "codeact_notes_compact": (CARDS / "0159-muse-in-loop-compaction-ls20-v24.md"),
    "codeact_notes_compact_world_model": (
        CARDS / "0160-muse-wm1-recovery-feedback-v1.md"
    ),
}
DEFAULT_MODEL_PROFILE = "muse-glimmer-30b"
LUNA_MODEL_PROFILE = "gpt-5.6-luna"
MODEL_PROFILES = {
    DEFAULT_MODEL_PROFILE: {
        "model": "meta/muse-glimmer-30b",
        "provider_route": "qualified-pool-v18",
        "provider_routes": [
            "deepinfra-bf16",
            "together",
            "fireworks",
            "phala",
        ],
        "context_window": 131_072,
    },
    LUNA_MODEL_PROFILE: {
        "model": "openai/gpt-5.6-luna",
        "provider_route": "openai",
        "provider_routes": ["openai"],
        "context_window": 1_050_000,
    },
}
LUNA_EXPERIMENT_CARD = CARDS / "0160-luna-wm1-recovery-feedback-v1.md"
LUNA_CAPABILITY_PROBE_CARD = CARDS / "0160-luna-wm1-capability-probe-v3.md"
MODEL = MODEL_PROFILES[DEFAULT_MODEL_PROFILE]["model"]
NOOA_REVISION = "55a4b57872b49bfe9698cda6fc8e1f3b0bdd723e"
READY_PREFIX = "0159_READY\t"
RESPONSE_PREFIX = "0159_RESPONSE\t"
DEFAULT_ACTION_CAP = 44
ANALYSIS_MAX_ITERATIONS = 11
COMMIT_MAX_ATTEMPTS = 3
MAX_PROVIDER_CALLS_PER_TURN = ANALYSIS_MAX_ITERATIONS + COMMIT_MAX_ATTEMPTS
WORLD_MODEL_VARIANT = "codeact_notes_compact_world_model"
COMPACT_VARIANTS = {"codeact_notes_compact", WORLD_MODEL_VARIANT}
KNOWLEDGE_VARIANTS = {"codeact_notes", *COMPACT_VARIANTS}
VARIANTS = {"predict", "codeact", *KNOWLEDGE_VARIANTS}
COMPACT_VARIANT = "codeact_notes_compact"
DEFAULT_COMPACTION_RECENT_TURNS = 6
DEFAULT_COMPACTION_TRIGGER_FRACTION = 0.80
DEFAULT_COMPACTION_TARGET_FRACTION = 0.30
AUTO_COMPACTION_SUMMARY_TOKEN_ALLOWANCE = 4_096
MAX_IN_LOOP_COMPACTIONS_PER_TURN = 1
QUALIFIED_PROVIDER_POOL = [
    "deepinfra-bf16",
    "together",
    "fireworks",
    "phala",
]
PROVIDER_ATTEMPT_TIMEOUT_SECONDS = 90.0
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

_E0_SPEC = importlib.util.spec_from_file_location("e0_evidence_0159", E0_EVIDENCE_PATH)
if _E0_SPEC is None or _E0_SPEC.loader is None:
    raise RuntimeError(f"cannot load E0 evidence adapter {E0_EVIDENCE_PATH}")
E0_EVIDENCE = importlib.util.module_from_spec(_E0_SPEC)
sys.modules[_E0_SPEC.name] = E0_EVIDENCE
_E0_SPEC.loader.exec_module(E0_EVIDENCE)


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
    "compaction_failure",
    "evidence_adapter_failure",
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


def source_commitments(
    experiment_card: Path,
    *,
    include_compaction: bool = False,
    include_world_model: bool = False,
    model_profile: str = DEFAULT_MODEL_PROFILE,
) -> tuple[dict[str, str], str]:
    paths = {
        "host": Path(__file__).resolve(),
        "e0_evidence": E0_EVIDENCE_PATH.resolve(),
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
    if include_compaction:
        paths.update(
            {
                "compaction": (EXPERIMENT / "compaction.py").resolve(),
                "checkpoint_comparison": (
                    EXPERIMENT / "compare_compaction.py"
                ).resolve(),
            }
        )
        if model_profile == DEFAULT_MODEL_PROFILE:
            paths.update(
                {
                    "provider_qualification_card": (
                        CARDS / "0159-muse-provider-qualification-v18.md"
                    ).resolve(),
                    "provider_qualification_summary": (
                        EXPERIMENT / "artifacts/provider-qualification-v18/summary.json"
                    ).resolve(),
                    "provider_qualification_events": (
                        EXPERIMENT
                        / "artifacts/provider-qualification-v18/provider-events.jsonl"
                    ).resolve(),
                    "provider_qualification_runner": (
                        RUNTIME / "qualify_providers.py"
                    ).resolve(),
                }
            )
    if include_world_model:
        paths.update(
            {
                "world_model_lane": (RUNTIME / "world_model_lane.py").resolve(),
                "world_model_evaluator": (
                    RUNTIME / "world_model_evaluator.py"
                ).resolve(),
                "executable_theory_lane_spec": (
                    EXPERIMENT / "EXECUTABLE-THEORY-LANE-SPEC.md"
                ).resolve(),
                "world_model_provenance": (EXPERIMENT / "PROVENANCE.md").resolve(),
            }
        )
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


def evidence_observation(frame: Any) -> Any:
    frames = tuple(
        tuple(tuple(int(value) for value in row) for row in grid)
        for grid in frame_sequence(frame)
    )
    state = frame.state.value if hasattr(frame.state, "value") else str(frame.state)
    return E0_EVIDENCE.PublicObservationInput(
        frames=frames,
        public_state=state,
        levels_completed=int(frame.levels_completed),
        full_reset=bool(frame.full_reset),
        legal_actions=tuple(
            action_name(int(action)) for action in frame.available_actions
        ),
    )


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
    world_model_dir: Path | None = None,
    model_profile: str = DEFAULT_MODEL_PROFILE,
    capability_probe: bool = False,
) -> dict[str, str]:
    if model_profile not in MODEL_PROFILES:
        raise ValueError(f"unknown model profile {model_profile!r}")
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
    environment["NOOA_0159_MODEL_PROFILE"] = model_profile
    if capability_probe:
        environment["NOOA_0159_CAPABILITY_PROBE"] = "1"
    if variant in KNOWLEDGE_VARIANTS:
        if knowledge_dir is None:
            raise ValueError(f"{variant} requires a run-local knowledge directory")
        environment["NOOA_0159_KNOWLEDGE_DIR"] = str(knowledge_dir.resolve())
    if variant == WORLD_MODEL_VARIANT:
        if world_model_dir is None:
            raise ValueError(f"{variant} requires a run-local world-model directory")
        environment["NOOA_0159_WORLD_MODEL_DIR"] = str(world_model_dir.resolve())
    return environment


class Worker:
    """A single worker process containing a single persistent NOOA Agent."""

    def __init__(
        self,
        run_dir: Path,
        *,
        variant: str = "predict",
        model_profile: str = DEFAULT_MODEL_PROFILE,
        capability_probe: bool = False,
        timeout_seconds: float = 3_600.0,
    ):
        if variant not in VARIANTS:
            raise ValueError(f"unknown worker variant {variant!r}")
        if model_profile not in MODEL_PROFILES:
            raise ValueError(f"unknown model profile {model_profile!r}")
        if model_profile == LUNA_MODEL_PROFILE and variant != WORLD_MODEL_VARIANT:
            raise ValueError("the Luna first-contact profile is restricted to WM0")
        self.variant = variant
        self.model_profile = model_profile
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
                        run_dir / "knowledge"
                        if variant in KNOWLEDGE_VARIANTS
                        else None
                    ),
                    world_model_dir=(
                        run_dir / "world-model"
                        if variant == WORLD_MODEL_VARIANT
                        else None
                    ),
                    model_profile=model_profile,
                    capability_probe=capability_probe,
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
            model_config = MODEL_PROFILES[model_profile]
            if self.ready.get("model_profile") != model_profile:
                raise WorkerIdentityError(
                    "NOOA worker started with the wrong model profile"
                )
            if self.ready.get("model") != model_config["model"]:
                raise WorkerIdentityError("NOOA worker started with the wrong model")
            expected_route = model_config["provider_route"]
            if model_profile == DEFAULT_MODEL_PROFILE and variant not in COMPACT_VARIANTS:
                expected_route = "deepinfra/bf16"
            if self.ready.get("provider_route") != expected_route:
                raise WorkerIdentityError(
                    "NOOA worker started with the wrong provider route"
                )
            expected_routes = model_config["provider_routes"]
            if model_profile == DEFAULT_MODEL_PROFILE and variant not in COMPACT_VARIANTS:
                expected_routes = None
            if variant in COMPACT_VARIANTS and (
                self.ready.get("provider_routes") != expected_routes
                or self.ready.get("provider_attempt_timeout_seconds")
                != PROVIDER_ATTEMPT_TIMEOUT_SECONDS
                or self.ready.get("implicit_context_archival") is not False
            ):
                raise WorkerIdentityError(
                    "NOOA worker started with the wrong compact-runtime policy"
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

    def compact(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.variant not in COMPACT_VARIANTS:
            raise ValueError("compact requests require a compact variant")
        assert self.process is not None
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        response = self._read(RESPONSE_PREFIX)
        if response.get("sequence") != request["sequence"]:
            raise RuntimeError("NOOA worker returned a stale compaction response")
        if response.get("type") != "compaction_applied":
            raise RuntimeError(f"NOOA worker rejected compaction: {response}")
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
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self._stop_process()
        if not self.stderr_handle.closed:
            self.stderr_handle.close()


def default_run_dir(variant: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return EXPERIMENT / "artifacts" / f"{stamp}-{variant}-nooa-ls20"


def _load_local_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load local module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def freeze_compaction_sources(run_dir: Path, destination: Path) -> Path:
    """Copy the completed evidence prefixes used by one compaction transaction."""
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("transcript.jsonl", "trajectory.jsonl", "provider-events.jsonl"):
        source = run_dir / name
        if source.is_file():
            shutil.copy2(source, destination / name)
    knowledge = run_dir / "knowledge"
    if knowledge.is_dir():
        shutil.copytree(knowledge, destination / "knowledge")
    return destination


def prepare_live_compaction(
    run_dir: Path,
    *,
    resume_turn: int,
    recent_turns: int = DEFAULT_COMPACTION_RECENT_TURNS,
    compaction_index: int = 1,
    model_profile: str = DEFAULT_MODEL_PROFILE,
) -> dict[str, Any]:
    """Prepare and validate a live compaction without mutating NOOA context."""
    if recent_turns < 1 or resume_turn <= recent_turns:
        raise ValueError("compaction needs at least one old turn beyond the suffix")
    if compaction_index < 1:
        raise ValueError("compaction index must be positive")
    compaction_root = run_dir / "compactions" / f"{compaction_index:04d}"
    compaction_root.mkdir(parents=True, exist_ok=False)
    frozen_run = freeze_compaction_sources(run_dir, compaction_root / "frozen-run")
    compaction = _load_local_module("compaction", EXPERIMENT / "compaction.py")
    comparison = _load_local_module(
        "compare_compaction_live_0159", EXPERIMENT / "compare_compaction.py"
    )
    archive = compaction.HistoryArchive(frozen_run)
    api_key = os.environ.get("ARC_LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("ARC_LLM_API_KEY or OPENROUTER_API_KEY is required")
    model_config = MODEL_PROFILES[model_profile]
    providers = (
        (("openai", "openai", None),)
        if model_profile == LUNA_MODEL_PROFILE
        else comparison.PROVIDERS
    )
    client = comparison.OpenRouter(
        api_key,
        compaction_root / "raw-consolidation",
        model=model_config["model"],
        providers=providers,
    )
    checkpoint = comparison.generate_checkpoint(
        client, archive, resume_turn=resume_turn
    )
    checkpoint_path = compaction_root / "model-checkpoint.json"
    write_json(checkpoint_path, checkpoint)
    bundle = compaction_root / "bundle"
    compaction.write_bundle(
        frozen_run,
        checkpoint_path,
        bundle,
        resume_turn=resume_turn,
        recent_turns=recent_turns,
    )
    retained_start_turn = resume_turn - recent_turns
    snapshot = compaction.build_history_snapshot(archive, through_turn=resume_turn)
    summary_text = compaction.render_live_recovery_summary(
        checkpoint,
        archive,
        resume_turn=resume_turn,
        retained_start_turn=retained_start_turn,
    )
    write_json(compaction_root / "history-snapshot.json", snapshot)
    (compaction_root / "active-summary.md").write_text(summary_text, encoding="utf-8")
    result = {
        "type": "compact",
        "sequence": resume_turn,
        "retain_from_turn": retained_start_turn,
        "summary_text": summary_text,
        "history_snapshot": snapshot,
    }
    write_json(
        compaction_root / "transaction.json",
        {
            "schema": "0159-live-compaction-transaction-v1",
            "resume_turn": resume_turn,
            "retained_start_turn": retained_start_turn,
            "recent_turns": recent_turns,
            "checkpoint_chars": len(json.dumps(checkpoint, sort_keys=True)),
            "summary_chars": len(summary_text),
            "archive_event_count": len(snapshot["events"]),
            "consolidation_provider_calls": client.call_count,
            "status": "prepared",
        },
    )
    return result


def should_auto_compact(
    context: dict[str, Any] | None, *, trigger_fraction: float
) -> bool:
    """Use NOOA's provider-anchored usable-window utilization at a safe boundary."""
    if context is None:
        return False
    utilization = context.get("utilization")
    return isinstance(utilization, int | float) and utilization >= trigger_fraction


def choose_recent_turns(
    prompt_samples: list[int],
    *,
    effective_window: int,
    target_fraction: float,
    summary_token_allowance: int = AUTO_COMPACTION_SUMMARY_TOKEN_ALLOWANCE,
) -> int:
    """Keep the largest complete suffix projected to fit the post-compact target.

    Provider counts are exact at each completed boundary. Their positive deltas
    estimate the token weight of complete environment turns; the first sample
    estimates the fixed prompt. A final recent delta reserves room for the next
    turn. The result never splits a turn and always retains at least one.
    """
    if not prompt_samples:
        raise ValueError("adaptive compaction requires prompt samples")
    if effective_window < 1:
        raise ValueError("effective window must be positive")
    if not 0 < target_fraction < 1:
        raise ValueError("target fraction must be between zero and one")
    samples = [max(0, int(value)) for value in prompt_samples]
    fixed_tokens = min(samples)
    deltas = [max(0, right - left) for left, right in pairwise(samples)]
    next_turn_reserve = max(deltas[-3:], default=0)
    target_tokens = int(effective_window * target_fraction)
    for recent_turns in range(len(samples), 0, -1):
        suffix_growth = sum(deltas[-(recent_turns - 1) :]) if recent_turns > 1 else 0
        projected = (
            fixed_tokens
            + suffix_growth
            + next_turn_reserve
            + summary_token_allowance
        )
        if projected <= target_tokens:
            return recent_turns
    return 1


def apply_live_compaction(
    run_dir: Path,
    worker: Worker,
    transcript: Path,
    *,
    resume_turn: int,
    recent_turns: int,
    compaction_index: int,
    trigger: str,
    model_profile: str = DEFAULT_MODEL_PROFILE,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prepare, commit, and record one evidence-preserving compaction."""
    prepared = prepare_live_compaction(
        run_dir,
        resume_turn=resume_turn,
        recent_turns=recent_turns,
        compaction_index=compaction_index,
        model_profile=model_profile,
    )
    result = worker.compact(prepared)
    append_jsonl(
        transcript,
        {**result, "type": "context_compaction", "trigger": trigger},
    )
    transaction_path = (
        run_dir / "compactions" / f"{compaction_index:04d}" / "transaction.json"
    )
    transaction = json.loads(transaction_path.read_text(encoding="utf-8"))
    transaction["status"] = "applied"
    transaction["trigger"] = trigger
    transaction["worker_result"] = result
    write_json(transaction_path, transaction)
    return prepared, result


def run(
    run_dir: Path,
    *,
    action_cap: int = DEFAULT_ACTION_CAP,
    variant: str = "predict",
    compact_at_turn: int | None = None,
    compaction_recent_turns: int = DEFAULT_COMPACTION_RECENT_TURNS,
    auto_compact: bool = False,
    compaction_trigger_fraction: float = DEFAULT_COMPACTION_TRIGGER_FRACTION,
    compaction_target_fraction: float = DEFAULT_COMPACTION_TARGET_FRACTION,
    model_profile: str = DEFAULT_MODEL_PROFILE,
    capability_probe: bool = False,
    experiment_card_override: Path | None = None,
) -> dict[str, Any]:
    arc_key = os.environ.get("ARC_API_KEY")
    if not arc_key:
        raise RuntimeError("ARC_API_KEY is required")
    if action_cap < 1:
        raise ValueError("action cap must be positive")
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    if model_profile not in MODEL_PROFILES:
        raise ValueError(f"unknown model profile {model_profile!r}")
    if model_profile == LUNA_MODEL_PROFILE and variant != WORLD_MODEL_VARIANT:
        raise ValueError("the Luna first-contact profile is restricted to WM0")
    if capability_probe and variant != WORLD_MODEL_VARIANT:
        raise ValueError("the capability probe requires the world-model variant")
    if capability_probe and action_cap != 1:
        raise ValueError("the capability probe is bounded to exactly one action")
    if experiment_card_override is not None:
        experiment_card_override = experiment_card_override.resolve()
        if not experiment_card_override.is_file():
            raise ValueError("explicit experiment card does not exist")
        if len(experiment_card_override.read_bytes()) > 1_024:
            raise ValueError("explicit experiment card exceeds 1 KiB")
    if variant in COMPACT_VARIANTS:
        if compaction_recent_turns < 1:
            raise ValueError("compaction recent turns must be positive")
        if auto_compact and compact_at_turn is not None:
            raise ValueError("choose either automatic or forced compaction")
        if not auto_compact and compact_at_turn is None:
            raise ValueError(
                "compact variants require --auto-compact or --compact-at-turn"
            )
        if compact_at_turn is not None and compact_at_turn <= compaction_recent_turns:
            raise ValueError(
                "compact-at-turn must leave at least one older turn beyond the suffix"
            )
        if compact_at_turn is not None and compact_at_turn >= action_cap:
            raise ValueError("compact-at-turn must occur before the action cap")
        if not 0 < compaction_trigger_fraction < 1:
            raise ValueError("compaction trigger fraction must be between zero and one")
        if not 0 < compaction_target_fraction < compaction_trigger_fraction:
            raise ValueError("compaction target must be below the trigger fraction")
    elif compact_at_turn is not None:
        raise ValueError("--compact-at-turn is only valid for compact variants")

    run_dir.mkdir(parents=True, exist_ok=False)
    frames_dir = run_dir / "frames"
    frames_dir.mkdir()
    animations_dir = run_dir / "animations"
    animations_dir.mkdir()
    transcript = run_dir / "transcript.jsonl"
    trajectory_trace = run_dir / "trajectory.jsonl"
    evidence_run_id = "run-" + uuid4().hex
    evidence_ledger = E0_EVIDENCE.HostEvidenceLedger(
        run_dir / "evidence", run_id=evidence_run_id
    )
    experiment_card = experiment_card_override or (
        LUNA_CAPABILITY_PROBE_CARD
        if capability_probe and model_profile == LUNA_MODEL_PROFILE
        else (
            LUNA_EXPERIMENT_CARD
            if model_profile == LUNA_MODEL_PROFILE
            else EXPERIMENT_CARDS[variant]
        )
    )
    model_config = MODEL_PROFILES[model_profile]
    source_sha256, source_bundle_sha256 = source_commitments(
        experiment_card,
        include_compaction=variant in COMPACT_VARIANTS,
        include_world_model=variant == WORLD_MODEL_VARIANT,
        model_profile=model_profile,
    )
    write_json(
        run_dir / "run-config.json",
        {
            "schema": (
                "0159-nooa-ls20-run-v20"
                if variant == WORLD_MODEL_VARIANT
                else "0159-nooa-ls20-run-v17"
            ),
            "game": "ls20",
            "seed": 0,
            "operation_mode": "online",
            "fresh_reset": True,
            "evidence_adapter": {
                "schema": "0159-host-evidence-v1",
                "run_id": evidence_run_id,
                "event_taxonomy": "host_observable_only",
                "attempt_ids_model_visible": False,
                "input_and_manifest_commitments": True,
            },
            "decision_observation": {
                "image": (
                    {
                        "source": "host settled public grid",
                        "transport": "private JSON row-major integer cells",
                        "worker_view": "64x64 palette-mode PIL image",
                        "coordinates": "np.asarray(image)[y, x]",
                        "visual_use": "show(image)",
                        "host_authored_semantics": False,
                    }
                    if variant != "predict"
                    else "rendered settled public grid for provider vision"
                ),
            },
            "model_profile": model_profile,
            "model": model_config["model"],
            "model_context_window": model_config["context_window"],
            "capability_probe": capability_probe,
            "provider_route": (
                model_config["provider_route"]
                if variant in COMPACT_VARIANTS
                else "deepinfra/bf16"
            ),
            "provider_routes": (
                model_config["provider_routes"] if variant in COMPACT_VARIANTS else None
            ),
            "provider_attempt_timeout_seconds": (
                PROVIDER_ATTEMPT_TIMEOUT_SECONDS if variant in COMPACT_VARIANTS else None
            ),
            "action_cap": action_cap,
            "variant": variant,
            "compact_at_turn": compact_at_turn,
            "auto_compact": auto_compact,
            "compaction_recent_turns": (
                compaction_recent_turns if variant in COMPACT_VARIANTS else None
            ),
            "capability": {
                "predict": "persistent conversation only",
                "codeact": "persistent conversation plus read-only trajectory CodeAct",
                "codeact_notes": (
                    "persistent conversation plus read-only trajectory CodeAct plus "
                    "model-authored durable notes"
                ),
                "codeact_notes_compact": (
                    "persistent CodeAct plus durable notes, namespaced immutable-history "
                    "retrieval, model checkpoint, explicit recovery Summary, and recent "
                    "trajectory suffix"
                ),
                "codeact_notes_compact_world_model": (
                    "the compact persistent investigator plus brokered Muse-authored "
                    "quick partial predictors or full Moore commitments, with isolated "
                    "replay against E0 evidence"
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
                if variant in KNOWLEDGE_VARIANTS
                else None
            ),
            "executable_theory": (
                {
                    "interface": [
                        "read_world_model",
                        "write_world_model",
                        "edit_world_model_function",
                    ],
                    "location": "world-model/",
                    "activation": "operational_validity_only",
                    "evaluation": "fresh_isolated_process",
                    "host_authored_semantics": False,
                    "shadow_prediction": False,
                }
                if variant == WORLD_MODEL_VARIANT
                else None
            ),
            "context_compaction": (
                {
                    "trigger": (
                        "provider_reported_utilization_after_complete_environment_turn"
                        if auto_compact
                        else "forced_environment_turn_boundary"
                    ),
                    "prepare_validate_commit": True,
                    "immutable_frozen_source_prefix": True,
                    "checkpoint_generator": "separate_stateless_model_call",
                    "history_interface": "namespaced history object",
                    "recent_complete_turns": compaction_recent_turns,
                    "automatic": auto_compact,
                    "trigger_fraction_of_usable_window": (
                        compaction_trigger_fraction if auto_compact else None
                    ),
                    "target_fraction_of_usable_window": (
                        compaction_target_fraction if auto_compact else None
                    ),
                    "suffix_policy": (
                        "largest_complete_suffix_projected_under_target"
                        if auto_compact
                        else "fixed_complete_suffix"
                    ),
                    "implicit_nooa_context_archival": False,
                    "in_loop_overflow_recovery": (
                        "compact_then_retry_same_environment_turn"
                        if auto_compact
                        else None
                    ),
                    "max_in_loop_compactions_per_turn": (
                        MAX_IN_LOOP_COMPACTIONS_PER_TURN if auto_compact else None
                    ),
                }
                if variant in COMPACT_VARIANTS
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
    evidence_snapshot_id: str | None = None
    compactions: list[dict[str, Any]] = []
    context_samples: list[int] = []
    last_context: dict[str, Any] | None = None
    retained_trajectory_start = 0
    compaction_module: Any | None = None
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
        evidence_ledger.record_initial_reset(evidence_observation(current))
        evidence_snapshot_id = evidence_ledger.snapshot().snapshot_id

        worker_options: dict[str, Any] = {
            "variant": variant,
            "model_profile": model_profile,
        }
        if capability_probe:
            worker_options["capability_probe"] = True
        worker = Worker(run_dir, **worker_options)
        append_jsonl(transcript, {"type": "worker_ready", **worker.ready})
        print(json.dumps({"event": "worker_ready", **worker.ready}), flush=True)

        for turn in range(action_cap):
            evidence_ledger.record_decision_observation(
                evidence_observation(current), global_turn=turn
            )
            decision_evidence_snapshot = evidence_ledger.snapshot()
            evidence_snapshot_id = decision_evidence_snapshot.snapshot_id
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
            prepared_compaction: dict[str, Any] | None = None
            auto_boundary = (
                variant in COMPACT_VARIANTS
                and auto_compact
                and turn > 1
                and should_auto_compact(
                    last_context, trigger_fraction=compaction_trigger_fraction
                )
            )
            forced_boundary = (
                variant in COMPACT_VARIANTS and turn == compact_at_turn
            )
            if auto_boundary or forced_boundary:
                try:
                    recent_turns = compaction_recent_turns
                    if auto_boundary:
                        assert last_context is not None
                        effective_window = last_context.get("effective_window")
                        if type(effective_window) is not int:
                            raise RuntimeError(
                                "automatic compaction needs a known effective window"
                            )
                        recent_turns = choose_recent_turns(
                            context_samples,
                            effective_window=effective_window,
                            target_fraction=compaction_target_fraction,
                        )
                        recent_turns = min(recent_turns, turn - 1)
                    prepared_compaction, compaction_result = apply_live_compaction(
                        run_dir,
                        worker,
                        transcript,
                        resume_turn=turn,
                        recent_turns=recent_turns,
                        compaction_index=len(compactions) + 1,
                        trigger=(
                            "automatic_boundary" if auto_boundary else "forced_boundary"
                        ),
                        model_profile=model_profile,
                    )
                except Exception as error:  # noqa: BLE001 - preserve boundary evidence
                    stop_reason = "compaction_failure"
                    record_failure(run_dir, stage="context_compaction", error=error)
                    break
                compactions.append(compaction_result)
                retained_trajectory_start = prepared_compaction["retain_from_turn"]
                context_samples = []
                last_context = None
            request = {
                "type": "turn",
                "sequence": turn,
                "image_path": str(image_path.resolve()),
                "legal_actions": observation["legal_actions"],
                "previous_action": previous_action,
                "levels_completed": levels_completed,
            }
            if variant == WORLD_MODEL_VARIANT:
                request["evidence_snapshot_path"] = str(
                    (
                        run_dir
                        / "evidence"
                        / "snapshots"
                        / f"{decision_evidence_snapshot.snapshot_id}.json"
                    ).resolve()
                )
            if variant != "predict":
                request["image_cells"] = grid_values(current)
                request["trajectory"] = (
                    (
                        trajectory[retained_trajectory_start:]
                        if auto_compact
                        else trajectory[-compaction_recent_turns:]
                    )
                    if variant in COMPACT_VARIANTS
                    else trajectory
                )
            if variant in COMPACT_VARIANTS:
                if prepared_compaction is not None:
                    request["history_snapshot"] = prepared_compaction[
                        "history_snapshot"
                    ]
                else:
                    if compaction_module is None:
                        compaction_module = _load_local_module(
                            "compaction", EXPERIMENT / "compaction.py"
                        )
                    archive = compaction_module.HistoryArchive(run_dir)
                    request["history_snapshot"] = (
                        compaction_module.build_history_snapshot(
                            archive, through_turn=turn
                        )
                    )
            decision = worker.decide(request)
            append_jsonl(transcript, decision)
            in_loop_compactions = 0
            while decision.get("type") == "needs_compaction" and auto_compact:
                if in_loop_compactions >= MAX_IN_LOOP_COMPACTIONS_PER_TURN:
                    break
                try:
                    if turn <= 1 or not context_samples or last_context is None:
                        raise RuntimeError(
                            "in-loop overflow occurred before an old complete turn "
                            "could be compacted"
                        )
                    effective_window = last_context.get("effective_window")
                    if type(effective_window) is not int:
                        raise RuntimeError(
                            "in-loop compaction needs a known effective window"
                        )
                    recent_turns = choose_recent_turns(
                        context_samples,
                        effective_window=effective_window,
                        target_fraction=compaction_target_fraction,
                    )
                    recent_turns = min(recent_turns, turn - 1)
                    prepared_compaction, compaction_result = apply_live_compaction(
                        run_dir,
                        worker,
                        transcript,
                        resume_turn=turn,
                        recent_turns=recent_turns,
                        compaction_index=len(compactions) + 1,
                        trigger="in_loop_context_rejection",
                        model_profile=model_profile,
                    )
                except Exception as error:  # noqa: BLE001 - preserve boundary evidence
                    stop_reason = "compaction_failure"
                    record_failure(run_dir, stage="in_loop_compaction", error=error)
                    break
                compactions.append(compaction_result)
                in_loop_compactions += 1
                retained_trajectory_start = prepared_compaction["retain_from_turn"]
                context_samples = []
                last_context = None
                request["trajectory"] = trajectory[retained_trajectory_start:]
                request["history_snapshot"] = prepared_compaction["history_snapshot"]
                decision = worker.decide(request)
                append_jsonl(transcript, decision)
            if stop_reason == "compaction_failure":
                break
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
            if variant in COMPACT_VARIANTS:
                context = decision.get("context")
                last_context = context if isinstance(context, dict) else None
                prompt_tokens = (
                    last_context.get("prompt_tokens") if last_context else None
                )
                if type(prompt_tokens) is int:
                    context_samples.append(prompt_tokens)
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
            try:
                evidence_result = evidence_ledger.record_action_result(
                    global_turn=turn,
                    action=E0_EVIDENCE.Action(action, x=x, y=y),
                    successor=evidence_observation(successor),
                )
            except Exception as error:  # noqa: BLE001 - action already happened
                stop_reason = "evidence_adapter_failure"
                record_failure(run_dir, stage="evidence_adapter", error=error)
                break
            evidence_snapshot_id = evidence_result.snapshot.snapshot_id
            append_jsonl(
                transcript,
                {
                    "type": "evidence_projection",
                    "turn": turn,
                    "source_action_result_turn": turn,
                    "evidence_event_id": (
                        evidence_result.snapshot.transitions[-1].source_event_id
                    ),
                    "evidence_snapshot_id": evidence_snapshot_id,
                },
            )
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
        "schema": "0159-nooa-ls20-summary-v4",
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
        "compactions": compactions,
        "animation_events": animation_events,
        "evidence_snapshot_id": evidence_snapshot_id,
        "source_bundle_sha256": source_bundle_sha256,
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--action-cap", type=int, default=DEFAULT_ACTION_CAP)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="predict")
    parser.add_argument(
        "--model-profile",
        choices=sorted(MODEL_PROFILES),
        default=DEFAULT_MODEL_PROFILE,
    )
    parser.add_argument("--compact-at-turn", type=int)
    parser.add_argument("--auto-compact", action="store_true")
    parser.add_argument("--capability-probe", action="store_true")
    parser.add_argument("--experiment-card", type=Path)
    parser.add_argument(
        "--compaction-trigger-fraction",
        type=float,
        default=DEFAULT_COMPACTION_TRIGGER_FRACTION,
    )
    parser.add_argument(
        "--compaction-target-fraction",
        type=float,
        default=DEFAULT_COMPACTION_TARGET_FRACTION,
    )
    parser.add_argument(
        "--compaction-recent-turns",
        type=int,
        default=DEFAULT_COMPACTION_RECENT_TURNS,
    )
    args = parser.parse_args()
    summary = run(
        (args.run_dir or default_run_dir(args.variant)).resolve(),
        action_cap=args.action_cap,
        variant=args.variant,
        compact_at_turn=args.compact_at_turn,
        compaction_recent_turns=args.compaction_recent_turns,
        auto_compact=args.auto_compact,
        compaction_trigger_fraction=args.compaction_trigger_fraction,
        compaction_target_fraction=args.compaction_target_fraction,
        model_profile=args.model_profile,
        capability_probe=args.capability_probe,
        experiment_card_override=args.experiment_card,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
