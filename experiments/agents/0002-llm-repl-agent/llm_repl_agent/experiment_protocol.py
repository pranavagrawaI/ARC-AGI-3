"""Frozen, machine-checkable contracts for the LS20 levels 1-2 experiment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from statistics import median
from typing import Any, Mapping, Sequence
from urllib.request import Request, urlopen

from .agent import SYSTEM_PROMPT
from .evidence import validate_run_evidence


AGENT_ROOT = Path(__file__).resolve().parent.parent
REPO = AGENT_ROOT.parents[2]
EXPERIMENT_FILES = AGENT_ROOT / "experiment"
_RUNTIME_PATH = EXPERIMENT_FILES / "runtime-config.yaml"
_EXPECTED_RUNTIME_SHA256 = "841608744a5c3158f7df251c185636b80b02858c71d86cc607c295bfbdbd3400"
_EXPECTED_PROMPTS = {
    "enabled": {
        "file": "reconstruction-enabled.txt",
        "sha256": "ad5e8fc75ef6d08e04a723f65c4b543c49cdc92589dbd169fb1f5c5d044ba247",
        "bytes": 2848,
    },
    "disabled": {
        "file": "reconstruction-disabled.txt",
        "sha256": "e990ac30f4af49326530e1fc7308d72a8d48a3201a1ac1ce1dd226f10dc2c07b",
        "bytes": 3040,
    },
}


class ExperimentCondition(str, Enum):
    ENABLED = "enabled"
    DISABLED = "disabled"


@dataclass(frozen=True)
class ProviderConfig:
    order: tuple[str, ...]
    only: tuple[str, ...]
    allow_fallbacks: bool
    require_parameters: bool
    quantizations: tuple[str, ...]

    def request_value(self) -> dict[str, Any]:
        return {
            "order": list(self.order),
            "only": list(self.only),
            "allow_fallbacks": self.allow_fallbacks,
            "require_parameters": self.require_parameters,
            "quantizations": list(self.quantizations),
        }


@dataclass(frozen=True)
class RequestConfig:
    temperature: float
    top_p: float
    top_k: int
    presence_penalty: float
    reasoning_enabled: bool
    reasoning_returned: bool
    max_completion_tokens: int


@dataclass(frozen=True)
class RuntimeLimits:
    max_context_turns: int
    max_model_rounds_per_action: int
    max_decision_repairs: int
    max_model_calls: int
    max_python_output_chars: int
    workspace_timeout_seconds: int
    python_context_policy: str
    retrospective: str


@dataclass(frozen=True)
class ActionBudgets:
    level_1_actions: int
    level_2_actions_after_transition: int
    total_actions_invariant: int


@dataclass(frozen=True)
class FrozenRuntime:
    schema: str
    endpoint: str
    model: str
    model_profile: str
    provider: ProviderConfig
    request: RequestConfig
    runtime: RuntimeLimits
    budgets: ActionBudgets


@dataclass(frozen=True)
class PairSchedule:
    pair_id: str
    environment_seed: int
    requested_seed: int
    color: str
    order: tuple[str, str]
    included: bool


@dataclass(frozen=True)
class RunOutcome:
    levels_completed: int
    level_1_completion_action: int | None
    level_2_completion_action: int | None


@dataclass(frozen=True)
class PairTelemetry:
    actions: int
    stop_reason: str
    reported_tokens: int | None
    model_rounds: tuple[int, ...]


@dataclass(frozen=True)
class InterfaceDamageAssessment:
    automatic_damage: bool
    annotated_damage: bool | None
    severe_interface_damage: bool | None
    criterion_1: bool
    criterion_2: bool


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "true":
        return True
    if value == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [] if not inner else [_parse_scalar(item) for item in inner.split(",")]
    try:
        return float(value) if "." in value else int(value)
    except ValueError:
        return value


def _parse_frozen_yaml(payload: bytes) -> dict[str, Any]:
    """Parse the deliberately tiny two-level YAML subset used by the frozen file."""

    result: dict[str, Any] = {}
    section: dict[str, Any] | None = None
    for line_number, raw_line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not raw_line or raw_line.lstrip().startswith("#"):
            continue
        indentation = len(raw_line) - len(raw_line.lstrip(" "))
        if indentation not in {0, 2} or ":" not in raw_line:
            raise ValueError(f"unsupported frozen YAML syntax on line {line_number}")
        key, raw_value = raw_line.strip().split(":", 1)
        target = section if indentation and section is not None else result
        if not key or key in target:
            raise ValueError(f"duplicate or blank key on line {line_number}: {key!r}")
        if indentation == 0:
            if raw_value.strip():
                result[key] = _parse_scalar(raw_value)
                section = None
            else:
                child: dict[str, Any] = {}
                result[key] = child
                section = child
        else:
            if section is None or not raw_value.strip():
                raise ValueError(f"orphan or nested value on line {line_number}")
            section[key] = _parse_scalar(raw_value)
    return result


def _require_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys differ from frozen schema: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def load_frozen_runtime(path: Path = _RUNTIME_PATH) -> FrozenRuntime:
    payload = path.read_bytes()
    if path == _RUNTIME_PATH and sha256(payload).hexdigest() != _EXPECTED_RUNTIME_SHA256:
        raise ValueError("participating runtime-config.yaml differs from its frozen hash")
    raw = _parse_frozen_yaml(payload)
    _require_keys(
        raw,
        {"schema", "endpoint", "model", "model_profile", "provider", "request", "runtime", "budgets"},
        "runtime",
    )
    provider = raw["provider"]
    request = raw["request"]
    runtime = raw["runtime"]
    budgets = raw["budgets"]
    for label, value in (("provider", provider), ("request", request), ("runtime", runtime), ("budgets", budgets)):
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} must be an object")
    _require_keys(provider, {"order", "only", "allow_fallbacks", "require_parameters", "quantizations"}, "provider")
    _require_keys(
        request,
        {"temperature", "top_p", "top_k", "presence_penalty", "reasoning_enabled", "reasoning_returned", "max_completion_tokens"},
        "request",
    )
    _require_keys(
        runtime,
        {"max_context_turns", "max_model_rounds_per_action", "max_decision_repairs", "max_model_calls", "max_python_output_chars", "workspace_timeout_seconds", "python_context_policy", "retrospective"},
        "runtime limits",
    )
    _require_keys(budgets, {"level_1_actions", "level_2_actions_after_transition", "total_actions_invariant"}, "budgets")
    config = FrozenRuntime(
        schema=str(raw["schema"]),
        endpoint=str(raw["endpoint"]),
        model=str(raw["model"]),
        model_profile=str(raw["model_profile"]),
        provider=ProviderConfig(
            order=tuple(str(item) for item in provider["order"]),
            only=tuple(str(item) for item in provider["only"]),
            allow_fallbacks=bool(provider["allow_fallbacks"]),
            require_parameters=bool(provider["require_parameters"]),
            quantizations=tuple(str(item) for item in provider["quantizations"]),
        ),
        request=RequestConfig(**request),
        runtime=RuntimeLimits(**runtime),
        budgets=ActionBudgets(**budgets),
    )
    if config.budgets.level_1_actions + config.budgets.level_2_actions_after_transition != config.budgets.total_actions_invariant:
        raise ValueError("total action invariant does not equal the two level budgets")
    return config


def frozen_request_options(config: FrozenRuntime | None = None) -> dict[str, Any]:
    config = config or load_frozen_runtime()
    return {
        "temperature": config.request.temperature,
        # Chutes advertises and routes this semantic cap under max_tokens.
        # Sending OpenRouter's max_completion_tokens alias yields a pre-dispatch 404.
        "max_tokens": config.request.max_completion_tokens,
        "max_completion_tokens": None,
        "top_p": config.request.top_p,
        "presence_penalty": config.request.presence_penalty,
        "extra_body": {
            "top_k": config.request.top_k,
            "reasoning": {
                "enabled": config.request.reasoning_enabled,
                "exclude": not config.request.reasoning_returned,
            },
            "provider": config.provider.request_value(),
        },
    }


def _condition_block(condition: ExperimentCondition) -> bytes:
    record = _EXPECTED_PROMPTS[condition.value]
    payload = (EXPERIMENT_FILES / record["file"]).read_bytes()
    if len(payload) != record["bytes"] or sha256(payload).hexdigest() != record["sha256"]:
        raise ValueError(f"{condition.value} prompt block differs from its frozen commitment")
    return payload


def compose_condition_prompt(condition: ExperimentCondition | str) -> str:
    condition = ExperimentCondition(condition)
    block = _condition_block(condition).decode("utf-8")
    return SYSTEM_PROMPT.rstrip() + "\n\n" + block


def experiment_artifact_templates(
    condition: ExperimentCondition | str,
    *,
    evidence_hashes: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    """Return frozen post-game annotation/candidate scaffolds for one run."""

    condition = ExperimentCondition(condition)
    terminal_statuses = (
        [
            "demonstrated",
            "activated_but_not_demonstrated",
            "not_activated",
            "theater_or_unidentifiable",
        ]
        if condition is ExperimentCondition.ENABLED
        else ["compliant", "violated", "unidentifiable"]
    )
    common = {
        "condition": condition.value,
        "evidence_hashes": dict(sorted(evidence_hashes.items())),
        "excluded_from_annotators": [
            "final_score",
            "paired_run_outcome",
            "intervention_responses",
            "game_source",
            "other_annotator_label",
        ],
    }
    annotation = {
        "schema": "scientist-game-designer-player-run-annotation-v1",
        **common,
        "authority": "exact_independent_agreement_without_adjudication",
        "terminal_status_values": terminal_statuses,
        "annotators": [
            {
                "annotator_id": "A",
                "terminal_status": None,
                "criterion_3_interface_damage": None,
                "evidence_event_indices": [],
                "notes": None,
            },
            {
                "annotator_id": "B",
                "terminal_status": None,
                "criterion_3_interface_damage": None,
                "evidence_event_indices": [],
                "notes": None,
            },
        ],
        "agreed_terminal_status": None,
        "agreed_criterion_3_interface_damage": None,
    }
    candidate = {
        "schema": "scientist-game-designer-player-candidate-workflow-v1",
        **common,
        "selector": {
            "selection_rule": "earliest_eligible_candidate",
            "candidate_id": None,
            "run_id": None,
            "selected_process_event_index": None,
            "selected_tool_call_id": None,
            "source_request_payload_archive": None,
            "evidence_event_indices": [],
            "selected_result_content_path": None,
            "selected_result_content": None,
            "grounded_perturbation": None,
            "rationale": None,
            "original_signature": None,
            "expected_changed_signature": None,
            "source_request_payload_sha256": None,
        },
        "auditor": {
            "status": None,
            "allowed_statuses": ["accepted", "rejected"],
            "evidence_event_indices": [],
            "reason": None,
            "source_reconstruction_matches_selected_content": None,
            "rejection_ends_intervention": True,
        },
        "second_candidate_permitted": False,
    }
    return {"annotation": annotation, "candidate": candidate}


def resolve_run_annotation(
    annotation: Mapping[str, Any],
    *,
    condition: ExperimentCondition | str,
    evidence_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Validate two blind labels and derive the authoritative no-adjudication result."""

    condition = ExperimentCondition(condition)
    expected_statuses = (
        {
            "demonstrated",
            "activated_but_not_demonstrated",
            "not_activated",
            "theater_or_unidentifiable",
        }
        if condition is ExperimentCondition.ENABLED
        else {"compliant", "violated", "unidentifiable"}
    )
    if annotation.get("schema") != "scientist-game-designer-player-run-annotation-v1":
        raise ValueError("annotation schema mismatch")
    if annotation.get("condition") != condition.value:
        raise ValueError("annotation condition mismatch")
    if annotation.get("evidence_hashes") != dict(sorted(evidence_hashes.items())):
        raise ValueError("annotation does not bind the exact immutable run evidence")
    rows = annotation.get("annotators")
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("annotation requires exactly two independent labels")
    by_id = {
        row.get("annotator_id"): row
        for row in rows
        if isinstance(row, Mapping) and row.get("annotator_id") in {"A", "B"}
    }
    if set(by_id) != {"A", "B"}:
        raise ValueError("annotation requires unique annotator IDs A and B")
    statuses: list[str] = []
    damage_labels: list[bool] = []
    for annotator_id in ("A", "B"):
        row = by_id[annotator_id]
        status = row.get("terminal_status")
        damage = row.get("criterion_3_interface_damage")
        indices = row.get("evidence_event_indices")
        if status not in expected_statuses:
            raise ValueError(f"annotator {annotator_id} has an invalid terminal status")
        if not isinstance(damage, bool):
            raise ValueError(f"annotator {annotator_id} must label criterion 3 true or false")
        if not isinstance(indices, list) or any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0
            for index in indices
        ):
            raise ValueError(f"annotator {annotator_id} has invalid evidence indices")
        statuses.append(str(status))
        damage_labels.append(damage)
    terminal_status = (
        statuses[0]
        if statuses[0] == statuses[1]
        else (
            "theater_or_unidentifiable"
            if condition is ExperimentCondition.ENABLED
            else "unidentifiable"
        )
    )
    agreed_damage = damage_labels[0] if damage_labels[0] == damage_labels[1] else None
    return {
        "terminal_status": terminal_status,
        "criterion_3_interface_damage": agreed_damage,
        "terminal_status_exact_agreement": statuses[0] == statuses[1],
        "criterion_3_exact_agreement": damage_labels[0] == damage_labels[1],
    }


def validate_prompt_pair(enabled: bytes, disabled: bytes) -> dict[str, Any]:
    enabled_hash = sha256(enabled).hexdigest()
    disabled_hash = sha256(disabled).hexdigest()
    valid = (
        enabled_hash == _EXPECTED_PROMPTS["enabled"]["sha256"]
        and disabled_hash == _EXPECTED_PROMPTS["disabled"]["sha256"]
        and len(enabled) == _EXPECTED_PROMPTS["enabled"]["bytes"]
        and len(disabled) == _EXPECTED_PROMPTS["disabled"]["bytes"]
    )
    return {
        "valid": valid,
        "enabled_sha256": enabled_hash,
        "disabled_sha256": disabled_hash,
        "enabled_bytes": len(enabled),
        "disabled_bytes": len(disabled),
        "authorized_diff_sha256": sha256(enabled + b"\0" + disabled).hexdigest(),
    }


def frozen_pair_schedule() -> tuple[PairSchedule, ...]:
    return (
        PairSchedule("P", 0, 1000, "identity", ("disabled", "enabled"), False),
        PairSchedule("1", 0, 0, "identity", ("enabled", "disabled"), True),
        PairSchedule("2", 0, 1, "identity", ("disabled", "enabled"), True),
        PairSchedule("3", 0, 2, "identity", ("enabled", "disabled"), True),
    )


def validate_pair_commitments(enabled: Mapping[str, Any], disabled: Mapping[str, Any]) -> None:
    required_fields = {
        "schema",
        "generation",
        "pair_id",
        "included",
        "model",
        "endpoint",
        "model_profile",
        "runtime_sha256",
        "source_hashes",
        "endpoint_metadata_sha256",
        "environment_seed",
        "requested_seed",
        "color_mapping",
        "pair_order",
        "authorized_prompt_diff_sha256",
        "condition",
        "condition_label",
        "prompt_sha256",
        "condition_block_sha256",
    }
    condition_fields = {
        "condition",
        "condition_label",
        "prompt_sha256",
        "condition_block_sha256",
    }
    shared_diff_field = "authorized_prompt_diff_sha256"
    all_keys = set(enabled) | set(disabled)
    if set(enabled) != required_fields or set(disabled) != required_fields:
        raise ValueError("pair commitment fields differ")
    for key in sorted(all_keys - condition_fields):
        if enabled[key] != disabled[key]:
            raise ValueError(f"unauthorized within-pair difference: {key}")
    if enabled.get(shared_diff_field) != disabled.get(shared_diff_field):
        raise ValueError("authorized prompt diff commitment differs within pair")
    if enabled.get("condition") != "enabled" or disabled.get("condition") != "disabled":
        raise ValueError("pair commitments have wrong assigned conditions")
    if enabled.get("prompt_sha256") == disabled.get("prompt_sha256"):
        raise ValueError("condition prompts unexpectedly have the same hash")

    config = load_frozen_runtime()
    pair_id = enabled.get("pair_id")
    try:
        schedule = next(row for row in frozen_pair_schedule() if row.pair_id == pair_id)
    except StopIteration as error:
        raise ValueError(f"unknown pair_id: {pair_id}") from error
    prompt_report = validate_prompt_pair(
        (EXPERIMENT_FILES / "reconstruction-enabled.txt").read_bytes(),
        (EXPERIMENT_FILES / "reconstruction-disabled.txt").read_bytes(),
    )
    expected_shared = {
        "schema": "scientist-game-designer-player-pair-v1",
        "pair_id": schedule.pair_id,
        "included": schedule.included,
        "model": config.model,
        "endpoint": config.endpoint,
        "model_profile": config.model_profile,
        "runtime_sha256": sha256(
            (EXPERIMENT_FILES / "runtime-config.yaml").read_bytes()
        ).hexdigest(),
        "source_hashes": participating_source_hashes(),
        "environment_seed": schedule.environment_seed,
        "requested_seed": schedule.requested_seed,
        "color_mapping": list(range(16)),
        "pair_order": list(schedule.order),
        "authorized_prompt_diff_sha256": prompt_report["authorized_diff_sha256"],
    }
    for key, expected in expected_shared.items():
        if enabled.get(key) != expected:
            raise ValueError(f"pair commitment does not match frozen {key}")
    if not isinstance(enabled.get("generation"), str) or not enabled["generation"].strip():
        raise ValueError("pair commitment generation must not be blank")
    endpoint_hash = enabled.get("endpoint_metadata_sha256")
    if (
        not isinstance(endpoint_hash, str)
        or len(endpoint_hash) != 64
        or any(character not in "0123456789abcdef" for character in endpoint_hash)
    ):
        raise ValueError("endpoint_metadata_sha256 must be a lowercase SHA-256 digest")
    for condition, member in (("enabled", enabled), ("disabled", disabled)):
        block_path = EXPERIMENT_FILES / f"reconstruction-{condition}.txt"
        expected_condition = {
            "condition": condition,
            "condition_label": f"reconstruction-{condition}",
            "prompt_sha256": sha256(
                compose_condition_prompt(ExperimentCondition(condition)).encode("utf-8")
            ).hexdigest(),
            "condition_block_sha256": sha256(block_path.read_bytes()).hexdigest(),
        }
        for key, expected in expected_condition.items():
            if member.get(key) != expected:
                raise ValueError(f"pair commitment does not match frozen {key}")


def gameplay_attempt_validity(
    manifest: Mapping[str, Any],
    commitment: Mapping[str, Any] | None = None,
    *,
    run_dir: Path | None = None,
) -> str:
    """Recompute whether a run is valid evidence under the frozen pair contract."""

    status = manifest.get("status")
    stop_reason = manifest.get("stop_reason")
    valid_stop_reasons = {
        "terminal",
        "no_available_actions",
        "level_limit",
        "level_action_budget",
        "action_budget",
        "model_budget_exhausted",
        "unrepaired_invalid_decision",
    }
    if status not in {"complete", "budget_exhausted"} or stop_reason not in valid_stop_reasons:
        return "infrastructure_invalid"
    if commitment is None:
        return "valid_contestant_result"
    experiment = manifest.get("experiment")
    telemetry_fields = (
        "level_action_counts",
        "completion_action_counts",
        "reset_count",
        "game_over_count",
        "model_calls",
        "python_calls",
        "model_rounds_per_action",
        "decision_repairs",
        "host_score_components",
        "usage",
        "provider_usage",
        "evidence_hashes",
    )
    matches = (
        isinstance(experiment, Mapping)
        and manifest.get("model") == commitment.get("model")
        and manifest.get("endpoint") == commitment.get("endpoint")
        and manifest.get("model_profile") == commitment.get("model_profile")
        and manifest.get("env_seed") == commitment.get("environment_seed")
        and manifest.get("model_seed") == commitment.get("requested_seed")
        and manifest.get("color_mapping") == commitment.get("color_mapping")
        and manifest.get("policy_prompt_sha256") == commitment.get("prompt_sha256")
        and experiment.get("generation") == commitment.get("generation")
        and experiment.get("pair_id") == commitment.get("pair_id")
        and experiment.get("included") == commitment.get("included")
        and experiment.get("condition") == commitment.get("condition")
        and experiment.get("pair_order") == commitment.get("pair_order")
        and experiment.get("runtime_config_sha256") == commitment.get("runtime_sha256")
        and experiment.get("condition_block_sha256")
        == commitment.get("condition_block_sha256")
        and experiment.get("prompt_sha256") == commitment.get("prompt_sha256")
        and experiment.get("source_hashes") == commitment.get("source_hashes")
        and experiment.get("endpoint_metadata_sha256")
        == commitment.get("endpoint_metadata_sha256")
        and all(field in manifest for field in telemetry_fields)
        and run_dir is not None
        and isinstance(manifest.get("evidence_hashes"), Mapping)
        and validate_run_evidence(run_dir, manifest["evidence_hashes"])
    )
    return "valid_contestant_result" if matches else "infrastructure_invalid"


def _endpoint_rows(response: Mapping[str, Any]) -> tuple[str | None, Sequence[Mapping[str, Any]]]:
    data = response.get("data")
    if isinstance(data, Mapping):
        model_id = data.get("id") or data.get("model")
        rows = data.get("endpoints")
    else:
        model_id = response.get("id") or response.get("model")
        rows = data if isinstance(data, list) else response.get("endpoints")
    if not isinstance(rows, list):
        raise ValueError("endpoint metadata response has no endpoint list")
    return str(model_id) if model_id is not None else None, tuple(
        row for row in rows if isinstance(row, Mapping)
    )


def canonical_endpoint_metadata(
    response: Mapping[str, Any], *, model: str, tag: str
) -> dict[str, Any]:
    returned_model, rows = _endpoint_rows(response)
    matches = [row for row in rows if row.get("tag") == tag or row.get("name") == tag]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one endpoint tagged {tag!r}, found {len(matches)}")
    row = matches[0]
    quantization = row.get("quantization")
    supported = row.get("supported_parameters")
    if not isinstance(supported, list):
        raise ValueError("target endpoint has no supported_parameters list")
    if returned_model not in {None, model}:
        raise ValueError(f"endpoint metadata returned wrong model: {returned_model}")
    context_limit = row.get("context_length", row.get("context_length_tokens"))
    revision = next(
        (
            row.get(key)
            for key in (
                "revision",
                "model_revision",
            )
            if row.get(key) is not None
        ),
        None,
    )
    endpoint_id = row.get("endpoint_id", row.get("id"))
    endpoint_name = row.get("name")
    revision_identifier = revision or endpoint_id or endpoint_name
    revision_identifier_source = (
        "revision" if revision is not None
        else "endpoint_id" if endpoint_id is not None
        else "endpoint_name"
    )
    return {
        "schema": "openrouter-endpoint-allowlist-v1",
        "model": model,
        "provider_slug": tag,
        "provider_name": row.get("provider_name"),
        "variant": row.get("variant", row.get("tag", tag).split("/", 1)[-1]),
        "quantization": quantization,
        "context_limit": context_limit,
        "supported_parameters": sorted(str(item) for item in supported),
        "revision": revision,
        "endpoint_id": endpoint_id,
        "endpoint_name": endpoint_name,
        "endpoint_updated_at": row.get("updated_at"),
        "revision_identifier": revision_identifier,
        "revision_identifier_source": revision_identifier_source,
    }


def endpoint_metadata_sha256(metadata: Mapping[str, Any]) -> str:
    return sha256(json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def participating_source_hashes() -> dict[str, str]:
    """Hash every frozen source, contract, and gate test used by this protocol."""

    planning_root = REPO / ".scratch" / "scientist-game-designer-player"
    paths = {
        AGENT_ROOT / "run.py",
        AGENT_ROOT / "probe_model.py",
        AGENT_ROOT / "release_gate.py",
        AGENT_ROOT / "run_experiment_pair.py",
        AGENT_ROOT / "run_shadow_intervention.py",
        AGENT_ROOT / "freeze_generation.py",
        AGENT_ROOT / "analyze_generation.py",
        AGENT_ROOT / "generate_prompt_token_counts.py",
        REPO / "pyproject.toml",
        REPO / "uv.lock",
        REPO / "experiments" / "harness" / "runner.py",
        REPO / "experiments" / "harness" / "trace.py",
        REPO / "experiments" / "harness" / "types.py",
        REPO / "experiments" / "harness" / "tests" / "test_harness.py",
        REPO / "experiments" / "agents" / "tests" / "test_llm_repl_agent.py",
        REPO / "experiments" / "agents" / "tests" / "test_falsifiable_experiment.py",
        REPO / "environment_files" / "ls20" / "9607627b" / "ls20.py",
        REPO / "environment_files" / "ls20" / "9607627b" / "metadata.json",
        planning_root / "falsifiable-experiment-specification.md",
    }
    paths.update((AGENT_ROOT / "llm_repl_agent").glob("*.py"))
    paths.update(EXPERIMENT_FILES.glob("*"))
    paths.update((planning_root / "falsifiable-experiment").glob("*"))
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise ValueError("participating source files are missing: " + ", ".join(missing))
    return {
        path.relative_to(REPO).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def fetch_endpoint_metadata(
    config: FrozenRuntime | None = None, *, timeout_seconds: float = 30.0
) -> dict[str, Any]:
    config = config or load_frozen_runtime()
    endpoint = config.endpoint.rstrip("/")
    request = Request(
        f"{endpoint}/models/{config.model}/endpoints",
        headers={"Accept": "application/json"},
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("endpoint metadata API returned a non-object")
    return value


def validate_endpoint_metadata(
    metadata: Mapping[str, Any], config: FrozenRuntime | None = None
) -> None:
    config = config or load_frozen_runtime()
    required_parameters = {
        "reasoning",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "presence_penalty",
        "seed",
        "tools",
        "tool_choice",
    }
    supported = metadata.get("supported_parameters")
    revision_identifier = metadata.get("revision_identifier")
    revision_source = metadata.get("revision_identifier_source")
    dated_name = isinstance(revision_identifier, str) and bool(
        re.search(r"-[0-9]{8}$", revision_identifier)
    )
    checks = {
        "model": metadata.get("model") == config.model,
        "provider_slug": metadata.get("provider_slug") == config.provider.only[0],
        "quantization": metadata.get("quantization") in config.provider.quantizations,
        "context_limit": isinstance(metadata.get("context_limit"), int),
        "revision": (
            metadata.get("revision") is None
            or isinstance(metadata.get("revision"), str)
        ),
        "endpoint_identity": bool(
            metadata.get("endpoint_id")
            or metadata.get("endpoint_name")
            or metadata.get("revision")
        ),
        "revision_identifier": (
            isinstance(revision_identifier, str)
            and bool(revision_identifier.strip())
            and revision_source in {"revision", "endpoint_id", "endpoint_name"}
            and (revision_source != "endpoint_name" or dated_name)
        ),
        "supported_parameters": (
            isinstance(supported, list)
            and required_parameters.issubset(str(item) for item in supported)
        ),
    }
    failures = sorted(key for key, passed in checks.items() if not passed)
    if failures:
        raise ValueError("endpoint metadata violates frozen allowlist: " + ", ".join(failures))


def assess_pair(enabled: RunOutcome, disabled: RunOutcome) -> str:
    if enabled.levels_completed != disabled.levels_completed:
        return "enabled" if enabled.levels_completed > disabled.levels_completed else "disabled"
    if enabled.levels_completed == 0:
        return "tie"
    if enabled.levels_completed == 1:
        left = enabled.level_1_completion_action
        right = disabled.level_1_completion_action
    else:
        left = enabled.level_2_completion_action
        right = disabled.level_2_completion_action
    if left == right:
        return "tie"
    if left is None or right is None:
        raise ValueError("completed run lacks its completion action")
    return "enabled" if left < right else "disabled"


def assess_interface_damage(
    disabled: PairTelemetry,
    enabled: PairTelemetry,
    *,
    annotated_damage: bool | None,
) -> InterfaceDamageAssessment:
    model_ceiling = {"model_budget_exhausted"}
    pre_action_interface_stop = model_ceiling | {"unrepaired_invalid_decision"}
    criterion_1 = (
        disabled.actions == 0
        and disabled.stop_reason in pre_action_interface_stop
        and enabled.actions >= 4
    )
    criterion_2 = False
    if (
        disabled.actions >= 1
        and disabled.stop_reason in model_ceiling
        and enabled.actions >= 4
        and disabled.actions * 2 < enabled.actions
    ):
        rounds_limb = bool(disabled.model_rounds and enabled.model_rounds) and (
            median(disabled.model_rounds) >= 2 * median(enabled.model_rounds)
        )
        tokens_limb = (
            disabled.reported_tokens is not None
            and enabled.reported_tokens is not None
            and disabled.reported_tokens / disabled.actions
            >= 2 * (enabled.reported_tokens / enabled.actions)
        )
        criterion_2 = rounds_limb or tokens_limb
    automatic = criterion_1 or criterion_2
    severe = True if automatic else annotated_damage
    return InterfaceDamageAssessment(
        automatic_damage=automatic,
        annotated_damage=annotated_damage,
        severe_interface_damage=severe,
        criterion_1=criterion_1,
        criterion_2=criterion_2,
    )


def assess_component_profile(
    *,
    enabled_statuses: Sequence[str],
    clean_disabled_runs: Sequence[bool],
    intervention_passes: Sequence[bool],
    pair_signs: Sequence[str],
) -> dict[str, bool]:
    if not all(len(values) == 3 for values in (enabled_statuses, clean_disabled_runs, intervention_passes, pair_signs)):
        raise ValueError("component calculation requires exactly three valid included pairs")
    activation = sum(status in {"demonstrated", "activated_but_not_demonstrated"} for status in enabled_statuses) >= 2
    clean_contrast = sum(clean_disabled_runs) >= 2
    local = any(intervention_passes)
    gameplay = sum(sign == "enabled" for sign in pair_signs) >= 2
    return {
        "activation": activation,
        "clean_contrast": clean_contrast,
        "local_content_dependence": local,
        "preliminary_gameplay_benefit": gameplay,
        "mechanism_consistent_preliminary_gameplay_benefit": activation and clean_contrast and local and gameplay,
    }
