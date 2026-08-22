"""Frozen, one-factor decision-point shadows over committed model requests.

The evaluator restores the credential-free request captured immediately before
one model call.  It produces an intact control and one precommitted content
perturbation, then samples both without executing any returned tool or touching
the game.  Its claim therefore ends at immediate response dependence.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import random
import shutil
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from nooa.unifiedllm import LLMResponse, UnifiedLLM
from nooa.unifiedllm.retry_config import RetryConfig

from tycho_nooa.decision_checkpoints import _replay_context
from tycho_nooa.llm import build_llm

SPEC_SCHEMA = "tycho-nooa-shadow-perturbation-v1"
PACKET_SCHEMA = "tycho-nooa-shadow-packet-v1"
REPORT_SCHEMA = "tycho-nooa-shadow-report-v1"
REQUEST_BOUNDARY = "nooa.llm_call"
SUPPORTED_EFFECTS = {
    "semantic_response",
    "content",
    "first_tool_call",
    "take_action",
}


class ShadowValidationError(ValueError):
    """The source checkpoint, perturbation, or prepared packet is not exact."""


@dataclass(frozen=True)
class CapturedTool:
    """Read-only tool schema sufficient for a single model response."""

    name: str
    description: str
    parameters: dict[str, Any]

    def get_parameter_schema(self, *, strict: bool = False) -> dict[str, Any]:
        del strict
        return copy.deepcopy(self.parameters)


@dataclass(frozen=True)
class PreparedShadow:
    path: Path
    manifest: dict[str, Any]
    control: dict[str, Any]
    treatment: dict[str, Any]
    perturbation: dict[str, Any]
    source_response: dict[str, Any]


def inspect_checkpoint(checkpoint_root: Path, *, call_index: int) -> dict[str, Any]:
    """Expose message pointers and commitments needed to precommit one treatment."""

    source = _load_checkpoint(checkpoint_root, call_index)
    request = source["request"]
    body = request["request_body"]
    return {
        "schema": SPEC_SCHEMA,
        "call_index": call_index,
        "source": request.get("source"),
        "decision_point": request.get("decision_point"),
        "source_request_payload_commitment": request["request_payload_commitment"],
        "model": body["model"],
        "message_text_targets": [
            {
                "pointer": pointer,
                "target_value_commitment": _commitment(value),
                "value": value,
            }
            for pointer, value in _message_text_leaves(body["messages"])
        ],
    }


def prepare_shadow_packet(
    checkpoint_root: Path,
    *,
    call_index: int,
    perturbation: Mapping[str, Any],
    destination: Path,
) -> dict[str, Any]:
    """Create one self-contained, immutable input packet before model spend."""

    if destination.exists():
        raise FileExistsError(destination)
    source = _load_checkpoint(checkpoint_root, call_index)
    spec = _validate_perturbation(dict(perturbation), source["request"])
    control = copy.deepcopy(source["request"]["request_body"])
    treatment = _apply_operation(control, spec["operation"])
    changed = _different_leaf_paths(control, treatment)
    pointer = spec["operation"]["pointer"]
    if changed != [pointer]:
        raise ShadowValidationError(
            f"one-factor isolation failed: changed paths are {changed!r}"
        )

    manifest = {
        "schema": PACKET_SCHEMA,
        "source_checkpoint": str(checkpoint_root.resolve()),
        "call_index": call_index,
        "source": source["request"].get("source"),
        "decision_point": source["request"].get("decision_point"),
        "source_request_payload_commitment": _commitment(control),
        "source_workspace_commitment": _commitment(source["workspace"]),
        "source_response_commitment": _commitment(source["response"]),
        "perturbation_commitment": _commitment(spec),
        "control_request_commitment": _commitment(control),
        "treatment_request_commitment": _commitment(treatment),
        "changed_paths": changed,
        "expected_effect": spec["expected_effect"],
        "model": control["model"],
        "replay_context": source["request"]["replay_context"],
        "claim_ceiling": (
            "immediate response dependence at this committed model-call "
            "checkpoint; no tool execution, gameplay consequence, or transfer"
        ),
    }

    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    try:
        _write_json(temporary / "manifest.json", manifest)
        _write_json(temporary / "perturbation.json", spec)
        _write_json(temporary / "control-request.json", control)
        _write_json(temporary / "treatment-request.json", treatment)
        source_dir = temporary / "source"
        source_dir.mkdir()
        for name in ("request.json", "response.json", "workspace.json"):
            shutil.copy2(source["call_dir"] / name, source_dir / name)
        _copy_blobs(checkpoint_root, source["workspace"], temporary)
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    validate_shadow_packet(destination)
    return manifest


def validate_shadow_packet(path: Path) -> PreparedShadow:
    """Recheck every commitment and the one-factor transform before execution."""

    manifest = _read_object(path / "manifest.json")
    if manifest.get("schema") != PACKET_SCHEMA:
        raise ShadowValidationError("unsupported shadow packet schema")
    spec = _read_object(path / "perturbation.json")
    control = _read_object(path / "control-request.json")
    treatment = _read_object(path / "treatment-request.json")
    source_request = _read_object(path / "source" / "request.json")
    source_response = _read_object(path / "source" / "response.json")
    workspace = _read_object(path / "source" / "workspace.json")

    checks = {
        "source_request_payload_commitment": _commitment(control),
        "source_workspace_commitment": _commitment(workspace),
        "source_response_commitment": _commitment(source_response),
        "perturbation_commitment": _commitment(spec),
        "control_request_commitment": _commitment(control),
        "treatment_request_commitment": _commitment(treatment),
    }
    for field, actual in checks.items():
        if manifest.get(field) != actual:
            raise ShadowValidationError(f"stale packet commitment: {field}")
    if source_request.get("request_payload_commitment") != _commitment(control):
        raise ShadowValidationError("control differs from the committed source request")
    if source_request.get("request_body") != control:
        raise ShadowValidationError("control differs from source/request.json")
    if source_request.get("workspace_commitment") != _commitment(workspace):
        raise ShadowValidationError("source workspace commitment is stale")
    _validate_perturbation(spec, source_request)
    if _apply_operation(control, spec["operation"]) != treatment:
        raise ShadowValidationError("treatment does not follow the committed operation")
    changed = _different_leaf_paths(control, treatment)
    if changed != manifest.get("changed_paths") or len(changed) != 1:
        raise ShadowValidationError(
            "packet no longer contains exactly one changed leaf"
        )
    manifest_fields = {
        "call_index": source_request.get("call_index"),
        "source": source_request.get("source"),
        "decision_point": source_request.get("decision_point"),
        "expected_effect": spec["expected_effect"],
        "model": control["model"],
        "replay_context": source_request.get("replay_context"),
    }
    for field, expected in manifest_fields.items():
        if manifest.get(field) != expected:
            raise ShadowValidationError(f"packet manifest drift: {field}")
    _verify_packet_blobs(path, workspace)
    _validate_request_body(control)
    _validate_request_body(treatment)
    return PreparedShadow(
        path=path,
        manifest=manifest,
        control=control,
        treatment=treatment,
        perturbation=spec,
        source_response=source_response,
    )


async def execute_shadow(
    packet_path: Path,
    *,
    llm: UnifiedLLM,
    repeats: int,
    max_calls: int,
    schedule_seed: int,
    max_cost_usd: float | None = None,
    cost_reserve_per_call_usd: float | None = None,
    record_call: Callable[[dict[str, Any]], None] | None = None,
    enforce_replay_context: bool = True,
) -> dict[str, Any]:
    """Sample both arms under call and observed-cost stops; never execute tools."""

    packet = validate_shadow_packet(packet_path)
    if repeats < 1:
        raise ShadowValidationError("repeats must be positive")
    planned_calls = repeats * 2
    if planned_calls > max_calls:
        raise ShadowValidationError(
            f"planned calls {planned_calls} exceed hard cap {max_calls}"
        )
    cost_limit_enabled = _validate_cost_limit(
        max_cost_usd,
        cost_reserve_per_call_usd,
    )
    if llm.model != packet.manifest["model"]:
        raise ShadowValidationError(
            f"model drift: packet={packet.manifest['model']!r}, client={llm.model!r}"
        )
    if enforce_replay_context:
        actual_context = _replay_context(llm)
        if actual_context != packet.manifest["replay_context"]:
            raise ShadowValidationError("client or provider configuration drift")

    schedule = [
        {"pair_index": pair, "arm": arm}
        for pair in range(repeats)
        for arm in ("control", "treatment")
    ]
    random.Random(schedule_seed).shuffle(schedule)
    observations: list[dict[str, Any]] = []
    observed_cost_usd = 0.0
    cost_stop_reason: str | None = None
    for order_index, item in enumerate(schedule):
        if (
            cost_limit_enabled
            and observed_cost_usd + cost_reserve_per_call_usd > max_cost_usd
        ):
            cost_stop_reason = "insufficient_reserved_cost"
            break
        body = packet.control if item["arm"] == "control" else packet.treatment
        observation = await _sample_once(
            llm,
            body,
            order_index=order_index,
            pair_index=item["pair_index"],
            arm=item["arm"],
        )
        observations.append(observation)
        if record_call is not None:
            record_call(observation)
        if cost_limit_enabled:
            call_cost = _observation_cost(observation)
            if call_cost is None:
                cost_stop_reason = "missing_observed_cost"
                break
            observed_cost_usd += call_cost
            if observed_cost_usd > max_cost_usd:
                cost_stop_reason = "observed_cost_exceeded"
                break

    assessment = _assess(
        observations,
        effect=packet.perturbation["expected_effect"],
        repeats=repeats,
        source_response=packet.source_response,
    )
    return {
        "schema": REPORT_SCHEMA,
        "status": "stopped_cost_limit" if cost_stop_reason else "complete",
        "packet": str(packet_path.resolve()),
        "packet_manifest_commitment": _commitment(packet.manifest),
        "perturbation_id": packet.perturbation["id"],
        "question": packet.perturbation["question"],
        "hypothesis": packet.perturbation["hypothesis"],
        "expected_effect": packet.perturbation["expected_effect"],
        "schedule_seed": schedule_seed,
        "repeats_per_arm": repeats,
        "max_calls": max_calls,
        "executed_calls": len(observations),
        "cost_budget": (
            {
                "max_cost_usd": max_cost_usd,
                "reserve_per_call_usd": cost_reserve_per_call_usd,
                "observed_cost_usd": round(observed_cost_usd, 12),
                "remaining_cost_usd": round(
                    max(0.0, max_cost_usd - observed_cost_usd),
                    12,
                ),
                "calls_withheld": planned_calls - len(observations),
                "stop_reason": cost_stop_reason,
            }
            if cost_limit_enabled
            else None
        ),
        "assessment": assessment,
        "promotion_allowed": assessment["judgment"] == "supported",
        "observations": observations,
        "claim_ceiling": packet.manifest["claim_ceiling"],
    }


async def _sample_once(
    llm: UnifiedLLM,
    body: dict[str, Any],
    *,
    order_index: int,
    pair_index: int,
    arm: str,
) -> dict[str, Any]:
    base = {
        "order_index": order_index,
        "pair_index": pair_index,
        "arm": arm,
        "request_commitment": _commitment(body),
    }
    try:
        response = await llm.acall(
            copy.deepcopy(body["messages"]),
            tools=_captured_tools(body["tools"]),
            output_model=None,
            **copy.deepcopy(body["parameters"]),
        )
    # A provider failure is an experimental observation, not a reason to lose
    # the other arm or silently retry past the precommitted call cap.
    except Exception as error:  # noqa: BLE001
        return {
            **base,
            "status": "error",
            "error_type": type(error).__name__,
            "message": str(error),
        }
    return {**base, "status": "complete", "response": _response_record(response)}


def _assess(
    observations: list[dict[str, Any]],
    *,
    effect: str,
    repeats: int,
    source_response: dict[str, Any],
) -> dict[str, Any]:
    completed = [row for row in observations if row["status"] == "complete"]
    errors = len(observations) - len(completed)
    by_arm = {
        arm: [row for row in completed if row["arm"] == arm]
        for arm in ("control", "treatment")
    }
    signatures = {
        arm: [_effect_signature(row["response"], effect) for row in rows]
        for arm, rows in by_arm.items()
    }
    unique = {arm: set(values) for arm, values in signatures.items()}
    route_signatures = [
        _canonical_json(row["response"]["provider_route"]).decode("utf-8")
        for row in completed
        if row["response"].get("provider_route")
    ]
    missing_routes = len(completed) - len(route_signatures)
    route_identities = set(route_signatures)

    if errors:
        judgment = "inconclusive"
        reason = "one or more sampled calls failed"
    elif any(len(rows) != repeats for rows in by_arm.values()):
        judgment = "inconclusive"
        reason = "the cost limit stopped the matched schedule before completion"
    elif len(route_identities) > 1:
        judgment = "inconclusive"
        reason = "observed provider route drift prevents matched attribution"
    elif repeats < 2:
        judgment = "inconclusive"
        reason = "one pair cannot separate content dependence from model variability"
    elif all(len(values) == 1 for values in unique.values()):
        if unique["control"] != unique["treatment"]:
            judgment = "supported"
            reason = (
                "each arm was internally stable and the expected signature differed"
            )
        else:
            judgment = "contradicted"
            reason = "both arms were internally stable with the same expected signature"
    else:
        judgment = "inconclusive"
        reason = "within-arm variability prevents attribution to the perturbation"

    pairs = []
    for pair_index in range(repeats):
        rows = {row["arm"]: row for row in completed if row["pair_index"] == pair_index}
        if set(rows) != {"control", "treatment"}:
            continue
        pairs.append(
            {
                "pair_index": pair_index,
                "semantic_response_changed": _effect_signature(
                    rows["control"]["response"], "semantic_response"
                )
                != _effect_signature(
                    rows["treatment"]["response"], "semantic_response"
                ),
                "content_changed": _effect_signature(
                    rows["control"]["response"], "content"
                )
                != _effect_signature(rows["treatment"]["response"], "content"),
                "first_tool_call_changed": _effect_signature(
                    rows["control"]["response"], "first_tool_call"
                )
                != _effect_signature(rows["treatment"]["response"], "first_tool_call"),
                "take_action_changed": _effect_signature(
                    rows["control"]["response"], "take_action"
                )
                != _effect_signature(rows["treatment"]["response"], "take_action"),
            }
        )
    source_signature = (
        _effect_signature(_normalize_recorded_response(source_response), effect)
        if source_response.get("status") == "complete"
        else None
    )
    control_matches_source = (
        sum(value == source_signature for value in signatures["control"])
        if source_signature is not None
        else 0
    )
    return {
        "judgment": judgment,
        "reason": reason,
        "error_calls": errors,
        "arm_signature_counts": {
            arm: _signature_counts(values) for arm, values in signatures.items()
        },
        "control_recorded_response_match_rate": (
            control_matches_source / len(signatures["control"])
            if signatures["control"] and source_signature is not None
            else None
        ),
        "paired_change_rates": {
            field: sum(pair[field] for pair in pairs) / len(pairs) if pairs else None
            for field in (
                "semantic_response_changed",
                "content_changed",
                "first_tool_call_changed",
                "take_action_changed",
            )
        },
        "provider_route": {
            "verifiable": bool(completed) and missing_routes == 0,
            "stable": bool(completed)
            and missing_routes == 0
            and len(route_identities) == 1,
            "missing_calls": missing_routes,
            "identity_counts": _signature_counts(route_signatures),
        },
        "usage": {arm: _sum_usage(rows) for arm, rows in by_arm.items()},
    }


def _validate_cost_limit(
    max_cost_usd: float | None,
    reserve_per_call_usd: float | None,
) -> bool:
    if (max_cost_usd is None) != (reserve_per_call_usd is None):
        raise ShadowValidationError(
            "max cost and per-call reservation must be supplied together"
        )
    if max_cost_usd is None:
        raise ShadowValidationError(
            "shadow execution requires an explicit monetary limit and per-call reserve"
        )
    if max_cost_usd <= 0 or reserve_per_call_usd <= 0:
        raise ShadowValidationError("cost limits must be positive")
    if reserve_per_call_usd > max_cost_usd:
        raise ShadowValidationError("per-call reservation exceeds the total cost limit")
    return True


def _observation_cost(observation: dict[str, Any]) -> float | None:
    if observation.get("status") != "complete":
        return None
    usage = observation["response"].get("usage")
    cost = usage.get("cost") if isinstance(usage, dict) else None
    if isinstance(cost, bool) or not isinstance(cost, (int, float)) or cost < 0:
        return None
    return float(cost)


def _effect_signature(response: dict[str, Any], effect: str) -> str:
    tools = response.get("tool_calls", [])
    if effect == "content":
        value: Any = response.get("content")
    elif effect == "first_tool_call":
        value = tools[0] if tools else None
    elif effect == "take_action":
        value = next(
            (tool for tool in tools if tool.get("name") == "take_action"), None
        )
    else:
        value = {
            "content": response.get("content"),
            "finish_reason": response.get("finish_reason"),
            "tool_calls": tools,
        }
    return _canonical_json(value).decode("utf-8")


def _signature_counts(values: list[str]) -> list[dict[str, Any]]:
    return [
        {"signature": signature, "count": count}
        for signature, count in sorted(Counter(values).items())
    ]


def _sum_usage(rows: list[dict[str, Any]]) -> dict[str, int]:
    total: Counter[str] = Counter()
    for row in rows:
        usage = row["response"].get("usage") or {}
        for key, value in usage.items():
            if isinstance(value, int):
                total[key] += value
    return dict(sorted(total.items()))


def _response_record(response: LLMResponse) -> dict[str, Any]:
    return {
        "content": _jsonable(response.content),
        "reasoning": response.reasoning,
        "finish_reason": response.finish_reason,
        "assistant_message": _jsonable(response.assistant_message),
        "usage": _jsonable(response.usage),
        "provider_route": _provider_route(response.raw_response),
        "tool_calls": [
            {
                "name": call.name,
                "arguments": _normalized_arguments(call.arguments),
            }
            for call in response.tool_calls
        ],
    }


def _provider_route(raw_response: Any) -> dict[str, str]:
    """Keep non-secret route identifiers exposed by the provider or LiteLLM."""

    hidden = _raw_field(raw_response, "_hidden_params")
    fields = {
        "reported_provider": _raw_field(raw_response, "provider"),
        "reported_model": _raw_field(raw_response, "model"),
        "system_fingerprint": _raw_field(raw_response, "system_fingerprint"),
        "service_tier": _raw_field(raw_response, "service_tier"),
        "litellm_provider": _raw_field(hidden, "custom_llm_provider"),
        "model_id": _raw_field(hidden, "model_id")
        or _raw_field(raw_response, "model_id"),
    }
    return {
        name: value
        for name, value in fields.items()
        if isinstance(value, str) and value
    }


def _raw_field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _normalize_recorded_response(response: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": response.get("content"),
        "finish_reason": response.get("finish_reason"),
        "tool_calls": [
            {
                "name": call.get("name"),
                "arguments": _normalized_arguments(call.get("arguments", "")),
            }
            for call in response.get("tool_calls", [])
            if isinstance(call, dict)
        ],
    }


def _normalized_arguments(value: Any) -> Any:
    if not isinstance(value, str):
        return _jsonable(value)
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {"malformed_json": value}


def _captured_tools(rows: list[dict[str, Any]]) -> list[CapturedTool]:
    return [
        CapturedTool(
            name=row["name"],
            description=row["description"],
            parameters=copy.deepcopy(row["parameters"]),
        )
        for row in rows
    ]


def _validate_perturbation(
    spec: dict[str, Any], request: dict[str, Any]
) -> dict[str, Any]:
    required = {
        "schema",
        "id",
        "question",
        "hypothesis",
        "expected_effect",
        "invalid_if",
        "source_request_payload_commitment",
        "operation",
    }
    if set(spec) != required:
        raise ShadowValidationError(
            f"perturbation fields must be exactly {sorted(required)!r}"
        )
    if spec["schema"] != SPEC_SCHEMA:
        raise ShadowValidationError("unsupported perturbation schema")
    for field in ("id", "question", "hypothesis", "invalid_if"):
        if not isinstance(spec[field], str) or not spec[field].strip():
            raise ShadowValidationError(f"perturbation {field} must be non-empty text")
    if spec["expected_effect"] not in SUPPORTED_EFFECTS:
        raise ShadowValidationError(
            f"expected_effect must be one of {sorted(SUPPORTED_EFFECTS)!r}"
        )
    committed = request.get("request_payload_commitment")
    if spec["source_request_payload_commitment"] != committed:
        raise ShadowValidationError("perturbation targets a different source request")
    operation = spec["operation"]
    if not isinstance(operation, dict) or set(operation) != {
        "type",
        "pointer",
        "target_value_commitment",
        "old",
        "new",
    }:
        raise ShadowValidationError("operation must be one exact replace_exact_text")
    if operation["type"] != "replace_exact_text":
        raise ShadowValidationError("only replace_exact_text is supported")
    if not isinstance(operation["old"], str) or not isinstance(operation["new"], str):
        raise ShadowValidationError("operation old and new must be text")
    if not operation["old"] or operation["old"] == operation["new"]:
        raise ShadowValidationError(
            "operation must replace non-empty text with a change"
        )
    pointer = operation["pointer"]
    if not isinstance(pointer, str) or not pointer.startswith("/messages/"):
        raise ShadowValidationError("operation pointer must target request messages")
    target = _resolve_pointer(request["request_body"], pointer)
    if not isinstance(target, str):
        raise ShadowValidationError("operation pointer must resolve to text")
    if operation["target_value_commitment"] != _commitment(target):
        raise ShadowValidationError("operation target value commitment is stale")
    if target.count(operation["old"]) != 1:
        raise ShadowValidationError("old text must occur exactly once at the target")
    return spec


def _apply_operation(body: dict[str, Any], operation: dict[str, Any]) -> dict[str, Any]:
    candidate = copy.deepcopy(body)
    parent, key = _resolve_pointer_parent(candidate, operation["pointer"])
    target = parent[key]
    if not isinstance(target, str) or target.count(operation["old"]) != 1:
        raise ShadowValidationError(
            "committed exact-text replacement no longer applies"
        )
    parent[key] = target.replace(operation["old"], operation["new"], 1)
    return candidate


def _resolve_pointer(value: Any, pointer: str) -> Any:
    parent, key = _resolve_pointer_parent(value, pointer)
    return parent[key]


def _resolve_pointer_parent(value: Any, pointer: str) -> tuple[Any, str | int]:
    if not pointer.startswith("/"):
        raise ShadowValidationError("JSON pointer must begin with /")
    parts = [
        part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")
    ]
    current = value
    for part in parts[:-1]:
        try:
            current = current[int(part)] if isinstance(current, list) else current[part]
        except (IndexError, KeyError, TypeError, ValueError) as error:
            raise ShadowValidationError(
                f"JSON pointer does not resolve: {pointer}"
            ) from error
    final: str | int = int(parts[-1]) if isinstance(current, list) else parts[-1]
    try:
        current[final]
    except (IndexError, KeyError, TypeError) as error:
        raise ShadowValidationError(
            f"JSON pointer does not resolve: {pointer}"
        ) from error
    return current, final


def _different_leaf_paths(left: Any, right: Any, pointer: str = "") -> list[str]:
    if type(left) is not type(right):
        return [pointer or "/"]
    if isinstance(left, dict):
        if set(left) != set(right):
            return [pointer or "/"]
        changed: list[str] = []
        for key in sorted(left):
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            changed.extend(
                _different_leaf_paths(left[key], right[key], f"{pointer}/{escaped}")
            )
        return changed
    if isinstance(left, list):
        if len(left) != len(right):
            return [pointer or "/"]
        changed = []
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            changed.extend(_different_leaf_paths(a, b, f"{pointer}/{index}"))
        return changed
    return [] if left == right else [pointer or "/"]


def _message_text_leaves(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    leaves: list[tuple[str, str]] = []

    def visit(value: Any, pointer: str) -> None:
        if isinstance(value, str):
            leaves.append((pointer, value))
        elif isinstance(value, dict):
            for key in sorted(value):
                escaped = str(key).replace("~", "~0").replace("/", "~1")
                visit(value[key], f"{pointer}/{escaped}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{pointer}/{index}")

    visit(messages, "/messages")
    return leaves


def _load_checkpoint(root: Path, call_index: int) -> dict[str, Any]:
    if call_index < 1:
        raise ShadowValidationError("call index must be positive")
    call_dir = root / "calls" / f"{call_index:06d}"
    request = _read_object(call_dir / "request.json")
    response = _read_object(call_dir / "response.json")
    workspace = _read_object(call_dir / "workspace.json")
    body = request.get("request_body")
    _validate_request_body(body)
    if request.get("call_index") != call_index:
        raise ShadowValidationError("checkpoint call index mismatch")
    if request.get("request_boundary") != REQUEST_BOUNDARY:
        raise ShadowValidationError("checkpoint was captured at a different boundary")
    if request.get("source") not in {"actor", "builder"}:
        raise ShadowValidationError("checkpoint source lane is invalid")
    if not isinstance(request.get("replay_context"), dict):
        raise ShadowValidationError("checkpoint replay context is missing")
    if request.get("request_payload_commitment") != _commitment(body):
        raise ShadowValidationError("checkpoint request commitment is stale")
    if request.get("workspace_commitment") != _commitment(workspace):
        raise ShadowValidationError("checkpoint workspace commitment is stale")
    if response.get("status") not in {"complete", "error"}:
        raise ShadowValidationError("checkpoint response status is invalid")
    if body.get("output_model") is not None:
        raise ShadowValidationError(
            "structured output checkpoints are not replayable yet"
        )
    _verify_source_blobs(root, workspace)
    return {
        "call_dir": call_dir,
        "request": request,
        "response": response,
        "workspace": workspace,
    }


def _validate_request_body(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {
        "model",
        "messages",
        "tools",
        "output_model",
        "parameters",
    }:
        raise ShadowValidationError("request body has an unsupported shape")
    if not isinstance(value["model"], str) or not value["model"]:
        raise ShadowValidationError("request model is missing")
    if not isinstance(value["messages"], list) or not all(
        isinstance(row, dict) for row in value["messages"]
    ):
        raise ShadowValidationError("request messages must be objects")
    if not isinstance(value["tools"], list):
        raise ShadowValidationError("request tools must be a list")
    for tool in value["tools"]:
        if not isinstance(tool, dict) or set(tool) != {
            "name",
            "description",
            "parameters",
        }:
            raise ShadowValidationError("captured tool schema is malformed")
    if not isinstance(value["parameters"], dict):
        raise ShadowValidationError("request parameters must be an object")


def _verify_source_blobs(root: Path, workspace: dict[str, Any]) -> None:
    for entry in _workspace_files(workspace):
        blob = root / "blobs" / "sha256" / entry["sha256"]
        _verify_blob(blob, entry)


def _verify_packet_blobs(packet: Path, workspace: dict[str, Any]) -> None:
    for entry in _workspace_files(workspace):
        blob = packet / "blobs" / "sha256" / entry["sha256"]
        _verify_blob(blob, entry)


def _workspace_files(workspace: dict[str, Any]) -> list[dict[str, Any]]:
    files = workspace.get("files")
    if not isinstance(files, list):
        raise ShadowValidationError("workspace files must be a list")
    for entry in files:
        if not isinstance(entry, dict):
            raise ShadowValidationError("workspace file entry must be an object")
        relative = entry.get("path")
        if not _safe_relative(relative):
            raise ShadowValidationError(f"unsafe workspace path: {relative!r}")
        digest = entry.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ShadowValidationError("workspace blob digest is malformed")
        if not isinstance(entry.get("size"), int) or entry["size"] < 0:
            raise ShadowValidationError("workspace blob size is malformed")
    return files


def _safe_relative(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts


def _verify_blob(path: Path, entry: dict[str, Any]) -> None:
    if not path.is_file():
        raise ShadowValidationError(f"checkpoint blob is missing: {entry['sha256']}")
    content = path.read_bytes()
    if len(content) != entry["size"]:
        raise ShadowValidationError(f"checkpoint blob size is stale: {entry['sha256']}")
    if hashlib.sha256(content).hexdigest() != entry["sha256"]:
        raise ShadowValidationError(f"checkpoint blob is corrupt: {entry['sha256']}")


def _copy_blobs(root: Path, workspace: dict[str, Any], destination: Path) -> None:
    for entry in _workspace_files(workspace):
        source = root / "blobs" / "sha256" / entry["sha256"]
        target = destination / "blobs" / "sha256" / entry["sha256"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ShadowValidationError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ShadowValidationError(f"{path} root is not an object")
    return value


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    raise TypeError(f"response value is not JSON-safe: {type(value).__name__}")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _commitment(value: Any) -> str:
    return f"sha256:{hashlib.sha256(_canonical_json(value)).hexdigest()}"


def canonical_commitment(value: Any) -> str:
    """Public helper for binding perturbation specs to exact JSON values."""

    return _commitment(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json(value) + b"\n")


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(_canonical_json(value).decode("utf-8") + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _execution_plan(
    packet: PreparedShadow,
    repeats: int,
    max_calls: int,
    seed: int,
    max_cost_usd: float | None,
    reserve_per_call_usd: float | None,
) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "status": "planned",
        "packet": str(packet.path.resolve()),
        "packet_manifest_commitment": _commitment(packet.manifest),
        "repeats_per_arm": repeats,
        "planned_calls": repeats * 2,
        "max_calls": max_calls,
        "max_cost_usd": max_cost_usd,
        "cost_reserve_per_call_usd": reserve_per_call_usd,
        "schedule_seed": seed,
        "tools_will_execute": False,
        "game_will_be_called": False,
        "provider_retries_will_be_disabled": True,
        "claim_ceiling": packet.manifest["claim_ceiling"],
    }


def _disable_provider_retries(llm: UnifiedLLM) -> None:
    if hasattr(llm, "retry_config"):
        llm.retry_config = RetryConfig(max_retries=0, rate_limit_extra_retries=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser(
        "inspect", help="list committed message targets without model spend"
    )
    inspect.add_argument("checkpoint_root", type=Path)
    inspect.add_argument("--call-index", type=int, required=True)
    prepare = commands.add_parser("prepare", help="create a zero-cost shadow packet")
    prepare.add_argument("checkpoint_root", type=Path)
    prepare.add_argument("destination", type=Path)
    prepare.add_argument("--call-index", type=int, required=True)
    prepare.add_argument("--perturbation", type=Path, required=True)
    run = commands.add_parser("run", help="execute a capped paired shadow")
    run.add_argument("packet", type=Path)
    run.add_argument("output", type=Path)
    run.add_argument("--repeats", type=int, default=2)
    run.add_argument("--max-calls", type=int, required=True)
    run.add_argument("--max-cost-usd", type=float)
    run.add_argument("--cost-reserve-per-call-usd", type=float)
    run.add_argument("--schedule-seed", type=int, default=0)
    run.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if args.command == "inspect":
        result = inspect_checkpoint(
            args.checkpoint_root,
            call_index=args.call_index,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "prepare":
        spec = _read_object(args.perturbation)
        result = prepare_shadow_packet(
            args.checkpoint_root,
            call_index=args.call_index,
            perturbation=spec,
            destination=args.destination,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    packet = validate_shadow_packet(args.packet)
    plan = _execution_plan(
        packet,
        args.repeats,
        args.max_calls,
        args.schedule_seed,
        args.max_cost_usd,
        args.cost_reserve_per_call_usd,
    )
    if args.repeats < 1 or args.repeats * 2 > args.max_calls:
        raise ShadowValidationError("planned calls exceed the explicit hard cap")
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        print("refusing model calls without --execute", flush=True)
        return 2
    if args.max_cost_usd is None or args.cost_reserve_per_call_usd is None:
        raise ShadowValidationError(
            "paid execution requires an explicit monetary limit and per-call reserve"
        )
    _validate_cost_limit(args.max_cost_usd, args.cost_reserve_per_call_usd)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    _write_json(args.output / "execution-plan.json", plan)
    llm, mode = build_llm()
    if mode != "live":
        raise ShadowValidationError("CLI shadow execution requires a live model client")
    _disable_provider_retries(llm)

    async def execute_and_close() -> dict[str, Any]:
        try:
            return await execute_shadow(
                args.packet,
                llm=llm,
                repeats=args.repeats,
                max_calls=args.max_calls,
                schedule_seed=args.schedule_seed,
                max_cost_usd=args.max_cost_usd,
                cost_reserve_per_call_usd=args.cost_reserve_per_call_usd,
                record_call=lambda row: _append_jsonl(args.output / "calls.jsonl", row),
            )
        finally:
            await llm.aclose()

    report = asyncio.run(execute_and_close())
    _write_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
