"""Cross-linked acceptance evidence for the bounded two-action smoke."""

from __future__ import annotations

import ast
import json
from hashlib import sha256
from typing import Any

import numpy as np
from competence import (
    CompetenceController,
    _assertion_value,
    _compare,
    _transition_output,
    parse_model_reason,
)
from competence_runner import EXPECTED_PROJECTION_DTYPE, attach_competence_context
from llm_repl_agent.observations import action_requirements

COMPETENCE_SCHEMA = "0006-nooa-competence-v1"
PROCESS_PROTOCOL = "0002-nooa-ipc-v1"
PUBLIC_OBSERVATION_KEYS = {
    "step",
    "pixels",
    "available_actions",
    "levels_completed",
    "state",
    "full_reset",
}


def canonical_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def competence_packet_digest(packet: object) -> str:
    return canonical_digest({"domain": "context_packet", "value": packet})


def _public_grid(public: object) -> list[list[int]] | None:
    if not isinstance(public, dict):
        return None
    pixels = public.get("pixels")
    if not isinstance(pixels, list) or not pixels:
        return None
    width: int | None = None
    height: int | None = None
    for layer in pixels:
        if not isinstance(layer, list) or not layer:
            return None
        if height is None:
            height = len(layer)
        if len(layer) != height:
            return None
        for row in layer:
            if not isinstance(row, list) or not row:
                return None
            if width is None:
                width = len(row)
            if len(row) != width or any(
                type(value) is not int or value not in range(256) for value in row
            ):
                return None
    return pixels[-1]


def _public_observation_valid(public: object, *, expected_step: int) -> bool:
    if not isinstance(public, dict) or set(public) != PUBLIC_OBSERVATION_KEYS:
        return False
    available_actions = public.get("available_actions")
    return (
        type(public.get("step")) is int
        and public.get("step") == expected_step
        and _public_grid(public) is not None
        and isinstance(available_actions, list)
        and bool(available_actions)
        and all(type(action) is int for action in available_actions)
        and len(set(available_actions)) == len(available_actions)
        and type(public.get("levels_completed")) is int
        and public["levels_completed"] >= 0
        and isinstance(public.get("state"), str)
        and bool(public["state"])
        and type(public.get("full_reset")) is bool
    )


def _projection_digest(public: object, projection: object) -> str | None:
    if not isinstance(public, dict) or not isinstance(projection, dict):
        return None
    if set(projection) != {"digest", "dtype", "shape"}:
        return None
    dtype_name = projection.get("dtype")
    shape = projection.get("shape")
    if dtype_name != EXPECTED_PROJECTION_DTYPE or not isinstance(shape, list):
        return None
    dtype = np.dtype(EXPECTED_PROJECTION_DTYPE)
    pixels = np.asarray(public.get("pixels"), dtype=dtype)
    if list(pixels.shape) != shape or pixels.ndim != 3:
        return None
    digest = sha256(pixels.tobytes()).hexdigest()[:16]
    return digest if projection.get("digest") == digest else None


def _sha256_string(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _model_call_audit_valid(
    call: object,
    *,
    expected_provider_profile: object,
) -> bool:
    if not isinstance(call, dict):
        return False
    if set(call) != {
        "actual_provider_crossings",
        "attempts",
        "canonical_audit_request_sha256",
        "canonical_request",
        "canonical_request_sha256",
        "canonicalization",
        "messages",
        "outcome",
        "schema",
        "tools",
    }:
        return False
    canonical_request = call.get("canonical_request")
    crossings = call.get("actual_provider_crossings")
    attempts = call.get("attempts")
    outcome = call.get("outcome")
    return not (
        not isinstance(canonical_request, dict)
        or set(canonical_request)
        != {"provider_request_profile", "messages", "tools", "kwargs"}
        or call.get("messages") != canonical_request.get("messages")
        or call.get("tools") != canonical_request.get("tools")
        or not isinstance(call.get("messages"), list)
        or not call["messages"]
        or not isinstance(call.get("tools"), list)
        or not isinstance(canonical_request.get("kwargs"), dict)
        or canonical_request.get("provider_request_profile")
        != expected_provider_profile
        or call.get("canonical_audit_request_sha256")
        != canonical_digest(canonical_request)
        or not _sha256_string(call.get("canonical_request_sha256"))
        or call.get("canonicalization") != "json-utf8-sort-keys-compact-v1"
        or call.get("schema") != "0002-nooa-provider-crossing-v1"
        or not isinstance(crossings, list)
        or not crossings
        or any(
            not isinstance(crossing, dict)
            or set(crossing) != {"request", "response", "sequence"}
            or not isinstance(crossing.get("request"), dict)
            or not isinstance(crossing.get("response"), dict)
            or type(crossing.get("sequence")) is not int
            # Zero-based. The frozen recorder assigns this as an index into its
            # own list (0002-nooa/nooa-runtime/llm.py:470), so the first HTTP
            # crossing a worker makes is legitimately sequence 0. Demanding a
            # positive value here failed every run at its very first crossing.
            or crossing["sequence"] < 0
            for crossing in crossings
        )
        or attempts
        != [
            {
                "attempt": 1,
                "outcome": "success",
                "actual_provider_crossings": crossings,
            }
        ]
        or not isinstance(outcome, dict)
        or set(outcome) != {"kind", "response"}
        or outcome.get("kind") != "success"
        or not isinstance(outcome.get("response"), dict)
    )


def _inline_return_closes_the_turn(final_tool_call: dict[str, object]) -> bool:
    """Accept the inline route, which the competence packet now requires.

    The packet tells the model to call ``return_result`` as the final statement
    of its ``execute_python`` cell, and forbids the separate tool, because the
    tool route makes it retype a serialized string by hand -- one surplus brace
    discarded a complete commitment in run 20260802T120006Z. This verifier still
    demanded the tool route, so it contradicted the instruction the treatment
    gives and failed the first two-action run for following it.

    The inline route cannot bind the literal payload the way the tool route can:
    the cell calls ``return_result`` with variables it computed, so the decision
    is not present as text. What is checkable, and what matters, is that the turn
    ended because the model invoked ``return_result`` itself -- the same final
    statement the frozen worker detects at agent.py's inline-return handling --
    rather than because the host supplied a decision of its own.
    """

    if set(final_tool_call) != {"arguments", "id", "name"}:
        return False
    identifier = final_tool_call.get("id")
    arguments = final_tool_call.get("arguments")
    if not isinstance(identifier, str) or not identifier:
        return False
    if not isinstance(arguments, str):
        return False
    try:
        payload = json.loads(arguments)
    except json.JSONDecodeError:
        return False
    if set(payload) != {"code"} or not isinstance(payload["code"], str):
        return False
    try:
        tree = ast.parse(payload["code"])
    except SyntaxError:
        return False
    return bool(tree.body) and _statement_calls_return_result(tree.body[-1])


def _statement_calls_return_result(statement: ast.stmt) -> bool:
    """True when running this statement calls ``return_result``.

    Not necessarily as a bare expression. Run 20260803T064129Z wrapped the call
    defensively -- ``try: return_result(...) except NameError: ...`` -- to check
    the tool was in scope before relying on it. The host received that decision
    and released the action; only this verifier objected, and it objected to the
    shape of a guard rather than to anything about the decision.

    Definitions are stepped over, because naming a call is not making one. A
    final statement whose only ``return_result`` sits inside a ``def`` body has
    not invoked anything, and the turn did not end because the model returned.
    """

    deferred = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
    if isinstance(statement, deferred):
        return False
    pending: list[ast.AST] = [statement]
    while pending:
        node = pending.pop()
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "return_result"
        ):
            return True
        pending.extend(
            child for child in ast.iter_child_nodes(node)
            if not isinstance(child, deferred)
        )
    return False


def _last_model_call_returns_decision(
    model_calls: list[object], worker_decision: object
) -> bool:
    if not model_calls or not isinstance(worker_decision, dict):
        return False
    final_call = model_calls[-1]
    if not isinstance(final_call, dict):
        return False
    outcome = final_call.get("outcome")
    response = outcome.get("response") if isinstance(outcome, dict) else None
    tool_calls = response.get("tool_calls") if isinstance(response, dict) else None
    if not isinstance(tool_calls, list) or not tool_calls:
        return False
    final_tool_call = tool_calls[-1]
    if (
        isinstance(final_tool_call, dict)
        and final_tool_call.get("name") == "execute_python"
    ):
        return _inline_return_closes_the_turn(final_tool_call)
    if (
        not isinstance(final_tool_call, dict)
        or set(final_tool_call) != {"arguments", "id", "name"}
        or final_tool_call.get("name") != "return_result"
        or not isinstance(final_tool_call.get("id"), str)
        or not final_tool_call["id"]
        or not isinstance(final_tool_call.get("arguments"), str)
    ):
        return False
    try:
        arguments = json.loads(final_tool_call["arguments"])
    except json.JSONDecodeError:
        return False
    return arguments == {"result": worker_decision}


def _representations_valid(value: object, *, public: dict[str, Any]) -> bool:
    if not isinstance(value, dict) or set(value) != {"grid", "rendered_image"}:
        return False
    grid = value.get("grid")
    rendered = value.get("rendered_image")
    if not isinstance(grid, dict) or not isinstance(rendered, dict):
        return False
    annotation = rendered.get("annotation")
    pixels = np.asarray(public["pixels"])
    return (
        set(grid)
        == {
            "axes",
            "encoding",
            "json_pointer",
            "json_value_type",
            "path",
            "shape",
            "values",
        }
        and grid.get("axes") == ["layer", "row", "column"]
        and grid.get("encoding") == "nested_integer_array"
        and grid.get("json_pointer") == "/pixels"
        and grid.get("json_value_type") == "integer"
        and grid.get("path") == "current_observation.json"
        and grid.get("shape") == list(pixels.shape)
        and grid.get("values") == "integer_color_ids"
        and set(rendered)
        == {
            "annotation",
            "axes",
            "dtype",
            "encoding",
            "path",
            "shape",
            "source_grid_layer",
            "source_grid_scale",
        }
        and rendered.get("axes") == ["row", "column", "channel"]
        and rendered.get("dtype") == "uint8"
        and rendered.get("encoding") == "rgb"
        and rendered.get("path") == "current_frame.png"
        and rendered.get("shape")
        == [len(public["pixels"][-1]) * 8, len(public["pixels"][-1][0]) * 8, 3]
        and rendered.get("source_grid_layer") == len(public["pixels"]) - 1
        and rendered.get("source_grid_scale") == 8
        and isinstance(annotation, dict)
        and set(annotation)
        == {
            "cell_pixels",
            "gridline_every",
            "label_every",
            "margin_pixels",
            "note",
            "rulers",
            "x_axis",
            "y_axis",
        }
        and annotation.get("cell_pixels") == 7
        and annotation.get("gridline_every") == 10
        and annotation.get("label_every") == 5
        and annotation.get("margin_pixels") == 24
        and isinstance(annotation.get("note"), str)
        and annotation.get("rulers")
        == "grid coordinates on the top and left margins"
        and annotation.get("x_axis") == "column, labelled along the top"
        and annotation.get("y_axis") == "row, labelled down the left"
    )


def _changed_pixels(before: object, after: object) -> int | None:
    before_grid = _public_grid(before)
    after_grid = _public_grid(after)
    if before_grid is None or after_grid is None:
        return None
    if len(before_grid) != len(after_grid) or len(before_grid[0]) != len(after_grid[0]):
        return None
    return sum(
        before_value != after_value
        for before_row, after_row in zip(before_grid, after_grid, strict=True)
        for before_value, after_value in zip(before_row, after_row, strict=True)
    )


def _expected_environment_transition(
    *,
    action_point_id: int,
    commitment: dict[str, Any],
    before: dict[str, Any],
    after: dict[str, Any],
    before_projection: object,
    after_projection: object,
) -> dict[str, Any] | None:
    before_public = before.get("public")
    after_public = after.get("public")
    if not isinstance(before_public, dict) or not isinstance(after_public, dict):
        return None
    before_digest = _projection_digest(before_public, before_projection)
    after_digest = _projection_digest(after_public, after_projection)
    if before_digest is None or after_digest is None:
        return None
    step = action_point_id + 1
    return {
        "step": step,
        "action": commitment.get("action"),
        "action_data": commitment.get("data"),
        "before_digest": before_digest,
        "after_digest": after_digest,
        "changed_pixels": _changed_pixels(before_public, after_public),
        "levels_before": before_public.get("levels_completed"),
        "levels_after": after_public.get("levels_completed"),
        "state_before": before_public.get("state"),
        "state_after": after_public.get("state"),
        "full_reset": after_public.get("full_reset"),
        "available_actions_after": after_public.get("available_actions"),
        "before_file": f"observations/step-{action_point_id:05d}.json",
        "after_file": f"observations/step-{step:05d}.json",
    }


def _content_id(prefix: str, domain: str, value: object) -> str:
    return prefix + canonical_digest({"domain": domain, "value": value})


def _apply_ledger_events(
    initial: dict[str, dict[str, object]],
    events: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, object]], bool]:
    claims = {claim_id: dict(claim) for claim_id, claim in initial.items()}
    valid = True
    for event in events:
        operation = event.get("operation")
        if not isinstance(operation, dict) or not isinstance(
            operation.get("claim_id"), str
        ):
            valid = False
            continue
        claim_id = operation["claim_id"]
        op = operation.get("op")
        if op == "create" and set(operation) == {"op", "claim_id", "statement"}:
            statement = operation.get("statement")
            if claim_id in claims or not isinstance(statement, str) or not statement:
                valid = False
                continue
            claims[claim_id] = {
                "claim_id": claim_id,
                "statement": statement,
                "status": "active",
                "revision": 0,
            }
        elif op == "revise" and set(operation) == {"op", "claim_id", "statement"}:
            statement = operation.get("statement")
            revision = claims.get(claim_id, {}).get("revision")
            if (
                claim_id not in claims
                or not isinstance(statement, str)
                or not statement
                or type(revision) is not int
            ):
                valid = False
                continue
            claims[claim_id] = {
                **claims[claim_id],
                "statement": statement,
                "revision": revision + 1,
            }
        elif op == "retire" and set(operation) == {"op", "claim_id"}:
            if claim_id not in claims:
                valid = False
                continue
            claims[claim_id] = {**claims[claim_id], "status": "retired"}
        else:
            valid = False
    return claims, valid


def _prediction_mismatches(
    commitment: dict[str, Any],
    prediction: dict[str, Any],
    before_observation: object,
    after_observation: object,
) -> tuple[list[dict[str, object]], bool]:
    assertions = commitment.get("assertions")
    checks = prediction.get("checks")
    if (
        not isinstance(assertions, list)
        or not isinstance(checks, list)
        or not isinstance(before_observation, dict)
        or not isinstance(after_observation, dict)
        or not isinstance(before_observation.get("public"), dict)
        or not isinstance(after_observation.get("public"), dict)
    ):
        return [], False
    if len(assertions) != len(checks):
        return [], False
    mismatches: list[dict[str, object]] = []
    checks_valid = True
    transition_output = _transition_output(
        before_observation["public"], after_observation["public"]
    )
    for index, (assertion, check) in enumerate(zip(assertions, checks, strict=True)):
        if not isinstance(assertion, dict) or not isinstance(check, dict):
            checks_valid = False
            continue
        matched = check.get("matched")
        observed = check.get("observed")
        expected_observed = _assertion_value(
            assertion,
            transition=transition_output,
            after=after_observation["public"],
        )
        expected_matched = _compare(
            expected_observed, str(assertion.get("op")), assertion.get("value")
        )
        if (
            type(matched) is not bool
            or observed != expected_observed
            or matched != expected_matched
            or {key: value for key, value in check.items() if key not in {"observed", "matched"}}
            != assertion
        ):
            checks_valid = False
            continue
        if matched is False:
            mismatches.append(
                {
                    "mismatch_id": f"mismatch-{commitment.get('action_id')}-{index}",
                    "action_id": commitment.get("action_id"),
                    "assertion": assertion,
                    "observed": observed,
                }
            )
    expected_outcome = "mismatch" if mismatches else "match"
    return mismatches, checks_valid and prediction.get("outcome") == expected_outcome


def _process_action_linked(
    *,
    action_point_id: int,
    packet: object,
    available_actions: object,
    commitment: dict[str, Any],
    before_observation: object,
    after_observation: object,
    expected_recent_transitions: list[object],
    context_record: dict[str, Any],
    request_record: dict[str, Any],
    crossing_record: dict[str, Any],
    validated_record: dict[str, Any],
    admitted_record: dict[str, Any],
    transition_record: dict[str, Any],
    successor_record: dict[str, Any],
) -> bool:
    request = request_record.get("request")
    if not isinstance(request, dict) or not isinstance(packet, dict):
        return False
    observation = request.get("observation")
    if not isinstance(observation, dict):
        return False
    competence_wrapper = (
        observation.get("competence")
    )
    legal_actions = request.get("legal_actions")
    worker_decision = crossing_record.get("worker_decision")
    audit = crossing_record.get("audit")
    model_calls = audit.get("model_calls") if isinstance(audit, dict) else None
    decision = validated_record.get("decision")
    reasoning = decision.get("reasoning") if isinstance(decision, dict) else None
    environment_transition = transition_record.get("transition")
    if (
        not isinstance(worker_decision, dict)
        or not isinstance(before_observation, dict)
        or not isinstance(after_observation, dict)
        or not isinstance(environment_transition, dict)
    ):
        return False
    worker_action = worker_decision.get("action")
    if type(worker_action) is not int:
        return False
    try:
        authored_decision = parse_model_reason(
            worker_decision.get("reason"),
            action=worker_action,
            data=worker_decision.get("data"),
        )
    except (TypeError, ValueError):
        return False
    expected_commitment = {
        key: value for key, value in commitment.items() if key != "type"
    }
    expected_transition = _expected_environment_transition(
        action_point_id=action_point_id,
        commitment=commitment,
        before=before_observation,
        after=after_observation,
        before_projection=context_record.get("projection"),
        after_projection=successor_record.get("projection"),
    )
    request_keys = {
        "protocol",
        "run_id",
        "action_point_id",
        "observation",
        "legal_actions",
        "coordinate_actions",
        "frame_width",
        "frame_height",
        "paired_run_spec",
    }
    observation_keys = {
        "step",
        "current_metadata",
        "representations",
        "recent_transitions",
        "full_history_file",
        "current_observation_file",
        "observations_glob",
        "competence",
    }
    expected_wrapper = attach_competence_context({}, packet).get("competence")
    host_observation = request_record.get("host_observation")
    successor_host_observation = successor_record.get("host_observation")
    workspace_commitment = validated_record.get("workspace_commitment")
    before_public = before_observation.get("public")
    current_metadata = observation.get("current_metadata")
    grid = _public_grid(before_public)
    if (
        not isinstance(before_public, dict)
        or not isinstance(current_metadata, dict)
        or grid is None
    ):
        return False
    expected_metadata = {
        "available_actions": before_public["available_actions"],
        "action_requirements": action_requirements(
            tuple(before_public["available_actions"])
        ),
        "levels_completed": before_public["levels_completed"],
        "state": before_public["state"],
        "full_reset": before_public["full_reset"],
        "frame": {
            "layers": len(before_public["pixels"]),
            "height": len(grid),
            "width": len(grid[0]),
        },
    }
    paired_run_spec_keys = {
        "allowed_distributions",
        "allowed_python_policy",
        "coordinate_convention",
        "environment_profile",
        "environment_seed",
        "final_action_schema",
        "host_workspace_files",
        "host_workspace_initialization",
        "include_changed_pixel_counts",
        "legal_event_schema",
        "max_actions",
        "max_context_turns",
        "max_decision_repairs",
        "max_model_calls",
        "max_model_rounds_per_action",
        "max_recent_transitions",
        "max_reported_tokens",
        "model_profile",
        "model_seed",
        "observation_representations",
        "omission_notice_schema",
        "provider_endpoint_profile",
        "provider_request_profile",
        "public_status_fields",
        "request_profile",
        "transition_observation_policy",
        "transport_max_retries",
        "usability_probe_instruction",
        "workspace_timeout_seconds",
    }
    before_host_digest = sha256(
        json.dumps(before_public, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    after_public = after_observation.get("public")
    if not isinstance(after_public, dict):
        return False
    after_host_digest = sha256(
        json.dumps(after_public, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return (
        set(request) == request_keys
        and request.get("protocol") == PROCESS_PROTOCOL
        and request.get("run_id") == request_record.get("run_id")
        # The IPC body carries the crossing id, which advances on every attempt.
        # The process record carries both, and its step is what ties this
        # crossing to the environment action it produced.
        and request.get("action_point_id") == request_record.get("action_point_id")
        and request_record.get("step") == action_point_id
        and isinstance(observation, dict)
        and set(observation) == observation_keys
        and observation.get("step") == action_point_id
        and observation.get("recent_transitions") == expected_recent_transitions
        and current_metadata == expected_metadata
        and _representations_valid(
            observation.get("representations"), public=before_public
        )
        and request.get("frame_height") == len(grid)
        and request.get("frame_width") == len(grid[0])
        and isinstance(request.get("coordinate_actions"), list)
        and isinstance(request.get("paired_run_spec"), dict)
        and set(request["paired_run_spec"]) == paired_run_spec_keys
        and isinstance(legal_actions, list)
        and legal_actions == available_actions
        and all(type(action) is int for action in legal_actions)
        and commitment.get("action") in legal_actions
        and isinstance(competence_wrapper, dict)
        and competence_wrapper == expected_wrapper
        and isinstance(host_observation, dict)
        and set(host_observation) == {"path", "sha256"}
        and host_observation.get("path")
        == f"host-observations/step-{action_point_id:05d}.json"
        and host_observation.get("sha256") == before_host_digest
        and set(successor_record)
        == {
            "protocol",
            "kind",
            "run_id",
            "action_point_id",
            "successor_step",
            "evidence_id",
            "host_observation",
            "projection",
        }
        and successor_record.get("protocol") == PROCESS_PROTOCOL
        and successor_record.get("kind") == "competence_successor_observed"
        and successor_record.get("run_id") == request_record.get("run_id")
        and successor_record.get("action_point_id") == action_point_id
        and successor_record.get("successor_step") == action_point_id + 1
        and successor_record.get("evidence_id") == after_observation.get("evidence_id")
        and isinstance(successor_host_observation, dict)
        and set(successor_host_observation) == {"path", "sha256"}
        and successor_host_observation.get("path")
        == f"host-observations/step-{action_point_id + 1:05d}.json"
        and successor_host_observation.get("sha256") == after_host_digest
        and _projection_digest(before_public, context_record.get("projection"))
        == environment_transition.get("before_digest")
        and _projection_digest(after_public, successor_record.get("projection"))
        == environment_transition.get("after_digest")
        and request_record.get("request_sha256") == canonical_digest(request)
        and isinstance(audit, dict)
        and set(audit) == {"model_calls"}
        and crossing_record.get("worker_error") is None
        and isinstance(model_calls, list)
        and bool(model_calls)
        and all(
            _model_call_audit_valid(
                call,
                expected_provider_profile=request["paired_run_spec"].get(
                    "provider_request_profile"
                ),
            )
            for call in model_calls
        )
        and _last_model_call_returns_decision(model_calls, worker_decision)
        and worker_decision.get("action") == commitment.get("action")
        and set(worker_decision) == {"action", "data", "reason"}
        and worker_decision.get("data") == commitment.get("data")
        and isinstance(worker_decision.get("reason"), str)
        and bool(worker_decision["reason"])
        and authored_decision.get("action") == commitment.get("action")
        and authored_decision.get("data") == commitment.get("data")
        and isinstance(decision, dict)
        and set(decision) == {"action", "data", "reasoning"}
        and decision.get("action") == commitment.get("action")
        and decision.get("data") == commitment.get("data")
        and isinstance(reasoning, dict)
        and set(reasoning) == {"reason", "competence"}
        and reasoning.get("reason") == worker_decision.get("reason")
        and reasoning.get("competence") == expected_commitment
        and admitted_record.get("commitment") == expected_commitment
        and isinstance(workspace_commitment, str)
        and len(workspace_commitment) == 64
        and context_record.get("current_evidence_id")
        == commitment.get("before_evidence_id")
        and environment_transition == expected_transition
    )


def _model_authored_replay_matches(
    *,
    events: list[dict[str, Any]],
    observations: list[object],
    requests: list[dict[str, Any]],
    crossings: list[dict[str, Any]],
    actions: int,
) -> bool:
    """Replay the raw model output through a fresh controller for every action.

    The counts are checked against the expected action count rather than each
    other, so a run that is internally consistent but shorter than the one that
    was approved cannot satisfy this by replaying only the part it completed.
    """

    if (
        len(observations) != actions + 1
        or len(requests) != actions
        or len(crossings) != actions
    ):
        return False
    try:
        controller = CompetenceController()
        if not all(isinstance(observation, dict) for observation in observations):
            return False
        typed_observations = [observation for observation in observations if isinstance(observation, dict)]
        initial_public = typed_observations[0].get("public")
        if not isinstance(initial_public, dict):
            return False
        controller.receive_observation(initial_public)
        for index in range(actions):
            request = requests[index].get("request")
            crossing_decision = crossings[index].get("worker_decision")
            if not isinstance(request, dict) or not isinstance(crossing_decision, dict):
                return False
            observation = request.get("observation")
            wrapper = (
                observation.get("competence")
                if isinstance(observation, dict)
                else None
            )
            packet = controller.context_packet()
            if not isinstance(wrapper, dict) or wrapper.get("context") != packet:
                return False
            action = crossing_decision.get("action")
            if type(action) is not int:
                return False
            authored = parse_model_reason(
                crossing_decision.get("reason"),
                action=action,
                data=crossing_decision.get("data"),
            )
            legal_actions = request.get("legal_actions")
            if not isinstance(legal_actions, list):
                return False
            controller.admit(authored, legal_actions=legal_actions)
            successor_public = typed_observations[index + 1].get("public")
            if not isinstance(successor_public, dict):
                return False
            controller.receive_observation(successor_public)
    except (TypeError, ValueError):
        return False
    return controller.events == events


def chain_report(
    events: list[dict[str, Any]],
    process_records: list[dict[str, Any]],
    *,
    actions: int,
) -> dict[str, bool]:
    """Validate one linked chain of ``actions`` commitment-successor-context steps.

    Every check here was written against two actions and is now applied across
    the whole run: each action must link to its own successor, and each successor
    must be the observation the next commitment was authored against. Nothing is
    checked pairwise-only, because a chain that holds at its ends and sags in the
    middle is exactly the artifact this is meant to reject.

    ``two_action_chain_report`` is this function at ``actions=2``, so the case the
    2-action tests already cover is the same code the 44-action run is judged by.
    """

    if actions < 1:
        raise ValueError("a chain needs at least one action")
    observation_count = actions + 1

    by_type = {
        event_type: [
            (index, event)
            for index, event in enumerate(events)
            if event.get("type") == event_type
        ]
        for event_type in (
            "commitment",
            "transition_occurrence",
            "prediction_result",
            "observation",
            "context_assembled",
        )
    }
    expected_counts = {
        "commitment": actions,
        "transition_occurrence": actions,
        "prediction_result": actions,
        "observation": observation_count,
        "context_assembled": actions,
    }
    shape_ok = all(
        len(by_type[event_type]) == count
        for event_type, count in expected_counts.items()
    )
    if not shape_ok:
        return {"linked_event_shape": False}

    commitment_indexed = by_type["commitment"]
    context_indexed = by_type["context_assembled"]
    transition_indexed = by_type["transition_occurrence"]
    prediction_indexed = by_type["prediction_result"]
    observation_indexed = by_type["observation"]

    commitments = [event for _index, event in commitment_indexed]
    predictions = [event for _index, event in prediction_indexed]
    transition_events = [event for _index, event in transition_indexed]
    observation_events = [event for _index, event in observation_indexed]
    context_events = [event for _index, event in context_indexed]

    transitions = [event.get("record") for event in transition_events]
    observation_records = [event.get("record") for event in observation_events]

    ids_linked = all(
        commitment.get("action_id") == action_id + 1
        for action_id, commitment in enumerate(commitments)
    ) and all(
        isinstance(record, dict) and record.get("step") == step
        for step, record in enumerate(observation_records)
    )
    for action_id in range(actions):
        commitment = commitments[action_id]
        transition = transitions[action_id]
        prediction = predictions[action_id]
        before = observation_records[action_id]
        after = observation_records[action_id + 1]
        ids_linked = (
            ids_linked
            and isinstance(before, dict)
            and isinstance(after, dict)
            and isinstance(transition, dict)
            and commitment.get("before_step") == before.get("step")
            and commitment.get("before_evidence_id") == before.get("evidence_id")
            and transition.get("action_id") == action_id + 1
            and prediction.get("action_id") == action_id + 1
            and transition.get("before_evidence_id")
            == commitment.get("before_evidence_id")
            and transition.get("before_evidence_id")
            == prediction.get("before_evidence_id")
            and transition.get("after_evidence_id")
            == prediction.get("after_evidence_id")
            and transition.get("after_evidence_id") == after.get("evidence_id")
            and transition.get("action") == commitment.get("action")
            and transition.get("data") == commitment.get("data")
        )

    content_ids_valid = all(
        isinstance(record, dict)
        and record.get("evidence_id")
        == _content_id(
            "obs-",
            "observation_occurrence",
            {key: value for key, value in record.items() if key != "evidence_id"},
        )
        for record in observation_records
    ) and all(
        isinstance(event.get("record"), dict)
        and event.get("transition_id")
        == _content_id("trn-", "transition_occurrence", event["record"])
        for event in transition_events
    )

    # observation 0, then for each action: context, commitment, transition,
    # prediction, successor observation.
    expected_event_order = [observation_indexed[0][0]]
    for action_id in range(actions):
        expected_event_order.extend(
            [
                context_indexed[action_id][0],
                commitment_indexed[action_id][0],
                transition_indexed[action_id][0],
                prediction_indexed[action_id][0],
                observation_indexed[action_id + 1][0],
            ]
        )
    event_order = all(
        earlier < later
        for earlier, later in zip(expected_event_order, expected_event_order[1:])
    )

    contexts_indexed = [
        (index, record)
        for index, record in enumerate(process_records)
        if record.get("kind") == "competence_context_assembled"
    ]
    requests_indexed = [
        (index, record)
        for index, record in enumerate(process_records)
        if record.get("kind") == "action_request"
    ]
    admitted_indexed = [
        (index, record)
        for index, record in enumerate(process_records)
        if record.get("kind") == "competence_action_admitted"
    ]
    crossings_indexed = [
        (index, record)
        for index, record in enumerate(process_records)
        if record.get("kind") == "model_crossing"
    ]
    validated_indexed = [
        (index, record)
        for index, record in enumerate(process_records)
        if record.get("kind") == "decision_validated"
    ]
    recorded_transitions_indexed = [
        (index, record)
        for index, record in enumerate(process_records)
        if record.get("kind") == "transition_recorded"
    ]
    successor_process_indexed = [
        (index, record)
        for index, record in enumerate(process_records)
        if record.get("kind") == "competence_successor_observed"
    ]
    rejected_indexed = [
        (index, record)
        for index, record in enumerate(process_records)
        if record.get("kind") == "competence_rejected"
    ]
    # A refused crossing is recorded in full, as its own request/crossing/
    # rejection triple, so the log shows how many attempts an action took.
    # Removing those triples must leave exactly the admitted spine.
    rejected_crossing_ids = {
        record.get("action_point_id") for _index, record in rejected_indexed
    }
    repair_records = [
        record
        for record in process_records
        if record.get("kind")
        in {"action_request", "model_crossing", "competence_rejected"}
        and record.get("action_point_id") in rejected_crossing_ids
    ]
    spine_records = [
        record for record in process_records if record not in repair_records
    ]
    # Downstream counts and ordering describe the admitted spine. The refused
    # crossings are checked separately, by repairs_linked below.
    requests_indexed = [
        (index, record)
        for index, record in requests_indexed
        if record.get("action_point_id") not in rejected_crossing_ids
    ]
    crossings_indexed = [
        (index, record)
        for index, record in crossings_indexed
        if record.get("action_point_id") not in rejected_crossing_ids
    ]
    started_indexed = [
        (index, record)
        for index, record in enumerate(process_records)
        if record.get("kind") == "worker_started"
    ]
    stopped_indexed = [
        (index, record)
        for index, record in enumerate(process_records)
        if record.get("kind") == "worker_stopped"
    ]
    contexts = [record for _index, record in contexts_indexed]
    requests = [record for _index, record in requests_indexed]
    admitted = [record for _index, record in admitted_indexed]
    crossings = [record for _index, record in crossings_indexed]
    validated = [record for _index, record in validated_indexed]
    recorded_transitions = [
        record for _index, record in recorded_transitions_indexed
    ]
    successor_process_records = [
        record for _index, record in successor_process_indexed
    ]
    expected_process_kinds = [
        "worker_started",
        *(
            kind
            for _action_id in range(actions)
            for kind in (
                "competence_context_assembled",
                "action_request",
                "model_crossing",
                "decision_validated",
                "competence_action_admitted",
                "transition_recorded",
                "competence_successor_observed",
            )
        ),
        "worker_stopped",
    ]
    process_keys = {
        "worker_started": {"protocol", "kind", "run_id", "pid", "runtime_python"},
        "competence_context_assembled": {
            "protocol",
            "kind",
            "run_id",
            "action_point_id",
            "step",
            "current_evidence_id",
            "competence_packet_sha256",
            "projection",
        },
        "action_request": {
            "protocol",
            "kind",
            "run_id",
            "action_point_id",
            "step",
            "host_observation",
            "request_sha256",
            "request",
        },
        "model_crossing": {
            "protocol",
            "kind",
            "run_id",
            "action_point_id",
            "step",
            "audit",
            "worker_decision",
            "worker_error",
        },
        "decision_validated": {
            "protocol",
            "kind",
            "run_id",
            "action_point_id",
            "step",
            "decision",
            "workspace_commitment",
        },
        "competence_action_admitted": {
            "protocol",
            "kind",
            "run_id",
            "action_point_id",
            "step",
            "commitment",
        },
        "transition_recorded": {
            "protocol",
            "kind",
            "run_id",
            "action_point_id",
            "transition",
        },
        "competence_successor_observed": {
            "protocol",
            "kind",
            "run_id",
            "action_point_id",
            "successor_step",
            "evidence_id",
            "host_observation",
            "projection",
        },
        "competence_rejected": {
            "protocol",
            "kind",
            "run_id",
            "action_point_id",
            "step",
            "error",
            "attempts_remaining",
        },
        "worker_stopped": {
            "protocol",
            "kind",
            "run_id",
            "returncode",
            "stderr",
        },
    }
    process_run_ids = {
        record.get("run_id")
        for record in process_records
        if isinstance(record.get("run_id"), str) and record.get("run_id")
    }
    expected_step_sequence = list(range(actions))
    # A truncated log is the artifact this exists to reject, so the ordering
    # sequence is only built once every spine is known to be complete. Indexing
    # into a short list would raise out of a verifier whose whole job is to
    # return False.
    spine_counts_ok = (
        len(started_indexed)
        == len(stopped_indexed)
        == 1
        and len(contexts_indexed)
        == len(requests_indexed)
        == len(crossings_indexed)
        == len(validated_indexed)
        == len(admitted_indexed)
        == len(recorded_transitions_indexed)
        == len(successor_process_indexed)
        == actions
    )
    # One interleaved spine per action, in the one order that means the model saw
    # the context before it decided and the environment moved after it did.
    expected_process_order: list[int] = []
    if spine_counts_ok:
        expected_process_order.append(started_indexed[0][0])
        for action_id in range(actions):
            expected_process_order.extend(
                [
                    contexts_indexed[action_id][0],
                    requests_indexed[action_id][0],
                    crossings_indexed[action_id][0],
                    validated_indexed[action_id][0],
                    admitted_indexed[action_id][0],
                    recorded_transitions_indexed[action_id][0],
                    successor_process_indexed[action_id][0],
                ]
            )
        expected_process_order.append(stopped_indexed[0][0])
    process_shape = (
        spine_counts_ok
        and [record.get("kind") for record in spine_records] == expected_process_kinds
        and all(
            isinstance((kind := record.get("kind")), str)
            and kind in process_keys
            and set(record) == process_keys[kind]
            for record in process_records
        )
        and len(process_run_ids) == 1
        and all(record.get("run_id") in process_run_ids for record in process_records)
        and all(record.get("protocol") == PROCESS_PROTOCOL for record in process_records)
        and type(started_indexed[0][1].get("pid")) is int
        and isinstance(started_indexed[0][1].get("runtime_python"), str)
        and bool(started_indexed[0][1]["runtime_python"])
        and isinstance(stopped_indexed[0][1].get("stderr"), str)
        and [record.get("action_point_id") for record in contexts]
        == expected_step_sequence
        and [record.get("step") for record in requests] == expected_step_sequence
        and [record.get("step") for record in crossings] == expected_step_sequence
        and [record.get("step") for record in validated] == expected_step_sequence
        and [record.get("step") for record in admitted] == expected_step_sequence
        and [record.get("action_point_id") for record in recorded_transitions]
        == expected_step_sequence
        and [record.get("action_point_id") for record in successor_process_records]
        == expected_step_sequence
        and all(
            earlier < later
            for earlier, later in zip(
                expected_process_order, expected_process_order[1:]
            )
        )
    )
    schemas_valid = all(
        event.get("schema") == COMPETENCE_SCHEMA for event in events
    ) and (
        len(process_run_ids) == 1
        and all(
            record.get("protocol") == PROCESS_PROTOCOL
            and record.get("run_id") in process_run_ids
            for record in process_records
        )
    )
    packets: list[object] = [None] * actions
    competence_wrappers: list[object] = []
    if process_shape:
        collected: list[object] = []
        for request_record in requests:
            request = request_record.get("request")
            observation = request.get("observation") if isinstance(request, dict) else None
            competence = (
                observation.get("competence") if isinstance(observation, dict) else None
            )
            competence_wrappers.append(competence)
            collected.append(
                competence.get("context") if isinstance(competence, dict) else None
            )
        packets = collected
    schemas_valid = schemas_valid and all(
        isinstance(value, dict) and value.get("schema") == COMPETENCE_SCHEMA
        for value in (
            *commitments,
            *observation_records,
            *transitions,
            *predictions,
            *packets,
        )
    )
    schemas_valid = schemas_valid and all(
        isinstance(wrapper, dict)
        and wrapper.get("schema") == COMPETENCE_SCHEMA
        for wrapper in competence_wrappers
    )
    schemas_valid = schemas_valid and all(
        isinstance(record, dict) and record.get("kind") == "observation_occurrence"
        for record in observation_records
    ) and all(
        isinstance(record, dict) and record.get("kind") == "transition_occurrence"
        for record in transitions
    )

    prediction_cores = [
        {key: value for key, value in prediction.items() if key != "type"}
        for prediction in predictions
    ]
    packet_linked = all(isinstance(packet, dict) for packet in packets) and all(
        isinstance(record, dict) for record in (*observation_records, *transitions)
    )
    if packet_linked:
        for action_id in range(actions):
            packet = packets[action_id]
            before = observation_records[action_id]
            expected_current = {
                "step": before.get("step"),
                "evidence_id": before.get("evidence_id"),
            }
            if action_id == 0:
                expected_result = None
            else:
                expected_result = prediction_cores[action_id - 1]
                previous = transitions[action_id - 1]
                continuity = {
                    "action_id": action_id,
                    "before_evidence_id": previous.get("before_evidence_id"),
                    "after_evidence_id": previous.get("after_evidence_id"),
                    "outcome": predictions[action_id - 1].get("outcome"),
                }
                if continuity not in packet.get("recent_continuity", []):
                    packet_linked = False
                    break
            if not (
                packet.get("current_observation") == expected_current
                and packet.get("last_prediction_result") == expected_result
                and contexts[action_id].get("competence_packet_sha256")
                == canonical_digest(packet)
                and context_events[action_id].get("packet_sha256")
                == competence_packet_digest(packet)
            ):
                packet_linked = False
                break

    observation_legal_actions = [
        record.get("public", {}).get("available_actions")
        if isinstance(record, dict) and isinstance(record.get("public"), dict)
        else None
        for record in observation_records[:actions]
    ]
    event_step_authority = (
        [
            record.get("step") if isinstance(record, dict) else None
            for record in observation_records
        ]
        == list(range(observation_count))
        and all(
            isinstance(record, dict)
            and _public_observation_valid(record.get("public"), expected_step=step)
            for step, record in enumerate(observation_records)
        )
        and all(
            commitments[action_id].get("before_step")
            == observation_records[action_id].get("step")
            for action_id in range(actions)
        )
    )
    # The frozen worker shows the model only the last max_recent_transitions of
    # them, so past that window the expected list is the tail, not the whole
    # history. Before action 9 of a live run the two are the same list, which is
    # why two actions never needed this.
    recent_transition_window: int | None = None
    if requests:
        spec = requests[0].get("request", {}).get("paired_run_spec")
        if isinstance(spec, dict):
            window = spec.get("max_recent_transitions")
            if type(window) is int and window >= 0:
                recent_transition_window = window

    def _expected_recent(action_id: int) -> list[object]:
        preceding = [
            record.get("transition")
            for record in recorded_transitions[:action_id]
            if isinstance(record.get("transition"), dict)
        ]
        if recent_transition_window is None:
            return preceding
        if recent_transition_window == 0:
            return []
        return preceding[-recent_transition_window:]

    process_linked = (
        process_shape
        and event_step_authority
        and all(
            record.get("request", {}).get("paired_run_spec")
            == requests[0].get("request", {}).get("paired_run_spec")
            for record in requests
        )
        and stopped_indexed[0][1].get("returncode") == 0
        # A refusal used to disqualify the chain outright, because it could only
        # end the run. It can now be answered, so it disqualifies nothing on its
        # own; refused_crossings_precede_their_action is what constrains it, and
        # a refusal with no later admission for its step still fails there.
        and not any(
            record.get("kind")
            in {
                "host_execution_failure",
                "temporal_seam_violation",
            }
            for record in process_records
        )
        and all(
            _process_action_linked(
                action_point_id=action_id,
                packet=packets[action_id],
                available_actions=observation_legal_actions[action_id],
                commitment=commitments[action_id],
                before_observation=observation_records[action_id],
                after_observation=observation_records[action_id + 1],
                expected_recent_transitions=_expected_recent(action_id),
                context_record=contexts[action_id],
                request_record=requests[action_id],
                crossing_record=crossings[action_id],
                validated_record=validated[action_id],
                admitted_record=admitted[action_id],
                transition_record=recorded_transitions[action_id],
                successor_record=successor_process_records[action_id],
            )
            for action_id in range(actions)
        )
    )
    model_authored_replay = _model_authored_replay_matches(
        events=events,
        observations=observation_records,
        requests=requests,
        crossings=crossings,
        actions=actions,
    )

    raw_claim_ids = [commitment.get("claim_ids") for commitment in commitments]
    claim_lists = [
        raw if isinstance(raw, list) else [] for raw in raw_claim_ids
    ]
    claim_id_shapes = all(
        isinstance(raw, list)
        and all(isinstance(claim_id, str) for claim_id in claim_list)
        and len(claim_list) == len(set(claim_list))
        for raw, claim_list in zip(raw_claim_ids, claim_lists)
    ) and bool(claim_lists[0])
    ledger_indexed = [
        (index, event)
        for index, event in enumerate(events)
        if event.get("type") == "ledger"
    ]
    ledger_segments = [
        [
            event
            for index, event in ledger_indexed
            if context_indexed[action_id][0] < index < commitment_indexed[action_id][0]
        ]
        for action_id in range(actions)
    ]
    ledger_placement = len(ledger_indexed) == sum(
        len(segment) for segment in ledger_segments
    ) and all(
        event.get("action_id") == action_id + 1
        for action_id, segment in enumerate(ledger_segments)
        for event in segment
    )
    ledger_states: list[dict[str, Any]] = []
    ledger_valid = True
    state: dict[str, Any] = {}
    for segment in ledger_segments:
        state, applied_ok = _apply_ledger_events(state, segment)
        ledger_valid = ledger_valid and applied_ok
        ledger_states.append(state)

    claim_linked = claim_id_shapes and ledger_placement and ledger_valid
    if claim_linked:
        for action_id in range(actions):
            selected = claim_lists[action_id]
            state_here = ledger_states[action_id]
            if not all(
                claim_id in state_here
                and state_here[claim_id].get("status") == "active"
                for claim_id in selected
            ):
                claim_linked = False
                break
            if action_id + 1 >= actions:
                break
            # The claims an action worked from must be answered by the next one:
            # carried forward, or explicitly revised or retired. Silently dropping
            # a claim is how a ledger stops being a ledger.
            next_packet = packets[action_id + 1]
            expected_working_claims = [
                state_here[claim_id] for claim_id in selected
            ]
            changed = {
                operation["claim_id"]
                for event in ledger_segments[action_id + 1]
                if isinstance((operation := event.get("operation")), dict)
                and operation.get("claim_id") in set(selected)
                and operation.get("op") in {"revise", "retire"}
            }
            if not (
                isinstance(next_packet, dict)
                and next_packet.get("working_claims") == expected_working_claims
                and set(selected) <= (set(claim_lists[action_id + 1]) | changed)
            ):
                claim_linked = False
                break

    predictions_valid = True
    expected_unresolved_by_action: list[list[Any]] = []
    for action_id in range(actions):
        expected_unresolved, valid = _prediction_mismatches(
            commitments[action_id],
            predictions[action_id],
            observation_records[action_id],
            observation_records[action_id + 1],
        )
        expected_unresolved_by_action.append(expected_unresolved)
        predictions_valid = predictions_valid and valid

    reconciliation_linked = True
    for action_id in range(1, actions):
        packet = packets[action_id]
        unresolved = (
            packet.get("unresolved_mismatches", []) if isinstance(packet, dict) else []
        )
        expected_unresolved = expected_unresolved_by_action[action_id - 1]
        reconciliation_events = [
            event
            for index, event in enumerate(events)
            if context_indexed[action_id][0] < index < commitment_indexed[action_id][0]
            and event.get("type") == "reconciliation"
        ]
        reconciliation = commitments[action_id].get("reconciliation")
        if expected_unresolved:
            unresolved_ids = {
                item["mismatch_id"]
                for item in unresolved
                if isinstance(item, dict) and isinstance(item.get("mismatch_id"), str)
            }
            reconciliation_ids = (
                {
                    item["mismatch_id"]
                    for item in reconciliation
                    if isinstance(item, dict)
                    and isinstance(item.get("mismatch_id"), str)
                }
                if isinstance(reconciliation, list)
                else set()
            )
            reconciliation_entries_valid = isinstance(reconciliation, list) and all(
                isinstance(item, dict)
                and set(item) == {"mismatch_id", "disposition", "note"}
                and isinstance(item.get("mismatch_id"), str)
                and 1 <= len(item["mismatch_id"]) <= 100
                and item.get("disposition") in {"ABANDON", "NARROW", "RETAIN"}
                and isinstance(item.get("note"), str)
                and 1 <= len(item["note"]) <= 1_000
                for item in reconciliation
            )
            step_ok = (
                unresolved == expected_unresolved
                and bool(unresolved_ids)
                and reconciliation_ids == unresolved_ids
                and reconciliation_entries_valid
                and (
                    len(reconciliation) if isinstance(reconciliation, list) else -1
                )
                == len(expected_unresolved)
                and len(reconciliation_events) == 1
                and reconciliation_events[0].get("items") == reconciliation
            )
        else:
            step_ok = (
                unresolved == []
                and reconciliation is None
                and not reconciliation_events
            )
        reconciliation_linked = reconciliation_linked and step_ok
    reconciliation_linked = reconciliation_linked and predictions_valid

    # A refused crossing counts as evidence only where it belongs: before the
    # crossing that carried the same action, under a distinct id, and never
    # after an action was already released.
    admitted_index_by_step = {
        record.get("step"): index for index, record in admitted_indexed
    }
    repairs_linked = (
        len(rejected_crossing_ids) == len(rejected_indexed)
        and all(
            record.get("action_point_id") not in admitted_index_by_step.values()
            and isinstance(record.get("error"), dict)
            and record["error"].get("correctable") is True
            and record.get("step") in admitted_index_by_step
            and index < admitted_index_by_step[record.get("step")]
            for index, record in rejected_indexed
        )
        and len(
            {record.get("action_point_id") for record in requests}
            & rejected_crossing_ids
        )
        == 0
    )
    return {
        "linked_event_shape": shape_ok,
        "linked_schema_versions": schemas_valid,
        "linked_action_and_evidence_ids": ids_linked,
        "content_addressed_evidence_ids": content_ids_valid,
        "linked_event_order": event_order,
        "linked_process_records": process_linked,
        # Booleans only: callers reduce this report with all(). The count of
        # refused crossings is a quantity, not a verdict, and is reported in the
        # run summary instead.
        "refused_crossings_precede_their_action": repairs_linked,
        "model_authored_semantic_replay": model_authored_replay,
        "successor_context_packet_linked": packet_linked,
        "claim_continuity_linked_across_actions": claim_linked,
        "prediction_results_derived_from_commitments": predictions_valid,
        "reconciliation_linked_to_observed_mismatch": reconciliation_linked,
    }


def two_action_chain_report(
    events: list[dict[str, Any]],
    process_records: list[dict[str, Any]],
) -> dict[str, bool]:
    """Validate one linked commitment-successor-context-commitment chain."""

    return chain_report(events, process_records, actions=2)
