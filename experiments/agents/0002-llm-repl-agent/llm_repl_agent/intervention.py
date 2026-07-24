"""Pure, frozen logic for the two-branch same-decision intervention."""

from __future__ import annotations

from base64 import b64decode, b64encode
from binascii import Error as Base64Error
from copy import deepcopy
from hashlib import sha256
import json
import random
from typing import Any, Callable, Mapping, Sequence

from .agent import DecisionFormatError, parse_decision_message
from .client import ChatResponse, ModelRequestError, serialize_request_payload


INTERVENTION_SCHEMA = "reconstruction-content-intervention-v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def candidate_id(*, generation: str, run_id: str, event_index: int, tool_call_id: str) -> str:
    if not generation or not run_id or not tool_call_id or event_index < 1:
        raise ValueError("candidate identity fields must be non-empty and event_index positive")
    source = canonical_json({
        "generation": generation,
        "run_id": run_id,
        "event_index": event_index,
        "tool_call_id": tool_call_id,
    })
    return f"candidate:{sha256(source.encode()).hexdigest()}"


def _derived_seed(*, candidate_id: str, generation: str, replicate: int) -> int:
    material = f"{INTERVENTION_SCHEMA}\0{generation}\0{candidate_id}\0{replicate}".encode()
    return int.from_bytes(sha256(material).digest()[:4], "big") & 0x7FFF_FFFF


def build_shadow_schedule(*, candidate_id: str, generation: str) -> tuple[dict[str, Any], ...]:
    """Create five adjacent, seed-matched, frozen two-branch replicates."""

    rows: list[dict[str, Any]] = []
    for replicate in range(1, 6):
        seed = _derived_seed(
            candidate_id=candidate_id,
            generation=generation,
            replicate=replicate,
        )
        branch_order = ["intact", "perturbed"]
        order_material = f"{generation}\0{candidate_id}\0order\0{replicate}"
        random.Random(sha256(order_material.encode()).digest()).shuffle(branch_order)
        for within_replicate_order, branch in enumerate(branch_order, 1):
            rows.append({
                "request_order": len(rows) + 1,
                "replicate": replicate,
                "within_replicate_order": within_replicate_order,
                "branch": branch,
                "seed": seed,
            })
    return tuple(rows)


def _parse_action(content: Any, decision_state: Mapping[str, Any] | None) -> str | None:
    if not isinstance(decision_state, Mapping):
        return None
    try:
        available = decision_state.get("available_actions")
        width = decision_state.get("frame_width")
        height = decision_state.get("frame_height")
        if not isinstance(available, list) or not isinstance(width, int) or not isinstance(height, int):
            return None
        action, data, _reason = parse_decision_message(
            {"content": content},
            available_actions=tuple(available),
            frame_width=width,
            frame_height=height,
        )
    except (DecisionFormatError, ValueError):
        return None
    return "ACT:" + canonical_json({"action": action, "data": data})


def _valid_python_calls(tool_calls: Any) -> bool:
    if not isinstance(tool_calls, list) or not tool_calls:
        return False
    for call in tool_calls:
        if not isinstance(call, Mapping):
            return False
        function = call.get("function")
        if not isinstance(function, Mapping) or function.get("name") != "python":
            return False
        arguments = function.get("arguments")
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError:
            return False
        if not isinstance(parsed, Mapping) or not isinstance(parsed.get("code"), str):
            return False
    return True


def response_signature(
    message: Mapping[str, Any],
    *,
    decision_state: Mapping[str, Any] | None = None,
) -> str:
    action = _parse_action(message.get("content"), decision_state)
    has_tool_calls = isinstance(message.get("tool_calls"), list) and bool(message["tool_calls"])
    if action is not None and not has_tool_calls:
        return action
    content = message.get("content")
    no_action_content = content is None or content == ""
    if no_action_content and _valid_python_calls(message.get("tool_calls")):
        return "CONTINUE"
    return "UNIDENTIFIABLE"


def _valid_frozen_signature(value: str, decision_state: Mapping[str, Any]) -> bool:
    if value == "CONTINUE":
        return True
    if not value.startswith("ACT:"):
        return False
    content = value.removeprefix("ACT:")
    return response_signature({"content": content}, decision_state=decision_state) == value


def _path_differences(left: Any, right: Any, path: str = "") -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        paths: list[str] = []
        for key in sorted(set(left) | set(right), key=str):
            child = f"{path}/{key}"
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(_path_differences(left[key], right[key], child))
        return paths
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [path]
        paths = []
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            paths.extend(_path_differences(left_item, right_item, f"{path}/{index}"))
        return paths
    return [] if left == right else [path]


def _pointer_tokens(path: str) -> tuple[str, ...]:
    if not path.startswith("/"):
        raise ValueError("selected result content path must be a JSON pointer")
    return tuple(token.replace("~1", "/").replace("~0", "~") for token in path[1:].split("/"))


def _selected_content(payload: Mapping[str, Any], path: str) -> tuple[Any, Any, str | int]:
    tokens = _pointer_tokens(path)
    if len(tokens) < 4 or tokens[0] != "messages" or tokens[2] != "content":
        raise ValueError("selected result content path must name a field inside a tool result")
    try:
        message_index = int(tokens[1])
    except ValueError as error:
        raise ValueError("selected tool message index must be an integer") from error
    messages = payload.get("messages")
    if not isinstance(messages, list) or not (0 <= message_index < len(messages)):
        raise ValueError("selected tool message index is outside the request")
    message = messages[message_index]
    if not isinstance(message, Mapping) or message.get("role") != "tool":
        raise ValueError("selected source message is not a tool result")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("selected tool result content is not JSON text")
    try:
        envelope = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("selected tool result content is not valid JSON") from error
    current = envelope
    for token in tokens[3:-1]:
        if isinstance(current, list):
            try:
                current = current[int(token)]
            except (ValueError, IndexError) as error:
                raise ValueError("selected result content path is outside the result envelope") from error
        elif isinstance(current, Mapping) and token in current:
            current = current[token]
        else:
            raise ValueError("selected result content path is outside the result envelope")
    leaf_token = tokens[-1]
    if isinstance(current, list):
        try:
            leaf: str | int = int(leaf_token)
            current[leaf]
        except (ValueError, IndexError) as error:
            raise ValueError("selected result content path is outside the result envelope") from error
    elif isinstance(current, Mapping) and leaf_token in current:
        leaf = leaf_token
    else:
        raise ValueError("selected result content path is outside the result envelope")
    return envelope, current, leaf


def _leaf(container: Any, key: str | int) -> Any:
    return container[key]


def validate_branch_payloads(
    intact: Mapping[str, Any],
    perturbed: Mapping[str, Any],
    *,
    selected_result_content_path: str,
    expected_intact_content: Any,
    expected_perturbed_content: Any,
) -> None:
    """Prove that matched branches differ at one frozen tool-content field only."""

    if intact.get("seed") != perturbed.get("seed"):
        raise ValueError("matched intervention branches must use the same derived seed")
    intact_envelope, intact_parent, intact_key = _selected_content(
        intact, selected_result_content_path
    )
    changed_envelope, changed_parent, changed_key = _selected_content(
        perturbed, selected_result_content_path
    )
    if _leaf(intact_parent, intact_key) != expected_intact_content:
        raise ValueError("intact branch does not contain the frozen selected content")
    if _leaf(changed_parent, changed_key) != expected_perturbed_content:
        raise ValueError("perturbed branch does not contain the frozen perturbation")
    tokens = _pointer_tokens(selected_result_content_path)
    allowed = f"/messages/{tokens[1]}/content"
    differences = _path_differences(intact, perturbed)
    if differences != [allowed]:
        raise ValueError(f"unauthorized branch payload differences: {differences}")
    envelope_path = "/" + "/".join(tokens[3:])
    if _path_differences(intact_envelope, changed_envelope) != [envelope_path]:
        raise ValueError("unauthorized result-envelope differences")


def build_branch_payloads(
    source_payload: Mapping[str, Any],
    *,
    selected_result_content_path: str,
    intact_content: Any,
    perturbed_content: Any,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    intact = deepcopy(dict(source_payload))
    perturbed = deepcopy(dict(source_payload))
    intact["seed"] = seed
    perturbed["seed"] = seed
    source_envelope, source_parent, source_key = _selected_content(
        source_payload, selected_result_content_path
    )
    if _leaf(source_parent, source_key) != intact_content:
        raise ValueError("source payload does not contain the frozen intact content")
    tokens = _pointer_tokens(selected_result_content_path)
    message_index = int(tokens[1])
    source_message = source_payload["messages"][message_index]
    if source_message["content"] != canonical_json(source_envelope):
        raise ValueError("source result envelope is not canonically encoded")
    changed_envelope, changed_parent, changed_key = _selected_content(
        perturbed, selected_result_content_path
    )
    changed_parent[changed_key] = perturbed_content
    perturbed["messages"][message_index]["content"] = canonical_json(changed_envelope)
    validate_branch_payloads(
        intact,
        perturbed,
        selected_result_content_path=selected_result_content_path,
        expected_intact_content=intact_content,
        expected_perturbed_content=perturbed_content,
    )
    return intact, perturbed


def execute_shadow_bundle(
    *,
    source_payload: Mapping[str, Any],
    source_payload_sha256: str,
    candidate_id: str,
    generation: str,
    selected_result_content_path: str,
    intact_content: Any,
    perturbed_content: Any,
    original_signature: str,
    expected_changed_signature: str,
    decision_state: Mapping[str, Any],
    complete_payload: Callable[[Mapping[str, Any]], ChatResponse],
) -> dict[str, Any]:
    """Execute the frozen ten-call shadow bundle without a game environment."""

    source_bytes = serialize_request_payload(source_payload)
    actual_source_hash = sha256(source_bytes).hexdigest()
    if actual_source_hash != source_payload_sha256:
        raise ValueError("source payload does not match its pre-redaction commitment")
    valid_expected = _valid_frozen_signature(expected_changed_signature, decision_state)
    valid_original = _valid_frozen_signature(original_signature, decision_state)
    if not valid_original or not valid_expected or original_signature == expected_changed_signature:
        raise ValueError("candidate signatures must be distinct frozen mechanical signatures")

    schedule = build_shadow_schedule(candidate_id=candidate_id, generation=generation)
    responses: list[dict[str, Any]] = []
    failed_attempts: list[dict[str, Any]] = []
    signatures: dict[str, list[str]] = {"intact": [], "perturbed": []}
    seeds: dict[str, int] = {}
    for row in schedule:
        replicate = int(row["replicate"])
        seed = int(row["seed"])
        seeds[str(replicate)] = seed
        intact, perturbed = build_branch_payloads(
            source_payload,
            selected_result_content_path=selected_result_content_path,
            intact_content=intact_content,
            perturbed_content=perturbed_content,
            seed=seed,
        )
        payload = intact if row["branch"] == "intact" else perturbed
        payload_bytes = serialize_request_payload(payload)
        payload_hash = sha256(payload_bytes).hexdigest()
        response: ChatResponse | None = None
        errors: list[str] = []
        for attempt in range(1, 3):
            try:
                response = complete_payload(deepcopy(payload))
                break
            except ModelRequestError as error:
                errors.append(str(error))
                failed_attempts.append({
                    **dict(row),
                    "attempt": attempt,
                    "request_payload": payload,
                    "request_payload_sha256": payload_hash,
                    "request_payload_encoding": "json-utf8-sort-keys-compact-v1",
                    "response_body_raw": error.response_body_raw,
                    "response_body_base64": error.response_body_base64
                    or (
                        b64encode(error.response_body_raw.encode("utf-8")).decode("ascii")
                        if error.response_body_raw is not None
                        else None
                    ),
                    "response_body_sha256": error.response_body_sha256,
                    "error": str(error),
                })
        if response is None:
            return {
                "schema": INTERVENTION_SCHEMA,
                "status": "content_dependence_unidentifiable",
                "reason": "shadow_transport_failure",
                "source_reconstruction_matches": True,
                "schedule": schedule,
                "seeds": seeds,
                "responses": responses,
                "failed_attempts": failed_attempts,
                "errors": errors,
            }
        if response.request_payload_sha256 != payload_hash:
            return {
                "schema": INTERVENTION_SCHEMA,
                "status": "content_dependence_unidentifiable",
                "reason": "shadow_payload_commitment_failure",
                "source_reconstruction_matches": True,
                "schedule": schedule,
                "seeds": seeds,
                "responses": responses,
                "failed_attempts": failed_attempts,
            }
        signature = response_signature(response.message, decision_state=decision_state)
        signatures[str(row["branch"])].append(signature)
        responses.append({
            **dict(row),
            "request_payload": payload,
            "request_payload_sha256": payload_hash,
            "response": response.message,
            "response_id": response.response_id,
            "response_metadata": response.response_metadata,
            "response_body": response.response_body,
            "response_body_sha256": response.response_body_sha256,
            "response_body_raw": response.response_body_raw,
            "response_body_base64": response.response_body_base64
            or (
                b64encode(response.response_body_raw.encode("utf-8")).decode("ascii")
                if response.response_body_raw is not None
                else None
            ),
            "usage": dict(response.usage),
            "signature": signature,
            "attempts": len(errors) + 1,
        })
    status = assess_content_dependence(
        source_reconstruction_matches=True,
        original_signature=original_signature,
        changed_signature=expected_changed_signature,
        intact_signatures=tuple(signatures["intact"]),
        perturbed_signatures=tuple(signatures["perturbed"]),
    )
    return {
        "schema": INTERVENTION_SCHEMA,
        "status": status,
        "source_reconstruction_matches": True,
        "source_payload_sha256": actual_source_hash,
        "candidate_id": candidate_id,
        "generation": generation,
        "decision_state": dict(decision_state),
        "original_signature": original_signature,
        "expected_changed_signature": expected_changed_signature,
        "schedule": schedule,
        "seeds": seeds,
        "responses": responses,
        "failed_attempts": failed_attempts,
        "intact_signatures": signatures["intact"],
        "perturbed_signatures": signatures["perturbed"],
    }


def validate_shadow_bundle_result(result: Mapping[str, Any]) -> str:
    """Recompute a completed shadow result from its credential-free raw evidence."""

    if result.get("schema") != INTERVENTION_SCHEMA:
        raise ValueError("shadow result schema mismatch")
    candidate = result.get("candidate_id")
    generation = result.get("generation")
    decision_state = result.get("decision_state")
    original = result.get("original_signature")
    changed = result.get("expected_changed_signature")
    if (
        not isinstance(candidate, str)
        or not candidate
        or not isinstance(generation, str)
        or not generation
        or not isinstance(original, str)
        or not original
        or not isinstance(changed, str)
        or not changed
    ):
        raise ValueError("shadow result is missing frozen identity or signatures")
    if not isinstance(decision_state, Mapping):
        raise ValueError("shadow result is missing the production decision state")
    expected_schedule = list(build_shadow_schedule(candidate_id=candidate, generation=generation))
    raw_schedule = result.get("schedule")
    if raw_schedule != expected_schedule and raw_schedule != tuple(expected_schedule):
        raise ValueError("shadow result schedule differs from its frozen derivation")
    responses = result.get("responses")
    if not isinstance(responses, list) or len(responses) != 10:
        raise ValueError("completed shadow result requires exactly ten responses")
    signatures: dict[str, list[str]] = {"intact": [], "perturbed": []}
    for expected, response in zip(expected_schedule, responses, strict=True):
        if not isinstance(response, Mapping) or any(
            response.get(field) != expected[field]
            for field in (
                "request_order",
                "replicate",
                "within_replicate_order",
                "branch",
                "seed",
            )
        ):
            raise ValueError("shadow response order differs from the frozen schedule")
        payload = response.get("request_payload")
        if not isinstance(payload, Mapping):
            raise ValueError("shadow response is missing its exact request payload")
        payload_hash = sha256(serialize_request_payload(payload)).hexdigest()
        if response.get("request_payload_sha256") != payload_hash:
            raise ValueError("shadow request payload hash mismatch")
        body_base64 = response.get("response_body_base64")
        if not isinstance(body_base64, str):
            raise ValueError("shadow response omits exact response bytes")
        try:
            body_bytes = b64decode(body_base64, validate=True)
        except (Base64Error, ValueError) as error:
            raise ValueError("shadow response body base64 is invalid") from error
        if response.get("response_body_sha256") != sha256(body_bytes).hexdigest():
            raise ValueError("shadow response body hash mismatch")
        try:
            parsed_body = json.loads(body_bytes)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("shadow response body is not the committed JSON") from error
        if parsed_body != response.get("response_body"):
            raise ValueError("shadow parsed response differs from exact response bytes")
        message = response.get("response")
        if not isinstance(message, Mapping):
            raise ValueError("shadow response message is missing")
        signature = response_signature(message, decision_state=decision_state)
        if response.get("signature") != signature:
            raise ValueError("shadow response signature does not match raw response evidence")
        signatures[str(expected["branch"])].append(signature)
    expected_status = assess_content_dependence(
        source_reconstruction_matches=result.get("source_reconstruction_matches") is True,
        original_signature=original,
        changed_signature=changed,
        intact_signatures=tuple(signatures["intact"]),
        perturbed_signatures=tuple(signatures["perturbed"]),
    )
    if result.get("status") != expected_status:
        raise ValueError("shadow status does not match the recomputed response evidence")
    if result.get("intact_signatures") != signatures["intact"] or result.get(
        "perturbed_signatures"
    ) != signatures["perturbed"]:
        raise ValueError("shadow signature summaries do not match raw response evidence")
    return expected_status


def assess_content_dependence(
    *,
    source_reconstruction_matches: bool,
    original_signature: str,
    changed_signature: str,
    intact_signatures: Sequence[str],
    perturbed_signatures: Sequence[str],
) -> str:
    if len(intact_signatures) != 5 or len(perturbed_signatures) != 5:
        raise ValueError("the intervention requires exactly five responses per branch")
    valid_signatures = lambda values: all(
        value == "CONTINUE" or value.startswith("ACT:") for value in values
    )
    if not source_reconstruction_matches or not valid_signatures(intact_signatures) or not valid_signatures(perturbed_signatures):
        return "content_dependence_unidentifiable"
    intact_original = sum(value == original_signature for value in intact_signatures)
    perturbed_changed = sum(value == changed_signature for value in perturbed_signatures)
    intact_cross = sum(value == changed_signature for value in intact_signatures)
    perturbed_cross = sum(value == original_signature for value in perturbed_signatures)
    if intact_original < 4:
        return "content_dependence_unidentifiable"
    if (
        original_signature != changed_signature
        and perturbed_changed >= 4
        and intact_cross <= 1
        and perturbed_cross <= 1
    ):
        return "demonstrated"
    return "no_local_content_dependence_detected"
