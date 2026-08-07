from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


TOOL_ARGUMENT_FIELDS = {
    "EXECUTE": ("plan_id", "first_action", "predicted_public_fields"),
    "PROBE": ("discriminator_id", "action", "predicted_outcomes"),
    "ABSTAIN": ("reason_code", "missing_evidence"),
}

CONTRACT = {
    "tools": {tool: list(fields) for tool, fields in TOOL_ARGUMENT_FIELDS.items()},
    "exactly_one_call": True,
}


class ContractError(ValueError):
    pass


@dataclass(frozen=True)
class Decision:
    tool: str
    arguments: dict[str, Any]

    @property
    def signature(self) -> tuple[Any, ...]:
        identifying_fields = {
            "EXECUTE": ("plan_id", "first_action"),
            "PROBE": ("discriminator_id", "action"),
            "ABSTAIN": ("reason_code",),
        }
        return (self.tool, *(self.arguments[field] for field in identifying_fields[self.tool]))


def _exact_fields(value: dict[str, Any], required: tuple[str, ...] | set[str]) -> None:
    if set(value) != set(required):
        raise ContractError(f"fields must be exactly {sorted(required)}")


def _integer(value: Any) -> bool:
    return type(value) is int


def parse_decision(raw: str | bytes | dict[str, Any]) -> Decision:
    try:
        value = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    except json.JSONDecodeError as exc:
        raise ContractError("response is not one JSON object") from exc
    if not isinstance(value, dict):
        raise ContractError("response is not one JSON object")
    _exact_fields(value, {"tool", "arguments"})
    tool, arguments = value["tool"], value["arguments"]
    if tool not in TOOL_ARGUMENT_FIELDS or not isinstance(arguments, dict):
        raise ContractError("undeclared tool or non-object arguments")
    _exact_fields(arguments, TOOL_ARGUMENT_FIELDS[tool])

    if tool == "EXECUTE":
        if not isinstance(arguments["plan_id"], str) or not _integer(arguments["first_action"]):
            raise ContractError("invalid EXECUTE identifiers")
        predictions = arguments["predicted_public_fields"]
        if (
            not isinstance(predictions, dict)
            or not predictions
            or any(value == "UNKNOWN" for value in predictions.values())
        ):
            raise ContractError("EXECUTE requires concrete public predictions")
    elif tool == "PROBE":
        if not isinstance(arguments["discriminator_id"], str) or not _integer(arguments["action"]):
            raise ContractError("invalid PROBE identifiers")
        outcomes = arguments["predicted_outcomes"]
        if (
            not isinstance(outcomes, dict)
            or len(outcomes) < 2
            or any(value == "UNKNOWN" for value in outcomes.values())
        ):
            raise ContractError("PROBE requires at least two concrete outcomes")
    else:
        missing = arguments["missing_evidence"]
        if (
            not isinstance(arguments["reason_code"], str)
            or not arguments["reason_code"]
            or not isinstance(missing, list)
            or not missing
            or any(not isinstance(item, str) or not item for item in missing)
            or len(set(missing)) != len(missing)
        ):
            raise ContractError("ABSTAIN requires a reason and non-empty unique missing evidence")
    return Decision(tool, arguments)
