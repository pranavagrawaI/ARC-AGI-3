"""Closed offline Game Judgment contract, request assembler, and scorer."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


ARMS = tuple("ABCDEFGH")


class ContractError(ValueError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(value: Any) -> str:
    return sha256(canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class Decision:
    tool: str
    arguments: dict[str, Any]

    @property
    def signature(self) -> tuple[Any, ...]:
        if self.tool == "EXECUTE":
            return (self.tool, self.arguments["plan_id"], self.arguments["first_action"])
        if self.tool == "PROBE":
            return (self.tool, self.arguments["discriminator_id"], self.arguments["action"])
        return (self.tool, self.arguments["reason_code"])


def _keys(value: dict[str, Any], required: set[str]) -> None:
    if set(value) != required:
        raise ContractError(f"fields must be exactly {sorted(required)}")


def parse_decision(raw: str | bytes | dict[str, Any]) -> Decision:
    try:
        value = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    except json.JSONDecodeError as exc:
        raise ContractError("response is not one JSON object") from exc
    if not isinstance(value, dict):
        raise ContractError("response is not one JSON object")
    _keys(value, {"tool", "arguments"})
    tool, args = value["tool"], value["arguments"]
    if tool not in {"EXECUTE", "PROBE", "ABSTAIN"} or not isinstance(args, dict):
        raise ContractError("undeclared tool or non-object arguments")
    if tool == "EXECUTE":
        _keys(args, {"plan_id", "first_action", "predicted_public_fields"})
        if not isinstance(args["plan_id"], str) or not isinstance(args["first_action"], int):
            raise ContractError("invalid EXECUTE identifiers")
        predictions = args["predicted_public_fields"]
        if not isinstance(predictions, dict) or not predictions or any(v == "UNKNOWN" for v in predictions.values()):
            raise ContractError("EXECUTE requires concrete public predictions")
    elif tool == "PROBE":
        _keys(args, {"discriminator_id", "action", "predicted_outcomes"})
        if not isinstance(args["discriminator_id"], str) or not isinstance(args["action"], int):
            raise ContractError("invalid PROBE identifiers")
        outcomes = args["predicted_outcomes"]
        if not isinstance(outcomes, dict) or len(outcomes) < 2 or any(v == "UNKNOWN" for v in outcomes.values()):
            raise ContractError("PROBE requires at least two concrete outcomes")
    else:
        _keys(args, {"reason_code", "missing_evidence"})
        if not isinstance(args["reason_code"], str) or not isinstance(args["missing_evidence"], list):
            raise ContractError("invalid ABSTAIN arguments")
    return Decision(tool, args)


def visible_case(case: dict[str, Any], arm: str) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm}")
    rank = ARMS.index(arm)
    out: dict[str, Any] = {
        "schema": "0003-model-request-v1",
        "case_id": case["case_id"],
        "burden": case["burden"],
        "legal_actions": case["legal_actions"],
        "current_public": case["current_public"],
        "images": [case["images"][-1]],
        "contract": case["contract"],
    }
    if rank >= 1: out["images"] = case["images"]
    if rank >= 2: out["step_labels"] = case["step_labels"]
    if rank >= 3: out["numeric_transitions"] = case["numeric_transitions"]
    if arm == "E": out["reflection"] = case["reflection"]
    if rank >= 5: out["candidate_scene"] = case["candidate_scene"]; out["executable_conjectures"] = case["executable_conjectures"]
    if rank >= 6: out["candidate_plans"] = case["candidate_plans"]
    if rank >= 7: out["verified_route"] = case["verified_route"]
    return out


def score(case: dict[str, Any], labels: dict[str, Any], decision: Decision) -> dict[str, Any]:
    legal = decision.tool == "ABSTAIN" or decision.arguments.get("action", decision.arguments.get("first_action")) in case["legal_actions"]
    semantics = True
    prediction = None
    decision_ok = decision.signature in {tuple(row) for row in labels["valid_decision_signatures"]}
    if decision.tool == "EXECUTE":
        plan = next((p for p in case["candidate_plans"] if p["plan_id"] == decision.arguments["plan_id"]), None)
        semantics = plan is not None and plan["actions"][0] == decision.arguments["first_action"]
        prediction = decision.arguments["predicted_public_fields"] == labels["protected_successor"]
    elif decision.tool == "PROBE":
        semantics = any(d["discriminator_id"] == decision.arguments["discriminator_id"] for d in case["discriminators"])
    return {"contract": True, "semantics": semantics, "legality": legal, "prediction": prediction,
            "decision": decision_ok, "pass": bool(semantics and legal and decision_ok and prediction is not False)}
