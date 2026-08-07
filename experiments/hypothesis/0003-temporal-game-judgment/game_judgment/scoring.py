from __future__ import annotations

from typing import Any

from .contract import Decision
from .machinery import discriminator_is_decision_relevant


def _find(rows: list[dict[str, Any]], field: str, value: Any) -> dict[str, Any] | None:
    return next((row for row in rows if row[field] == value), None)


def score(case: dict[str, Any], labels: dict[str, Any], decision: Decision) -> dict[str, Any]:
    action = decision.arguments.get("action", decision.arguments.get("first_action"))
    legal = decision.tool == "ABSTAIN" or action in case["legal_actions"]
    labelled = decision.signature in {tuple(row) for row in labels["valid_decision_signatures"]}
    semantics = False
    prediction: bool | None = None

    if decision.tool == "EXECUTE":
        plan = _find(case["candidate_plans"], "plan_id", decision.arguments["plan_id"])
        statuses = {row["id"]: row["replay"]["status"] for row in case["executable_conjectures"]}
        dependencies_work = bool(plan) and all(
            statuses.get(dependency) == "working" for dependency in plan["conjecture_ids"]
        )
        semantics = bool(
            plan
            and plan["actions"]
            and plan["actions"][0] == decision.arguments["first_action"]
            and dependencies_work
            and decision.arguments["predicted_public_fields"] == plan["predicted_public_fields"]
        )
        prediction = decision.arguments["predicted_public_fields"] == labels["protected_successor"]
    elif decision.tool == "PROBE":
        discriminator = _find(
            case["discriminators"], "discriminator_id", decision.arguments["discriminator_id"]
        )
        semantics = bool(
            discriminator
            and discriminator_is_decision_relevant(discriminator)
            and discriminator["action"] == decision.arguments["action"]
            and discriminator["predicted_outcomes"] == decision.arguments["predicted_outcomes"]
        )
        prediction = bool(
            discriminator
            and labels["protected_successor"] in discriminator["predicted_outcomes"].values()
        )
    else:
        expected = {
            (row["reason_code"], tuple(row["missing_evidence"]))
            for row in labels.get("valid_abstentions", [])
        }
        semantics = (
            decision.arguments["reason_code"], tuple(decision.arguments["missing_evidence"])
        ) in expected

    passed = bool(semantics and legal and labelled and prediction is not False)
    return {
        "contract": True,
        "semantics": semantics,
        "legality": legal,
        "prediction": prediction,
        "decision": labelled,
        "pass": passed,
    }
