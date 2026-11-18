"""Generic sealed visible ABI replay. Not the LS20 semantic observer."""

from __future__ import annotations

import hashlib
from typing import Any

LIVE = __import__("pathlib").Path(__file__).resolve().parents[1] / "construction-policy-live-20260914T093351Z"


def evaluate_visible_abi(source: str, case: Any, include_suffix: bool = False, boundary=None) -> dict[str, Any]:
    import sys

    if str(LIVE) not in sys.path:
        sys.path.insert(0, str(LIVE))
    from integration_boundary import CandidateRPC, SealedBoundary

    if include_suffix:
        raise ValueError("this evaluator does not assess prospective suffix evidence")
    fixture = case.evaluator_payload(include_suffix=False)
    rows = fixture["visible"]
    module = CandidateRPC(boundary or SealedBoundary(), source)
    digest = hashlib.sha256(source.encode()).hexdigest()
    start = {"grid": rows[0]["before"]["grid"]}
    try:
        state = module.initialize(start)
        if not isinstance(state, dict):
            raise ValueError("initialize must return an object")
        checks = []
        for row in rows:
            produced = module.predict(state, row["action"])
            if not isinstance(produced, list) or not produced:
                raise ValueError("predict must return a nonempty list")
            if len(produced) != 1:
                raise ValueError("predict returned multiple branches; host does not drop branches")
            item = produced[0]
            if not isinstance(item, dict) or not {"state", "commitments"} <= set(item):
                raise ValueError("predict branch must contain state and commitments")
            deps = item.get("declared_dependencies", [])
            if not isinstance(item["commitments"], dict) or not isinstance(deps, list):
                raise ValueError("commitments and declared_dependencies have bad types")
            state = module.assimilate(
                item["state"],
                {"grid": row["after"]["grid"]},
                {"status": "unscored"},
            )
            if not isinstance(state, dict):
                raise ValueError("assimilate must return an object")
            checks.append({"step": row["step"], "action": row["action"], "status": "abi_replayed"})
        visible = {"status": "passed", "kind": "visible_abi_replay", "checks": checks}
    except Exception as error:  # fail closed; constructor may repair
        visible = {"status": "failed", "kind": "visible_abi_replay", "error": f"{type(error).__name__}: {error}"}
    return {
        "model_sha256": digest,
        "rpc_methods": module.calls,
        "visible": visible,
        "suffix": {"status": "not_run", "kind": "prospective_not_assessed"},
        "coverage": {"required": [], "missing_required": []},
        "evidence_claim": "ABI_VISIBLE_REPLAY_ONLY_NOT_SUPPORTED_MODEL",
        "dependencies": {
            "declared_on_visible_cases": [],
            "encountered_on_proposed_rollout": [],
            "static_completeness_established": False,
        },
    }
