"""Host-side visible replay using the canonical observation object."""

from __future__ import annotations

import copy
from typing import Any

from observation_abi import observation_from_state, validate_observation
from real_trace import measure


def replay_visible(module: Any, fixture: dict[str, Any]) -> dict[str, Any]:
    rows = fixture["visible"]
    initial = validate_observation(observation_from_state(rows[0]["before"]))
    state = module.initialize(copy.deepcopy(initial))
    checks = []
    for row in rows:
        produced = module.predict(copy.deepcopy(state), row["action"])
        if not isinstance(produced, list) or not produced:
            raise ValueError("predict must return nonempty list")
        measurement = measure(row["after"]["grid"])
        item = produced[0]
        commitments = item["commitments"]
        compatible = all(commitments.get(k) == measurement[k] for k in fixture["required_targets"])
        checks.append(
            {
                "step": row["step"],
                "action": row["action"],
                "measurement": measurement,
                "status": "passed" if compatible else "contradicted",
            }
        )
        if not compatible:
            return {"status": "contradicted", "checks": checks}
        observation = validate_observation(observation_from_state(row["after"]))
        state = module.assimilate(
            copy.deepcopy(item["state"]),
            observation,
            {"status": "compatible", "measurement": measurement},
        )
    return {"status": "passed", "checks": checks}
