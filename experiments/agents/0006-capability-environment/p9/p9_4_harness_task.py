"""P9.4 — does the harness carry a real task through injected faults?

Question: with a reactive policy pursuing one objective across an episode, does
0002-R still deliver a correct result when the transcript evicts, the interpreter
times out, the transport fails, output overflows, and tool arguments are
malformed?

This is the escalation of the verifier. P9.0–P9.3 assert that the arm matches its
declared contract, which can only ever confirm the contract. Here the measurement
is task completion: the policy must finish with an ``index.json`` that matches
digests the host computed itself. A harness weakness shows up as a wrong answer,
not as a failed assertion.

Each condition runs the identical policy and task. Only the injected fault differs.

Run: ``uv run python3 experiments/agents/0006-capability-environment/p9/p9_4_harness_task.py``
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "p9"

from p9.arms import build_arm, control_spec
from p9.evidence import Checks, artifacts_root
from p9.failures import classify_tool_result
from p9.fixtures import RecordedEnvironment, default_recording, load_recorded_frames
from p9.task_policy import (
    OVERSIZED_PROBE,
    TIMEOUT_PROBE,
    TaskPolicy,
    expected_index,
    grade_index,
)

from llm_repl_agent.observations import ColorPermutation

from experiments.harness import run_episode


ACTIONS = 4
FRAMES = ACTIONS + 1


@dataclass(frozen=True)
class Condition:
    name: str
    stresses: str
    context_turns: int = 6
    probe_at_step: dict[int, str] | None = None
    malformed_at_step: set[int] | None = None
    transport_failure_at_call: int | None = None
    workspace_timeout_seconds: float = 20.0


CONDITIONS = (
    Condition("baseline", "nothing; establishes the task is achievable"),
    Condition("transcript_eviction", "every turn but one leaves the context", context_turns=1),
    Condition(
        "interpreter_timeout",
        "the interpreter is killed and restarted mid-task",
        probe_at_step={2: TIMEOUT_PROBE},
        workspace_timeout_seconds=2.0,
    ),
    Condition("retryable_transport_failure", "one provider failure mid-task", transport_failure_at_call=4),
    Condition("oversized_tool_output", "a tool result overflows the model output limit",
              probe_at_step={2: OVERSIZED_PROBE}),
    Condition("malformed_tool_arguments", "the model emits unparsable tool arguments",
              malformed_at_step={2}),
    Condition(
        "everything_at_once",
        "eviction, timeout, transport failure and malformed arguments together",
        context_turns=1,
        probe_at_step={2: TIMEOUT_PROBE},
        malformed_at_step={3},
        transport_failure_at_call=4,
        workspace_timeout_seconds=2.0,
    ),
)


def run_condition(condition: Condition, frames) -> dict[str, Any]:
    policy = TaskPolicy(
        probe_at_step=dict(condition.probe_at_step or {}),
        malformed_at_step=set(condition.malformed_at_step or set()),
        transport_failure_at_call=condition.transport_failure_at_call,
    )
    spec = control_spec(
        max_actions=ACTIONS,
        max_model_calls=60,
        max_model_rounds_per_action=8,
        max_context_turns=condition.context_turns,
        workspace_timeout_seconds=condition.workspace_timeout_seconds,
    )
    with TemporaryDirectory(prefix=f"p9-4-{condition.name}-") as scratch:
        with build_arm(spec, run_dir=Path(scratch) / "arm", inner_client=policy) as arm:
            result = run_episode(
                RecordedEnvironment(frames=frames), arm.agent, max_actions=ACTIONS
            )
            files = arm.workspace.evidence_state()
            log = (arm.run_dir / "process.jsonl").read_text(encoding="utf-8")
    events = [json.loads(line) for line in log.splitlines()]
    tool_outputs = [json.loads(e["output"]) for e in events if e["type"] == "python_tool"]
    grade = grade_index(
        files.get("index.json", {}).get("text", ""),
        expected_index(
            frames, colors=ColorPermutation.identity(), through_step=result.actions - 1
        ),
    )
    return {
        "condition": condition.name,
        "stresses": condition.stresses,
        "actions": result.actions,
        "model_calls": policy.calls,
        "grade": grade,
        "tool_kinds": sorted({classify_tool_result(o).kind for o in tool_outputs}),
        "fault_observed": {
            "evictions": sum(1 for e in events if e["type"] == "context_eviction"),
            "restarts": sum(1 for o in tool_outputs if o.get("workspace_restarted")),
            "truncations": sum(1 for o in tool_outputs if o.get("truncated")),
            "transport_retries": sum(1 for e in events if e["type"] == "model_transport_retry"),
            "decision_repairs": sum(1 for e in events if e["type"] == "decision_repair"),
        },
    }


def main() -> int:
    checks = Checks(
        prototype="P9.4-harness-task",
        question=(
            "Does a reactive policy pursuing one objective finish with a correct "
            "answer under eviction, timeout, transport failure, oversized output "
            "and malformed tool arguments?"
        ),
    )
    repo = Path(__file__).resolve().parents[4]
    frames = load_recorded_frames(
        default_recording(repo, minimum_frames=FRAMES), count=FRAMES, crop=8
    )

    outcomes = [run_condition(condition, frames) for condition in CONDITIONS]
    by_name = {outcome["condition"]: outcome for outcome in outcomes}

    baseline = by_name["baseline"]
    checks.record(
        "the_task_is_achievable_at_all",
        baseline["grade"]["complete"] and baseline["grade"]["expected_entries"] >= ACTIONS,
        baseline["grade"],
    )
    for outcome in outcomes:
        checks.record(
            f"task_completes_under_{outcome['condition']}",
            outcome["grade"]["complete"],
            {"grade": outcome["grade"], "observed": outcome["fault_observed"]},
        )

    # A condition that never triggered its fault would pass vacuously.
    checks.record(
        "the_eviction_condition_actually_evicted",
        by_name["transcript_eviction"]["fault_observed"]["evictions"] > 0,
        by_name["transcript_eviction"]["fault_observed"],
    )
    checks.record(
        "the_timeout_condition_actually_restarted_the_interpreter",
        by_name["interpreter_timeout"]["fault_observed"]["restarts"] > 0
        and "execution_timeout_restart" in by_name["interpreter_timeout"]["tool_kinds"],
        by_name["interpreter_timeout"],
    )
    checks.record(
        "the_transport_condition_actually_retried",
        by_name["retryable_transport_failure"]["fault_observed"]["transport_retries"] == 1,
        by_name["retryable_transport_failure"]["fault_observed"],
    )
    checks.record(
        "the_oversized_condition_actually_truncated",
        by_name["oversized_tool_output"]["fault_observed"]["truncations"] > 0,
        by_name["oversized_tool_output"]["fault_observed"],
    )
    checks.record(
        "the_malformed_condition_actually_hit_validation",
        "tool_argument_validation_failure" in by_name["malformed_tool_arguments"]["tool_kinds"],
        by_name["malformed_tool_arguments"]["tool_kinds"],
    )
    combined = by_name["everything_at_once"]["fault_observed"]
    checks.record(
        "the_combined_condition_triggered_more_than_one_fault",
        sum(1 for value in combined.values() if value) >= 3,
        combined,
    )
    checks.record(
        "no_condition_silently_lost_an_action",
        all(outcome["actions"] == ACTIONS for outcome in outcomes),
        {outcome["condition"]: outcome["actions"] for outcome in outcomes},
    )

    checks.finish(
        artifacts_root() / "p9-4-harness-task" / "evidence.json",
        conditions=outcomes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
