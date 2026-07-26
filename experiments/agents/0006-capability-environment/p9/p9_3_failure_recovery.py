"""P9.3 — six failure modes, six distinguishable behaviors, one invisible retry.

Question: does every failure the loop can produce come back distinguishable and
reconstructable, does each land with whoever can repair it, and is the automatic
retry confined to the transport failures the client itself marked retryable?

Cases exercised: malformed tool arguments, a Python syntax error, a Python
runtime error, oversized output, an execution timeout with interpreter restart,
an illegal action, a retryable transport failure, and — as the discriminator for
the last one — a fatal transport failure.

Run: ``uv run python3 experiments/agents/0006-capability-environment/p9/p9_3_failure_recovery.py``
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "p9"

from p9.arms import build_arm, control_spec
from p9.evidence import Checks, artifacts_root
from p9.failures import (
    classify_decision_error,
    classify_tool_result,
    classify_transport_error,
)
from p9.fixtures import (
    RecordedEnvironment,
    ScriptedChatClient,
    action_turn,
    default_recording,
    load_recorded_frames,
    malformed_tool_turn,
    tool_turn,
    transport_failure,
)

from llm_repl_agent.client import ModelRequestError

from experiments.harness import Decision, run_episode
from experiments.harness.runner import IllegalActionError


SETUP = """\
in_memory_marker = 'PRESENT'
open('persist.txt', 'w').write('kept')
print('SETUP_OK')
"""

SYNTAX_ERROR = "def broken(:\n    return 1\n"
RUNTIME_ERROR = "value = 1 / 0\n"
OVERSIZED = "print('x' * 200000)\n"
TIMEOUT = "while True:\n    pass\n"

AFTER_RESTART = """\
print('IN_MEMORY_MARKER_PRESENT', 'in_memory_marker' in dir())
print('FILE_SURVIVED', open('persist.txt').read())
print('OBSERVATIONS_READABLE', len(open('observations/step-00000.json').read()) > 0)
"""

AFTER_RETRY = "print('AFTER_RETRY_OK')\n"

SCRIPT = (
    tool_turn(SETUP),                                                    # round 1
    malformed_tool_turn("{not json"),                                    # round 2
    tool_turn(SYNTAX_ERROR),                                             # round 3
    tool_turn(RUNTIME_ERROR),                                            # round 4
    tool_turn(OVERSIZED),                                                # round 5
    tool_turn(TIMEOUT),                                                  # round 6
    tool_turn(AFTER_RESTART),                                            # round 7
    transport_failure("simulated 503 from provider", retryable=True, http_status=503),
    tool_turn(AFTER_RETRY),                                              # round 8 (after retry)
    action_turn(99, reason="illegal on purpose"),                        # round 9
    action_turn(1, reason="repaired"),                                   # round 10
)

FATAL_SCRIPT = (
    transport_failure("simulated 400 bad request", retryable=False, http_status=400),
)


class _IllegalAgent:
    """Minimal agent that returns an action the environment never offered."""

    def act(self, observation) -> Decision:
        return Decision(action=99)

    def observe(self, transition) -> None:  # pragma: no cover - never reached
        raise AssertionError("an illegal action must not reach observe")


def _events(path: Path, event_type: str) -> list[dict]:
    return [
        record
        for record in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
        if record["type"] == event_type
    ]


def _exception_type(output: dict) -> str | None:
    traceback = output.get("traceback")
    if not isinstance(traceback, str) or not traceback.strip():
        return None
    return traceback.strip().splitlines()[-1].split(":", 1)[0].strip()


def _signature(output: dict) -> tuple:
    outcome = classify_tool_result(output)
    return (outcome.kind, outcome.truncated, _exception_type(output))


def main() -> int:
    checks = Checks(
        prototype="P9.3-failure-recovery",
        question=(
            "Is every failure mode distinguishable and reconstructable, routed to "
            "whoever can repair it, and retried only when the transport said it was safe?"
        ),
    )
    repo = Path(__file__).resolve().parents[4]
    frames = load_recorded_frames(default_recording(repo, minimum_frames=2), count=2, crop=8)

    with TemporaryDirectory(prefix="p9-3-") as scratch:
        root = Path(scratch)
        client = ScriptedChatClient(SCRIPT)
        spec = control_spec(
            max_actions=1,
            max_model_calls=20,
            max_model_rounds_per_action=12,
            workspace_timeout_seconds=3.0,
        )
        with build_arm(spec, run_dir=root / "arm", inner_client=client) as arm:
            result = run_episode(
                RecordedEnvironment(frames=frames), arm.agent, max_actions=spec.max_actions
            )
            log = arm.run_dir / "process.jsonl"
            tool_outputs = [json.loads(event["output"]) for event in _events(log, "python_tool")]
            repairs = _events(log, "decision_repair")
            retries = _events(log, "model_transport_retry")
            attempt_failures = _events(log, "model_transport_attempt_failed")
            request_failures = _events(log, "model_request_failed")
            payloads = client.request_payloads()
            transport_attempts = arm.client.attempts
            transport_retries = arm.client.retries

        fatal_client = ScriptedChatClient(FATAL_SCRIPT)
        with build_arm(
            control_spec(max_actions=1, max_model_calls=20),
            run_dir=root / "fatal",
            inner_client=fatal_client,
        ) as fatal_arm:
            fatal_frames = load_recorded_frames(
                default_recording(repo, minimum_frames=2), count=2, crop=8
            )
            fatal_environment = RecordedEnvironment(frames=fatal_frames)
            fatal_error: ModelRequestError | None = None
            try:
                run_episode(fatal_environment, fatal_arm.agent, max_actions=1)
            except ModelRequestError as error:
                fatal_error = error
            fatal_log = fatal_arm.run_dir / "process.jsonl"
            fatal_retries = _events(fatal_log, "model_transport_retry")
            fatal_attempts = fatal_arm.client.attempts
            fatal_request_failures = _events(fatal_log, "model_request_failed")

    setup, malformed, syntax, runtime, oversized, timed_out, after_restart, after_retry = (
        tool_outputs
    )

    checks.record(
        "the_script_ran_to_completion_and_the_run_recovered",
        client.remaining == 0 and result.actions == 1,
        {"remaining_script_entries": client.remaining, "actions": result.actions},
    )

    cases = {
        "malformed_tool_arguments": malformed,
        "python_syntax_error": syntax,
        "python_runtime_error": runtime,
        "oversized_output": oversized,
        "execution_timeout": timed_out,
    }
    signatures = {name: _signature(output) for name, output in cases.items()}
    checks.record(
        "every_tool_failure_mode_has_a_distinct_model_visible_signature",
        len(set(signatures.values())) == len(signatures),
        {name: list(signature) for name, signature in signatures.items()},
    )
    checks.record(
        "malformed_tool_arguments_are_a_validation_failure_not_an_interpreter_failure",
        signatures["malformed_tool_arguments"][0] == "tool_argument_validation_failure"
        and malformed["traceback"] is None
        and not malformed["workspace_restarted"],
        {"stderr": malformed["stderr"], "signature": list(signatures["malformed_tool_arguments"])},
    )
    checks.record(
        "syntax_and_runtime_failures_share_a_kind_but_name_different_exceptions",
        signatures["python_syntax_error"][0] == signatures["python_runtime_error"][0]
        == "python_execution_failure"
        and signatures["python_syntax_error"][2] == "SyntaxError"
        and signatures["python_runtime_error"][2] == "ZeroDivisionError",
        {
            "syntax": list(signatures["python_syntax_error"]),
            "runtime": list(signatures["python_runtime_error"]),
        },
    )
    checks.record(
        "oversized_output_is_truncated_and_says_so_with_its_original_size",
        oversized["ok"]
        and oversized["truncated"]
        and oversized["original_chars"] > oversized["limit"],
        {
            "original_chars": oversized["original_chars"],
            "limit": oversized["limit"],
            "delivered_chars": len(json.dumps(oversized)),
        },
    )
    timeout_outcome = classify_tool_result(timed_out)
    checks.record(
        "the_timeout_restarted_the_interpreter_and_said_so",
        timed_out["timed_out"]
        and timed_out["workspace_restarted"]
        and timeout_outcome.kind == "execution_timeout_restart"
        and timeout_outcome.workspace_restarted
        and timeout_outcome.in_memory_state_lost,
        {
            "stderr": timed_out["stderr"],
            "signature": list(signatures["execution_timeout"]),
            "workspace_restarted": timeout_outcome.workspace_restarted,
            "in_memory_state_lost": timeout_outcome.in_memory_state_lost,
        },
    )
    checks.record(
        "the_restart_lost_in_memory_state_and_kept_files_and_observations",
        "IN_MEMORY_MARKER_PRESENT False" in after_restart["stdout"]
        and "FILE_SURVIVED kept" in after_restart["stdout"]
        and "OBSERVATIONS_READABLE True" in after_restart["stdout"],
        {"stdout": after_restart["stdout"].strip()},
    )
    checks.record(
        "setup_succeeded_before_the_failures_so_the_losses_are_attributable",
        classify_tool_result(setup).kind == "none",
        {"stdout": setup["stdout"].strip()},
    )

    checks.record(
        "the_illegal_action_returned_to_the_model_as_a_repairable_decision_error",
        len(repairs) == 1
        and classify_decision_error(repairs[0]["error"]) == "decision_illegal_action",
        {
            "repairs": [
                {"error": item["error"], "kind": classify_decision_error(item["error"])}
                for item in repairs
            ]
        },
    )
    checks.record(
        "the_repaired_action_is_the_one_the_environment_executed",
        result.transitions[0].decision.action == 1,
        {"executed_action": result.transitions[0].decision.action},
    )

    checks.record(
        "the_retryable_transport_failure_was_retried_exactly_once",
        transport_retries == 1
        and len(retries) == 1
        and len(attempt_failures) == 1
        and attempt_failures[0]["failure_kind"] == "transport_retryable",
        {
            "client_attempts": transport_attempts,
            "client_retries": transport_retries,
            "logged_retries": len(retries),
        },
    )
    checks.record(
        "the_retry_is_reconstructable_from_the_host_log_alone",
        bool(attempt_failures)
        and attempt_failures[0]["http_status"] == 503
        and isinstance(attempt_failures[0].get("request_payload_sha256"), str),
        {"attempt_failure": attempt_failures[0] if attempt_failures else None},
    )
    checks.record(
        "the_retry_left_no_trace_in_anything_the_model_received",
        not any(
            "simulated 503" in json.dumps(payload) or "transport" in json.dumps(payload).lower()
            for payload in payloads
        )
        and "AFTER_RETRY_OK" in after_retry["stdout"],
        {"requests": len(payloads)},
    )
    checks.record(
        "no_model_request_failure_escaped_on_the_retryable_run",
        not request_failures,
        {"model_request_failed_events": len(request_failures)},
    )

    checks.record(
        "the_fatal_transport_failure_was_not_retried",
        fatal_error is not None
        and not fatal_retries
        and fatal_attempts == 1
        and classify_transport_error(retryable=bool(fatal_error.retryable)) == "transport_fatal",
        {
            "attempts": fatal_attempts,
            "retries": len(fatal_retries),
            "http_status": getattr(fatal_error, "http_status", None),
        },
    )
    checks.record(
        "the_fatal_transport_failure_is_recorded_before_it_propagates",
        len(fatal_request_failures) == 1,
        {"model_request_failed_events": len(fatal_request_failures)},
    )

    illegal_frames = load_recorded_frames(
        default_recording(repo, minimum_frames=2), count=2, crop=8
    )
    try:
        run_episode(RecordedEnvironment(frames=illegal_frames), _IllegalAgent(), max_actions=1)
    except IllegalActionError as error:
        checks.record(
            "an_unrepaired_illegal_action_stops_the_run_rather_than_being_coerced",
            True,
            str(error),
        )
    else:
        checks.record(
            "an_unrepaired_illegal_action_stops_the_run_rather_than_being_coerced",
            False,
            "run_episode accepted an illegal action",
        )

    checks.finish(
        artifacts_root() / "p9-3-failure-recovery" / "evidence.json",
        tool_signatures={name: list(signature) for name, signature in signatures.items()},
        decision_repairs=repairs,
        transport_events={
            "attempt_failures": attempt_failures,
            "retries": retries,
            "fatal_request_failures": fatal_request_failures,
        },
        after_restart_stdout=after_restart["stdout"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
