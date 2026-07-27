"""Tests for the P9 shared arm seam (issue #9).

The four prototypes are the evidence, so each one runs end to end here with its
artifacts redirected to a temporary directory. The pure pieces — path
normalization, authorization, failure classification, the omission template —
are pinned separately so a regression names itself instead of surfacing as a
prototype failure.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


EXPERIMENT_ROOT = (
    Path(__file__).resolve().parents[1] / "0006-capability-environment"
)
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from p9 import (
    p9_0_shared_arm_seam,
    p9_1_cumulative_loop,
    p9_2_context_loss,
    p9_3_failure_recovery,
    p9_4_harness_task,
    run_arm,
)
from p9.arm_diff import compare_arms, flatten, is_authorized, normalize_request_payload
from p9.arms import ArmSpec, control_spec, treatment_spec
from p9.capabilities import (
    TREATMENT_CAPABILITIES,
    authorized_manifest_paths,
    resolve_capabilities,
)
from p9.conversation import (
    OMISSION_NOTICE_TEMPLATE,
    AnnouncedConversation,
    compact_steps,
    omission_notice_text,
)
from p9.probe import detect_workarounds
from p9.failures import (
    FRICTION_KINDS,
    TOOL_FAILURE_KINDS,
    classify_friction,
    classify_decision_error,
    classify_tool_result,
    classify_transport_error,
)
from p9.transport import RetryingChatClient

from llm_repl_agent.client import ChatResponse, ModelRequestError
from llm_repl_agent.process_log import ProcessLog


def _tool_output(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "ok": True,
        "stdout": "",
        "stderr": "",
        "traceback": None,
        "timed_out": False,
        "workspace_restarted": False,
        "truncated": False,
    }
    record.update(overrides)
    return record


class FlattenTests(unittest.TestCase):
    def test_scalar_lists_stay_one_leaf(self) -> None:
        self.assertEqual(flatten({"a": [1, 2]}), {"a": [1, 2]})

    def test_object_lists_are_indexed(self) -> None:
        self.assertEqual(flatten({"a": [{"b": 1}]}), {"a.0.b": 1})

    def test_empty_object_is_its_own_leaf(self) -> None:
        self.assertEqual(flatten({"a": {}}), {"a": {}})


class AuthorizationTests(unittest.TestCase):
    def test_an_ancestor_path_authorizes_its_descendants(self) -> None:
        self.assertTrue(is_authorized("a.b.c", {"a.b"}))

    def test_a_descendant_path_does_not_authorize_its_ancestor(self) -> None:
        self.assertFalse(is_authorized("a.b", {"a.b.c"}))

    def test_wildcards_match_one_segment(self) -> None:
        self.assertTrue(is_authorized("messages.3.content.0.x", {"messages.*.content.*.x"}))
        self.assertFalse(is_authorized("messages.3.tools.0.x", {"messages.*.content.*.x"}))

    def test_an_undeclared_change_is_unauthorized(self) -> None:
        report = compare_arms({"a": 1, "b": 2}, {"a": 1, "b": 3}, authorized_paths={"a"})
        self.assertFalse(report.ok)
        self.assertEqual([item.path for item in report.unauthorized], ["b"])

    def test_an_authorized_path_that_never_moves_is_reported(self) -> None:
        report = compare_arms({"a": 1}, {"a": 1}, authorized_paths={"a"})
        self.assertTrue(report.ok)
        self.assertEqual(report.authorized_unused, ("a",))


class ArmSpecTests(unittest.TestCase):
    def test_the_treatment_inherits_every_shared_field(self) -> None:
        control = control_spec(max_context_turns=3, workspace_timeout_seconds=7.0)
        treatment = treatment_spec(control)
        for name in ArmSpec.__dataclass_fields__:
            if name in {"arm_id", "treatment_capabilities"}:
                continue
            self.assertEqual(getattr(control, name), getattr(treatment, name), name)

    def test_unknown_capabilities_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_capabilities(("not_a_capability",))

    def test_every_capability_declares_at_least_one_path(self) -> None:
        for capability in TREATMENT_CAPABILITIES.values():
            self.assertTrue(
                capability.manifest_paths or capability.request_payload_paths,
                capability.name,
            )

    def test_the_control_authorizes_only_identity_paths(self) -> None:
        self.assertEqual(
            authorized_manifest_paths(()),
            frozenset(
                {"arm_id", "host_only.spec.arm_id", "host_only.spec.treatment_capabilities"}
            ),
        )


class FailureClassificationTests(unittest.TestCase):
    def test_each_kind_is_reachable_from_a_distinct_record(self) -> None:
        records = {
            "none": _tool_output(),
            "tool_argument_validation_failure": _tool_output(ok=False),
            "python_execution_failure": _tool_output(ok=False, traceback="ZeroDivisionError: x"),
            "execution_timeout_restart": _tool_output(
                ok=False, timed_out=True, workspace_restarted=True
            ),
            "worker_crash_restart": _tool_output(ok=False, workspace_restarted=True),
        }
        self.assertEqual(set(records), set(TOOL_FAILURE_KINDS))
        for expected, record in records.items():
            self.assertEqual(classify_tool_result(record).kind, expected)

    def test_a_restart_is_reported_as_in_memory_state_loss(self) -> None:
        outcome = classify_tool_result(
            _tool_output(ok=False, timed_out=True, workspace_restarted=True)
        )
        self.assertTrue(outcome.in_memory_state_lost)

    def test_decision_errors_map_to_distinct_kinds(self) -> None:
        self.assertEqual(
            classify_decision_error("action 99 is not legal; legal actions are (1,)"),
            "decision_illegal_action",
        )
        self.assertEqual(
            classify_decision_error("action must be an integer"),
            "decision_action_not_integer",
        )
        self.assertEqual(
            classify_decision_error("coordinates (1, 2) are outside width=8, height=8"),
            "decision_coordinates_out_of_bounds",
        )
        self.assertEqual(
            classify_decision_error("action 6 requires integer x and y data"),
            "decision_missing_coordinates",
        )
        self.assertEqual(
            classify_decision_error("Python is unavailable on the final deliberation round"),
            "decision_tool_after_final_round",
        )
        self.assertEqual(
            classify_decision_error("response is not valid JSON: nope"),
            "decision_not_json",
        )

    def test_only_a_retryable_transport_error_is_named_retryable(self) -> None:
        self.assertEqual(classify_transport_error(retryable=True), "transport_retryable")
        self.assertEqual(
            classify_transport_error(retryable=False, http_status=400), "transport_fatal"
        )

    def test_a_malformed_response_body_is_retryable_though_the_client_says_otherwise(
        self,
    ) -> None:
        # Observed live: the provider returned 1001 bytes of whitespace, the frozen
        # client marked it fatal, and the run died at action 12 of 44.
        self.assertEqual(
            classify_transport_error(
                retryable=False, http_status=None, response_body="   \n   \n"
            ),
            "transport_malformed_response",
        )

    def test_an_http_error_with_a_body_stays_fatal(self) -> None:
        self.assertEqual(
            classify_transport_error(
                retryable=False, http_status=400, response_body='{"error":"bad"}'
            ),
            "transport_fatal",
        )


class FrictionClassificationTests(unittest.TestCase):
    """The reader that turns a recorded run into a measurement."""

    def test_a_successful_call_has_no_friction(self) -> None:
        self.assertIsNone(classify_friction("print(1)", _tool_output()))

    def test_a_constructed_observation_path_is_named_separately(self) -> None:
        friction = classify_friction(
            "open('observations/step-0007.json')",
            _tool_output(
                ok=False,
                traceback="FileNotFoundError: No such file: 'observations/step-0007.json'",
            ),
        )
        assert friction is not None
        self.assertEqual(friction.kind, "constructed_observation_path_not_found")
        self.assertEqual(friction.attributable_to, "interface")

    def test_a_missing_package_is_interface_friction(self) -> None:
        friction = classify_friction(
            "import sklearn",
            _tool_output(ok=False, traceback="ModuleNotFoundError: No module named 'sklearn'"),
        )
        assert friction is not None
        self.assertEqual(friction.kind, "module_unavailable")
        self.assertEqual(friction.attributable_to, "interface")

    def test_a_guessed_schema_key_is_interface_friction(self) -> None:
        friction = classify_friction(
            "obs['current']", _tool_output(ok=False, traceback="KeyError: 'current'")
        )
        assert friction is not None
        self.assertEqual(friction.kind, "observation_schema_key_guessed")

    def test_a_timeout_is_not_blamed_on_the_interface(self) -> None:
        friction = classify_friction(
            "while True: pass", _tool_output(ok=False, timed_out=True)
        )
        assert friction is not None
        self.assertEqual(friction.attributable_to, "model")

    def test_every_kind_it_can_emit_is_declared(self) -> None:
        cases = [
            _tool_output(ok=False, traceback="FileNotFoundError: 'observations/step-1.json'"),
            _tool_output(ok=False, traceback="FileNotFoundError: 'notes.md'"),
            _tool_output(ok=False, traceback="ModuleNotFoundError: x"),
            _tool_output(ok=False, traceback="KeyError: 'x'"),
            _tool_output(ok=False, traceback="NameError: x"),
            _tool_output(ok=False, timed_out=True),
            _tool_output(ok=False),
            _tool_output(ok=False, traceback="ValueError: x"),
        ]
        emitted = {classify_friction(None, case).kind for case in cases}
        self.assertEqual(emitted, set(FRICTION_KINDS))


class OmissionNoticeTests(unittest.TestCase):
    def test_steps_compact_into_ranges(self) -> None:
        self.assertEqual(compact_steps((1, 2, 3, 5)), "1-3, 5")
        self.assertEqual(compact_steps(()), "none")

    def test_the_notice_is_the_template_with_only_substitutions(self) -> None:
        text = omission_notice_text(steps=(4,), observations_glob="episode/o/*")
        self.assertEqual(
            text,
            OMISSION_NOTICE_TEMPLATE.format(steps="4", observations_glob="episode/o/*"),
        )

    def test_no_notice_appears_before_an_eviction(self) -> None:
        conversation = AnnouncedConversation(
            system_message={"role": "system", "content": "s"},
            max_turns=2,
            observations_glob="episode/o/*",
        )
        conversation.append(step=1, messages=[{"role": "user", "content": "a"}])
        self.assertIsNone(conversation.omission_notice())
        self.assertEqual(len(conversation.messages()), 2)

    def test_the_notice_names_every_evicted_step(self) -> None:
        conversation = AnnouncedConversation(
            system_message={"role": "system", "content": "s"},
            max_turns=1,
            observations_glob="episode/o/*",
        )
        for step in (1, 2, 3):
            conversation.append(step=step, messages=[{"role": "user", "content": str(step)}])
        notice = conversation.omission_notice()
        assert notice is not None
        self.assertIn("action steps 1-2", notice["content"])
        self.assertEqual(conversation.messages()[1], notice)


class RequestNormalizationTests(unittest.TestCase):
    def test_image_data_urls_collapse_to_a_digest(self) -> None:
        normalized = normalize_request_payload(
            {"messages": [{"content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}]}]}
        )
        url = normalized["messages"][0]["content"][0]["image_url"]["url"]
        self.assertTrue(url.startswith("sha256:"))

    def test_the_observation_payload_is_parsed_into_structure(self) -> None:
        normalized = normalize_request_payload(
            {"messages": [{"content": [{"type": "text", "text": '{"step": 2}'}]}]}
        )
        part = normalized["messages"][0]["content"][0]
        self.assertEqual(part["observation_payload"], {"step": 2})
        self.assertNotIn("text", part)


class _StubClient:
    def __init__(self, failures: list[ModelRequestError]) -> None:
        self.failures = list(failures)
        self.calls = 0

    def complete(self, *, messages: object, tools: object) -> ChatResponse:
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return ChatResponse(message={"role": "assistant", "content": "ok"}, usage={})


class RetryingChatClientTests(unittest.TestCase):
    def _client(self, failures: list[ModelRequestError], scratch: str, retries: int = 2):
        log = ProcessLog(Path(scratch) / "process.jsonl")
        inner = _StubClient(failures)
        return RetryingChatClient(inner, process_log=log, max_retries=retries), inner, log

    def _events(self, log: ProcessLog) -> list[dict]:
        log.close()
        return [
            json.loads(line)
            for line in log.path.read_text(encoding="utf-8").splitlines()
        ]

    def test_a_retryable_failure_is_retried_then_succeeds(self) -> None:
        with TemporaryDirectory() as scratch:
            failure = ModelRequestError("503", retryable=True, http_status=503)
            client, inner, log = self._client([failure], scratch)
            client.complete(messages=[], tools=[])
            self.assertEqual(inner.calls, 2)
            self.assertEqual(client.retries, 1)
            kinds = [event["type"] for event in self._events(log)]
            self.assertEqual(kinds.count("model_transport_retry"), 1)

    def test_a_retryable_failure_still_raises_once_the_budget_is_gone(self) -> None:
        with TemporaryDirectory() as scratch:
            failures = [
                ModelRequestError("503", retryable=True, http_status=503) for _ in range(4)
            ]
            client, inner, log = self._client(failures, scratch, retries=2)
            with self.assertRaises(ModelRequestError):
                client.complete(messages=[], tools=[])
            self.assertEqual(inner.calls, 3)
            self.assertEqual(client.retries, 2)
            self.assertEqual(
                [event["type"] for event in self._events(log)].count(
                    "model_transport_attempt_failed"
                ),
                3,
            )

    def test_a_malformed_response_body_is_retried(self) -> None:
        with TemporaryDirectory() as scratch:
            failure = ModelRequestError(
                "not json", retryable=False, response_body_raw="        "
            )
            client, inner, log = self._client([failure], scratch)
            client.complete(messages=[], tools=[])
            self.assertEqual(inner.calls, 2)
            self.assertEqual(client.retries, 1)
            kinds = [
                event.get("failure_kind") for event in self._events(log)
            ]
            self.assertIn("transport_malformed_response", kinds)

    def test_a_fatal_failure_is_never_retried(self) -> None:
        with TemporaryDirectory() as scratch:
            failure = ModelRequestError("400", retryable=False, http_status=400)
            client, inner, log = self._client([failure], scratch)
            with self.assertRaises(ModelRequestError):
                client.complete(messages=[], tools=[])
            self.assertEqual(inner.calls, 1)
            self.assertEqual(client.retries, 0)
            events = self._events(log)
            self.assertEqual(
                [event["failure_kind"] for event in events], ["transport_fatal"]
            )

    def test_a_negative_retry_budget_is_refused(self) -> None:
        with TemporaryDirectory() as scratch:
            log = ProcessLog(Path(scratch) / "process.jsonl")
            with self.assertRaises(ValueError):
                RetryingChatClient(_StubClient([]), process_log=log, max_retries=-1)


class RunnerTests(unittest.TestCase):
    """The runner runs offline end to end, so it is not first exercised live."""

    def test_a_single_arm_run_produces_the_full_artifact_set(self) -> None:
        with TemporaryDirectory(prefix="p9-run-") as scratch:
            run_dir = Path(scratch) / "run"
            self.assertEqual(
                run_arm.main(["--offline", "--max-actions", "3", "--run-dir", str(run_dir)]), 0
            )
            for name in (
                "manifest.json",
                "process.jsonl",
                "trace.jsonl",
                "friction.json",
                "process-annotation.json",
            ):
                self.assertTrue((run_dir / name).is_file(), name)
            self.assertTrue((run_dir / "workspace").is_dir())
            manifest = json.loads((run_dir / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["actions"], 3)
            self.assertIn("arm_manifest", manifest)
            self.assertTrue(manifest["evidence_hashes"])

    def test_the_fixture_policy_finishes_the_task_through_the_runner(self) -> None:
        with TemporaryDirectory(prefix="p9-run-") as scratch:
            run_dir = Path(scratch) / "run"
            run_arm.main(["--offline", "--max-actions", "3", "--run-dir", str(run_dir)])
            index = json.loads((run_dir / "workspace" / "index.json").read_text())
            self.assertEqual(sorted(index), ["00000", "00001", "00002"])

    def test_a_pair_derives_both_arms_and_emits_an_authorized_diff(self) -> None:
        with TemporaryDirectory(prefix="p9-pair-") as scratch:
            root = Path(scratch) / "pair"
            self.assertEqual(
                run_arm.main(["--offline", "--pair", "--max-actions", "3", "--run-dir", str(root)]),
                0,
            )
            diff = json.loads((root / "pair-diff.json").read_text())
            self.assertEqual(diff["diff"]["unauthorized"], [])
            self.assertGreater(diff["diff"]["shared_paths"], 0)
            self.assertEqual(
                json.loads((root / "control" / "manifest.json").read_text())["arm_id"],
                "0002-R",
            )

    def test_a_live_run_without_a_game_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            run_arm.main(["--max-actions", "1"])

    def test_the_control_arm_refuses_a_capability(self) -> None:
        with self.assertRaises(SystemExit):
            run_arm.main(["--offline", "--capability", "sentinel_probe_note"])


class ProbeTests(unittest.TestCase):
    """Play-test mode: the signal the friction log structurally cannot see."""

    def test_hedging_in_model_code_is_detected(self) -> None:
        events = [
            {
                "type": "python_tool",
                "action_step": 3,
                "code": (
                    "# Since I don't have the files explicitly, I'll use current_frame.png\n"
                    "img = open('current_frame.png')\n"
                ),
            }
        ]
        found = detect_workarounds(events)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].action_step, 3)

    def test_clean_code_is_not_flagged(self) -> None:
        events = [{"type": "python_tool", "action_step": 1, "code": "print(sum(range(3)))"}]
        self.assertEqual(detect_workarounds(events), [])

    def test_a_probe_run_is_marked_inadmissible_and_reports_three_signals(self) -> None:
        with TemporaryDirectory(prefix="p9-probe-") as scratch:
            run_dir = Path(scratch) / "run"
            self.assertEqual(
                run_arm.main(
                    ["--offline", "--probe", "--max-actions", "3", "--run-dir", str(run_dir)]
                ),
                0,
            )
            manifest = json.loads((run_dir / "manifest.json").read_text())
            self.assertEqual(manifest["arm_id"], "0002-R-probe-pure")
            self.assertTrue(manifest["probe"])
            self.assertFalse(manifest["admissible_contestant_claim"])
            report = json.loads((run_dir / "probe-report.json").read_text())
            self.assertEqual(
                sorted(report["signals"]), ["failed", "reported", "worked_around"]
            )

    def test_the_probe_prompt_keeps_the_contract_under_test_intact(self) -> None:
        from p9.arms import ARM_SYSTEM_PROMPT

        spec = run_arm.spec_from_args(
            run_arm.build_parser().parse_args(["--offline", "--probe"])
        )
        self.assertIn(ARM_SYSTEM_PROMPT.rstrip(), spec.system_prompt)
        self.assertIn("You are not playing to win", spec.system_prompt)

    def test_the_two_probe_styles_share_the_arc_brief_verbatim(self) -> None:
        from p9.probe import ARC_AGI_3_BRIEF, PLAY_ALONG_ROLE, PURE_PROBE_ROLE

        # The comparison is only attributable to the role if the brief is identical.
        self.assertIn(ARC_AGI_3_BRIEF, PURE_PROBE_ROLE)
        self.assertIn(ARC_AGI_3_BRIEF, PLAY_ALONG_ROLE)
        self.assertNotEqual(PURE_PROBE_ROLE, PLAY_ALONG_ROLE)

    def test_play_along_keeps_the_contestant_repair_budget(self) -> None:
        parse = run_arm.build_parser().parse_args
        pure = run_arm.spec_from_args(parse(["--offline", "--probe"]))
        play = run_arm.spec_from_args(
            parse(["--offline", "--probe", "--probe-style", "play-along"])
        )
        control = run_arm.spec_from_args(parse(["--offline"]))
        self.assertEqual(pure.max_decision_repairs, 20)
        self.assertEqual(play.max_decision_repairs, control.max_decision_repairs)
        self.assertEqual(play.arm_id, "0002-R-probe-play-along")

    def test_probe_refuses_a_treatment_arm(self) -> None:
        with self.assertRaises(SystemExit):
            run_arm.main(["--offline", "--probe", "--arm", "treatment"])


class PrototypeTests(unittest.TestCase):
    """Each prototype runs end to end; a failed check raises PrototypeFailure."""

    def _run(self, module: object) -> None:
        with TemporaryDirectory(prefix="p9-test-") as scratch:
            with patch.object(module, "artifacts_root", lambda: Path(scratch)):
                self.assertEqual(module.main(), 0)  # type: ignore[attr-defined]

    def test_p9_0_shared_arm_seam(self) -> None:
        self._run(p9_0_shared_arm_seam)

    def test_p9_1_cumulative_loop(self) -> None:
        self._run(p9_1_cumulative_loop)

    def test_p9_2_context_loss(self) -> None:
        self._run(p9_2_context_loss)

    def test_p9_3_failure_recovery(self) -> None:
        self._run(p9_3_failure_recovery)

    def test_p9_4_harness_task(self) -> None:
        self._run(p9_4_harness_task)


if __name__ == "__main__":
    unittest.main()
