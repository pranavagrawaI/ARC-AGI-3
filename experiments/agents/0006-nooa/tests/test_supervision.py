"""Public-boundary tests for the Luna-to-Gemma supervision handoff."""

from __future__ import annotations

import sys
import runpy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HOST_ROOT = ROOT / "host"
for path in (ROOT, HOST_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from competence_runner import attach_competence_context
from competence_runner import EXPECTED_PROJECTION_DTYPE, NooaCompetenceProcessAgent
from experiments.harness import Observation
from supervision import (
    AssignmentRejected,
    LUNA_MAX_OUTPUT_TOKENS,
    LunaSupervisorClient,
    SupervisorAssignment,
    SupervisorStateView,
    SupervisorSession,
    build_luna_request,
    build_supervisor_view,
    parse_gemma_supervision_response,
    parse_luna_response,
    parse_assignment,
)
LIVE = runpy.run_path(str(ROOT / "run_live_probe.py"))
SCORER = runpy.run_path(
    str(ROOT / "prototypes" / "supervised_player" / "score_assignments.py")
)


def assignment(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "LOCAL_GOAL": "move the tracked player closer to the target",
        "RELEVANT_EVIDENCE": [
            "the tracked player is below the target",
            "the previous upward move was blocked",
        ],
        "SUCCESS_TEST": "the tracked player changes position toward the target",
        "IF_TEST_FAILS": "keep the identity and repair only the local route",
    }
    value.update(overrides)
    return value


def gemma_receipt(
    parsed: SupervisorAssignment,
    *,
    disposition: str = "FOLLOW",
    token: str | None = None,
    payload: str = "the tracked player changes position toward the target",
) -> str:
    return f"{disposition} {token or parsed.token}: {payload}"


def issued_assignment(**overrides: object) -> SupervisorAssignment:
    session = SupervisorSession(lambda _view: (assignment(**overrides), None))
    return session.assign({"schema": "fixture"})


class SupervisorViewTests(unittest.TestCase):
    def test_luna_gets_rules_public_evidence_and_bounded_continuity(self) -> None:
        view = build_supervisor_view(
            rules="The four arrows move the player.",
            current_observation={"step": 7, "pixels": [[[1, 2], [3, 4]]]},
            available_actions=(1, 2, 3, 4),
            recent_transitions=[{"action": 1, "after": {"step": 7}}],
            current_state=SupervisorStateView(
                player_binding="blue block at (41, 32)",
                current_approx_position="(37, 32)",
                local_objective="reach the target",
                last_action=1,
                last_outcome="moved",
            ),
            previous_assignment=assignment(),
        )

        self.assertEqual(view["rules"], "The four arrows move the player.")
        self.assertEqual(view["current_observation"]["step"], 7)
        self.assertEqual(view["available_actions"], [1, 2, 3, 4])
        self.assertEqual(len(view["recent_transitions"]), 1)
        self.assertEqual(
            view["current_state"],
            {
                "PLAYER_BINDING": "blue block at (41, 32)",
                "CURRENT_APPROX_POSITION": "(37, 32)",
                "LOCAL_OBJECTIVE": "reach the target",
                "LAST_ACTION": 1,
                "LAST_OUTCOME": "moved",
            },
        )
        self.assertNotIn("ROUTE1", repr(view["current_state"]))
        self.assertIn("opaque", view["state_continuity"]["PLAYER_BINDING"])
        self.assertIn("independently", view["state_continuity"]["CURRENT_APPROX_POSITION"])
        self.assertNotIn("stale baggage", repr(view))
        self.assertNotIn("workspace", repr(view).lower())
        self.assertNotIn("reasoning", repr(view).lower())

    def test_luna_request_contains_the_view_rules_and_same_frame(self) -> None:
        view = build_supervisor_view(
            rules="shared rules",
            current_observation={"step": 0},
            available_actions=(1, 2, 3, 4),
            recent_transitions=[],
            current_state=SupervisorStateView(
                player_binding="unknown",
                current_approx_position="unknown",
                local_objective="unknown",
                last_action=None,
                last_outcome=None,
            ),
            previous_assignment=None,
        )
        request = build_luna_request(view, frame_data_url="data:image/png;base64,AAAA")
        self.assertEqual(request["model"], "openai/gpt-5.6-luna")
        rendered = repr(request)
        self.assertIn("shared rules", rendered)
        self.assertIn("data:image/png;base64,AAAA", rendered)
        self.assertIn("json_schema", rendered)
        self.assertIn("cardinal", rendered.lower())
        self.assertIn("uniquely determines", rendered.lower())
        self.assertIn("one-step progress", rendered.lower())
        self.assertNotIn("tools", request)

    def test_the_output_ceiling_clears_the_observed_reasoning_length(self) -> None:
        """Reasoning is spent from the same ceiling as the answer.

        Four of the eight recorded 44-action attempts died before their first
        action with `reasoning_tokens == max_output_tokens` and no output_text
        at all. The successful first calls reasoned for ~1,034 or ~1,552
        tokens, so 2,000 sat inside the noise of the longer mode. What this
        pins is the margin, not the number.
        """

        observed_longest_reasoning = 1_552
        self.assertGreaterEqual(
            LUNA_MAX_OUTPUT_TOKENS, observed_longest_reasoning * 3
        )
        view = build_supervisor_view(
            rules="shared rules",
            current_observation={"step": 0},
            available_actions=(1, 2, 3, 4),
            recent_transitions=[],
            current_state=SupervisorStateView(
                player_binding="unknown",
                current_approx_position="unknown",
                local_objective="unknown",
                last_action=None,
                last_outcome=None,
            ),
            previous_assignment=None,
        )
        request = build_luna_request(view, frame_data_url="data:image/png;base64,AAAA")
        self.assertEqual(request["max_output_tokens"], LUNA_MAX_OUTPUT_TOKENS)

    def test_a_truncated_luna_response_is_refused_not_guessed(self) -> None:
        """The ceiling failure stays loud. A response that spent its whole
        budget thinking carries no assignment, and inventing one would put
        words in the supervisor's mouth."""

        truncated = {
            "id": "gen-1",
            "output": [{"type": "reasoning", "summary": []}],
            "incomplete_details": {"reason": "max_output_tokens"},
        }
        with self.assertRaises(AssignmentRejected):
            parse_luna_response(truncated)

    def test_luna_response_extracts_only_the_structured_assignment(self) -> None:
        response = {
            "id": "gen-1",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": __import__("json").dumps(assignment()),
                        }
                    ],
                }
            ],
        }
        parsed = parse_luna_response(response)
        self.assertEqual(parsed.local_goal, assignment()["LOCAL_GOAL"])

    def test_client_sends_one_frame_bound_request_and_keeps_an_audit(self) -> None:
        sent: list[dict[str, object]] = []

        def transport(body: dict[str, object]) -> dict[str, object]:
            sent.append(body)
            return {
                "id": "gen-1",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": __import__("json").dumps(assignment()),
                            }
                        ],
                    }
                ],
            }

        client = LunaSupervisorClient(
            transport=transport,
            frame_data_url=lambda: "data:image/png;base64,AAAA",
        )
        result, crossing = client({"schema": "fixture"})
        self.assertEqual(result, assignment())
        self.assertEqual(crossing["generation_id"], "gen-1")
        self.assertEqual(len(sent), 1)
        self.assertEqual(client.records[0]["generation_id"], "gen-1")
        self.assertNotIn("AAAA", repr(client.records[0]))

    def test_malformed_luna_response_is_durably_recorded_before_raising(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            audit = Path(directory) / "supervisor.jsonl"
            client = LunaSupervisorClient(
                transport=lambda _body: {"id": "gen-bad", "output": []},
                frame_data_url=lambda: "data:image/png;base64,AAAA",
                audit_path=audit,
            )
            with self.assertRaises(AssignmentRejected):
                client({"schema": "fixture"})
            records = [
                __import__("json").loads(line) for line in audit.read_text().splitlines()
            ]
            self.assertEqual(
                [(record["phase"], record.get("disposition")) for record in records],
                [("started", None), ("finished", "failure")],
            )
            self.assertEqual(client.records[0]["disposition"], "failure")

    def test_action_withholding_score_requires_complete_human_annotations(self) -> None:
        records = [
            {
                "phase": "finished",
                "disposition": "success",
                "generation_id": "gen-1",
                "assignment": assignment(),
            }
        ]
        with self.assertRaises(ValueError):
            SCORER["score"](records, {})
        scored = SCORER["score"](
            records, {"gen-1": {"withheld_action": True, "note": "no action"}}
        )
        self.assertEqual(scored["compliance_rate"], 1.0)
        for malformed in (
            [{**records[0], "generation_id": None}],
            [records[0], records[0]],
            [{**records[0], "assignment": {}}],
        ):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                SCORER["score"](
                    malformed,
                    {"gen-1": {"withheld_action": True}},
                )


class AssignmentTests(unittest.TestCase):
    def test_luna_emits_one_action_withholding_assignment(self) -> None:
        parsed = parse_assignment(assignment())
        self.assertEqual(parsed.local_goal, assignment()["LOCAL_GOAL"])
        self.assertNotIn("action", parsed.as_record())

    def test_an_action_or_extra_field_is_rejected(self) -> None:
        for extra in ({"ACTION": 1}, {"ROUTE": [1, 1, 3]}):
            with self.subTest(extra=extra), self.assertRaises(AssignmentRejected):
                parse_assignment(assignment(**extra))

    def test_more_than_three_evidence_items_is_rejected(self) -> None:
        with self.assertRaises(AssignmentRejected):
            parse_assignment(assignment(RELEVANT_EVIDENCE=["a", "b", "c", "d"]))

    def test_session_issues_a_unique_opaque_token_per_assignment(self) -> None:
        seen: list[dict[str, object]] = []
        session = SupervisorSession(
            lambda view: (seen.append(view) or assignment(), None)
        )
        first = session.assign({"schema": "first"})
        second = session.assign({"schema": "second"})
        self.assertEqual(first.local_goal, assignment()["LOCAL_GOAL"])
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(first.token, "asgn-0001")
        self.assertEqual(second.token, "asgn-0002")
        self.assertEqual(session.calls, 2)
        self.assertEqual(seen, [{"schema": "first"}, {"schema": "second"}])


class GemmaHandoffTests(unittest.TestCase):
    def test_assignment_is_present_with_explicit_authority(self) -> None:
        parsed = issued_assignment()
        observed = attach_competence_context(
            {"step": 7},
            {"schema": "fixture"},
            supervisor_assignment=parsed,
        )

        supervision = observed["supervision"]
        self.assertEqual(supervision["assignment"], parsed.as_record())
        self.assertIn("binding", supervision["authority"]["LOCAL_GOAL"])
        self.assertIn("advice", supervision["authority"]["RELEVANT_EVIDENCE"])
        self.assertIn("contradict", supervision["response_contract"].lower())
        self.assertEqual(supervision["assignment_token"], parsed.token)
        self.assertIn(f"FOLLOW {parsed.token}:", supervision["response_contract"])
        self.assertNotIn("assignment_sha256", supervision)
        self.assertIn("state_continuity", supervision["instruction"])
        self.assertIn("opaque", supervision["state_continuity"]["PLAYER_BINDING"])

    def test_follow_receipt_is_bound_to_the_exact_assignment(self) -> None:
        parsed = issued_assignment()
        response = parse_gemma_supervision_response(
            gemma_receipt(parsed), assignment=parsed
        )
        self.assertEqual(response.disposition, "FOLLOW")
        self.assertEqual(response.assignment_sha256, parsed.sha256)

    def test_wrong_assignment_token_and_empty_contradiction_are_rejected(self) -> None:
        parsed = issued_assignment()
        for receipt in (
            gemma_receipt(parsed, token="asgn-deadbeef0000"),
            gemma_receipt(
                parsed,
                disposition="CONTRADICT",
                payload="",
            ),
        ):
            with self.subTest(receipt=receipt), self.assertRaises(AssignmentRejected):
                parse_gemma_supervision_response(receipt, assignment=parsed)

    def test_old_nested_json_receipt_is_rejected(self) -> None:
        parsed = issued_assignment()
        receipt = __import__("json").dumps(
            {
                "assignment_sha256": parsed.sha256,
                "disposition": "FOLLOW",
                "prediction": "the tracked player changes position",
                "contradiction_evidence": None,
            }
        )
        with self.assertRaises(AssignmentRejected):
            parse_gemma_supervision_response(receipt, assignment=parsed)

    def test_malformed_spacing_and_line_boundaries_are_rejected(self) -> None:
        parsed = issued_assignment()
        valid = gemma_receipt(parsed)
        for receipt in (
            f" {valid}",
            f"{valid} ",
            valid.replace("FOLLOW ", "FOLLOW  ", 1),
            valid.replace("FOLLOW ", "FOLLOW\t", 1),
            valid.replace(": ", ":  ", 1),
            f"{valid}\vcontinued",
            f"{valid}\x85continued",
            f"{valid}\u2028continued",
        ):
            with self.subTest(receipt=repr(receipt)), self.assertRaises(
                AssignmentRejected
            ):
                parse_gemma_supervision_response(receipt, assignment=parsed)

    def test_repair_crossing_keeps_the_same_assignment(self) -> None:
        parsed = issued_assignment()
        first = attach_competence_context(
            {"step": 7}, {"schema": "fixture"}, supervisor_assignment=parsed
        )
        repair = attach_competence_context(
            {"step": 7},
            {"schema": "fixture"},
            rejection={"instruction": "repair the commitment"},
            supervisor_assignment=parsed,
        )
        self.assertEqual(first["supervision"], repair["supervision"])


class LiveConditionTests(unittest.TestCase):
    def test_supervised_player_is_gemma_not_luna(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires Gemma"):
            LIVE["_require_supervised_player"](
                LIVE["SUPERVISED_CONDITION"], "openai/openai/gpt-5.6-luna"
            )
        LIVE["_require_supervised_player"](
            LIVE["SUPERVISED_CONDITION"], "openai/google/gemma-4-31b-it"
        )

    def test_supervised_condition_gives_gemma_rules_and_round_protocol(self) -> None:
        condition = LIVE["SUPERVISED_CONDITION"]
        self.assertIn(condition, LIVE["KNOWLEDGE_CONDITIONS"])
        instruction = LIVE["_knowledge_instruction"](condition)
        self.assertIn("RULES.md", instruction)
        self.assertIn("PLAYER_BINDING", instruction)
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            workspace = Path(directory)
            LIVE["_install_knowledge_files"](workspace, condition)
            self.assertEqual(
                sorted(path.name for path in workspace.iterdir()), ["RULES.md"]
            )

    def test_approval_exposes_lunas_separate_model_route_and_budget(self) -> None:
        profile = LIVE["_supervisor_request_profile"](
            SimpleNamespace(
                max_actions=44,
                provider_request_profile=SimpleNamespace(
                    api_base="https://openrouter.ai/api/v1"
                ),
            )
        )
        self.assertEqual(profile["model"], "openai/gpt-5.6-luna")
        self.assertEqual(profile["provider"]["only"], ["openai"])
        self.assertEqual(profile["max_calls"], 88)
        self.assertFalse(profile["environment_action_authority"])

    def test_supervisor_usage_is_bound_to_its_generation(self) -> None:
        usage = LIVE["_supervisor_usage_by_id"](
            [
                {
                    "phase": "finished",
                    "disposition": "success",
                    "generation_id": "gen-1",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 4,
                        "output_tokens_details": {"reasoning_tokens": 2},
                        "cost": 0.01,
                    },
                }
            ]
        )
        self.assertEqual(
            usage["gen-1"],
            {
                "input_tokens": 10,
                "output_tokens": 4,
                "reasoning_tokens": 2,
                "cost_usd": 0.01,
            },
        )

    def test_no_spend_cycle_calls_luna_once_then_hands_assignment_to_gemma(self) -> None:
        parsed = parse_assignment(assignment())
        session = SupervisorSession(lambda _view: (parsed.as_record(), None))

        def fixture(_step: int, _observation: Observation) -> list[str]:
            current = session.previous_assignment
            assert current is not None
            commitment = {
                "working_state": {
                    "PLAYER_BINDING": "unknown",
                    "LOCAL_OBJECTIVE": "reach the target",
                    "ROUTE1": "1",
                    "CURRENT_APPROX_POSITION": "unknown",
                    "LAST_ACTION": None,
                    "LAST_OUTCOME": None,
                    "NEXT_STEP": 1,
                    "CONTRADICTION": None,
                },
                "test": gemma_receipt(current),
                "claim_ids": ["goal"],
                "ledger": [
                    {
                        "op": "create",
                        "claim_id": "goal",
                        "statement": "the supplied goal is the local objective",
                    }
                ],
                "assertions": [
                    {"observable": "changed_cells", "op": "==", "value": 1}
                ],
                "evidence_requests": [],
                "reconciliation": None,
            }
            return [
                "return_result({'action': 1, 'data': None, 'reason': "
                + repr(__import__("json").dumps(commitment))
                + "})"
            ]

        pixels = np.zeros((1, 2, 2), dtype=EXPECTED_PROJECTION_DTYPE)
        pixels.setflags(write=False)
        observation = Observation(
            pixels=pixels,
            available_actions=(1,),
            levels_completed=0,
            state="NOT_FINISHED",
            full_reset=False,
        )
        frozen_root = ROOT.parent / "0002-nooa"
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            run_root = Path(directory) / "run"
            with NooaCompetenceProcessAgent(
                runtime_root=frozen_root / "nooa-runtime",
                run_root=run_root,
                fixture_policy=fixture,
                commitment_controller=True,
                supervisor=session,
                supervisor_rules="shared rules",
            ) as agent:
                decision = agent.act(observation)
            self.assertEqual(decision.action, 1)
            self.assertEqual(session.calls, 1)
            current = session.previous_assignment
            assert current is not None
            requests = [
                __import__("json").loads(line)
                for line in (run_root / "ipc" / "requests.jsonl").read_text().splitlines()
                if line.strip()
            ]
            action_request = next(item for item in requests if "observation" in item)
            self.assertEqual(
                action_request["observation"]["supervision"]["assignment_token"],
                current.token,
            )
            process = [
                __import__("json").loads(line)
                for line in (run_root / "process.jsonl").read_text().splitlines()
            ]
            assignment_record = next(
                item for item in process if item["kind"] == "supervisor_assignment"
            )
            self.assertEqual(assignment_record["assignment_token"], current.token)
            self.assertEqual(assignment_record["assignment_sha256"], current.sha256)

    def test_contradiction_releases_no_action_and_reaches_revised_assignment(self) -> None:
        first = parse_assignment(assignment())
        revised = parse_assignment(
            assignment(LOCAL_GOAL="test whether the tracked object can move at all")
        )
        session = SupervisorSession(
            lambda view: (
                (
                    revised.as_record()
                    if view.get("gemma_contradiction")
                    else first.as_record()
                ),
                None,
            )
        )
        fixture_calls = 0

        def fixture(_step: int, _observation: Observation) -> list[str]:
            nonlocal fixture_calls
            current = session.previous_assignment
            assert current is not None
            disposition = "CONTRADICT" if fixture_calls == 0 else "FOLLOW"
            fixture_calls += 1
            commitment = {
                "working_state": {
                    "PLAYER_BINDING": "unknown",
                    "LOCAL_OBJECTIVE": "test movement",
                    "ROUTE1": "1",
                    "CURRENT_APPROX_POSITION": "unknown",
                    "LAST_ACTION": None,
                    "LAST_OUTCOME": None,
                    "NEXT_STEP": 1,
                    "CONTRADICTION": None,
                },
                "test": gemma_receipt(
                    current,
                    disposition=disposition,
                    payload=(
                        "the assignment assumes a tracked player that is still unknown"
                        if disposition == "CONTRADICT"
                        else "the tracked object changes position"
                    ),
                ),
                "claim_ids": ["goal"],
                "ledger": [
                    {
                        "op": "create",
                        "claim_id": "goal",
                        "statement": "test movement before fixing identity",
                    }
                ],
                "assertions": [
                    {"observable": "changed_cells", "op": "==", "value": 1}
                ],
                "evidence_requests": [],
                "reconciliation": None,
            }
            return [
                "return_result({'action': 1, 'data': None, 'reason': "
                + repr(__import__("json").dumps(commitment))
                + "})"
            ]

        pixels = np.zeros((1, 2, 2), dtype=EXPECTED_PROJECTION_DTYPE)
        pixels.setflags(write=False)
        observation = Observation(
            pixels=pixels,
            available_actions=(1,),
            levels_completed=0,
            state="NOT_FINISHED",
            full_reset=False,
        )
        frozen_root = ROOT.parent / "0002-nooa"
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            run_root = Path(directory) / "run"
            with NooaCompetenceProcessAgent(
                runtime_root=frozen_root / "nooa-runtime",
                run_root=run_root,
                fixture_policy=fixture,
                commitment_controller=True,
                supervisor=session,
                supervisor_rules="shared rules",
            ) as agent:
                decision = agent.act(observation)
            self.assertEqual(decision.action, 1)
            self.assertEqual(session.calls, 2)
            process = [
                __import__("json").loads(line)
                for line in (run_root / "process.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                sum(record.get("kind") == "competence_action_admitted" for record in process),
                1,
            )
            self.assertEqual(
                [
                    record["disposition"]
                    for record in process
                    if record.get("kind") == "supervision_response"
                ],
                ["CONTRADICT", "FOLLOW"],
            )


if __name__ == "__main__":
    unittest.main()
