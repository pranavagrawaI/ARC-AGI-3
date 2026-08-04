"""Tests for the fixed commitment-preserving controller and its wiring.

The offline replay in `prototypes/commitment_controller/probe.py` covers whether
the rules admit real play. These cover the parts a replay cannot: the parsing
the live host does to the model's authored block, the wiring that turns a
rejection into a repair round, and the guards that keep the condition from
silently degrading into a plain rules run.
"""

from __future__ import annotations

import runpy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOST_ROOT = ROOT / "host"
for path in (ROOT, HOST_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from commitment_controller import (
    FIELDS,
    CommitmentController,
    Rejected,
    classify,
)
from competence_runner import (
    ModelCorrectableError,
    NooaCompetenceProcessAgent,
    NooaWorkerError,
    admit_transactionally,
    attach_competence_context,
    rejection_notice,
)
from competence import ContractError

MODULE = runpy.run_path(str(ROOT / "run_live_probe.py"))

LEGAL = (1, 2, 3, 4)


def block(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "PLAYER_BINDING": "two-colour 5x5 block",
        "LOCAL_OBJECTIVE": "reach the shape changer",
        "ROUTE1": "1,1,3",
        "CURRENT_APPROX_POSITION": "x34 y45",
        "LAST_ACTION": None,
        "LAST_OUTCOME": None,
        "NEXT_STEP": 1,
        "CONTRADICTION": None,
    }
    base.update(overrides)
    return base


class WorkingStateParsingTests(unittest.TestCase):
    """The model authors one block; both surfaces must accept the same text."""

    def test_route_and_actions_accept_the_notes_text_form(self) -> None:
        controller = CommitmentController()
        state = controller.admit_working_state(
            block(ROUTE1="1, 1, 3", NEXT_STEP="1"),
            action=1,
            legal_actions=LEGAL,
        )
        self.assertEqual(state.ROUTE1, (1, 1, 3))
        self.assertEqual(state.NEXT_STEP, 1)

    def test_unknown_route_is_empty_not_a_parse_error(self) -> None:
        controller = CommitmentController()
        state = controller.admit_working_state(
            block(ROUTE1="unknown"), action=1, legal_actions=LEGAL
        )
        self.assertEqual(state.ROUTE1, ())

    def test_a_missing_field_names_itself(self) -> None:
        controller = CommitmentController()
        incomplete = block()
        del incomplete["CONTRADICTION"]
        with self.assertRaises(Rejected) as caught:
            controller.admit_working_state(incomplete, action=1, legal_actions=LEGAL)
        self.assertEqual(caught.exception.rule, "R0_FIELDS")
        self.assertIn("CONTRADICTION", caught.exception.detail)

    def test_next_step_must_be_the_action_actually_committed(self) -> None:
        """The block is only a commitment if it binds the action being taken."""

        controller = CommitmentController()
        with self.assertRaises(Rejected) as caught:
            controller.admit_working_state(
                block(NEXT_STEP=1), action=4, legal_actions=LEGAL
            )
        self.assertEqual(caught.exception.rule, "R7_ONE_ACTION")

    def test_malformed_action_text_is_rejected(self) -> None:
        controller = CommitmentController()
        for field in ("LAST_ACTION", "NEXT_STEP"):
            with self.subTest(field=field), self.assertRaises(Rejected) as caught:
                controller.admit_working_state(
                    block(**{field: "garbage"}), action=1, legal_actions=LEGAL
                )
            self.assertEqual(caught.exception.rule, "R0_FIELDS")

    def test_a_non_object_working_state_is_refused(self) -> None:
        controller = CommitmentController()
        with self.assertRaises(Rejected):
            controller.admit_working_state("route1", action=1, legal_actions=LEGAL)

    def test_a_malformed_route_is_model_correctable(self) -> None:
        controller = CommitmentController()
        for route in ("1,left", [1, True], [1, "3"]):
            with self.subTest(route=route), self.assertRaises(Rejected) as caught:
                controller.admit_working_state(
                    block(ROUTE1=route), action=1, legal_actions=LEGAL
                )
            self.assertEqual(caught.exception.rule, "R0_FIELDS")

    def test_textual_absence_is_rejected_in_favor_of_json_null(self) -> None:
        for contradiction in ("", "none", " None ", "null", " Null "):
            with self.subTest(contradiction=contradiction):
                controller = CommitmentController()
                with self.assertRaises(Rejected) as caught:
                    controller.admit_working_state(
                        block(CONTRADICTION=contradiction),
                        action=1,
                        legal_actions=LEGAL,
                    )
                self.assertEqual(caught.exception.rule, "R0_FIELDS")
                self.assertIn("JSON null", caught.exception.detail)

    def test_replay_and_live_paths_share_the_contradiction_contract(self) -> None:
        controller = CommitmentController()
        with self.assertRaises(Rejected) as caught:
            controller.propose(
                block(ROUTE1=(1, 1, 3), CONTRADICTION="none"),
                legal=LEGAL,
            )
        self.assertEqual(caught.exception.rule, "R0_FIELDS")
        self.assertIn("JSON null", caught.exception.detail)


class ContinuityRuleTests(unittest.TestCase):
    def _settle(self) -> CommitmentController:
        controller = CommitmentController()
        controller.admit_working_state(block(), action=1, legal_actions=LEGAL)
        return controller

    def test_an_ordinary_move_cannot_rebind_the_player(self) -> None:
        controller = self._settle()
        with self.assertRaises(Rejected) as caught:
            controller.admit_working_state(
                block(
                    PLAYER_BINDING="actually the plus sign",
                    LAST_ACTION=1,
                    LAST_OUTCOME="moved",
                ),
                action=1,
                legal_actions=LEGAL,
            )
        self.assertEqual(caught.exception.rule, "R1_BINDING_PERSISTS")

    def test_last_action_must_continue_the_prior_commitment(self) -> None:
        controller = self._settle()
        with self.assertRaises(Rejected) as caught:
            controller.admit_working_state(
                block(LAST_ACTION=4, LAST_OUTCOME="moved"),
                action=1,
                legal_actions=LEGAL,
            )
        self.assertEqual(caught.exception.rule, "R0_FIELDS")

    def test_unknown_binding_does_not_make_the_second_turn_first_again(self) -> None:
        controller = CommitmentController()
        controller.admit_working_state(
            block(PLAYER_BINDING="unknown"), action=1, legal_actions=LEGAL
        )
        state = controller.admit_working_state(
            block(PLAYER_BINDING="unknown", LAST_ACTION=1, LAST_OUTCOME="moved"),
            action=1,
            legal_actions=LEGAL,
        )
        self.assertEqual(state.LAST_ACTION, 1)

    def test_null_contradiction_remains_absent_across_turns(self) -> None:
        controller = CommitmentController()
        first = controller.admit_working_state(
            block(CONTRADICTION=None), action=1, legal_actions=LEGAL
        )
        second = controller.admit_working_state(
            block(CONTRADICTION=None, LAST_ACTION=1, LAST_OUTCOME="moved"),
            action=1,
            legal_actions=LEGAL,
        )
        self.assertIsNone(first.CONTRADICTION)
        self.assertIsNone(second.CONTRADICTION)

    def test_a_named_contradiction_licenses_exactly_one_rebinding(self) -> None:
        controller = self._settle()
        controller.admit_working_state(
            block(
                PLAYER_BINDING="the plus sign, not the block",
                LAST_ACTION=1,
                LAST_OUTCOME="moved",
                CONTRADICTION="the block I called the player did not move",
            ),
            action=1,
            legal_actions=LEGAL,
        )
        # Still open on the next action: the licence was never spent.
        with self.assertRaises(Rejected) as caught:
            controller.admit_working_state(
                block(
                    PLAYER_BINDING="the plus sign, not the block",
                    LAST_ACTION=1,
                    LAST_OUTCOME="moved",
                    CONTRADICTION="still not sure",
                ),
                action=1,
                legal_actions=LEGAL,
            )
        self.assertEqual(caught.exception.rule, "R8_CONTRADICTION_CONSUMED")

    def test_a_changer_event_rebinds_without_claiming_error(self) -> None:
        """R5. The player really did change; nothing earlier was wrong."""

        controller = self._settle()
        state = controller.admit_working_state(
            block(
                PLAYER_BINDING="two-colour 5x5 block, shape advanced",
                LAST_ACTION=1,
                LAST_OUTCOME="special_event",
            ),
            action=1,
            legal_actions=LEGAL,
        )
        self.assertIsNone(state.CONTRADICTION)
        self.assertEqual(controller.metrics()["identity_reversals"], 1)

    def test_a_wall_reroutes_without_touching_the_objective(self) -> None:
        controller = self._settle()
        controller.admit_working_state(
            block(ROUTE1="3,1,3", LAST_ACTION=1, LAST_OUTCOME="blocked", NEXT_STEP=3),
            action=3,
            legal_actions=LEGAL,
        )
        with self.assertRaises(Rejected) as caught:
            controller.admit_working_state(
                block(
                    LOCAL_OBJECTIVE="find some other goal",
                    LAST_ACTION=3,
                    LAST_OUTCOME="blocked",
                ),
                action=1,
                legal_actions=LEGAL,
            )
        self.assertEqual(caught.exception.rule, "R3_BLOCKED_IS_LOCAL")

    def test_a_wall_must_change_the_route(self) -> None:
        controller = self._settle()
        with self.assertRaises(Rejected) as caught:
            controller.admit_working_state(
                block(LAST_ACTION=1, LAST_OUTCOME="blocked"),
                action=1,
                legal_actions=LEGAL,
            )
        self.assertEqual(caught.exception.rule, "R3_BLOCKED_IS_LOCAL")

    def test_standing_doubt_is_bounded(self) -> None:
        controller = self._settle()
        for _ in range(2):
            controller.admit_working_state(
                block(LAST_ACTION=1, LAST_OUTCOME="uncertain"),
                action=1,
                legal_actions=LEGAL,
            )
        with self.assertRaises(Rejected) as caught:
            controller.admit_working_state(
                block(LAST_ACTION=1, LAST_OUTCOME="uncertain"),
                action=1,
                legal_actions=LEGAL,
            )
        self.assertEqual(caught.exception.rule, "R9_NO_STANDING_DOUBT")

    def test_replanning_the_route_is_counted_never_refused(self) -> None:
        """Demoted from a hard rule by the Luna replay; keep it demoted."""

        controller = self._settle()
        controller.admit_working_state(
            block(ROUTE1="4,4,2", LAST_ACTION=1, LAST_OUTCOME="moved", NEXT_STEP=4),
            action=4,
            legal_actions=LEGAL,
        )
        self.assertEqual(controller.metrics()["route_replacements"], 1)

    def test_a_rejected_replan_does_not_change_the_metric(self) -> None:
        controller = self._settle()
        controller.admit_working_state(
            block(LAST_ACTION=1, CONTRADICTION="route premise failed"),
            action=1,
            legal_actions=LEGAL,
        )
        with self.assertRaises(Rejected) as caught:
            controller.admit_working_state(
                block(
                    ROUTE1="4,4,2",
                    LAST_ACTION=1,
                    LAST_OUTCOME="moved",
                    NEXT_STEP=4,
                    CONTRADICTION="still open",
                ),
                action=4,
                legal_actions=LEGAL,
            )
        self.assertEqual(caught.exception.rule, "R8_CONTRADICTION_CONSUMED")
        self.assertEqual(controller.metrics()["route_replacements"], 0)


class OutcomeScoringTests(unittest.TestCase):
    def test_the_recorded_ls20_diff_counts_separate_cleanly(self) -> None:
        """52 move, 2 budget-bar only, 58 changer -- from the Luna trajectory."""

        self.assertEqual(classify(52, moved=52, quiet=2), "moved")
        self.assertEqual(classify(2, moved=52, quiet=2), "blocked")
        self.assertEqual(classify(58, moved=52, quiet=2), "special_event")


class PacketTests(unittest.TestCase):
    def test_the_schema_is_untouched_when_the_controller_is_off(self) -> None:
        packet = attach_competence_context({"step": 0}, {"schema": "x"})
        working_state = packet["competence"]["reason_authoring_schema"]["working_state"]
        self.assertEqual(working_state, {"type": "object", "maxEncodedCharacters": 4_000})

    def test_the_model_is_shown_the_fields_rules_and_its_own_block(self) -> None:
        controller = CommitmentController()
        controller.admit_working_state(block(), action=1, legal_actions=LEGAL)
        packet = attach_competence_context(
            {"step": 0}, {"schema": "x"}, commitment_controller=controller
        )
        working_state = packet["competence"]["reason_authoring_schema"]["working_state"]
        self.assertEqual(working_state["exactKeys"], list(FIELDS))
        self.assertIn("R1", working_state["rules"])
        self.assertIn("two-colour 5x5 block", working_state["current_block"])

    def test_the_model_sees_json_null_as_the_only_absent_contradiction(self) -> None:
        controller = CommitmentController()
        packet = attach_competence_context(
            {"step": 0}, {"schema": "x"}, commitment_controller=controller
        )
        working_state = packet["competence"]["reason_authoring_schema"]["working_state"]
        self.assertEqual(working_state["CONTRADICTION"]["type"], "string-or-null")
        self.assertEqual(
            working_state["CONTRADICTION"]["absence"],
            "use JSON null exactly",
        )
        self.assertIn("CONTRADICTION: null", working_state["current_block"])

    def test_the_host_never_labels_the_outcome_for_the_model(self) -> None:
        """The moved/blocked mapping is the thing being measured, not given."""

        controller = CommitmentController()
        packet = attach_competence_context(
            {"step": 0}, {"schema": "x"}, commitment_controller=controller
        )
        rendered = repr(packet)
        self.assertNotIn("52", rendered)
        self.assertIn("name the outcome yourself", rendered)


class TransactionTests(unittest.TestCase):
    def test_later_competence_rejection_does_not_publish_controller_state(self) -> None:
        controller = CommitmentController()

        class RefusingCompetence:
            def admit(self, raw: object, *, legal_actions: object) -> object:
                raise ContractError("claim id is invalid")

        with self.assertRaises(ContractError):
            admit_transactionally(
                controller,
                RefusingCompetence(),  # type: ignore[arg-type]
                {"working_state": block()},
                action=1,
                legal_actions=LEGAL,
            )
        self.assertEqual(controller.metrics()["turns"], 0)
        self.assertEqual(controller.state.PLAYER_BINDING, "unknown")

    def test_a_refused_block_is_counted_on_the_surviving_controller(self) -> None:
        """The copy protects state; it must not swallow the measurement.

        Run supervised-probe-rules_supervised-l1-44-v3 reported `rejections: 0`
        beside eight refused crossings and a fatal R3, because every refusal was
        logged onto the discarded candidate. A reader of that summary would
        conclude the controller never refused anything.
        """

        controller = CommitmentController()
        controller.admit_working_state(block(), action=1, legal_actions=LEGAL)

        class PassingCompetence:
            def admit(self, raw: object, *, legal_actions: object) -> object:
                return {}

        # R2: after a `moved` outcome the objective may not change without a
        # named contradiction.
        with self.assertRaises(Rejected):
            admit_transactionally(
                controller,
                PassingCompetence(),  # type: ignore[arg-type]
                {
                    "working_state": block(
                        LAST_ACTION=1,
                        LAST_OUTCOME="moved",
                        LOCAL_OBJECTIVE="something else entirely",
                    )
                },
                action=1,
                legal_actions=LEGAL,
            )

        report = MODULE["_controller_report"](controller)
        self.assertEqual(report["rejections"], 1)
        self.assertEqual(report["rejected_rules"], ["R2_NO_RESCAN_ON_MOVE"])
        # The refusal is recorded and the accepted block is left alone.
        self.assertEqual(report["turns"], 1)
        self.assertEqual(controller.state.LOCAL_OBJECTIVE, "reach the shape changer")

    def test_a_rule_raised_before_propose_is_also_counted(self) -> None:
        """R7 and the field parsers raise inside admit_working_state, so they
        never reached the logging path in propose on any controller at all."""

        controller = CommitmentController()

        class PassingCompetence:
            def admit(self, raw: object, *, legal_actions: object) -> object:
                return {}

        with self.assertRaises(Rejected):
            admit_transactionally(
                controller,
                PassingCompetence(),  # type: ignore[arg-type]
                {"working_state": block(NEXT_STEP=2)},
                action=1,
                legal_actions=LEGAL,
            )

        report = MODULE["_controller_report"](controller)
        self.assertEqual(report["rejections"], 1)
        self.assertEqual(report["rejected_rules"], ["R7_ONE_ACTION"])


class RoundExhaustionTests(unittest.TestCase):
    """A model that thinks too long gets asked again, not killed."""

    AUDIT = {"model_calls": [{"messages": [{"role": "user"}], "tools": []}]}

    def _validate(self, error: dict[str, object]) -> None:
        NooaCompetenceProcessAgent._validate_worker_outcome(self.AUDIT, error)

    def test_exhausted_rounds_are_correctable(self) -> None:
        """The exact failure that ended the recorded gemma+rules run at action 0."""

        with self.assertRaises(ModelCorrectableError) as caught:
            self._validate(
                {
                    "kind": "decision_failure",
                    "type": "GenerationError",
                    "message": (
                        "Generation failed after 8 iterations (max_iterations=8). "
                        "Unable to complete `deliberate`."
                    ),
                }
            )
        self.assertIn("deliberation rounds exhausted", str(caught.exception))

    def test_other_decision_failures_still_end_the_run(self) -> None:
        """Only running out of thinking time is forgiven, not any bad decision."""

        with self.assertRaises(NooaWorkerError) as caught:
            self._validate(
                {
                    "kind": "decision_failure",
                    "type": "InvariantError",
                    "message": "postcondition has no public action point",
                }
            )
        self.assertNotIsInstance(caught.exception, ModelCorrectableError)

    def test_transport_failure_is_still_fatal(self) -> None:
        """Not the model's fault and not fixable by asking again."""

        with self.assertRaises(NooaWorkerError) as caught:
            self._validate(
                {"kind": "provider_transport", "disposition": "fatal", "attempts": 1}
            )
        self.assertNotIsInstance(caught.exception, ModelCorrectableError)

    def test_the_notice_does_not_claim_a_commitment_was_refused(self) -> None:
        """There was none. The ordinary notice would send the model looking for it."""

        notice = rejection_notice(
            "deliberation rounds exhausted without a committed action: ...",
            attempts_remaining=1,
        )
        instruction = notice["instruction"]
        self.assertIn("without calling return_result", instruction)
        self.assertIn("Commit this round", instruction)
        self.assertNotIn("previous commitment was refused", instruction)

    def test_the_notice_still_authors_no_action(self) -> None:
        notice = rejection_notice(
            "deliberation rounds exhausted without a committed action: ...",
            attempts_remaining=1,
        )
        self.assertIn("will not choose an action for you", notice["instruction"])


class ConditionWiringTests(unittest.TestCase):
    def test_the_condition_supplies_the_same_files_as_plain_rules(self) -> None:
        """Otherwise it would test knowing more, not holding on to more."""

        install = MODULE["_install_knowledge_files"]
        import tempfile

        for condition in ("rules", MODULE["CONTROLLER_CONDITION"]):
            with tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                install(workspace, condition)
                self.assertEqual(
                    sorted(path.name for path in workspace.iterdir()), ["RULES.md"]
                )

    def test_the_seed_fits_the_frozen_workers_echo_budget(self) -> None:
        """Over the limit the model silently never sees the head of the protocol."""

        instruction = MODULE["_knowledge_instruction"](MODULE["CONTROLLER_CONDITION"])
        self.assertLessEqual(len(instruction), MODULE["NOTES_ECHO_LIMIT"])
        self.assertIn("PLAYER_BINDING", instruction)
        self.assertIn("RULES.md", instruction)

    def test_an_oversized_seed_fails_loudly(self) -> None:
        # runpy.run_path hands back a copy of the namespace, so the limit has to
        # be patched where the function actually resolves it.
        instruction = MODULE["_knowledge_instruction"]
        namespace = instruction.__globals__
        original = namespace["NOTES_ECHO_LIMIT"]
        namespace["NOTES_ECHO_LIMIT"] = 10
        try:
            with self.assertRaises(ValueError) as caught:
                instruction(MODULE["CONTROLLER_CONDITION"])
            self.assertIn("shorten STATE_BLOCK.md", str(caught.exception))
        finally:
            namespace["NOTES_ECHO_LIMIT"] = original

    def test_the_control_arm_cannot_select_the_condition(self) -> None:
        """It authors no working_state, so the run would be a mislabelled baseline."""

        args = MODULE["build_parser"]().parse_args(
            [
                "--game", "ls20",
                "--mode", "offline",
                "--environment-seed", "0",
                "--model", "openai/google/gemma-4-31b-it",
                "--wire-model", "google/gemma-4-31b-it",
                "--api-base", "https://openrouter.ai/api/v1",
                "--max-output-tokens", "16384",
                "--reasoning-effort", "medium",
                "--request-timeout-seconds", "180",
                "--max-actions", "44",
                "--max-model-calls", "400",
                "--max-reported-tokens", "8388608",
                "--max-model-rounds-per-action", "8",
                "--max-decision-repairs", "2",
                "--transport-max-retries", "2",
                "--workspace-timeout-seconds", "20",
                "--run-root", "/tmp/unused-controller-arm-check",
                "--arm", "control",
                "--provider-route", "pinned",
                "--knowledge-condition", "rules_controller",
            ]
        )
        with self.assertRaises(ValueError) as caught:
            MODULE["_validate_live_args"](args)
        self.assertIn("requires --arm treatment", str(caught.exception))

    def test_the_state_block_is_bound_into_the_approved_sources(self) -> None:
        """Otherwise the protocol could be rewritten after approval."""

        sources = MODULE["_participating_sources"]()
        self.assertIn(
            "experiments/agents/0006-nooa/prototypes/commitment_controller/"
            "STATE_BLOCK.md",
            sources,
        )

    def test_the_report_is_absent_when_the_controller_never_ran(self) -> None:
        self.assertIsNone(MODULE["_controller_report"](None))

    def test_the_report_carries_the_preregistered_measures(self) -> None:
        controller = CommitmentController()
        controller.admit_working_state(block(), action=1, legal_actions=LEGAL)
        report = MODULE["_controller_report"](controller)
        for measure in (
            "identity_reversals",
            "route_replacements",
            "rejections",
            "turns",
            "final_state",
            "rejected_rules",
        ):
            self.assertIn(measure, report)


if __name__ == "__main__":
    unittest.main()
