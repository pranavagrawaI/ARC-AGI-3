"""Contract tests for the first 0006-NOOA competence vertical slice."""

from __future__ import annotations

import json
import sys
import unittest
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from competence import (
    CompetenceController,
    ContractError,
    _assertion_is_at_risk,
    parse_model_reason,
)


def observation(step: int, *, pixels: list[list[int]], state: str = "NOT_FINISHED") -> dict:
    return {
        "step": step,
        "pixels": pixels,
        "state": state,
        "levels_completed": 0,
    }


class CompetenceControllerTests(unittest.TestCase):
    def test_a_commitment_that_predicts_or_holds_nothing_is_refused(self) -> None:
        """Both empty lists cost a live run before the contract refused them.

        Run 20260802T184631Z admitted actions whose assertions were empty, so
        nothing was checked against the successor. Run 20260803T061050Z admitted
        two actions whose claim_ids were empty, so action two had no claim from
        action one to continue and the chain could not qualify. Each field was
        individually optional and jointly the whole point, and in both cases the
        run completed, looked healthy, and proved nothing.
        """

        base = {
            "action": 1,
            "data": None,
            "working_state": {},
            "test": "prose the host never checks",
            "claim_ids": ["claim-1"],
            "ledger": [
                {
                    "op": "create",
                    "claim_id": "claim-1",
                    "statement": "action 1 has an observable effect",
                }
            ],
            "assertions": [{"observable": "changed_cells", "op": ">", "value": 0}],
            "evidence_requests": [],
            "reconciliation": None,
        }

        for field, message in (("assertions", "assertions"), ("claim_ids", "claim_ids")):
            controller = CompetenceController()
            controller.receive_observation(observation(0, pixels=[[0, 0], [0, 0]]))
            with self.subTest(field=field):
                with self.assertRaisesRegex(ContractError, message):
                    controller.admit({**base, field: []}, legal_actions=(1,))

    def test_an_assertion_earns_its_place_by_being_able_to_fail(self) -> None:
        """The boundary between a prediction and a restatement of the present.

        The current observation is NOT_FINISHED with no levels completed on a
        four-cell grid, so ``changed_cells`` can only land in 0..4. Assertions
        that hold across that whole range, or that repeat a value the model was
        just handed, cannot be falsified by any successor and are refused.
        Predicting an exact count, including zero, remains a real claim about
        the action rather than a restatement of the present.
        """

        base = {
            "action": 1,
            "data": None,
            "working_state": {},
            "test": "prose the host never checks",
            "claim_ids": ["claim-1"],
            "ledger": [
                {
                    "op": "create",
                    "claim_id": "claim-1",
                    "statement": "action 1 has an observable effect",
                }
            ],
            "evidence_requests": [],
            "reconciliation": None,
        }
        state_quo = {"observable": "state", "op": "==", "value": "NOT_FINISHED"}
        at_risk = {"observable": "changed_cells", "op": ">", "value": 0}

        for assertion, expected in (
            (state_quo, False),
            ({"observable": "state", "op": "!=", "value": "NOT_FINISHED"}, True),
            ({"observable": "levels_completed", "op": "==", "value": 0}, False),
            ({"observable": "levels_completed", "op": ">=", "value": 0}, False),
            ({"observable": "levels_completed", "op": ">", "value": 0}, True),
            ({"observable": "changed_cells", "op": ">=", "value": 0}, False),
            ({"observable": "changed_cells", "op": "<=", "value": 4}, False),
            ({"observable": "changed_cells", "op": "<=", "value": 3}, True),
            ({"observable": "changed_cells", "op": "==", "value": 0}, True),
            (at_risk, True),
        ):
            controller = CompetenceController()
            controller.receive_observation(observation(0, pixels=[[0, 0], [0, 0]]))
            decision = {**base, "assertions": [assertion]}
            with self.subTest(assertion=assertion):
                if expected:
                    commitment = controller.admit(decision, legal_actions=(1,))
                    self.assertTrue(commitment["assertions"][0]["at_risk"])
                    self.assertEqual(
                        commitment["prediction_coverage_status"], "complete"
                    )
                else:
                    with self.assertRaisesRegex(ContractError, "falsify"):
                        controller.admit(decision, legal_actions=(1,))

        controller = CompetenceController()
        controller.receive_observation(observation(0, pixels=[[0, 0], [0, 0]]))
        mixed = controller.admit(
            {**base, "assertions": [state_quo, at_risk]}, legal_actions=(1,)
        )
        self.assertEqual(
            [item["at_risk"] for item in mixed["assertions"]], [False, True]
        )
        self.assertEqual(mixed["prediction_coverage_status"], "partial")

    def test_the_rule_refuses_what_live_runs_actually_authored(self) -> None:
        """Drive the real commitments from four live ls20 runs through the rule.

        The invented fixtures above fix the boundary; these fix the target. Every
        assertion here was written by the model against a real 64x64 observation,
        and three of the six commitments assert nothing the successor could
        contradict - including both actions of 20260802T182855Z, which completed,
        reported two matches, and established nothing. The one that predicts
        ``state != NOT_FINISHED`` survives, as it should: it was a real claim,
        and the environment falsified it.
        """

        preserved = json.loads(
            (Path(__file__).parent / "live_commitments_ls20.json").read_text()
        )
        observations = preserved["observations"]
        verdicts = []
        for commitment in preserved["commitments"]:
            public = observations[commitment["observation_sha256"]]
            self.assertEqual(
                sha256(
                    json.dumps(public, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                commitment["observation_sha256"],
                "the preserved observation is not the one the model was shown",
            )
            at_risk = [
                _assertion_is_at_risk(assertion, current=public)
                for assertion in commitment["assertions"]
            ]
            verdicts.append(
                (
                    commitment["run"],
                    commitment["before_step"],
                    "refused" if not any(at_risk) else "admitted",
                )
            )

        self.assertEqual(
            verdicts,
            [
                ("20260802T114747Z", 0, "refused"),
                ("20260802T182855Z", 0, "refused"),
                ("20260802T182855Z", 1, "refused"),
                ("20260803T061050Z", 0, "admitted"),
                ("20260803T061050Z", 1, "admitted"),
                ("20260803T062052Z", 0, "admitted"),
            ],
        )

    def test_host_grid_diff_handles_the_real_multi_layer_observation_shape(self) -> None:
        controller = CompetenceController()
        controller.receive_observation(
            observation(0, pixels=[[[0, 0], [0, 0]]])
        )
        controller.admit(
            {
                "action": 1,
                "data": None,
                "working_state": {},
                "test": "test whether an immediate grid difference is visible",
                "claim_ids": ["claim-1"],
                "ledger": [
                    {
                        "op": "create",
                        "claim_id": "claim-1",
                        "statement": "this action has an observable effect",
                    }
                ],
                "assertions": [
                    {"observable": "changed_cells", "op": "==", "value": 1},
                ],
                "evidence_requests": [],
                "reconciliation": None,
            },
            legal_actions=(1,),
        )
        controller.receive_observation(
            observation(1, pixels=[[[0, 1], [0, 0]]])
        )

        self.assertEqual(
            controller.context_packet()["last_prediction_result"]["outcome"],
            "match",
        )

    def test_nooa_final_action_reason_carries_the_typed_commitment(self) -> None:
        payload = {
            "working_state": {"candidate": "test action 1"},
            "test": "test one action",
            "claim_ids": ["claim-1"],
            "ledger": [
                {
                    "op": "create",
                    "claim_id": "claim-1",
                    "statement": "this action has an observable effect",
                }
            ],
            "assertions": [{"observable": "changed_cells", "op": ">", "value": 0}],
            "evidence_requests": [],
            "reconciliation": None,
        }

        decision = parse_model_reason(json.dumps(payload), action=1, data=None)

        self.assertEqual(decision["action"], 1)
        self.assertIsNone(decision["data"])
        with self.assertRaisesRegex(ContractError, "JSON object"):
            parse_model_reason("a brief explanation", action=1, data=None)
        with self.assertRaisesRegex(ContractError, "must not override"):
            parse_model_reason(
                json.dumps({"action": 2}),
                action=1,
                data=None,
            )

    def test_model_claim_becomes_a_commitment_then_a_public_mismatch(self) -> None:
        controller = CompetenceController()
        controller.receive_observation(observation(0, pixels=[[0, 0], [0, 0]]))

        controller.admit(
            {
                "action": 1,
                "data": None,
                "working_state": {"candidate": "action 1 changes the public grid"},
                "test": "test the immediate effect of action 1",
                "claim_ids": ["claim-1"],
                "ledger": [
                    {
                        "op": "create",
                        "claim_id": "claim-1",
                        "statement": "action 1 changes at least one public cell",
                    }
                ],
                "assertions": [
                    {"observable": "changed_cells", "op": ">", "value": 0},
                ],
                "evidence_requests": [],
                "reconciliation": None,
            },
            legal_actions=(1, 2),
        )
        controller.receive_observation(observation(1, pixels=[[0, 0], [0, 0]]))

        packet = controller.context_packet()
        self.assertEqual(packet["working_claims"][0]["claim_id"], "claim-1")
        self.assertEqual(packet["last_prediction_result"]["outcome"], "mismatch")
        self.assertEqual(len(packet["unresolved_mismatches"]), 1)
        self.assertEqual(packet["requested_evidence"], [])

        with self.assertRaisesRegex(ContractError, "reconciliation"):
            controller.admit(
                {
                    "action": 2,
                    "data": None,
                    "working_state": {},
                    "test": "act again without accounting for the result",
                    "claim_ids": ["claim-1"],
                    "ledger": [],
                    "assertions": [{"observable": "changed_cells", "op": ">", "value": 0}],
                    "evidence_requests": [],
                    "reconciliation": None,
                },
                legal_actions=(1, 2),
            )

        controller.admit(
            {
                "action": 2,
                "data": None,
                "working_state": {"candidate": "the first action was inert"},
                "test": "try a distinct legal input",
                "claim_ids": ["claim-1"],
                "ledger": [
                    {
                        "op": "revise",
                        "claim_id": "claim-1",
                        "statement": "action 1 did not change the tested frame",
                    }
                ],
                "assertions": [{"observable": "changed_cells", "op": ">", "value": 0}],
                "evidence_requests": [
                    {"operator": "transition", "action_id": 1},
                ],
                "reconciliation": [
                    {
                        "mismatch_id": packet["unresolved_mismatches"][0]["mismatch_id"],
                        "disposition": "ABANDON",
                        "note": "The predicted public change did not occur.",
                    }
                ],
            },
            legal_actions=(1, 2),
        )

        resolved = controller.context_packet()
        self.assertEqual(resolved["unresolved_mismatches"], [])
        self.assertEqual(resolved["working_claims"][0]["revision"], 1)
        self.assertEqual(resolved["requested_evidence"][0]["operator"], "transition")
        self.assertEqual(resolved["requested_evidence"][0]["output"]["changed_cells"], 0)
        self.assertTrue(
            resolved["requested_evidence"][0]["source_transition"]["transition_id"].startswith(
                "trn-"
            )
        )

    def test_reconciliation_may_be_omitted_only_while_nothing_is_unresolved(self) -> None:
        """Reproduces the 20260802T052341Z live failure and its boundary.

        The model authored ``reconciliation: {}``, deleted the field on the
        reasoning that absence means nothing to reconcile, and the run died
        before admitting any action. With nothing unresolved, null is the only
        legal value, so omission is accepted. Once a mismatch is outstanding the
        field carries real content and stays mandatory.
        """

        controller = CompetenceController()
        controller.receive_observation(observation(0, pixels=[[0, 0], [0, 0]]))

        commitment = controller.admit(
            {
                "action": 1,
                "data": None,
                "working_state": {"candidate": "action 1 changes the public grid"},
                "test": "test the immediate effect of action 1",
                "claim_ids": ["claim-1"],
                "ledger": [
                    {
                        "op": "create",
                        "claim_id": "claim-1",
                        "statement": "action 1 changes at least one public cell",
                    }
                ],
                "assertions": [{"observable": "changed_cells", "op": ">", "value": 0}],
                "evidence_requests": [],
            },
            legal_actions=(1, 2),
        )
        self.assertIsNone(commitment["reconciliation"])

        controller.receive_observation(observation(1, pixels=[[0, 0], [0, 0]]))
        self.assertEqual(len(controller.context_packet()["unresolved_mismatches"]), 1)

        with self.assertRaisesRegex(ContractError, "reconciliation"):
            controller.admit(
                {
                    "action": 2,
                    "data": None,
                    "working_state": {},
                    "test": "omit the field while a mismatch is outstanding",
                    "claim_ids": ["claim-1"],
                    "ledger": [],
                    "assertions": [
                        {"observable": "changed_cells", "op": ">", "value": 0}
                    ],
                    "evidence_requests": [],
                },
                legal_actions=(1, 2),
            )

    def test_only_model_selected_claims_enter_the_bounded_context(self) -> None:
        controller = CompetenceController()
        controller.receive_observation(observation(0, pixels=[[0]]))
        controller.admit(
            {
                "action": 1,
                "data": None,
                "working_state": {},
                "test": "hold one selected claim in the next viewport",
                "claim_ids": ["claim-selected"],
                "ledger": [
                    {
                        "op": "create",
                        "claim_id": "claim-selected",
                        "statement": "this one is working state",
                    },
                    {
                        "op": "create",
                        "claim_id": "claim-unselected",
                        "statement": "this remains in the ledger only",
                    },
                ],
                "assertions": [{"observable": "changed_cells", "op": ">", "value": 0}],
                "evidence_requests": [],
                "reconciliation": None,
            },
            legal_actions=(1,),
        )
        controller.receive_observation(observation(1, pixels=[[0]]))

        packet = controller.context_packet()
        self.assertEqual(
            [claim["claim_id"] for claim in packet["working_claims"]],
            ["claim-selected"],
        )
        self.assertEqual(len(controller.ledger.active()), 2)

    def test_retain_keeps_an_anomaly_visible_after_releasing_the_gate(self) -> None:
        controller = CompetenceController()
        controller.receive_observation(observation(0, pixels=[[0]]))
        controller.admit(
            {
                "action": 1,
                "data": None,
                "working_state": {},
                "test": "test a change that will not occur",
                "claim_ids": ["claim-1"],
                "ledger": [
                    {
                        "op": "create",
                        "claim_id": "claim-1",
                        "statement": "this action has an observable effect",
                    }
                ],
                "assertions": [{"observable": "changed_cells", "op": ">", "value": 0}],
                "evidence_requests": [],
                "reconciliation": None,
            },
            legal_actions=(1,),
        )
        controller.receive_observation(observation(1, pixels=[[0]]))
        mismatch_id = controller.context_packet()["unresolved_mismatches"][0]["mismatch_id"]

        controller.admit(
            {
                "action": 1,
                "data": None,
                "working_state": {},
                "test": "continue while keeping the anomaly explicit",
                "claim_ids": ["claim-1"],
                "ledger": [],
                "assertions": [{"observable": "changed_cells", "op": ">", "value": 0}],
                "evidence_requests": [],
                "reconciliation": [
                    {
                        "mismatch_id": mismatch_id,
                        "disposition": "RETAIN",
                        "note": "The anomaly is still decision-relevant.",
                    }
                ],
            },
            legal_actions=(1,),
        )

        packet = controller.context_packet()
        self.assertEqual(packet["unresolved_mismatches"], [])
        self.assertEqual(packet["retained_mismatches"][0]["mismatch_id"], mismatch_id)

    def test_rejected_reconciliation_does_not_clear_a_mismatch(self) -> None:
        controller = CompetenceController()
        controller.receive_observation(observation(0, pixels=[[0]]))
        controller.admit(
            {
                "action": 1,
                "data": None,
                "working_state": {},
                "test": "test a change",
                "claim_ids": ["claim-1"],
                "ledger": [
                    {
                        "op": "create",
                        "claim_id": "claim-1",
                        "statement": "this action has an observable effect",
                    }
                ],
                "assertions": [{"observable": "changed_cells", "op": ">", "value": 0}],
                "evidence_requests": [],
                "reconciliation": None,
            },
            legal_actions=(1,),
        )
        controller.receive_observation(observation(1, pixels=[[0]]))
        mismatch_id = controller.context_packet()["unresolved_mismatches"][0]["mismatch_id"]

        with self.assertRaisesRegex(ContractError, "neutral transition"):
            controller.admit(
                {
                    "action": 1,
                    "data": None,
                    "working_state": {},
                    "test": "this candidate must be rejected",
                    "claim_ids": ["claim-1"],
                    "ledger": [],
                    "assertions": [{"observable": "changed_cells", "op": ">", "value": 0}],
                    "evidence_requests": [{"operator": "not-an-operator", "action_id": 1}],
                    "reconciliation": [
                        {
                            "mismatch_id": mismatch_id,
                            "disposition": "ABANDON",
                            "note": "The prediction failed.",
                        }
                    ],
                },
                legal_actions=(1,),
            )

        self.assertEqual(
            controller.context_packet()["unresolved_mismatches"][0]["mismatch_id"],
            mismatch_id,
        )


if __name__ == "__main__":
    unittest.main()
