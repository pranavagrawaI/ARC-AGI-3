"""Contract tests for the first 0006-NOOA competence vertical slice."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from competence import (
    CompetenceController,
    ContractError,
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
            "assertions": [{"observable": "state", "op": "==", "value": "NOT_FINISHED"}],
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
                    "assertions": [{"observable": "state", "op": "==", "value": "NOT_FINISHED"}],
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
                "assertions": [{"observable": "state", "op": "==", "value": "NOT_FINISHED"}],
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
                        {"observable": "state", "op": "==", "value": "NOT_FINISHED"}
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
                "assertions": [{"observable": "state", "op": "==", "value": "NOT_FINISHED"}],
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
                "assertions": [{"observable": "state", "op": "==", "value": "NOT_FINISHED"}],
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
                    "assertions": [{"observable": "state", "op": "==", "value": "NOT_FINISHED"}],
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
