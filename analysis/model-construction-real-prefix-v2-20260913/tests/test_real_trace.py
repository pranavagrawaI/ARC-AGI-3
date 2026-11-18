from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from constructor import ConstructionLimits, construction_messages, run_construction
from real_trace import load_case
from sandbox_eval import evaluate_candidate, execute_inspection


class RealTraceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = load_case()

    def test_split_is_chronological_and_suffix_is_sealed(self) -> None:
        self.assertEqual(self.case.frontier, 5)
        self.assertEqual([row["step"] for row in self.case.visible], [1, 2, 3, 4, 5])
        self.assertEqual([row["step"] for row in self.case.sealed_suffix], [6, 7, 8])
        self.assertEqual(self.case.planned_actions, ("UP", "UP", "LEFT", "LEFT", "LEFT"))
        self.assertEqual(self.case.incumbent_sha256, hashlib.sha256(self.case.incumbent_source.encode()).hexdigest())
        public = json.dumps(self.case.constructor_view(), sort_keys=True)
        self.assertNotIn(self.case.sealed_suffix_sha256, public)
        self.assertNotIn('"sealed_suffix"', public)

    def test_constructor_prompt_contains_policy_but_not_suffix(self) -> None:
        messages = construction_messages(self.case, feedback=None)
        rendered = json.dumps(messages, sort_keys=True)
        self.assertIn("ground", rendered)
        self.assertIn("candidate_source", rendered)
        self.assertIn("UP", rendered)
        self.assertNotIn(self.case.sealed_suffix_sha256, rendered)
        self.assertNotIn('"step": 6', rendered)
        self.assertNotIn('"step": 7', rendered)
        self.assertNotIn('"step": 8', rendered)

    def test_incumbent_replays_prefix_then_contradicts_suffix(self) -> None:
        result = evaluate_candidate(self.case.legacy_incumbent_source(), self.case, include_suffix=True)
        self.assertEqual(result["visible"]["status"], "passed")
        self.assertEqual(result["suffix"]["status"], "contradicted")
        self.assertEqual(result["suffix"]["compatible_depth"], 2)
        self.assertEqual(result["suffix"]["first_contradiction_step"], 8)

    def test_host_only_witness_proves_evaluator_can_accept_improvement(self) -> None:
        source = (HERE / "tests" / "witness_candidate.py").read_text()
        result = evaluate_candidate(source, self.case, include_suffix=True)
        self.assertEqual(result["visible"]["status"], "passed")
        self.assertEqual(result["suffix"]["status"], "passed")
        self.assertEqual(result["suffix"]["compatible_depth"], 3)
        self.assertEqual(result["coverage"]["missing_required"], [])

    def test_candidate_bundle_is_version_bound(self) -> None:
        source = (HERE / "tests" / "witness_candidate.py").read_text()
        result = evaluate_candidate(source, self.case, include_suffix=False)
        self.assertEqual(result["model_sha256"], hashlib.sha256(source.encode()).hexdigest())
        self.assertEqual(result["belief_model_sha256"], result["model_sha256"])

    def test_inspection_realm_sees_only_constructor_view(self) -> None:
        result = execute_inspection(
            "import json; data=json.load(open('/evidence/case.json')); "
            "print(data['frontier']); print(sorted(data))",
            self.case,
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["returncode"], 0)
        self.assertTrue(result["stdout"].startswith("5\n"))
        self.assertNotIn("sealed_suffix", result["stdout"])

    def test_bounded_tool_loop_inspects_then_submits_candidate(self) -> None:
        source = (HERE / "tests" / "witness_candidate.py").read_text()
        responses = iter(
            [
                {
                    "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [{
                        "id": "inspect-1",
                        "type": "function",
                        "function": {"name": "execute_python", "arguments": json.dumps({
                            "code": "import json; print(json.load(open('/evidence/case.json'))['frontier'])"
                        })},
                    }]}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10},
                },
                {
                    "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [{
                        "id": "submit-1",
                        "type": "function",
                        "function": {"name": "submit_candidate", "arguments": json.dumps({
                            "work_item": {
                                "decision": "assess fixed rollout",
                                "obstacle": "incumbent does not evaluate destination support",
                                "operation": "close",
                                "claim": "movement depends on maintained support evidence",
                                "evidence_refs": [1, 2, 3, 4, 5],
                                "dependencies": ["block", "bar_left", "terrain"],
                                "completion_checks": [1, 2, 3, 4, 5],
                                "remaining_scope": "covered terrain remains unresolved",
                                "reasoning_need": "needs_computation",
                            },
                            "candidate_source": source,
                        })},
                    }]}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10},
                },
            ]
        )
        seen = []

        def transport(body):
            seen.append(body)
            return next(responses)

        result = run_construction(
            self.case,
            transport=transport,
            limits=ConstructionLimits(reserve_requests=0, reserve_completion_tokens=0, max_model_requests=2, max_inspections=1, max_submissions=1),
        )

        self.assertEqual(result.status, "CANDIDATE_READY")
        self.assertEqual(result.model_requests, 2)
        self.assertEqual(result.inspections, 1)
        self.assertEqual(result.submissions, 1)
        self.assertEqual(result.candidate_source, source)
        self.assertTrue(all(body["reasoning"] == {"enabled": False} for body in seen))


if __name__ == "__main__":
    unittest.main()
