import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from planning import EVIDENCE_REFS,parse_response,tool_definitions
from run_planning import checkpoint


def response(tool,arguments):
 return {"choices":[{"message":{"tool_calls":[{"function":{"name":tool,"arguments":json.dumps(arguments)}}]}}]}


class PlanningTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  import freeze_planning;freeze_planning.main()
  import freeze_affordances;freeze_affordances.main()
  cls.case=json.loads((ROOT/"fixtures/planning-case.json").read_text())
  cls.affordance_case=json.loads((ROOT/"fixtures/planning-affordance-case.json").read_text())
  cls.compact_case=json.loads((ROOT/"fixtures/planning-affordance-compact-case.json").read_text())
  cls.label=json.loads((ROOT/"scorer/planning-label.json").read_text())
 def test_case_has_no_route_or_successor(self):
  self.assertTrue({"candidate_plans","verified_route","protected_successor","checkpoint_prefix_actions"}.isdisjoint(self.case))
 def test_closed_plan_parses(self):
  decision=parse_response(response("SUBMIT_PLAN",{"actions":[1,4,2],"evidence_refs":["chronological_frames"]}))
  self.assertEqual(decision.signature,("SUBMIT_PLAN",(1,4,2)))
  compact=parse_response(response("SUBMIT_PLAN",{"actions":[1,4,3],"evidence_refs":["action_model","causal_affordances","current_state","objective"]}))
  self.assertEqual(compact.arguments["actions"],[1,4,3])
 def test_malformed_plans_fail_closed(self):
  for actions in [[],[5],[True],[1]*65]:
   with self.assertRaises(ValueError):parse_response(response("SUBMIT_PLAN",{"actions":actions,"evidence_refs":["chronological_frames"]}))
 def test_evidence_schema_is_closed(self):
  schema=tool_definitions()[0]["function"]["parameters"]["properties"]["evidence_refs"]
  self.assertEqual(tuple(schema["items"]["enum"]),EVIDENCE_REFS)
 def test_affordance_is_prior_supported_and_route_free(self):
  affordance=self.affordance_case["causal_affordances"][0]
  self.assertEqual(affordance["current_trigger_top_left"],[10,49])
  self.assertEqual(affordance["objective_applications_required"],2)
  self.assertEqual(len(affordance["replay"]["supports"]),3)
  self.assertTrue(all(row["prior_level"]==1 for row in affordance["replay"]["supports"]))
  self.assertTrue({"candidate_plans","verified_route","protected_successor"}.isdisjoint(self.affordance_case))
 def test_compact_case_preserves_effect_without_route(self):
  self.assertEqual(self.compact_case["current_state"],{"controlled_cell":[30,24],"target_cell":[50,54]})
  self.assertEqual(self.compact_case["causal_affordances"][0]["trigger_cell"],[10,49])
  self.assertEqual(self.compact_case["causal_affordances"][0]["contacts_required"],2)
  self.assertIn("divide coordinate difference by 5",self.compact_case["planning_rule"])
  self.assertEqual(len(self.compact_case["images"]),1)
  self.assertTrue({"candidate_plans","verified_route","protected_successor"}.isdisjoint(self.compact_case))
 def test_offline_checkpoint_matches_frozen_prefix(self):
  arcade,_,frame=checkpoint(self.label)
  try:self.assertEqual(frame.state.value,"NOT_FINISHED")
  finally:arcade.close_scorecard()


if __name__=="__main__":unittest.main()
