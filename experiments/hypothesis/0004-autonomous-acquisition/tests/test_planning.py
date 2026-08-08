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
  cls.case=json.loads((ROOT/"fixtures/planning-case.json").read_text())
  cls.label=json.loads((ROOT/"scorer/planning-label.json").read_text())
 def test_case_has_no_route_or_successor(self):
  self.assertTrue({"candidate_plans","verified_route","protected_successor","checkpoint_prefix_actions"}.isdisjoint(self.case))
 def test_closed_plan_parses(self):
  decision=parse_response(response("SUBMIT_PLAN",{"actions":[1,4,2],"predicted_level_gain":1,"evidence_refs":["chronological_frames"]}))
  self.assertEqual(decision.signature,("SUBMIT_PLAN",(1,4,2)))
 def test_malformed_plans_fail_closed(self):
  for actions in [[],[5],[True],[1]*65]:
   with self.assertRaises(ValueError):parse_response(response("SUBMIT_PLAN",{"actions":actions,"predicted_level_gain":1,"evidence_refs":["chronological_frames"]}))
 def test_evidence_schema_is_closed(self):
  schema=tool_definitions()[0]["function"]["parameters"]["properties"]["evidence_refs"]
  self.assertEqual(tuple(schema["items"]["enum"]),EVIDENCE_REFS)
 def test_offline_checkpoint_matches_frozen_prefix(self):
  arcade,_,frame=checkpoint(self.label)
  try:self.assertEqual(frame.state.value,"NOT_FINISHED")
  finally:arcade.close_scorecard()


if __name__=="__main__":unittest.main()
