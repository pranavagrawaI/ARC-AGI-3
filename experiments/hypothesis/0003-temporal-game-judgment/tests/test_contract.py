import json, sys, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from game_judgment import ContractError, parse_decision, score, visible_case

class ContractTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  import freeze_batch; freeze_batch.main()
  cls.case=json.loads((ROOT/"fixtures/batch.json").read_text())["cases"][0]
  cls.label=json.loads((ROOT/"scorer/labels.json").read_text())["labels"][0]
 def test_valid_alternatives_and_multiple_objects(self):
  d=parse_decision({"tool":"EXECUTE","arguments":{"plan_id":"route-historical","first_action":1,"predicted_public_fields":{"public.levels_completed":1}}})
  self.assertTrue(score(self.case,self.label,d)["pass"])
  self.assertEqual(len(visible_case(self.case,"F")["candidate_scene"]["regions"]),2)
 def test_malformed_undeclared_illegal_unknown_and_contradiction_fail_closed(self):
  bad=["no",{"tool":"GO","arguments":{}},{"tool":"EXECUTE","arguments":{"plan_id":"route-historical","first_action":9,"predicted_public_fields":{"x":"UNKNOWN"}}}]
  for raw in bad:
   with self.subTest(raw=raw), self.assertRaises(ContractError): parse_decision(raw)
  d=parse_decision({"tool":"EXECUTE","arguments":{"plan_id":"route-historical","first_action":9,"predicted_public_fields":{"public.levels_completed":1}}})
  self.assertFalse(score(self.case,self.label,d)["pass"])
  d=parse_decision({"tool":"EXECUTE","arguments":{"plan_id":"route-historical","first_action":1,"predicted_public_fields":{"public.levels_completed":0}}})
  self.assertFalse(score(self.case,self.label,d)["pass"])
 def test_probe_and_abstention(self):
  self.assertEqual(parse_decision({"tool":"PROBE","arguments":{"discriminator_id":"cycle-entry","action":1,"predicted_outcomes":{"a":0,"b":1}}}).tool,"PROBE")
  self.assertEqual(parse_decision({"tool":"ABSTAIN","arguments":{"reason_code":"insufficient_evidence","missing_evidence":["objective"]}}).tool,"ABSTAIN")
 def test_scorer_deletion_does_not_change_requests(self):
  before=[visible_case(self.case,a) for a in "ABCDEFGH"]
  with tempfile.TemporaryDirectory() as t:
   Path(t,"labels.json").write_text(json.dumps(self.label)); Path(t,"labels.json").unlink()
  self.assertEqual(before,[visible_case(self.case,a) for a in "ABCDEFGH"])
 def test_arm_boundaries(self):
  self.assertEqual(len(visible_case(self.case,"A")["images"]),1); self.assertEqual(len(visible_case(self.case,"B")["images"]),4)
  self.assertNotIn("numeric_transitions",visible_case(self.case,"C")); self.assertIn("numeric_transitions",visible_case(self.case,"D"))
  self.assertIn("reflection",visible_case(self.case,"E")); self.assertIn("executable_conjectures",visible_case(self.case,"F")); self.assertIn("candidate_plans",visible_case(self.case,"G")); self.assertIn("verified_route",visible_case(self.case,"H"))
if __name__=="__main__": unittest.main()
