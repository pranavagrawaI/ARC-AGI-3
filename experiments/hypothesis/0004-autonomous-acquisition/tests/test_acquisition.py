from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from acquisition import assemble_request,parse_response,score

CONFIG={"model":"google/gemma-4-E4B-it","temperature":1.0,"top_p":0.95,"top_k":64,"max_completion_tokens":2048,"reasoning":{"enabled":False},"seed":0}


def response(tool,arguments):
 return {"choices":[{"message":{"tool_calls":[{"function":{"name":tool,"arguments":json.dumps(arguments)}}]}}]}


class AcquisitionTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  import freeze_acquisition;freeze_acquisition.main()
  cls.case=json.loads((ROOT/"fixtures/case.json").read_text())
  cls.label=json.loads((ROOT/"scorer/label.json").read_text())
 def test_frozen_boundary_and_transfer_audit(self):
  self.assertEqual([row["pair_id"] for row in self.case["candidate_hypotheses"]],[7,9,13,17,25,32,35])
  self.assertNotIn("valid_pair_id",json.dumps(self.case))
  transfer=json.loads((ROOT/"fixtures/machinery-transfer.json").read_text())["levels"]
  self.assertEqual([row["unique_scale_first_observation"] for row in transfer],[18,None,14,10])
 def test_select_compiles_and_heldout_scores(self):
  decision=parse_response(response("SELECT_RELATION",{"pair_id":32,"program_kind":"integer_scale_objective","factor":2,"evidence_refs":["invariant.patch","mutable.contact_change_rows"]}))
  self.assertTrue(score(self.case,self.label,decision)["acquired"])
  wrong=deepcopy(decision.arguments);wrong["pair_id"]=7
  self.assertFalse(score(self.case,self.label,parse_response(response("SELECT_RELATION",wrong)))["acquired"])
 def test_abstention_is_safe_but_not_acquisition(self):
  decision=parse_response(response("ABSTAIN",{"reason_code":"underdetermined","missing_evidence":["discriminating_contact"]}))
  result=score(self.case,self.label,decision)
  self.assertTrue(result["safe"]);self.assertFalse(result["acquired"])
 def test_request_is_scorer_independent(self):
  before=assemble_request(self.case,CONFIG)
  changed=deepcopy(self.label);changed["valid_pair_id"]=7
  self.assertEqual(before,assemble_request(self.case,CONFIG));self.assertNotEqual(changed,self.label)
 def test_malformed_or_invented_output_fails_closed(self):
  for payload in [response("SELECT_RELATION",{"pair_id":32}),response("INVENT",{}),response("ABSTAIN",{"reason_code":"x","missing_evidence":[]})]:
   with self.assertRaises(ValueError):parse_response(payload)


if __name__=="__main__":unittest.main()
