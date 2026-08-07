from hashlib import sha256
from copy import deepcopy
import json, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from game_judgment import ContractError, parse_decision, prepare_requests, prepared_bytes, score, visible_case
from game_judgment.experiment import thinking_config

REQUEST_CONFIG={"model":"google/gemma-4-31b-it","visual_token_budget":280,"temperature":1.0,"top_p":0.95,"top_k":64,"max_completion_tokens":2048,"reasoning":{"enabled":False},"provider":{"only":["coreweave/bf16"],"order":["coreweave/bf16"],"allow_fallbacks":False,"require_parameters":True},"seed":0}

class ContractTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  import freeze_batch; freeze_batch.main()
  cls.case=json.loads((ROOT/"fixtures/batch.json").read_text())["cases"][0]
  cls.label=json.loads((ROOT/"scorer/labels.json").read_text())["labels"][0]
 def test_valid_alternatives_and_multiple_objects(self):
  d=parse_decision({"tool":"EXECUTE","arguments":{"plan_id":"route-left-control","first_action":3,"predicted_public_fields":{"public.levels_completed":0}}})
  self.assertTrue(score(self.case,self.label,d)["pass"])
  self.assertEqual(len(visible_case(self.case,"F")["candidate_scene"]["regions"]),2)
 def test_malformed_undeclared_illegal_unknown_and_contradiction_fail_closed(self):
  bad=["no",{"tool":"GO","arguments":{}},{"tool":"EXECUTE","arguments":{"plan_id":"route-left-control","first_action":9,"predicted_public_fields":{"x":"UNKNOWN"}}}]
  for raw in bad:
   with self.subTest(raw=raw), self.assertRaises(ContractError): parse_decision(raw)
  d=parse_decision({"tool":"EXECUTE","arguments":{"plan_id":"route-left-control","first_action":9,"predicted_public_fields":{"public.levels_completed":0}}})
  self.assertFalse(score(self.case,self.label,d)["pass"])
  d=parse_decision({"tool":"EXECUTE","arguments":{"plan_id":"route-left-control","first_action":3,"predicted_public_fields":{"public.levels_completed":1}}})
  self.assertFalse(score(self.case,self.label,d)["pass"])
 def test_probe_and_abstention(self):
  self.assertEqual(parse_decision({"tool":"PROBE","arguments":{"discriminator_id":"cycle-entry","action":1,"predicted_outcomes":{"a":0,"b":1}}}).tool,"PROBE")
  self.assertEqual(parse_decision({"tool":"ABSTAIN","arguments":{"reason_code":"insufficient_evidence","missing_evidence":["objective"]}}).tool,"ABSTAIN")
  with self.assertRaises(ContractError):
   parse_decision({"tool":"ABSTAIN","arguments":{"reason_code":"insufficient_evidence","missing_evidence":[]}})
 def test_boolean_actions_fail_closed(self):
  for tool,arguments in (
   ("EXECUTE",{"plan_id":"route-left-control","first_action":True,"predicted_public_fields":{"public.levels_completed":0}}),
   ("PROBE",{"discriminator_id":"D","action":True,"predicted_outcomes":{"a":0,"b":1}}),
  ):
   with self.subTest(tool=tool),self.assertRaises(ContractError):
    parse_decision({"tool":tool,"arguments":arguments})
 def test_probe_predictions_and_abstention_are_scored_semantically(self):
  batch=json.loads((ROOT/"fixtures/batch.json").read_text())
  labels=json.loads((ROOT/"scorer/labels.json").read_text())["labels"]
  judgment=next(case for case in batch["cases"] if case["case_id"]=="game-judgment")
  label=next(row for row in labels if row["case_id"]=="game-judgment")
  discriminator=judgment["discriminators"][0]
  probe=parse_decision({"tool":"PROBE","arguments":{"discriminator_id":discriminator["discriminator_id"],"action":discriminator["action"],"predicted_outcomes":discriminator["predicted_outcomes"]}})
  self.assertTrue(score(judgment,label,probe)["pass"])
  wrong=parse_decision({"tool":"PROBE","arguments":{"discriminator_id":discriminator["discriminator_id"],"action":discriminator["action"],"predicted_outcomes":{"a":0,"b":1}}})
  self.assertFalse(score(judgment,label,wrong)["pass"])
  abstain=parse_decision({"tool":"ABSTAIN","arguments":{"reason_code":"insufficient_evidence","missing_evidence":["cycle_delta"]}})
  self.assertTrue(score(judgment,label,abstain)["pass"])
  self.assertFalse(score(self.case,self.label,abstain)["pass"])
 def test_unresolved_dependency_cannot_pass_execute_scoring(self):
  case=deepcopy(self.case); label=deepcopy(self.label)
  case["candidate_plans"][0]["conjecture_ids"].append("O-progress")
  decision=parse_decision({"tool":"EXECUTE","arguments":{"plan_id":"route-left-control","first_action":3,"predicted_public_fields":{"public.levels_completed":0}}})
  self.assertFalse(score(case,label,decision)["pass"])
 def test_scorer_deletion_does_not_change_requests(self):
  before=prepared_bytes(prepare_requests(ROOT/"fixtures/batch.json",ROOT,REQUEST_CONFIG))
  scorer=ROOT/"scorer"; hidden=ROOT/"scorer.contract-test-hidden"
  scorer.rename(hidden)
  try:
   after=prepared_bytes(prepare_requests(ROOT/"fixtures/batch.json",ROOT,REQUEST_CONFIG))
  finally:
   hidden.rename(scorer)
  self.assertEqual(before,after)
 def test_provider_requests_deliver_images_and_visual_provenance(self):
  rows=prepare_requests(ROOT/"fixtures/batch.json",ROOT,REQUEST_CONFIG)
  current=next(row for row in rows if row["case_id"]=="execution-control" and row["arm"]=="A")
  chronological=next(row for row in rows if row["case_id"]=="execution-control" and row["arm"]=="C")
  current_images=[part for part in current["request"]["messages"][1]["content"] if part["type"]=="image_url"]
  chronological_images=[part for part in chronological["request"]["messages"][1]["content"] if part["type"]=="image_url"]
  self.assertEqual(len(current_images),1); self.assertEqual(len(chronological_images),4)
  self.assertTrue(current_images[0]["image_url"]["url"].startswith("data:image/png;base64,"))
  self.assertEqual(chronological["audit"]["images"][0]["requested_visual_token_grid"],[16,16])
  self.assertEqual(chronological["audit"]["images"][0]["reference_realized_visual_tokens"],256)
  self.assertTrue(all(row["chronology_label_rendered"] for row in chronological["audit"]["images"]))
  self.assertNotIn("fixtures/images",chronological["request"]["messages"][1]["content"][-1]["text"])
  unlabelled=next(row for row in rows if row["case_id"]=="execution-control" and row["arm"]=="B")
  self.assertNotEqual(unlabelled["audit"]["images"][0]["delivered_sha256"],chronological["audit"]["images"][0]["delivered_sha256"])
  definitions={row["function"]["name"]:row for row in current["request"]["tools"]}
  predictions=definitions["EXECUTE"]["function"]["parameters"]["properties"]["predicted_public_fields"]
  self.assertEqual(predictions["properties"]["public.levels_completed"]["type"],"integer")
  self.assertFalse(predictions["additionalProperties"])
 def test_thinking_layout_uses_off_on_off_control_token(self):
  off=thinking_config(REQUEST_CONFIG,False); on=thinking_config(REQUEST_CONFIG,True)
  self.assertFalse(off["system_prompt"].startswith("<|think|>"))
  self.assertTrue(on["system_prompt"].startswith("<|think|>\n"))
  self.assertFalse(off["reasoning"]["enabled"]); self.assertTrue(on["reasoning"]["enabled"])
 def test_protected_successor_cannot_change_requests(self):
  import freeze_batch

  source_lines=freeze_batch.SOURCE.read_text().splitlines()
  successor=json.loads(source_lines[14])
  successor["data"]["action_input"]["id"]="ACTION4"
  successor["data"]["levels_completed"]+=7
  changed_lines=[*source_lines[:14],json.dumps(successor),*source_lines[15:]]

  def generate(lines):
   with tempfile.TemporaryDirectory(dir=ROOT) as directory:
    root=Path(directory); source=root/"source.jsonl"
    source.write_text("\n".join(lines)+"\n")
    with patch.object(freeze_batch,"ROOT",root), patch.object(freeze_batch,"SOURCE",source), patch.object(freeze_batch,"EXPECTED",sha256(source.read_bytes()).hexdigest()):
     freeze_batch.main()
    batch=json.loads((root/"fixtures/batch.json").read_text())
    labels=json.loads((root/"scorer/labels.json").read_text())
    case=next(case for case in batch["cases"] if case["case_id"]=="game-judgment")
    label=next(label for label in labels["labels"] if label["case_id"]=="game-judgment")
    provider_requests=prepared_bytes(prepare_requests(root/"fixtures/batch.json",root,REQUEST_CONFIG))
    return provider_requests,label

  original_requests,original_label=generate(source_lines)
  changed_requests,changed_label=generate(changed_lines)
  self.assertEqual(original_requests,changed_requests)
  self.assertNotEqual(original_label,changed_label)
 def test_only_replay_working_conjectures_authorize_plans(self):
  batch=json.loads((ROOT/"fixtures/batch.json").read_text())
  for case in batch["cases"]:
   statuses={row["id"]:row["replay"]["status"] for row in case["executable_conjectures"]}
   for plan in case["candidate_plans"]:
    self.assertTrue(plan["conjecture_ids"])
    self.assertTrue(all(statuses[dependency]=="working" for dependency in plan["conjecture_ids"]))
  judgment=next(case for case in batch["cases"] if case["case_id"]=="game-judgment")
  self.assertEqual(judgment["candidate_plans"],[])
  self.assertIsNone(judgment["verified_route"])
  self.assertEqual(len(judgment["discriminators"]),1)
 def test_cases_isolate_execution_planning_and_judgment(self):
  batch={case["case_id"]:case for case in json.loads((ROOT/"fixtures/batch.json").read_text())["cases"]}
  self.assertEqual(batch["execution-control"]["candidate_plans"][0]["actions"],[3])
  self.assertEqual(batch["planning-control"]["candidate_plans"][0]["actions"],[1,1,1])
  self.assertEqual(batch["game-judgment"]["candidate_plans"],[])
 def test_arm_boundaries(self):
  self.assertEqual(len(visible_case(self.case,"A")["images"]),1); self.assertEqual(len(visible_case(self.case,"B")["images"]),4)
  self.assertNotIn("numeric_transitions",visible_case(self.case,"C")); self.assertIn("numeric_transitions",visible_case(self.case,"D"))
  self.assertIn("reflection",visible_case(self.case,"E")); self.assertIn("executable_conjectures",visible_case(self.case,"F")); self.assertIn("candidate_plans",visible_case(self.case,"G")); self.assertIn("verified_route",visible_case(self.case,"H"))
if __name__=="__main__": unittest.main()
