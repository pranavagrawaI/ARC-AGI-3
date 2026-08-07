from hashlib import sha256
from copy import deepcopy
import json, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from game_judgment import ContractError, parse_decision, prepare_requests, prepared_bytes, score, visible_case
from game_judgment.provider import normalize_response
from game_judgment.experiment import perturb_judgment_case, thinking_config
from game_judgment.live import validate_current, validate_successor

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
  self.assertEqual(parse_decision({"tool":"PROBE","arguments":{"discriminator_id":"cycle-entry","action":1,"predicted_outcomes":{"a":{"public.levels_completed":0},"b":{"public.levels_completed":1}}}}).tool,"PROBE")
  self.assertEqual(parse_decision({"tool":"ABSTAIN","arguments":{"reason_code":"insufficient_evidence","missing_evidence":["objective"]}}).tool,"ABSTAIN")
  with self.assertRaises(ContractError):
   parse_decision({"tool":"ABSTAIN","arguments":{"reason_code":"insufficient_evidence","missing_evidence":[]}})
 def test_probe_transport_records_are_closed_typed_and_decode_to_canonical_outcomes(self):
  payload={"model":"google/gemma-4-31b-it","provider":"Novita","choices":[{"finish_reason":"tool_calls","message":{"tool_calls":[{"function":{"name":"PROBE","arguments":json.dumps({"discriminator_id":"D","action":1,"predicted_outcomes":[{"conjecture_id":"C-a","levels_completed":1},{"conjecture_id":"C-b","levels_completed":0}]})}}]}}]}
  normalized=normalize_response(payload)
  self.assertEqual(normalized["transport_decision"]["arguments"]["predicted_outcomes"][0]["conjecture_id"],"C-a")
  self.assertEqual(normalized["decision"]["arguments"]["predicted_outcomes"],{"C-a":{"public.levels_completed":1},"C-b":{"public.levels_completed":0}})
  for outcomes in (
   [{"conjecture_id":"C-a","levels_completed":"1"},{"conjecture_id":"C-b","levels_completed":0}],
   [{"conjecture_id":"C-a","levels_completed":1,"extra":0},{"conjecture_id":"C-b","levels_completed":0}],
   [{"conjecture_id":"C-a","levels_completed":1},{"conjecture_id":"C-a","levels_completed":0}],
  ):
   bad=deepcopy(payload)
   bad["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]=json.dumps({"discriminator_id":"D","action":1,"predicted_outcomes":outcomes})
   with self.subTest(outcomes=outcomes),self.assertRaises(ValueError):
    normalize_response(bad)
 def test_boolean_actions_fail_closed(self):
  for tool,arguments in (
   ("EXECUTE",{"plan_id":"route-left-control","first_action":True,"predicted_public_fields":{"public.levels_completed":0}}),
   ("PROBE",{"discriminator_id":"D","action":True,"predicted_outcomes":{"a":{"public.levels_completed":0},"b":{"public.levels_completed":1}}}),
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
  wrong=parse_decision({"tool":"PROBE","arguments":{"discriminator_id":discriminator["discriminator_id"],"action":discriminator["action"],"predicted_outcomes":{"a":{"public.levels_completed":0},"b":{"public.levels_completed":1}}}})
  self.assertFalse(score(judgment,label,wrong)["pass"])
  abstain=parse_decision({"tool":"ABSTAIN","arguments":{"reason_code":"insufficient_evidence","missing_evidence":["cycle_delta"]}})
  self.assertTrue(score(judgment,label,abstain)["pass"])
  self.assertFalse(score(self.case,self.label,abstain)["pass"])
 def test_non_discriminating_probe_is_not_exposed_or_accepted(self):
  batch=json.loads((ROOT/"fixtures/batch.json").read_text())
  labels=json.loads((ROOT/"scorer/labels.json").read_text())["labels"]
  judgment=next(case for case in batch["cases"] if case["case_id"]=="game-judgment")
  label=next(row for row in labels if row["case_id"]=="game-judgment")
  perturbed,_=perturb_judgment_case(judgment)
  discriminator=perturbed["discriminators"][0]
  self.assertEqual(visible_case(perturbed,"G")["discriminators"],[])
  probe=parse_decision({"tool":"PROBE","arguments":{"discriminator_id":discriminator["discriminator_id"],"action":discriminator["action"],"predicted_outcomes":discriminator["predicted_outcomes"]}})
  self.assertFalse(score(perturbed,label,probe)["semantics"])
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
  execution_g=next(row for row in rows if row["case_id"]=="execution-control" and row["arm"]=="G")
  judgment_g=next(row for row in rows if row["case_id"]=="game-judgment" and row["arm"]=="G")
  execution_definitions={row["function"]["name"]:row for row in execution_g["request"]["tools"]}
  judgment_definitions={row["function"]["name"]:row for row in judgment_g["request"]["tools"]}
  predictions=execution_definitions["EXECUTE"]["function"]["parameters"]["properties"]["predicted_public_fields"]
  self.assertEqual(predictions["properties"]["public.levels_completed"]["type"],"integer")
  self.assertFalse(predictions["additionalProperties"])
  outcomes=judgment_definitions["PROBE"]["function"]["parameters"]["properties"]["predicted_outcomes"]
  self.assertEqual(outcomes["type"],"array")
  self.assertEqual(outcomes["minItems"],2)
  self.assertEqual(outcomes["items"]["required"],["conjecture_id","levels_completed"])
  self.assertEqual(outcomes["items"]["properties"]["levels_completed"]["type"],"integer")
  self.assertFalse(outcomes["items"]["additionalProperties"])
 def test_provider_exposes_only_tools_authorized_by_visible_candidates(self):
  rows=prepare_requests(ROOT/"fixtures/batch.json",ROOT,REQUEST_CONFIG)
  tools={(row["case_id"],row["arm"]):{tool["function"]["name"] for tool in row["request"]["tools"]} for row in rows}
  self.assertEqual(tools[("execution-control","G")],{"EXECUTE","ABSTAIN"})
  self.assertEqual(tools[("game-judgment","G")],{"PROBE","ABSTAIN"})
  self.assertEqual(tools[("game-judgment","D")],{"ABSTAIN"})
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
 def test_widened_batch_adds_prefix_derived_action_and_frontier_coverage(self):
  batch={case["case_id"]:case for case in json.loads((ROOT/"fixtures/widened-batch.json").read_text())["cases"]}
  labels={row["case_id"]:row for row in json.loads((ROOT/"scorer/widened-labels.json").read_text())["labels"]}
  self.assertEqual(set(batch),{"execution-control","planning-control","game-judgment","execution-control-left","planning-control-down"})
  self.assertEqual(batch["execution-control-left"]["candidate_plans"][0]["actions"],[1])
  self.assertEqual(batch["planning-control-down"]["candidate_plans"][0]["actions"],[4,4])
  self.assertIn(["EXECUTE","route-left-control-frontier-5",1],labels["execution-control-left"]["valid_decision_signatures"])
  self.assertIn(["EXECUTE","route-down-to-supplied-goal",4],labels["planning-control-down"]["valid_decision_signatures"])
 def test_live_initial_certificate_matches_offline_engine_without_scorer_leakage(self):
  import arc_agi
  from arc_agi.base import OperationMode
  from arcengine import GameAction

  case=json.loads((ROOT/"fixtures/live-initial-case.json").read_text())
  label=json.loads((ROOT/"scorer/live-initial-label.json").read_text())
  self.assertNotIn("protected_successor",case)
  self.assertFalse(any("expected_" in key for key in case))
  self.assertEqual(case["candidate_plans"][0]["actions"],[3])
  arcade=arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
  environment=arcade.make("ls20",seed=0)
  current=environment.reset()
  self.assertTrue(validate_current(current,case,label)["pass"])
  successor=environment.step(GameAction.ACTION3)
  self.assertTrue(validate_successor(successor,case["candidate_plans"][0]["predicted_public_fields"],label)["pass"])
 def test_route_ceiling_replays_one_level_offline_with_exact_successors(self):
  import arc_agi
  from arc_agi.base import OperationMode
  from arcengine import GameAction

  cases=json.loads((ROOT/"fixtures/live-route-ceiling.json").read_text())["cases"]
  labels=json.loads((ROOT/"scorer/live-route-ceiling-labels.json").read_text())["labels"]
  self.assertEqual(len(cases),13)
  self.assertTrue(all("protected_successor" not in case for case in cases))
  environment=arc_agi.Arcade(operation_mode=OperationMode.OFFLINE).make("ls20",seed=0)
  current=environment.reset()
  for case,label in zip(cases,labels):
   self.assertTrue(validate_current(current,case,label)["pass"])
   plan=case["candidate_plans"][0]
   current=environment.step(GameAction.from_id(plan["actions"][0]))
   self.assertTrue(validate_successor(current,plan["predicted_public_fields"],label)["pass"])
  self.assertEqual(current.levels_completed,1)
 def test_arm_boundaries(self):
  self.assertEqual(len(visible_case(self.case,"A")["images"]),1); self.assertEqual(len(visible_case(self.case,"B")["images"]),4)
  self.assertNotIn("numeric_transitions",visible_case(self.case,"C")); self.assertIn("numeric_transitions",visible_case(self.case,"D"))
  self.assertIn("reflection",visible_case(self.case,"E")); self.assertIn("executable_conjectures",visible_case(self.case,"F")); self.assertIn("candidate_plans",visible_case(self.case,"G")); self.assertIn("verified_route",visible_case(self.case,"H"))
 def test_candidate_local_layout_moves_relevant_conjectures_beside_candidates(self):
  batch={case["case_id"]:case for case in json.loads((ROOT/"fixtures/batch.json").read_text())["cases"]}
  planning=visible_case(batch["planning-control"],"G","candidate-local")
  judgment=visible_case(batch["game-judgment"],"G","candidate-local")
  self.assertNotIn("executable_conjectures",planning)
  self.assertEqual([row["id"] for row in planning["candidate_plans"][0]["supporting_conjectures"]],["C-translation","C-public-persistence"])
  self.assertEqual({row["id"] for row in judgment["discriminators"][0]["supporting_conjectures"]},{"C-cycle","C-cycle-competitor","O-progress"})
  self.assertEqual(judgment["context_layout"],"candidate-local")
if __name__=="__main__": unittest.main()
