from __future__ import annotations

import json
from copy import deepcopy
from io import BytesIO
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from game_judgment.experiment import HTTPTransport, perturb_judgment_case, thinking_config  # noqa: E402
from game_judgment.provider import assemble_provider_request  # noqa: E402
from game_judgment.serialization import digest  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_ladder.py"
CONFIG = json.loads((ROOT / "runtime-config.json").read_text())


def provider_response(decision=None, model="google/gemma-4-31b-it", provider="Novita"):
    if decision is None:
        message = {"content": "{}"}
        finish = "stop"
    else:
        transport_decision = deepcopy(decision)
        if transport_decision["tool"] == "PROBE":
            transport_decision["arguments"]["predicted_outcomes"] = [
                {
                    "conjecture_id": conjecture_id,
                    "levels_completed": prediction["public.levels_completed"],
                }
                for conjecture_id, prediction in transport_decision["arguments"]["predicted_outcomes"].items()
            ]
        message = {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": transport_decision["tool"],
                        "arguments": json.dumps(transport_decision["arguments"]),
                    },
                }
            ]
        }
        finish = "tool_calls"
    return {
        "model": model,
        "provider": provider,
        "choices": [{"message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10, "cost": 0.001},
    }


def replay_row(
    *,
    phase,
    case,
    arm,
    repeat,
    response,
    variant="intact",
    sequence_index=0,
    request_config=None,
):
    request, _ = assemble_provider_request(
        case, arm, ROOT, request_config or CONFIG
    )
    return {
        "phase": phase,
        "case_id": case["case_id"],
        "arm": arm,
        "repeat": repeat,
        "variant": variant,
        "sequence_index": sequence_index,
        "request_sha256": digest(request),
        "response": response,
    }


class RunnerTests(unittest.TestCase):
    def test_http_transport_retries_declared_transient_status_and_audits_attempts(self):
        class Response:
            status = 200
            headers = {}

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self):
                return json.dumps(provider_response()).encode()

        throttled = HTTPError(
            "https://example.invalid",
            429,
            "Too Many Requests",
            {},
            BytesIO(b'{"error":{"code":429,"message":"slow down"}}'),
        )
        transport = HTTPTransport(
            "https://example.invalid", "key", 1, max_attempts=2
        )
        with patch(
            "game_judgment.experiment.urlopen", side_effect=[throttled, Response()]
        ), patch("game_judgment.experiment.sleep"):
            payload, audit = transport.call({}, {"model": "test"})
        self.assertIn("choices", payload)
        self.assertEqual(audit["attempt_count"], 2)
        self.assertEqual(
            [row["http_status"] for row in audit["attempts"]], [429, 200]
        )

    def test_provider_error_aborts_on_first_response(self):
        batch = json.loads((ROOT / "fixtures/batch.json").read_text())
        case = batch["cases"][0]
        row = replay_row(
            phase="ladder",
            case=case,
            arm="A",
            repeat=0,
            response={"error": {"code": 502, "message": "unsupported grammar"}},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            responses = root / "responses.jsonl"
            responses.write_text(json.dumps(row) + "\n")
            output = root / "run"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--replay-responses",
                    str(responses),
                    "--phase",
                    "ladder",
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            result = json.loads((output / "results.json").read_text())
            self.assertEqual(result["model_requests"], 1)
            self.assertEqual(len(result["results"]), 1)
            self.assertIn("provider error: 502", result["error"])

    def test_empty_replay_fails_nonzero_with_no_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            responses = root / "empty.jsonl"
            responses.write_text("")
            output = root / "run"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--replay-responses",
                    str(responses),
                    "--phase",
                    "ladder",
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            result = json.loads((output / "results.json").read_text())
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(any(row.get("score", {}).get("pass") for row in result["results"]))

    def test_repeated_ladder_stops_at_first_signature_stable_arm(self):
        batch = json.loads((ROOT / "fixtures/batch.json").read_text())
        cases = {case["case_id"]: case for case in batch["cases"]}
        decisions = {
            "execution-control": {
                "tool": "EXECUTE",
                "arguments": {
                    "plan_id": "route-left-control",
                    "first_action": 3,
                    "predicted_public_fields": {"public.levels_completed": 0},
                },
            },
            "planning-control": {
                "tool": "EXECUTE",
                "arguments": {
                    "plan_id": "route-to-supplied-goal",
                    "first_action": 1,
                    "predicted_public_fields": {"public.levels_completed": 0},
                },
            },
            "game-judgment": {
                "tool": "PROBE",
                "arguments": {
                    "discriminator_id": cases["game-judgment"]["discriminators"][0]["discriminator_id"],
                    "action": cases["game-judgment"]["discriminators"][0]["action"],
                    "predicted_outcomes": cases["game-judgment"]["discriminators"][0]["predicted_outcomes"],
                },
            },
        }
        rows = []
        for arm in "ABCDEFG":
            for case_id in cases:
                for repeat in range(2):
                    rows.append(
                        replay_row(
                            phase="ladder",
                            case=cases[case_id],
                            arm=arm,
                            repeat=repeat,
                            response=provider_response(
                                decisions[case_id] if arm == "G" else None
                            ),
                        )
                    )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            responses = root / "responses.jsonl"
            responses.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            output = root / "run"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--replay-responses",
                    str(responses),
                    "--phase",
                    "ladder",
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads((output / "ladder-summary.json").read_text())
            self.assertEqual(summary["minimal_successful_arm"], "G")
            result = json.loads((output / "results.json").read_text())
            self.assertEqual(result["model_requests"], 42)
            self.assertEqual(result["provider_calls"], 0)
            self.assertEqual(sum(row.get("score", {}).get("pass", False) for row in result["results"]), 6)

    def test_full_replay_runs_thinking_perturbation_and_e4b_transfer(self):
        batch = json.loads((ROOT / "fixtures/batch.json").read_text())
        cases = {case["case_id"]: case for case in batch["cases"]}
        discriminator = cases["game-judgment"]["discriminators"][0]
        decisions = {
            "execution-control": {"tool":"EXECUTE","arguments":{"plan_id":"route-left-control","first_action":3,"predicted_public_fields":{"public.levels_completed":0}}},
            "planning-control": {"tool":"EXECUTE","arguments":{"plan_id":"route-to-supplied-goal","first_action":1,"predicted_public_fields":{"public.levels_completed":0}}},
            "game-judgment": {"tool":"PROBE","arguments":{"discriminator_id":discriminator["discriminator_id"],"action":discriminator["action"],"predicted_outcomes":discriminator["predicted_outcomes"]}},
        }
        abstain={"tool":"ABSTAIN","arguments":{"reason_code":"insufficient_evidence","missing_evidence":["cycle_delta"]}}
        rows=[]
        for arm in "ABCDEFG":
            for case_id in cases:
                for repeat in range(2):
                    rows.append(replay_row(phase="ladder",case=cases[case_id],arm=arm,repeat=repeat,response=provider_response(decisions[case_id] if arm=="G" else None)))
        for sequence_index,enabled in enumerate((False,True,False)):
            for case_id in cases:
                condition=thinking_config(CONFIG,enabled)
                rows.append(replay_row(phase="thinking",case=cases[case_id],arm="G",repeat=0,variant="thinking-on" if enabled else "thinking-off",sequence_index=sequence_index,request_config=condition,response=provider_response(decisions[case_id])))
        perturbed,_=perturb_judgment_case(cases["game-judgment"])
        for sequence_index,variant in enumerate(("intact","perturbed","intact")):
            for repeat in range(2):
                request_case=perturbed if variant=="perturbed" else cases["game-judgment"]
                rows.append(replay_row(phase="perturbation",case=request_case,arm="G",repeat=repeat,variant=variant,sequence_index=sequence_index,response=provider_response(abstain if variant=="perturbed" else decisions["game-judgment"])))
        transfer_config=dict(CONFIG); transfer_config["model"]="google/gemma-4-E4B-it"; transfer_config.pop("provider",None)
        for case_id in cases:
            for repeat in range(2):
                rows.append(replay_row(phase="transfer",case=cases[case_id],arm="G",repeat=repeat,request_config=transfer_config,response=provider_response(decisions[case_id],model="google/gemma-4-E4B-it",provider="Local")))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); responses=root/"responses.jsonl"; output=root/"run"
            responses.write_text("\n".join(json.dumps(row) for row in rows)+"\n")
            completed=subprocess.run([sys.executable,str(RUNNER),"--replay-responses",str(responses),"--phase","full","--output",str(output)],capture_output=True,text=True)
            self.assertEqual(completed.returncode,0,completed.stderr)
            self.assertTrue(json.loads((output/"thinking-summary.json").read_text())["first_off_equals_last_off"])
            self.assertTrue(json.loads((output/"perturbation-summary.json").read_text())["pass"])
            self.assertTrue(json.loads((output/"transfer-summary.json").read_text())["stable_success"])
            result=json.loads((output/"results.json").read_text())
            self.assertEqual(result["status"],"PASS_REPLAY")
            self.assertEqual(result["transport_mode"],"replay")
            self.assertEqual(result["provider_calls"],0)

    def test_perturbation_phase_runs_only_g_baseline_and_sequence(self):
        batch = json.loads((ROOT / "fixtures/batch.json").read_text())
        cases = {case["case_id"]: case for case in batch["cases"]}
        discriminator = cases["game-judgment"]["discriminators"][0]
        decisions = {
            "execution-control": {"tool":"EXECUTE","arguments":{"plan_id":"route-left-control","first_action":3,"predicted_public_fields":{"public.levels_completed":0}}},
            "planning-control": {"tool":"EXECUTE","arguments":{"plan_id":"route-to-supplied-goal","first_action":1,"predicted_public_fields":{"public.levels_completed":0}}},
            "game-judgment": {"tool":"PROBE","arguments":{"discriminator_id":discriminator["discriminator_id"],"action":discriminator["action"],"predicted_outcomes":discriminator["predicted_outcomes"]}},
        }
        abstain={"tool":"ABSTAIN","arguments":{"reason_code":"insufficient_evidence","missing_evidence":["cycle_delta"]}}
        rows=[]
        for case_id in cases:
            for repeat in range(2):
                rows.append(replay_row(phase="ladder",case=cases[case_id],arm="G",repeat=repeat,response=provider_response(decisions[case_id])))
        perturbed,_=perturb_judgment_case(cases["game-judgment"])
        for sequence_index,variant in enumerate(("intact","perturbed","intact")):
            for repeat in range(2):
                request_case=perturbed if variant=="perturbed" else cases["game-judgment"]
                rows.append(replay_row(phase="perturbation",case=request_case,arm="G",repeat=repeat,variant=variant,sequence_index=sequence_index,response=provider_response(abstain if variant=="perturbed" else decisions["game-judgment"])))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); responses=root/"responses.jsonl"; output=root/"run"
            responses.write_text("\n".join(json.dumps(row) for row in rows)+"\n")
            completed=subprocess.run([sys.executable,str(RUNNER),"--replay-responses",str(responses),"--phase","perturbation","--only-arm","G","--output",str(output)],capture_output=True,text=True)
            self.assertEqual(completed.returncode,0,completed.stderr)
            self.assertTrue(json.loads((output/"perturbation-summary.json").read_text())["pass"])
            self.assertFalse((output/"thinking-summary.json").exists())
            self.assertFalse((output/"transfer-summary.json").exists())
            result=json.loads((output/"results.json").read_text())
            self.assertEqual(result["model_requests"],12)

    def test_thinking_phase_runs_only_g_baseline_and_off_on_off_sequence(self):
        batch = json.loads((ROOT / "fixtures/batch.json").read_text())
        cases = {case["case_id"]: case for case in batch["cases"]}
        discriminator = cases["game-judgment"]["discriminators"][0]
        decisions = {
            "execution-control": {"tool":"EXECUTE","arguments":{"plan_id":"route-left-control","first_action":3,"predicted_public_fields":{"public.levels_completed":0}}},
            "planning-control": {"tool":"EXECUTE","arguments":{"plan_id":"route-to-supplied-goal","first_action":1,"predicted_public_fields":{"public.levels_completed":0}}},
            "game-judgment": {"tool":"PROBE","arguments":{"discriminator_id":discriminator["discriminator_id"],"action":discriminator["action"],"predicted_outcomes":discriminator["predicted_outcomes"]}},
        }
        rows=[]
        for case_id in cases:
            for repeat in range(2):
                rows.append(replay_row(phase="ladder",case=cases[case_id],arm="G",repeat=repeat,response=provider_response(decisions[case_id])))
        for sequence_index,enabled in enumerate((False,True,False)):
            condition=thinking_config(CONFIG,enabled)
            for case_id in cases:
                rows.append(replay_row(phase="thinking",case=cases[case_id],arm="G",repeat=0,variant="thinking-on" if enabled else "thinking-off",sequence_index=sequence_index,request_config=condition,response=provider_response(decisions[case_id])))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); responses=root/"responses.jsonl"; output=root/"run"
            responses.write_text("\n".join(json.dumps(row) for row in rows)+"\n")
            completed=subprocess.run([sys.executable,str(RUNNER),"--replay-responses",str(responses),"--phase","thinking","--only-arm","G","--output",str(output)],capture_output=True,text=True)
            self.assertEqual(completed.returncode,0,completed.stderr)
            self.assertTrue(json.loads((output/"thinking-summary.json").read_text())["first_off_equals_last_off"])
            self.assertFalse((output/"perturbation-summary.json").exists())
            self.assertFalse((output/"transfer-summary.json").exists())
            result=json.loads((output/"results.json").read_text())
            self.assertEqual(result["model_requests"],15)

    def test_layout_phase_runs_only_g_baseline_and_layout_sequence(self):
        batch = json.loads((ROOT / "fixtures/batch.json").read_text())
        cases = {case["case_id"]: case for case in batch["cases"]}
        discriminator = cases["game-judgment"]["discriminators"][0]
        decisions = {
            "execution-control": {"tool":"EXECUTE","arguments":{"plan_id":"route-left-control","first_action":3,"predicted_public_fields":{"public.levels_completed":0}}},
            "planning-control": {"tool":"EXECUTE","arguments":{"plan_id":"route-to-supplied-goal","first_action":1,"predicted_public_fields":{"public.levels_completed":0}}},
            "game-judgment": {"tool":"PROBE","arguments":{"discriminator_id":discriminator["discriminator_id"],"action":discriminator["action"],"predicted_outcomes":discriminator["predicted_outcomes"]}},
        }
        rows=[]
        for case_id in cases:
            for repeat in range(2):
                rows.append(replay_row(phase="ladder",case=cases[case_id],arm="G",repeat=repeat,response=provider_response(decisions[case_id])))
        for sequence_index,layout in enumerate(("separated","candidate-local","separated")):
            condition=dict(CONFIG); condition["context_layout"]=layout
            for case_id in cases:
                rows.append(replay_row(phase="layout",case=cases[case_id],arm="G",repeat=0,variant=layout,sequence_index=sequence_index,request_config=condition,response=provider_response(decisions[case_id])))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); responses=root/"responses.jsonl"; output=root/"run"
            responses.write_text("\n".join(json.dumps(row) for row in rows)+"\n")
            completed=subprocess.run([sys.executable,str(RUNNER),"--replay-responses",str(responses),"--phase","layout","--only-arm","G","--output",str(output)],capture_output=True,text=True)
            self.assertEqual(completed.returncode,0,completed.stderr)
            self.assertTrue(json.loads((output/"layout-summary.json").read_text())["first_separated_equals_last_separated"])
            result=json.loads((output/"results.json").read_text())
            self.assertEqual(result["model_requests"],15)

    def test_transfer_phase_runs_only_g_baseline_and_e4b(self):
        batch = json.loads((ROOT / "fixtures/batch.json").read_text())
        cases = {case["case_id"]: case for case in batch["cases"]}
        discriminator = cases["game-judgment"]["discriminators"][0]
        decisions = {
            "execution-control": {"tool":"EXECUTE","arguments":{"plan_id":"route-left-control","first_action":3,"predicted_public_fields":{"public.levels_completed":0}}},
            "planning-control": {"tool":"EXECUTE","arguments":{"plan_id":"route-to-supplied-goal","first_action":1,"predicted_public_fields":{"public.levels_completed":0}}},
            "game-judgment": {"tool":"PROBE","arguments":{"discriminator_id":discriminator["discriminator_id"],"action":discriminator["action"],"predicted_outcomes":discriminator["predicted_outcomes"]}},
        }
        transfer_config=dict(CONFIG); transfer_config["model"]="google/gemma-4-E4B-it"; transfer_config.pop("provider",None)
        rows=[]
        for case_id in cases:
            for repeat in range(2):
                rows.append(replay_row(phase="ladder",case=cases[case_id],arm="G",repeat=repeat,response=provider_response(decisions[case_id])))
                rows.append(replay_row(phase="transfer",case=cases[case_id],arm="G",repeat=repeat,request_config=transfer_config,response=provider_response(decisions[case_id],model="google/gemma-4-E4B-it",provider="Local")))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); responses=root/"responses.jsonl"; output=root/"run"
            responses.write_text("\n".join(json.dumps(row) for row in rows)+"\n")
            completed=subprocess.run([sys.executable,str(RUNNER),"--replay-responses",str(responses),"--phase","transfer","--only-arm","G","--output",str(output)],capture_output=True,text=True)
            self.assertEqual(completed.returncode,0,completed.stderr)
            self.assertTrue(json.loads((output/"transfer-summary.json").read_text())["stable_success"])
            self.assertEqual(json.loads((output/"results.json").read_text())["model_requests"],12)

    def test_reconstruction_control_compares_e4b_g_with_disabled_d(self):
        batch = json.loads((ROOT / "fixtures/batch.json").read_text())
        cases = {case["case_id"]: case for case in batch["cases"]}
        discriminator = cases["game-judgment"]["discriminators"][0]
        decisions = {
            "execution-control": {"tool":"EXECUTE","arguments":{"plan_id":"route-left-control","first_action":3,"predicted_public_fields":{"public.levels_completed":0}}},
            "planning-control": {"tool":"EXECUTE","arguments":{"plan_id":"route-to-supplied-goal","first_action":1,"predicted_public_fields":{"public.levels_completed":0}}},
            "game-judgment": {"tool":"PROBE","arguments":{"discriminator_id":discriminator["discriminator_id"],"action":discriminator["action"],"predicted_outcomes":discriminator["predicted_outcomes"]}},
        }
        abstain={"tool":"ABSTAIN","arguments":{"reason_code":"no_executable_options","missing_evidence":["candidate_plans","discriminators"]}}
        transfer_config=dict(CONFIG); transfer_config["model"]="google/gemma-4-E4B-it"; transfer_config.pop("provider",None)
        rows=[]
        for case_id in cases:
            for repeat in range(2):
                rows.append(replay_row(phase="reconstruction-control",case=cases[case_id],arm="G",repeat=repeat,variant="enabled",request_config=transfer_config,response=provider_response(decisions[case_id],model="google/gemma-4-E4B-it",provider="Local")))
        for case_id in cases:
            for repeat in range(2):
                rows.append(replay_row(phase="reconstruction-control",case=cases[case_id],arm="D",repeat=repeat,variant="disabled",request_config=transfer_config,response=provider_response(abstain,model="google/gemma-4-E4B-it",provider="Local")))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); responses=root/"responses.jsonl"; output=root/"run"
            responses.write_text("\n".join(json.dumps(row) for row in rows)+"\n")
            completed=subprocess.run([sys.executable,str(RUNNER),"--replay-responses",str(responses),"--phase","reconstruction-control","--output",str(output)],capture_output=True,text=True)
            self.assertEqual(completed.returncode,0,completed.stderr)
            self.assertTrue(json.loads((output/"reconstruction-control-summary.json").read_text())["pass"])
            self.assertEqual(json.loads((output/"results.json").read_text())["model_requests"],12)

    def test_replay_response_is_bound_to_exact_request(self):
        batch = json.loads((ROOT / "fixtures/batch.json").read_text())
        case = batch["cases"][0]
        row = replay_row(
            phase="ladder",
            case=case,
            arm="A",
            repeat=0,
            response=provider_response(),
        )
        row["request_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            responses = root / "responses.jsonl"
            responses.write_text(json.dumps(row) + "\n")
            output = root / "run"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--replay-responses",
                    str(responses),
                    "--phase",
                    "ladder",
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            result = json.loads((output / "results.json").read_text())
            self.assertIn("replay request mismatch", result["error"])


if __name__ == "__main__":
    unittest.main()
