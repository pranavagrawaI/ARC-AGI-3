from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Barrier
from typing import Any
import unittest
from unittest.mock import patch

import numpy as np

from experiments.harness import Decision, JsonlTraceWriter, run_episode


AGENT_ROOT = Path(__file__).resolve().parents[1] / "0002-llm-repl-agent"
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from llm_repl_agent.agent import SYSTEM_PROMPT
from llm_repl_agent.experiment_protocol import (
    EXPERIMENT_FILES,
    ExperimentCondition,
    PairTelemetry,
    RunOutcome,
    assess_component_profile,
    assess_interface_damage,
    assess_pair,
    canonical_endpoint_metadata,
    compose_condition_prompt,
    endpoint_metadata_sha256,
    experiment_artifact_templates,
    frozen_pair_schedule,
    frozen_request_options,
    load_frozen_runtime,
    participating_source_hashes,
    resolve_run_annotation,
    validate_pair_commitments,
    validate_prompt_pair,
)
from llm_repl_agent.intervention import (
    assess_content_dependence,
    build_branch_payloads,
    build_shadow_schedule,
    candidate_id,
    execute_shadow_bundle,
    response_signature,
    validate_shadow_bundle_result,
    validate_branch_payloads,
)
from llm_repl_agent.client import ChatResponse, ModelRequestError, serialize_request_payload
from llm_repl_agent.authority import (
    build_final_freeze,
    consume_pair_authority,
    validate_final_freeze,
    validate_included_pair_result,
    validate_pilot_approval,
    validate_pilot_decision,
)
from llm_repl_agent.evidence import evidence_hashes
from llm_repl_agent.sandbox_admissibility import run_sandbox_admissibility_suite
from llm_repl_agent.workspace import PythonResult
from run_experiment_pair import (
    build_completed_pair_result,
    build_pair_commitments,
    gameplay_attempt_validity,
    validate_release_gate_bundle,
)
from run_shadow_intervention import load_shadow_preflight
from analyze_generation import ANALYSIS_INPUT_SCHEMA, analyze


def _valid_endpoint_metadata() -> dict[str, Any]:
    config = load_frozen_runtime()
    return {
        "schema": "openrouter-endpoint-allowlist-v1",
        "model": config.model,
        "provider_slug": config.provider.only[0],
        "provider_name": "Test Provider",
        "variant": "test",
        "quantization": config.provider.quantizations[0],
        "context_limit": 262144,
        "supported_parameters": [
            "max_tokens",
            "presence_penalty",
            "reasoning",
            "seed",
            "temperature",
            "tool_choice",
            "tools",
            "top_k",
            "top_p",
        ],
        "revision": None,
        "endpoint_id": "test-endpoint-id",
        "endpoint_name": "Test Endpoint",
        "endpoint_updated_at": None,
        "revision_identifier": "test-endpoint-id",
        "revision_identifier_source": "endpoint_id",
    }


class State(str):
    @property
    def value(self) -> str:
        return str(self)


class Frame:
    def __init__(
        self,
        value: int,
        *,
        levels_completed: int = 0,
        state: str = "NOT_FINISHED",
        available_actions: tuple[int, ...] = (1,),
    ) -> None:
        self.frame = [np.full((2, 2), value, dtype=np.int8)]
        self.available_actions = list(available_actions)
        self.levels_completed = levels_completed
        self.state = State(state)
        self.full_reset = False


class HoldoutSentinelFrame:
    available_actions = [1]
    levels_completed = 2
    state = State("NOT_FINISHED")
    full_reset = False

    @property
    def frame(self):
        raise AssertionError("holdout pixels were accessed")


class Environment:
    def __init__(self, frames: list[Frame]) -> None:
        self.frames = frames
        self.index = 0
        self.actions: list[int] = []

    def reset(self) -> Frame:
        return self.frames[0]

    def step(self, action, data=None, reasoning=None) -> Frame:
        self.actions.append(action.value)
        self.index += 1
        return self.frames[self.index]


class Agent:
    def __init__(self) -> None:
        self.seen_values: list[int] = []
        self.transitions = []

    def act(self, observation):
        self.seen_values.append(int(observation.pixels[0, 0, 0]))
        return Decision(1)

    def observe(self, transition) -> None:
        self.transitions.append(transition)


class FrozenConfigurationTests(unittest.TestCase):
    def test_annotation_disagreement_is_derived_as_unidentifiable_without_adjudication(self) -> None:
        evidence = {"process.jsonl": "a" * 64}
        annotation = experiment_artifact_templates(
            "disabled", evidence_hashes=evidence
        )["annotation"]
        annotation["annotators"] = [
            {
                "annotator_id": "A",
                "terminal_status": "compliant",
                "criterion_3_interface_damage": False,
                "evidence_event_indices": [1],
            },
            {
                "annotator_id": "B",
                "terminal_status": "violated",
                "criterion_3_interface_damage": True,
                "evidence_event_indices": [2],
            },
        ]

        result = resolve_run_annotation(
            annotation,
            condition="disabled",
            evidence_hashes=evidence,
        )

        self.assertEqual(result["terminal_status"], "unidentifiable")
        self.assertIsNone(result["criterion_3_interface_damage"])

    def test_runtime_and_request_exactly_match_the_frozen_contract(self) -> None:
        config = load_frozen_runtime()

        self.assertEqual(config.schema, "scientist-game-designer-player-runtime-v1")
        self.assertEqual(config.model, "qwen/qwen3.6-27b")
        self.assertEqual(config.runtime.max_model_rounds_per_action, 5)
        self.assertEqual(config.runtime.max_model_calls, 2030)
        self.assertEqual(config.budgets.level_1_actions, 44)
        self.assertEqual(config.budgets.level_2_actions_after_transition, 246)
        self.assertEqual(config.budgets.total_actions_invariant, 290)
        request_options = frozen_request_options(config)
        self.assertEqual(request_options["max_tokens"], 4096)
        self.assertIsNone(request_options["max_completion_tokens"])
        self.assertEqual(
            request_options["extra_body"]["provider"],
            {
                "order": ["chutes/fp8"],
                "only": ["chutes/fp8"],
                "allow_fallbacks": False,
                "require_parameters": True,
                "quantizations": ["fp8"],
            },
        )
        source_hashes = participating_source_hashes()
        self.assertIn(
            "experiments/agents/tests/test_falsifiable_experiment.py",
            source_hashes,
        )
        self.assertIn("experiments/harness/tests/test_harness.py", source_hashes)

    def test_annotation_and_candidate_templates_bind_to_immutable_evidence(self) -> None:
        templates = experiment_artifact_templates(
            ExperimentCondition.ENABLED,
            evidence_hashes={"process.jsonl": "abc", "trace.jsonl": "def"},
        )

        self.assertEqual(
            templates["annotation"]["authority"],
            "exact_independent_agreement_without_adjudication",
        )
        self.assertEqual(
            templates["candidate"]["selector"]["selection_rule"],
            "earliest_eligible_candidate",
        )
        self.assertFalse(templates["candidate"]["second_candidate_permitted"])
        self.assertIn("candidate_id", templates["candidate"]["selector"])
        self.assertIn("source_request_payload_archive", templates["candidate"]["selector"])
        self.assertIn(
            "source_reconstruction_matches_selected_content",
            templates["candidate"]["auditor"],
        )
        self.assertEqual(
            templates["candidate"]["evidence_hashes"]["process.jsonl"],
            "abc",
        )

    def test_condition_prompts_are_exact_and_only_the_authorized_passage_differs(self) -> None:
        enabled = EXPERIMENT_FILES / "reconstruction-enabled.txt"
        disabled = EXPERIMENT_FILES / "reconstruction-disabled.txt"

        report = validate_prompt_pair(enabled.read_bytes(), disabled.read_bytes())

        self.assertTrue(report["valid"])
        self.assertEqual(report["enabled_sha256"], "ad5e8fc75ef6d08e04a723f65c4b543c49cdc92589dbd169fb1f5c5d044ba247")
        self.assertEqual(report["disabled_sha256"], "e990ac30f4af49326530e1fc7308d72a8d48a3201a1ac1ce1dd226f10dc2c07b")
        self.assertEqual(report["enabled_bytes"], 2848)
        self.assertEqual(report["disabled_bytes"], 3040)
        self.assertTrue(compose_condition_prompt(ExperimentCondition.ENABLED).startswith(SYSTEM_PROMPT))

    def test_prompt_counts_are_pinned_to_the_official_tokenizer_revision(self) -> None:
        counts = json.loads(
            (EXPERIMENT_FILES / "prompt-token-counts.json").read_text(encoding="utf-8")
        )

        self.assertEqual(
            counts["tokenizer"],
            {
                "add_special_tokens": False,
                "model_repository": "Qwen/Qwen3.6-27B",
                "model_revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
                "tokenizer_json_sha256": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
                "tokenizer_url": (
                    "https://huggingface.co/Qwen/Qwen3.6-27B/resolve/"
                    "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9/tokenizer.json"
                ),
            },
        )
        self.assertEqual(counts["conditions"]["enabled"]["tokens"], 515)
        self.assertEqual(counts["conditions"]["enabled"]["full_prompt_tokens"], 810)
        self.assertEqual(counts["conditions"]["disabled"]["tokens"], 552)
        self.assertEqual(counts["conditions"]["disabled"]["full_prompt_tokens"], 847)

    def test_prompt_validator_rejects_a_shared_instruction_change(self) -> None:
        enabled = (EXPERIMENT_FILES / "reconstruction-enabled.txt").read_bytes()
        disabled = (EXPERIMENT_FILES / "reconstruction-disabled.txt").read_bytes().replace(
            b"Think opportunistically", b"Think procedurally", 1
        )

        report = validate_prompt_pair(enabled, disabled)

        self.assertFalse(report["valid"])

    def test_schedule_is_precommitted_and_adjacent(self) -> None:
        schedule = frozen_pair_schedule()

        self.assertEqual(
            [(pair.pair_id, pair.requested_seed, pair.order, pair.included) for pair in schedule],
            [
                ("P", 1000, ("disabled", "enabled"), False),
                ("1", 0, ("enabled", "disabled"), True),
                ("2", 1, ("disabled", "enabled"), True),
                ("3", 2, ("enabled", "disabled"), True),
            ],
        )

    def test_pair_validator_allows_only_condition_specific_fields(self) -> None:
        endpoint = {
            "schema": "openrouter-endpoint-allowlist-v1",
            "model": "qwen/qwen3.6-27b",
            "provider_slug": "chutes/fp8",
            "provider_name": "Chutes",
            "variant": "fp8",
            "quantization": "fp8",
            "context_limit": 262144,
            "supported_parameters": [
                "include_reasoning", "max_tokens", "presence_penalty", "reasoning",
                "seed", "temperature", "tool_choice", "tools", "top_k", "top_p",
            ],
            "revision": "model-20260422",
            "revision_identifier": "model-20260422",
            "revision_identifier_source": "revision",
        }
        members = build_pair_commitments(
            generation="g1",
            pair_id="1",
            endpoint_metadata=endpoint,
            source_hashes=participating_source_hashes(),
        )
        enabled = next(member for member in members if member["condition"] == "enabled")
        disabled = next(member for member in members if member["condition"] == "disabled")

        validate_pair_commitments(enabled, disabled)
        changed = deepcopy(disabled)
        changed["model"] = "other/model"
        with self.assertRaisesRegex(ValueError, "model"):
            validate_pair_commitments(enabled, changed)
        incomplete = deepcopy(disabled)
        del incomplete["endpoint_metadata_sha256"]
        with self.assertRaisesRegex(ValueError, "fields"):
            validate_pair_commitments(enabled, incomplete)
        arbitrary_enabled = deepcopy(enabled)
        arbitrary_sources = deepcopy(disabled)
        arbitrary_enabled["source_hashes"] = {"fabricated.py": "0" * 64}
        arbitrary_sources["source_hashes"] = {"fabricated.py": "0" * 64}
        with self.assertRaisesRegex(ValueError, "source_hashes"):
            validate_pair_commitments(arbitrary_enabled, arbitrary_sources)

    def test_endpoint_metadata_is_reduced_to_the_frozen_allowlist(self) -> None:
        response = {
            "data": {
                "id": "qwen/qwen3.6-27b",
                "endpoints": [{
                    "name": "Chutes | Qwen",
                    "tag": "chutes/fp8",
                    "provider_name": "Chutes",
                    "variant": "fp8",
                    "quantization": "fp8",
                    "context_length": 131072,
                    "supported_parameters": ["seed", "top_k", "tools"],
                    "updated_at": "revision-17",
                    "pricing": {"prompt": "secretly-dynamic"},
                }],
            }
        }

        metadata = canonical_endpoint_metadata(response, model="qwen/qwen3.6-27b", tag="chutes/fp8")

        self.assertEqual(metadata["model"], "qwen/qwen3.6-27b")
        self.assertEqual(metadata["provider_slug"], "chutes/fp8")
        self.assertEqual(metadata["provider_name"], "Chutes")
        self.assertEqual(metadata["variant"], "fp8")
        self.assertEqual(metadata["quantization"], "fp8")
        self.assertEqual(metadata["context_limit"], 131072)
        self.assertEqual(metadata["supported_parameters"], ["seed", "tools", "top_k"])
        self.assertIsNone(metadata["revision"])
        self.assertEqual(metadata["endpoint_name"], "Chutes | Qwen")
        self.assertEqual(metadata["endpoint_updated_at"], "revision-17")
        self.assertNotIn("pricing", metadata)


class TwoLevelRunnerTests(unittest.TestCase):
    def test_level_budgets_reset_and_unused_level_one_actions_do_not_transfer(self) -> None:
        env = Environment([
            Frame(0, levels_completed=0),
            Frame(1, levels_completed=1),
            Frame(2, levels_completed=1),
            Frame(3, levels_completed=1),
        ])
        agent = Agent()

        result = run_episode(
            env,
            agent,
            max_actions=99,
            level_action_budgets=(1, 2),
            stop_after_levels=2,
        )

        self.assertEqual(result.stop_reason, "level_action_budget")
        self.assertEqual(result.actions, 3)
        self.assertEqual(result.level_action_counts, (1, 2))

    def test_level_three_frame_is_suppressed_before_agent_and_trace(self) -> None:
        env = Environment([
            Frame(10, levels_completed=0),
            Frame(11, levels_completed=1),
            Frame(99, levels_completed=2),
        ])
        agent = Agent()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            with JsonlTraceWriter(path) as trace:
                result = run_episode(
                    env,
                    agent,
                    max_actions=290,
                    level_action_budgets=(44, 246),
                    stop_after_levels=2,
                    trace=trace,
                )
            records = [json.loads(line) for line in path.read_text().splitlines()]
            trace_text = path.read_text()

        self.assertEqual(result.stop_reason, "level_limit")
        self.assertEqual(result.final.levels_completed, 2)
        self.assertEqual(result.completion_action_counts, (1, 2))
        self.assertTrue(result.transitions[-1].observation_suppressed)
        self.assertEqual(agent.seen_values, [10, 11])
        self.assertEqual(len(agent.transitions), 1)
        self.assertEqual(records[-2]["type"], "suppressed_transition")
        self.assertNotIn("observation", records[-2])
        self.assertNotIn("state", records[-2])
        self.assertEqual(result.final.state, "NOT_FINISHED")
        self.assertNotIn("99", trace_text)
        self.assertEqual(records[-1]["completion_action_counts"], [1, 2])

    def test_level_three_holdout_pixels_are_never_accessed(self) -> None:
        env = Environment([
            Frame(10, levels_completed=0),
            Frame(11, levels_completed=1),
            HoldoutSentinelFrame(),
        ])

        result = run_episode(
            env,
            Agent(),
            max_actions=290,
            level_action_budgets=(44, 246),
            stop_after_levels=2,
        )

        self.assertEqual(result.stop_reason, "level_limit")
        self.assertTrue(result.transitions[-1].observation_suppressed)

    def test_total_action_limit_is_an_invariant_not_an_independent_stop(self) -> None:
        frames = [Frame(0, levels_completed=0)]
        frames.extend(Frame(index, levels_completed=0) for index in range(1, 44))
        frames.append(Frame(44, levels_completed=1))
        frames.extend(Frame(index % 16, levels_completed=1) for index in range(45, 291))

        result = run_episode(
            Environment(frames),
            Agent(),
            max_actions=290,
            level_action_budgets=(44, 246),
            stop_after_levels=2,
        )

        self.assertEqual(result.actions, 290)
        self.assertEqual(result.level_action_counts, (44, 246))
        self.assertEqual(result.stop_reason, "level_action_budget")

    def test_completion_on_the_final_permitted_action_counts(self) -> None:
        env = Environment([
            Frame(0, levels_completed=0),
            Frame(1, levels_completed=0),
            Frame(2, levels_completed=1),
            Frame(3, levels_completed=1),
            Frame(4, levels_completed=2),
        ])

        result = run_episode(
            env,
            Agent(),
            max_actions=4,
            level_action_budgets=(2, 2),
            stop_after_levels=2,
        )

        self.assertEqual(result.stop_reason, "level_limit")
        self.assertEqual(result.level_action_counts, (2, 2))
        self.assertEqual(result.completion_action_counts, (2, 4))
        self.assertEqual(result.final.levels_completed, 2)


class AnalysisTests(unittest.TestCase):
    def test_analysis_consumer_recomputes_the_profile_from_raw_pair_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pair_rows = []
            for pair_id in ("1", "2", "3"):
                members = []
                annotations = {}
                for condition in ("enabled", "disabled"):
                    run_dir = root / f"pair-{pair_id}" / condition
                    (run_dir / "workspace").mkdir(parents=True)
                    (run_dir / "trace.jsonl").write_text("{}\n", encoding="utf-8")
                    (run_dir / "process.jsonl").write_text("{}\n", encoding="utf-8")
                    (run_dir / "scorecard.json").write_text("{}\n", encoding="utf-8")
                    evidence = evidence_hashes(run_dir)
                    manifest = {
                        "run_id": f"run-{pair_id}-{condition}",
                        "model": "test/model",
                        "endpoint": "https://provider.example/v1",
                        "model_profile": "test-provider",
                        "actions": 4,
                        "levels_completed": 1,
                        "completion_action_counts": [3 if condition == "enabled" else 4],
                        "stop_reason": "level_action_budget",
                        "usage": {"total_tokens": 40},
                        "provider_usage": [{
                            "prompt_tokens": 12,
                            **({"cost": None} if condition == "enabled" else {}),
                        }],
                        "model_rounds_per_action": [1, 1, 1, 1],
                        "evidence_hashes": evidence,
                        "experiment": {
                            "generation": "g-analysis",
                            "pair_id": pair_id,
                            "condition": condition,
                            "included": True,
                            "endpoint_metadata_sha256": "e" * 64,
                            "endpoint_metadata": {
                                "provider_slug": "test-provider",
                                "revision_identifier": "endpoint-1",
                            },
                        },
                    }
                    manifest_path = run_dir / "manifest.json"
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                    members.append({
                        "condition": condition,
                        "validity": "valid_contestant_result",
                        "run_dir": str(run_dir),
                        "manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
                    })
                    annotation = experiment_artifact_templates(
                        condition, evidence_hashes=evidence
                    )["annotation"]
                    status = (
                        "activated_but_not_demonstrated"
                        if condition == "enabled"
                        else "compliant"
                    )
                    annotation["annotators"] = [
                        {
                            "annotator_id": annotator_id,
                            "terminal_status": status,
                            "criterion_3_interface_damage": False,
                            "evidence_event_indices": [1],
                        }
                        for annotator_id in ("A", "B")
                    ]
                    annotation_path = run_dir / "annotation.completed.json"
                    annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
                    annotations[condition] = str(annotation_path)
                pair_result = build_completed_pair_result(
                    preflight={
                        "schema": "scientist-game-designer-player-pair-v1",
                        "generation": "g-analysis",
                        "pair_id": pair_id,
                        "included": True,
                    },
                    valid_attempt=1,
                    pair_sign="enabled",
                    attempts=[{"attempt": 1, "valid": True, "members": members}],
                )
                pair_path = root / f"pair-{pair_id}" / "pair-result.json"
                pair_path.write_text(json.dumps(pair_result), encoding="utf-8")
                pair_rows.append({
                    "pair_id": pair_id,
                    "pair_result": str(pair_path),
                    "annotations": annotations,
                    "shadow_result": None,
                })
            release_dir = root / "release"
            release_dir.mkdir()
            release_results_path = release_dir / "release-gate-results.json"
            release_results_path.write_text("{}\n", encoding="utf-8")
            final_freeze_path = root / "final-freeze.json"
            final_freeze_path.write_text(json.dumps({
                "artifacts": {"release_gate_results": str(release_results_path)},
                "endpoint_metadata_sha256": "e" * 64,
            }), encoding="utf-8")
            input_path = root / "analysis-input.json"
            input_path.write_text(json.dumps({
                "schema": ANALYSIS_INPUT_SCHEMA,
                "generation": "g-analysis",
                "final_freeze": str(final_freeze_path),
                "final_freeze_sha256": sha256(final_freeze_path.read_bytes()).hexdigest(),
                "pairs": pair_rows,
            }), encoding="utf-8")

            with (
                patch("analyze_generation.validate_final_freeze"),
                patch("analyze_generation.validate_included_pair_result"),
            ):
                result = analyze(input_path)

        self.assertEqual(result["component_profile"], {
            "activation": True,
            "clean_contrast": True,
            "local_content_dependence": False,
            "preliminary_gameplay_benefit": True,
            "mechanism_consistent_preliminary_gameplay_benefit": False,
        })
        enabled_evidence = result["pairs"][0]["run_evidence"]["enabled"]
        disabled_evidence = result["pairs"][0]["run_evidence"]["disabled"]
        self.assertEqual(enabled_evidence["model"], "test/model")
        self.assertEqual(enabled_evidence["endpoint_metadata"]["revision_identifier"], "endpoint-1")
        self.assertIsNone(enabled_evidence["provider_usage"][0]["cost"])
        self.assertNotIn("cost", disabled_evidence["provider_usage"][0])

    def test_analysis_rejects_input_without_a_final_freeze_commitment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "analysis-input.json"
            input_path.write_text(json.dumps({
                "schema": ANALYSIS_INPUT_SCHEMA,
                "generation": "g-analysis",
                "pairs": [{"pair_id": pair_id} for pair_id in ("1", "2", "3")],
            }), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "final freeze"):
                analyze(input_path)

    def test_pair_sign_is_lexicographic_and_ignores_later_incomplete_actions(self) -> None:
        enabled = RunOutcome(levels_completed=1, level_1_completion_action=20, level_2_completion_action=None)
        disabled = RunOutcome(levels_completed=1, level_1_completion_action=21, level_2_completion_action=None)

        self.assertEqual(assess_pair(enabled, disabled), "enabled")
        tie = RunOutcome(levels_completed=0, level_1_completion_action=None, level_2_completion_action=None)
        self.assertEqual(assess_pair(tie, tie), "tie")

    def test_interface_damage_implements_the_automatic_criteria(self) -> None:
        zero_action = assess_interface_damage(
            PairTelemetry(actions=0, stop_reason="model_budget_exhausted", reported_tokens=None, model_rounds=()),
            PairTelemetry(actions=4, stop_reason="level_action_budget", reported_tokens=20, model_rounds=(1, 1, 1, 1)),
            annotated_damage=False,
        )
        ratio = assess_interface_damage(
            PairTelemetry(actions=2, stop_reason="model_budget_exhausted", reported_tokens=None, model_rounds=(4, 4)),
            PairTelemetry(actions=6, stop_reason="level_action_budget", reported_tokens=None, model_rounds=(2, 2, 2, 2, 2, 2)),
            annotated_damage=False,
        )

        self.assertTrue(zero_action.severe_interface_damage)
        self.assertTrue(ratio.severe_interface_damage)
        unrepaired_after_action = assess_interface_damage(
            PairTelemetry(
                actions=1,
                stop_reason="unrepaired_invalid_decision",
                reported_tokens=100,
                model_rounds=(5,),
            ),
            PairTelemetry(
                actions=4,
                stop_reason="level_action_budget",
                reported_tokens=100,
                model_rounds=(1, 1, 1, 1),
            ),
            annotated_damage=False,
        )
        self.assertFalse(unrepaired_after_action.criterion_2)
        self.assertFalse(unrepaired_after_action.severe_interface_damage)

    def test_component_profile_is_a_four_way_conjunction(self) -> None:
        profile = assess_component_profile(
            enabled_statuses=("demonstrated", "activated_but_not_demonstrated", "not_activated"),
            clean_disabled_runs=(True, True, False),
            intervention_passes=(True, False, False),
            pair_signs=("enabled", "disabled", "enabled"),
        )

        self.assertEqual(
            profile,
            {
                "activation": True,
                "clean_contrast": True,
                "local_content_dependence": True,
                "preliminary_gameplay_benefit": True,
                "mechanism_consistent_preliminary_gameplay_benefit": True,
            },
        )


class PairOrchestrationTests(unittest.TestCase):
    def test_final_freeze_cli_launches_from_the_repository_root(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(AGENT_ROOT / "freeze_generation.py"), "--help"],
            cwd=AGENT_ROOT.parents[2],
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--pilot-decision", completed.stdout)

    def test_rejected_pilot_decision_cannot_authorize_a_final_freeze(self) -> None:
        pair_dir = (
            AGENT_ROOT
            / "artifacts"
            / "generations"
            / "post-final-audit-regate"
            / "pair-P-20260719T195944Z"
        )

        with self.assertRaisesRegex(ValueError, "accepted pilot decision"):
            validate_pilot_decision(
                pair_dir / "pilot-inspection.rejected.json",
                pilot_pair_dir=pair_dir,
                generation="post-final-audit-regate",
            )

    def test_legacy_underbounded_pilot_approval_is_not_executable(self) -> None:
        approval = (
            AGENT_ROOT
            / "artifacts"
            / "approvals"
            / "pilot-P-post-final-audit-regate-approved-20260719T195917Z.json"
        )
        release_dir = AGENT_ROOT / "artifacts" / "release-gates" / "20260719T193032Z"

        with self.assertRaisesRegex(ValueError, "v2 pilot approval"):
            validate_pilot_approval(
                approval,
                generation="post-final-audit-regate",
                release_gate_dir=release_dir,
                source_hashes=participating_source_hashes(),
                endpoint_metadata_hash="8fc809bd32e8c3845916d740c506f0d5aef3f3bed7bc28379d9f07b0c8ee5a4c",
            )

    def test_pair_authority_is_consumed_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            generation_root = Path(directory) / "g1"
            pair_root = generation_root / "pair-P-test"
            authority = Path(directory) / "approval.json"
            authority.write_text('{"status":"approved"}\n', encoding="utf-8")

            record_path = consume_pair_authority(
                generation_root=generation_root,
                pair_root=pair_root,
                generation="g1",
                pair_id="P",
                authority_path=authority,
                authority_kind="pilot_approval",
                maximum_pair_attempts=2,
            )

            self.assertTrue(record_path.is_file())
            with self.assertRaisesRegex(ValueError, "already consumed"):
                consume_pair_authority(
                    generation_root=generation_root,
                    pair_root=pair_root,
                    generation="g1",
                    pair_id="P",
                    authority_path=authority,
                    authority_kind="pilot_approval",
                    maximum_pair_attempts=2,
                )

    def test_pair_authority_concurrent_consumers_have_one_winner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            generation_root = Path(directory) / "g1"
            pair_root = generation_root / "pair-P-test"
            authority = Path(directory) / "approval.json"
            authority.write_text('{"status":"approved"}\n', encoding="utf-8")
            barrier = Barrier(2)

            def consume() -> Path | ValueError:
                barrier.wait()
                try:
                    return consume_pair_authority(
                        generation_root=generation_root,
                        pair_root=pair_root,
                        generation="g1",
                        pair_id="P",
                        authority_path=authority,
                        authority_kind="pilot_approval",
                        maximum_pair_attempts=2,
                    )
                except ValueError as error:
                    return error

            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = (pool.submit(consume), pool.submit(consume))
                results = [future.result() for future in futures]

            self.assertEqual(sum(isinstance(result, Path) for result in results), 1)
            errors = [result for result in results if isinstance(result, ValueError)]
            self.assertEqual(len(errors), 1)
            self.assertIn("already consumed", str(errors[0]))

    def test_v2_pilot_approval_binds_exact_request_and_exposure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation = "g-new-pilot"
            sources = participating_source_hashes()
            endpoint_metadata = _valid_endpoint_metadata()
            endpoint_hash = endpoint_metadata_sha256(endpoint_metadata)
            release_dir = root / "release"
            release_dir.mkdir()
            artifacts = {
                "endpoint_metadata": "endpoint-metadata.json",
                "sandbox_results": "sandbox-results.json",
                "synthetic_test_results": "synthetic-test-results.json",
                "intervention_test_vectors": "intervention-test-vectors.json",
            }
            for filename in artifacts.values():
                (release_dir / filename).write_text('{"passed":true}\n', encoding="utf-8")
            (release_dir / "endpoint-metadata.json").write_text(
                json.dumps(endpoint_metadata), encoding="utf-8"
            )
            release = {
                "schema": "scientist-game-designer-player-release-gate-v1",
                "passed": True,
                "pilot_authorized_by_gate": True,
                "source_hashes": sources,
                "compatibility": {"endpoint_metadata_sha256": endpoint_hash},
                "artifacts": artifacts,
            }
            release_path = release_dir / "release-gate-results.json"
            release_path.write_text(json.dumps(release), encoding="utf-8")
            candidate = {
                "schema": "scientist-game-designer-player-release-candidate-v1",
                "release_gate_passed": True,
                "release_gate_results_sha256": sha256(release_path.read_bytes()).hexdigest(),
                "participating_source_hashes": sources,
                "generated_artifact_hashes": {
                    path.name: sha256(path.read_bytes()).hexdigest()
                    for path in release_dir.glob("*.json")
                },
            }
            candidate_path = release_dir / "release-candidate-manifest.json"
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

            config = load_frozen_runtime()
            schedule = next(row for row in frozen_pair_schedule() if row.pair_id == "P")
            exposure = {
                "maximum_pair_attempts": 2,
                "members_per_attempt": 2,
                "environment_actions": 4 * config.budgets.total_actions_invariant,
                "model_calls": 4 * config.runtime.max_model_calls,
            }
            request = {
                "schema": "scientist-game-designer-player-execution-approval-request-v2",
                "status": "awaiting_explicit_human_approval",
                "requested_action": "execute_excluded_pilot_pair_P",
                "generation": generation,
                "pair_id": "P",
                "included_in_confirmatory_analysis": False,
                "execution_order": list(schedule.order),
                "environment_seed": schedule.environment_seed,
                "requested_model_seed": schedule.requested_seed,
                "endpoint": config.endpoint,
                "model": config.model,
                "provider": config.provider.only[0],
                "release_gate_dir": str(release_dir),
                "release_gate_results_sha256": sha256(release_path.read_bytes()).hexdigest(),
                "release_candidate_manifest_sha256": sha256(candidate_path.read_bytes()).hexdigest(),
                "endpoint_metadata_sha256": endpoint_hash,
                "source_hashes": sources,
                "maximum_exposure": exposure,
            }
            request_path = root / "pilot-request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            approval = {
                "schema": "scientist-game-designer-player-execution-approval-v2",
                "status": "approved",
                "authority": "test-human",
                "approval_record": "explicit test approval",
                "approval_request": request_path.name,
                "approval_request_sha256": sha256(request_path.read_bytes()).hexdigest(),
                "generation": generation,
                "pair_id": "P",
                "scope": {
                    "execute_excluded_pilot": True,
                    "accept_pilot": False,
                    "create_final_freeze": False,
                    "execute_included_pairs": False,
                    "execute_shadow_interventions": False,
                },
                "maximum_exposure": exposure,
                "release_gate_results_sha256": sha256(release_path.read_bytes()).hexdigest(),
                "release_candidate_manifest_sha256": sha256(candidate_path.read_bytes()).hexdigest(),
                "endpoint_metadata_sha256": endpoint_hash,
                "source_hashes": sources,
            }
            approval_path = root / "pilot-approved.json"
            approval_path.write_text(json.dumps(approval), encoding="utf-8")

            validated = validate_pilot_approval(
                approval_path,
                generation=generation,
                release_gate_dir=release_dir,
                source_hashes=sources,
                endpoint_metadata_hash=endpoint_hash,
            )
            self.assertEqual(validated["maximum_pair_attempts"], 2)

            request["maximum_exposure"]["environment_actions"] += 1
            request_path.write_text(json.dumps(request), encoding="utf-8")
            approval["maximum_exposure"]["environment_actions"] += 1
            approval["approval_request_sha256"] = sha256(request_path.read_bytes()).hexdigest()
            approval_path.write_text(json.dumps(approval), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact execution"):
                validate_pilot_approval(
                    approval_path,
                    generation=generation,
                    release_gate_dir=release_dir,
                    source_hashes=sources,
                    endpoint_metadata_hash=endpoint_hash,
                )

    def test_final_freeze_consumer_recomputes_artifact_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation = "g-authority"
            sources = participating_source_hashes()
            config = load_frozen_runtime()
            endpoint_metadata = _valid_endpoint_metadata()
            endpoint_hash = endpoint_metadata_sha256(endpoint_metadata)
            release_dir = root / "release"
            release_dir.mkdir()
            supporting = {
                "endpoint_metadata": "endpoint-metadata.json",
                "sandbox_results": "sandbox-results.json",
                "synthetic_test_results": "synthetic-test-results.json",
                "intervention_test_vectors": "intervention-test-vectors.json",
            }
            for filename in supporting.values():
                (release_dir / filename).write_text('{"passed":true}\n', encoding="utf-8")
            (release_dir / "endpoint-metadata.json").write_text(
                json.dumps(endpoint_metadata), encoding="utf-8"
            )
            release = {
                "schema": "scientist-game-designer-player-release-gate-v1",
                "passed": True,
                "pilot_authorized_by_gate": True,
                "source_hashes": sources,
                "compatibility": {"endpoint_metadata_sha256": endpoint_hash},
                "artifacts": supporting,
            }
            release_path = release_dir / "release-gate-results.json"
            release_path.write_text(json.dumps(release), encoding="utf-8")
            candidate = {
                "schema": "scientist-game-designer-player-release-candidate-v1",
                "release_gate_passed": True,
                "release_gate_results_sha256": sha256(release_path.read_bytes()).hexdigest(),
                "participating_source_hashes": sources,
                "generated_artifact_hashes": {
                    path.name: sha256(path.read_bytes()).hexdigest()
                    for path in release_dir.glob("*.json")
                },
            }
            candidate_path = release_dir / "release-candidate-manifest.json"
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

            pair_dir = root / "pair-P"
            schedule = next(row for row in frozen_pair_schedule() if row.pair_id == "P")
            exposure = {
                "maximum_pair_attempts": 2,
                "members_per_attempt": 2,
                "environment_actions": 4 * config.budgets.total_actions_invariant,
                "model_calls": 4 * config.runtime.max_model_calls,
            }
            approval_request = {
                "schema": "scientist-game-designer-player-execution-approval-request-v2",
                "status": "awaiting_explicit_human_approval",
                "requested_action": "execute_excluded_pilot_pair_P",
                "generation": generation,
                "pair_id": "P",
                "included_in_confirmatory_analysis": False,
                "execution_order": list(schedule.order),
                "environment_seed": schedule.environment_seed,
                "requested_model_seed": schedule.requested_seed,
                "endpoint": config.endpoint,
                "model": config.model,
                "provider": config.provider.only[0],
                "release_gate_dir": str(release_dir),
                "release_gate_results_sha256": sha256(release_path.read_bytes()).hexdigest(),
                "release_candidate_manifest_sha256": sha256(candidate_path.read_bytes()).hexdigest(),
                "endpoint_metadata_sha256": endpoint_hash,
                "source_hashes": sources,
                "maximum_exposure": exposure,
            }
            approval_request_path = root / "pilot-request.json"
            approval_request_path.write_text(json.dumps(approval_request), encoding="utf-8")
            approval = {
                "schema": "scientist-game-designer-player-execution-approval-v2",
                "status": "approved",
                "authority": "test-human",
                "approval_record": "explicit pilot approval",
                "approval_request": approval_request_path.name,
                "approval_request_sha256": sha256(approval_request_path.read_bytes()).hexdigest(),
                "generation": generation,
                "pair_id": "P",
                "scope": {
                    "execute_excluded_pilot": True,
                    "accept_pilot": False,
                    "create_final_freeze": False,
                    "execute_included_pairs": False,
                    "execute_shadow_interventions": False,
                },
                "maximum_exposure": exposure,
                "release_gate_results_sha256": sha256(release_path.read_bytes()).hexdigest(),
                "release_candidate_manifest_sha256": sha256(candidate_path.read_bytes()).hexdigest(),
                "endpoint_metadata_sha256": endpoint_hash,
                "source_hashes": sources,
            }
            approval_path = root / "pilot-approved.json"
            approval_path.write_text(json.dumps(approval), encoding="utf-8")
            authority_consumption_path = consume_pair_authority(
                generation_root=root,
                pair_root=pair_dir,
                generation=generation,
                pair_id="P",
                authority_path=approval_path,
                authority_kind="pilot_approval",
                maximum_pair_attempts=2,
            )
            commitments = build_pair_commitments(
                generation=generation,
                pair_id="P",
                endpoint_metadata=endpoint_metadata,
                source_hashes=sources,
            )
            commitments_by_condition = {
                commitment["condition"]: commitment for commitment in commitments
            }
            commitments_path = pair_dir / "pair-commitments.json"
            commitments_path.parent.mkdir(parents=True, exist_ok=True)
            commitments_path.write_text(
                json.dumps({
                    "schema": "scientist-game-designer-player-pair-v1",
                    "commitments": list(commitments),
                }),
                encoding="utf-8",
            )
            members = []
            for condition in ("disabled", "enabled"):
                commitment = commitments_by_condition[condition]
                run_dir = pair_dir / "attempt-1" / condition
                run_dir.mkdir(parents=True)
                (run_dir / "trace.jsonl").write_text("{}\n", encoding="utf-8")
                (run_dir / "process.jsonl").write_text("{}\n", encoding="utf-8")
                (run_dir / "scorecard.json").write_text("{}\n", encoding="utf-8")
                evidence = evidence_hashes(run_dir)
                member_consumption_path = (
                    root
                    / ".authority"
                    / "members"
                    / f"pair-P-attempt-1-{condition}.json"
                )
                member_consumption_path.parent.mkdir(parents=True, exist_ok=True)
                member_consumption_path.write_text(
                    json.dumps({
                        "schema": "scientist-game-designer-player-member-authority-consumption-v1",
                        "generation": generation,
                        "pair_id": "P",
                        "attempt": 1,
                        "condition": condition,
                        "run_dir": str(run_dir),
                        "pair_authority_consumption": str(authority_consumption_path),
                        "pair_authority_consumption_sha256": sha256(
                            authority_consumption_path.read_bytes()
                        ).hexdigest(),
                    }),
                    encoding="utf-8",
                )
                manifest = {
                    "status": "complete",
                    "stop_reason": "level_action_budget",
                    "model": commitment["model"],
                    "endpoint": commitment["endpoint"],
                    "model_profile": commitment["model_profile"],
                    "env_seed": commitment["environment_seed"],
                    "model_seed": commitment["requested_seed"],
                    "color_mapping": commitment["color_mapping"],
                    "policy_prompt_sha256": commitment["prompt_sha256"],
                    "level_action_counts": [],
                    "completion_action_counts": [],
                    "reset_count": 0,
                    "game_over_count": 0,
                    "model_calls": 0,
                    "python_calls": 0,
                    "model_rounds_per_action": [],
                    "decision_repairs": 0,
                    "host_score_components": {},
                    "usage": {},
                    "provider_usage": {},
                    "evidence_hashes": evidence,
                    "experiment": {
                        "generation": generation,
                        "pair_id": "P",
                        "condition": condition,
                        "included": False,
                        "pair_order": commitment["pair_order"],
                        "runtime_config_sha256": commitment["runtime_sha256"],
                        "condition_block_sha256": commitment["condition_block_sha256"],
                        "prompt_sha256": commitment["prompt_sha256"],
                        "source_hashes": sources,
                        "endpoint_metadata_sha256": endpoint_hash,
                        "pair_authority_consumption": str(authority_consumption_path),
                        "pair_authority_consumption_sha256": sha256(
                            authority_consumption_path.read_bytes()
                        ).hexdigest(),
                        "member_authority_consumption": str(member_consumption_path),
                        "member_authority_consumption_sha256": sha256(
                            member_consumption_path.read_bytes()
                        ).hexdigest(),
                    },
                }
                manifest_path = run_dir / "manifest.json"
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                members.append({
                    "condition": condition,
                    "validity": "valid_contestant_result",
                    "run_dir": str(run_dir),
                    "manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
                })
            pair = {
                "schema": "scientist-game-designer-player-pair-v1",
                "generation": generation,
                "pair_id": "P",
                "included": False,
                "status": "pair_complete",
                "gameplay_executed": True,
                "valid_attempt": 1,
                "source_hashes": sources,
                "endpoint_metadata_sha256": endpoint_hash,
                "release_gate_results_sha256": sha256(release_path.read_bytes()).hexdigest(),
                "pair_commitments_sha256": sha256(commitments_path.read_bytes()).hexdigest(),
                "authority_consumption": str(authority_consumption_path),
                "authority_consumption_sha256": sha256(
                    authority_consumption_path.read_bytes()
                ).hexdigest(),
                "attempts": [{"attempt": 1, "valid": True, "members": members}],
            }
            pair_path = pair_dir / "pair-result.json"
            pair_path.write_text(json.dumps(pair), encoding="utf-8")
            decision = {
                "schema": "scientist-game-designer-player-pilot-inspection-v2",
                "generation": generation,
                "pair_id": "P",
                "excluded_pilot": True,
                "pilot_pair_dir": str(pair_dir),
                "pilot_pair_result_sha256": sha256(pair_path.read_bytes()).hexdigest(),
                "release_gate_results_sha256": sha256(release_path.read_bytes()).hexdigest(),
                "endpoint_metadata_sha256": endpoint_hash,
                "source_hashes": sources,
                "decision": "accepted",
                "accepted_for_final_freeze": True,
                "blocking_findings": [],
                "authority": "human acceptance authority",
                "approval_record": "accepted exact pilot and bounded included execution",
                "scope": {
                    "accept_pilot": True,
                    "create_final_freeze": True,
                    "execute_included_pairs": True,
                    "execute_shadow_interventions": False,
                },
                "maximum_exposure": {
                    "included_pairs": 3,
                    "members_per_pair": 2,
                    "maximum_pair_attempts_per_pair": 2,
                    "environment_actions": 3480,
                    "model_calls": 24360,
                },
            }
            decision_path = pair_dir / "pilot-inspection.accepted.json"
            decision_path.write_text(json.dumps(decision), encoding="utf-8")
            freeze = build_final_freeze(
                generation=generation,
                pilot_pair_dir=pair_dir,
                pilot_decision_path=decision_path,
                release_gate_dir=release_dir,
                source_hashes=sources,
            )
            freeze_path = root / "final-freeze.json"
            freeze_path.write_text(json.dumps(freeze), encoding="utf-8")

            validate_final_freeze(
                freeze_path,
                generation=generation,
                release_gate_dir=release_dir,
                source_hashes=sources,
                endpoint_metadata_hash=endpoint_hash,
            )

            included_pair_dir = root / "pair-1"
            included_consumption_path = consume_pair_authority(
                generation_root=root,
                pair_root=included_pair_dir,
                generation=generation,
                pair_id="1",
                authority_path=freeze_path,
                authority_kind="final_freeze",
                maximum_pair_attempts=2,
            )
            included_commitments = build_pair_commitments(
                generation=generation,
                pair_id="1",
                endpoint_metadata=endpoint_metadata,
                source_hashes=sources,
            )
            included_by_condition = {
                commitment["condition"]: commitment
                for commitment in included_commitments
            }
            included_commitments_path = included_pair_dir / "pair-commitments.json"
            included_commitments_path.parent.mkdir(parents=True, exist_ok=True)
            included_commitments_path.write_text(
                json.dumps({
                    "schema": "scientist-game-designer-player-pair-v1",
                    "commitments": list(included_commitments),
                }),
                encoding="utf-8",
            )
            included_members = []
            for condition in ("enabled", "disabled"):
                commitment = included_by_condition[condition]
                run_dir = included_pair_dir / "attempt-1" / condition
                run_dir.mkdir(parents=True)
                (run_dir / "trace.jsonl").write_text("{}\n", encoding="utf-8")
                (run_dir / "process.jsonl").write_text("{}\n", encoding="utf-8")
                (run_dir / "scorecard.json").write_text("{}\n", encoding="utf-8")
                evidence = evidence_hashes(run_dir)
                member_consumption_path = (
                    root
                    / ".authority"
                    / "members"
                    / f"pair-1-attempt-1-{condition}.json"
                )
                member_consumption_path.write_text(
                    json.dumps({
                        "schema": "scientist-game-designer-player-member-authority-consumption-v1",
                        "generation": generation,
                        "pair_id": "1",
                        "attempt": 1,
                        "condition": condition,
                        "run_dir": str(run_dir),
                        "pair_authority_consumption": str(included_consumption_path),
                        "pair_authority_consumption_sha256": sha256(
                            included_consumption_path.read_bytes()
                        ).hexdigest(),
                    }),
                    encoding="utf-8",
                )
                levels_completed = 1 if condition == "enabled" else 0
                completion_counts = [10] if condition == "enabled" else []
                manifest = {
                    "status": "complete",
                    "stop_reason": "level_action_budget",
                    "model": commitment["model"],
                    "endpoint": commitment["endpoint"],
                    "model_profile": commitment["model_profile"],
                    "env_seed": commitment["environment_seed"],
                    "model_seed": commitment["requested_seed"],
                    "color_mapping": commitment["color_mapping"],
                    "policy_prompt_sha256": commitment["prompt_sha256"],
                    "levels_completed": levels_completed,
                    "level_action_counts": completion_counts,
                    "completion_action_counts": completion_counts,
                    "reset_count": 0,
                    "game_over_count": 0,
                    "model_calls": 0,
                    "python_calls": 0,
                    "model_rounds_per_action": [],
                    "decision_repairs": 0,
                    "host_score_components": {},
                    "usage": {},
                    "provider_usage": {},
                    "evidence_hashes": evidence,
                    "experiment": {
                        "generation": generation,
                        "pair_id": "1",
                        "condition": condition,
                        "included": True,
                        "pair_order": commitment["pair_order"],
                        "runtime_config_sha256": commitment["runtime_sha256"],
                        "condition_block_sha256": commitment["condition_block_sha256"],
                        "prompt_sha256": commitment["prompt_sha256"],
                        "source_hashes": sources,
                        "endpoint_metadata_sha256": endpoint_hash,
                        "pair_authority_consumption": str(included_consumption_path),
                        "pair_authority_consumption_sha256": sha256(
                            included_consumption_path.read_bytes()
                        ).hexdigest(),
                        "member_authority_consumption": str(member_consumption_path),
                        "member_authority_consumption_sha256": sha256(
                            member_consumption_path.read_bytes()
                        ).hexdigest(),
                    },
                }
                manifest_path = run_dir / "manifest.json"
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                included_members.append({
                    "condition": condition,
                    "validity": "valid_contestant_result",
                    "run_dir": str(run_dir),
                    "manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
                })
            included_pair = build_completed_pair_result(
                preflight={
                    "schema": "scientist-game-designer-player-pair-v1",
                    "generation": generation,
                    "pair_id": "1",
                    "included": True,
                    "source_hashes": sources,
                    "endpoint_metadata_sha256": endpoint_hash,
                    "release_gate_results_sha256": sha256(
                        release_path.read_bytes()
                    ).hexdigest(),
                    "pair_commitments_sha256": sha256(
                        included_commitments_path.read_bytes()
                    ).hexdigest(),
                    "authority_consumption": str(included_consumption_path),
                    "authority_consumption_sha256": sha256(
                        included_consumption_path.read_bytes()
                    ).hexdigest(),
                },
                valid_attempt=1,
                pair_sign="enabled",
                attempts=[{
                    "attempt": 1,
                    "valid": True,
                    "members": included_members,
                }],
            )
            included_pair_path = included_pair_dir / "pair-result.json"
            included_pair_path.write_text(json.dumps(included_pair), encoding="utf-8")
            validate_included_pair_result(
                included_pair_path,
                final_freeze_path=freeze_path,
                generation=generation,
                release_gate_dir=release_dir,
                source_hashes=sources,
                endpoint_metadata_hash=endpoint_hash,
            )
            included_pair["pair_sign"] = "disabled"
            included_pair_path.write_text(json.dumps(included_pair), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pair sign"):
                validate_included_pair_result(
                    included_pair_path,
                    final_freeze_path=freeze_path,
                    generation=generation,
                    release_gate_dir=release_dir,
                    source_hashes=sources,
                    endpoint_metadata_hash=endpoint_hash,
                )
            forged = deepcopy(freeze)
            forged["artifact_hashes"]["pilot_pair_result"] = "0" * 64
            forged["artifact_hashes"]["release_gate_results"] = "0" * 64
            freeze_path.write_text(json.dumps(forged), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_final_freeze(
                    freeze_path,
                    generation=generation,
                    release_gate_dir=release_dir,
                    source_hashes=sources,
                    endpoint_metadata_hash=endpoint_hash,
                )

            enabled_manifest_path = Path(members[1]["run_dir"]) / "manifest.json"
            enabled_manifest = json.loads(enabled_manifest_path.read_text(encoding="utf-8"))
            enabled_manifest["model_seed"] = 999999
            enabled_manifest_path.write_text(json.dumps(enabled_manifest), encoding="utf-8")
            members[1]["manifest_sha256"] = sha256(
                enabled_manifest_path.read_bytes()
            ).hexdigest()
            pair_path.write_text(json.dumps(pair), encoding="utf-8")
            decision["pilot_pair_result_sha256"] = sha256(
                pair_path.read_bytes()
            ).hexdigest()
            decision_path.write_text(json.dumps(decision), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "raw evidence does not validate"):
                build_final_freeze(
                    generation=generation,
                    pilot_pair_dir=pair_dir,
                    pilot_decision_path=decision_path,
                    release_gate_dir=release_dir,
                    source_hashes=sources,
                )

    def test_release_gate_bundle_requires_every_committed_supporting_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = {
                "endpoint_metadata": "endpoint-metadata.json",
                "sandbox_results": "sandbox-results.json",
                "synthetic_test_results": "synthetic-test-results.json",
                "intervention_test_vectors": "intervention-test-vectors.json",
            }
            for filename in artifacts.values():
                (root / filename).write_text('{"passed":true}\n', encoding="utf-8")
            gate = {"artifacts": artifacts}
            candidate = {
                "generated_artifact_hashes": {
                    filename: sha256((root / filename).read_bytes()).hexdigest()
                    for filename in artifacts.values()
                }
            }

            validate_release_gate_bundle(root, gate, candidate)
            (root / "sandbox-results.json").write_text('{"passed":false}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "supporting artifact"):
                validate_release_gate_bundle(root, gate, candidate)


    def test_preflight_materializes_two_valid_adjacent_commitments(self) -> None:
        endpoint = {
            "schema": "openrouter-endpoint-allowlist-v1",
            "model": "qwen/qwen3.6-27b",
            "provider_slug": "chutes/fp8",
            "provider_name": "Chutes",
            "variant": "fp8",
            "quantization": "fp8",
            "context_limit": 262144,
            "supported_parameters": [
                "max_tokens", "presence_penalty", "reasoning", "seed",
                "temperature", "tool_choice", "tools", "top_k", "top_p",
            ],
            "revision": "Chutes | qwen/qwen3.6-27b-20260422",
            "revision_identifier": "Chutes | qwen/qwen3.6-27b-20260422",
            "revision_identifier_source": "revision",
        }

        commitments = build_pair_commitments(
            generation="generation-test",
            pair_id="2",
            endpoint_metadata=endpoint,
            source_hashes=participating_source_hashes(),
        )

        self.assertEqual(
            [row["condition"] for row in commitments], ["disabled", "enabled"]
        )
        self.assertEqual({row["requested_seed"] for row in commitments}, {1})
        self.assertNotEqual(
            commitments[0]["prompt_sha256"], commitments[1]["prompt_sha256"]
        )

    def test_only_frozen_infrastructure_failures_are_replaceable(self) -> None:
        self.assertEqual(
            gameplay_attempt_validity({
                "status": "complete",
                "stop_reason": "model_budget_exhausted",
            }),
            "valid_contestant_result",
        )
        self.assertEqual(
            gameplay_attempt_validity({"status": "failed", "stop_reason": "error"}),
            "infrastructure_invalid",
        )

    def test_member_manifest_must_match_its_source_and_condition_commitment(self) -> None:
        endpoint = {
            "schema": "openrouter-endpoint-allowlist-v1",
            "model": "qwen/qwen3.6-27b",
            "provider_slug": "chutes/fp8",
            "provider_name": "Chutes",
            "variant": "fp8",
            "quantization": "fp8",
            "context_limit": 131072,
            "supported_parameters": [
                "max_tokens", "presence_penalty", "reasoning", "seed",
                "temperature", "tool_choice", "tools", "top_k", "top_p",
            ],
            "revision": "Chutes | qwen/qwen3.6-27b-20260422",
            "revision_identifier": "Chutes | qwen/qwen3.6-27b-20260422",
            "revision_identifier_source": "revision",
        }
        commitment = build_pair_commitments(
            generation="generation-test",
            pair_id="1",
            endpoint_metadata=endpoint,
            source_hashes=participating_source_hashes(),
        )[0]
        manifest = {
            "status": "complete",
            "stop_reason": "level_limit",
            "model": commitment["model"],
            "endpoint": commitment["endpoint"],
            "model_profile": commitment["model_profile"],
            "env_seed": commitment["environment_seed"],
            "model_seed": commitment["requested_seed"],
            "color_mapping": commitment["color_mapping"],
            "policy_prompt_sha256": commitment["prompt_sha256"],
            "experiment": {
                "generation": commitment["generation"],
                "pair_id": commitment["pair_id"],
                "included": commitment["included"],
                "condition": commitment["condition"],
                "pair_order": commitment["pair_order"],
                "runtime_config_sha256": commitment["runtime_sha256"],
                "condition_block_sha256": commitment["condition_block_sha256"],
                "prompt_sha256": commitment["prompt_sha256"],
                "source_hashes": commitment["source_hashes"],
                "endpoint_metadata_sha256": commitment["endpoint_metadata_sha256"],
            },
            "level_action_counts": [1, 1],
            "completion_action_counts": [1, 2],
            "reset_count": 0,
            "game_over_count": 0,
            "model_calls": 2,
            "python_calls": 0,
            "model_rounds_per_action": [1, 1],
            "decision_repairs": 0,
            "host_score_components": {},
            "usage": {"total_tokens": 20},
            "provider_usage": [{"total_tokens": 20}],
            "evidence_hashes": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            evidence_payloads = {
                "trace.jsonl": b'{"type":"end"}\n',
                "process.jsonl": b'{"type":"run_finished"}\n',
                "scorecard.json": b'{}\n',
            }
            for relative, payload in evidence_payloads.items():
                (run_dir / relative).write_bytes(payload)
            manifest["evidence_hashes"] = {
                relative: sha256(payload).hexdigest()
                for relative, payload in evidence_payloads.items()
            }

            self.assertEqual(
                gameplay_attempt_validity(manifest, commitment, run_dir=run_dir),
                "valid_contestant_result",
            )
            (run_dir / "process.jsonl").write_text("corrupt", encoding="utf-8")
            self.assertEqual(
                gameplay_attempt_validity(manifest, commitment, run_dir=run_dir),
                "infrastructure_invalid",
            )

    def test_workspace_evidence_hashes_symlinks_without_following_targets(self) -> None:
        from llm_repl_agent.evidence import evidence_hashes

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            for name in ("trace.jsonl", "process.jsonl", "scorecard.json"):
                (run_dir / name).write_text("{}\n", encoding="utf-8")
            workspace = run_dir / "workspace"
            workspace.mkdir()
            (workspace / "outside").symlink_to("/etc/passwd")

            hashes = evidence_hashes(run_dir)

        expected = sha256(
            b'{"kind":"symlink","target":"/etc/passwd"}'
        ).hexdigest()
        self.assertEqual(hashes["workspace/outside"], expected)


class InterventionTests(unittest.TestCase):
    def test_signatures_are_mechanical(self) -> None:
        decision_state = {
            "available_actions": [1, 2, 3, 4, 5, 6, 7],
            "frame_width": 8,
            "frame_height": 6,
        }
        self.assertEqual(
            response_signature(
                {"content": '{"action":2,"data":{"x":1,"y":3},"reason":"go"}'},
                decision_state=decision_state,
            ),
            'ACT:{"action":2,"data":{"x":1,"y":3}}',
        )
        self.assertEqual(
            response_signature({
                "content": None,
                "tool_calls": [{"function": {"name": "python", "arguments": '{"code":"x=1"}'}}],
            }),
            "CONTINUE",
        )
        self.assertEqual(response_signature({"content": "maybe"}), "UNIDENTIFIABLE")
        self.assertEqual(
            response_signature(
                {"content": '{"action":6,"data":{"x":-1,"y":3}}'},
                decision_state=decision_state,
            ),
            "UNIDENTIFIABLE",
        )
        self.assertEqual(
            response_signature(
                {"content": '{"action":7,"data":null}'},
                decision_state={**decision_state, "available_actions": [1, 2]},
            ),
            "UNIDENTIFIABLE",
        )

    def test_schedule_has_five_matched_seeds_and_randomized_branch_order(self) -> None:
        schedule = build_shadow_schedule(candidate_id="run-1:event-7", generation="g1")

        self.assertEqual(len(schedule), 10)
        for replicate in range(1, 6):
            rows = [row for row in schedule if row["replicate"] == replicate]
            self.assertEqual({row["branch"] for row in rows}, {"intact", "perturbed"})
            self.assertEqual(len({row["seed"] for row in rows}), 1)

    def test_branch_validator_allows_only_selected_content_with_a_matched_seed(self) -> None:
        source = {
            "model": "m",
            "seed": 1,
            "messages": [{
                "role": "tool",
                "tool_call_id": "t",
                "content": '{"ok":true,"stderr":"","stdout":"A"}',
            }],
        }
        intact, perturbed = build_branch_payloads(
            source,
            selected_result_content_path="/messages/0/content/stdout",
            intact_content="A",
            perturbed_content="B",
            seed=2,
        )

        validate_branch_payloads(
            intact,
            perturbed,
            selected_result_content_path="/messages/0/content/stdout",
            expected_intact_content="A",
            expected_perturbed_content="B",
        )
        self.assertEqual(intact["messages"], source["messages"])
        self.assertEqual(
            json.loads(perturbed["messages"][0]["content"]),
            {"ok": True, "stderr": "", "stdout": "B"},
        )
        perturbed["model"] = "other"
        with self.assertRaisesRegex(ValueError, "unauthorized"):
            validate_branch_payloads(
                intact,
                perturbed,
                selected_result_content_path="/messages/0/content/stdout",
                expected_intact_content="A",
                expected_perturbed_content="B",
            )
        with self.assertRaisesRegex(ValueError, "intact"):
            build_branch_payloads(
                source,
                selected_result_content_path="/messages/0/content/stdout",
                intact_content="wrong",
                perturbed_content="B",
                seed=2,
            )

    def test_branch_builder_accepts_the_exact_production_python_result_envelope(self) -> None:
        content = PythonResult(ok=True, stdout="A", stderr="").model_text()
        source = {
            "model": "m",
            "seed": 1,
            "messages": [{"role": "tool", "tool_call_id": "t", "content": content}],
        }

        intact, perturbed = build_branch_payloads(
            source,
            selected_result_content_path="/messages/0/content/stdout",
            intact_content="A",
            perturbed_content="B",
            seed=2,
        )

        self.assertEqual(intact["messages"][0]["content"], content)
        self.assertEqual(json.loads(perturbed["messages"][0]["content"])["stdout"], "B")

    def test_shadow_executor_runs_ten_matched_direct_payload_calls(self) -> None:
        source = {
            "model": "m",
            "messages": [{
                "role": "tool",
                "tool_call_id": "t",
                "content": '{"ok":true,"stderr":"","stdout":"A"}',
            }],
            "seed": 1,
            "temperature": 0.2,
        }
        calls: list[dict] = []

        def complete(payload):
            record = deepcopy(dict(payload))
            calls.append(record)
            selected = json.loads(record["messages"][0]["content"])["stdout"]
            action = 1 if selected == "A" else 2
            request_bytes = serialize_request_payload(record)
            body = {
                "id": f"shadow-{len(calls)}",
                "choices": [{"message": {"content": json.dumps({"action": action, "data": None})}}],
            }
            body_raw = json.dumps(body, separators=(",", ":"), sort_keys=True)
            return ChatResponse(
                message={"content": json.dumps({"action": action, "data": None})},
                usage={"total_tokens": 1},
                request_payload=record,
                request_payload_sha256=sha256(request_bytes).hexdigest(),
                request_payload_encoding="json-utf8-sort-keys-compact-v1",
                response_body=body,
                response_body_raw=body_raw,
                response_body_sha256=sha256(body_raw.encode()).hexdigest(),
            )

        source_bytes = serialize_request_payload(source)
        result = execute_shadow_bundle(
            source_payload=source,
            source_payload_sha256=sha256(source_bytes).hexdigest(),
            candidate_id="candidate:test",
            generation="g1",
            selected_result_content_path="/messages/0/content/stdout",
            intact_content="A",
            perturbed_content="B",
            original_signature='ACT:{"action":1,"data":null}',
            expected_changed_signature='ACT:{"action":2,"data":null}',
            decision_state={"available_actions": [1, 2], "frame_width": 8, "frame_height": 6},
            complete_payload=complete,
        )

        self.assertEqual(result["status"], "demonstrated")
        self.assertEqual(len(calls), 10)
        self.assertEqual(len(result["responses"]), 10)
        self.assertEqual(validate_shadow_bundle_result(result), "demonstrated")
        tampered = deepcopy(result)
        tampered["responses"][0]["signature"] = "CONTINUE"
        with self.assertRaisesRegex(ValueError, "signature"):
            validate_shadow_bundle_result(tampered)
        for replicate in range(1, 6):
            matched = [call for call in calls if call["seed"] == result["seeds"][str(replicate)]]
            self.assertEqual(len(matched), 2)

    def test_shadow_retry_is_identical_and_failure_does_not_invalidate_gameplay(self) -> None:
        source = {
            "model": "m",
            "messages": [{
                "role": "tool",
                "tool_call_id": "t",
                "content": '{"ok":true,"stderr":"","stdout":"A"}',
            }],
            "seed": 1,
        }
        attempted: list[bytes] = []

        def fail(payload):
            attempted.append(serialize_request_payload(payload))
            raise ModelRequestError("transport unavailable")

        result = execute_shadow_bundle(
            source_payload=source,
            source_payload_sha256=sha256(serialize_request_payload(source)).hexdigest(),
            candidate_id="candidate:failure",
            generation="g1",
            selected_result_content_path="/messages/0/content/stdout",
            intact_content="A",
            perturbed_content="B",
            original_signature='ACT:{"action":1,"data":null}',
            expected_changed_signature='ACT:{"action":2,"data":null}',
            decision_state={"available_actions": [1, 2], "frame_width": 8, "frame_height": 6},
            complete_payload=fail,
        )

        self.assertEqual(len(attempted), 2)
        self.assertEqual(attempted[0], attempted[1])
        self.assertEqual(result["status"], "content_dependence_unidentifiable")
        self.assertEqual(result["reason"], "shadow_transport_failure")
        self.assertTrue(result["source_reconstruction_matches"])
        self.assertEqual(len(result["failed_attempts"]), 2)
        self.assertEqual(
            result["failed_attempts"][0]["request_payload_sha256"],
            result["failed_attempts"][1]["request_payload_sha256"],
        )

    def test_shadow_preflight_binds_candidate_to_immutable_predecision_evidence(self) -> None:
        source = {
            "model": "qwen/qwen3.6-27b",
            "messages": [{
                "role": "tool",
                "tool_call_id": "call-7",
                "content": '{"ok":true,"stderr":"","stdout":"A"}',
            }],
            "seed": 0,
        }
        raw_endpoint = {
            "data": {
                "id": "qwen/qwen3.6-27b",
                "endpoints": [{
                    "name": "Chutes | qwen/qwen3.6-27b-20260422",
                    "tag": "chutes/fp8",
                    "provider_name": "Chutes",
                    "variant": "fp8",
                    "quantization": "fp8",
                    "context_length": 262144,
                    "supported_parameters": [
                        "include_reasoning", "max_tokens", "presence_penalty", "reasoning",
                        "seed", "temperature", "tool_choice", "tools", "top_k", "top_p",
                    ],
                }],
            },
        }
        baseline = canonical_endpoint_metadata(
            raw_endpoint,
            model="qwen/qwen3.6-27b",
            tag="chutes/fp8",
        )
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            archive = run_dir / "request-payloads" / "000001.json"
            archive.parent.mkdir()
            archive.write_bytes(serialize_request_payload(source))
            archive_hash = sha256(archive.read_bytes()).hexdigest()
            event = {
                "event_index": 1,
                "type": "model_response",
                "request_payload_archive": "request-payloads/000001.json",
                "request_payload_sha256": archive_hash,
                "response": {"content": '{"action":1,"data":null}'},
                "decision_state": {
                    "available_actions": [1, 2],
                    "frame_width": 8,
                    "frame_height": 6,
                },
            }
            process = run_dir / "process.jsonl"
            process.write_text(json.dumps(event) + "\n", encoding="utf-8")
            evidence = {
                "process.jsonl": sha256(process.read_bytes()).hexdigest(),
                "request-payloads/000001.json": archive_hash,
            }
            source_hashes = participating_source_hashes()
            manifest = {
                "run_id": "run-1",
                "experiment": {
                    "generation": "g1",
                    "source_hashes": source_hashes,
                    "endpoint_metadata_sha256": endpoint_metadata_sha256(baseline),
                },
                "evidence_hashes": evidence,
            }
            (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            candidate = experiment_artifact_templates(
                "enabled", evidence_hashes=evidence
            )["candidate"]
            candidate["selector"].update({
                "candidate_id": candidate_id(
                    generation="g1", run_id="run-1", event_index=1, tool_call_id="call-7"
                ),
                "run_id": "run-1",
                "selected_process_event_index": 1,
                "selected_tool_call_id": "call-7",
                "source_request_payload_archive": "request-payloads/000001.json",
                "evidence_event_indices": [1],
                "selected_result_content_path": "/messages/0/content/stdout",
                "selected_result_content": "A",
                "grounded_perturbation": "B",
                "original_signature": 'ACT:{"action":1,"data":null}',
                "expected_changed_signature": 'ACT:{"action":2,"data":null}',
                "source_request_payload_sha256": archive_hash,
            })
            candidate["auditor"].update({
                "status": "accepted",
                "evidence_event_indices": [1],
                "source_reconstruction_matches_selected_content": True,
            })
            candidate_path = run_dir / "candidate.json"
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            endpoint_path = run_dir / "endpoint.json"
            endpoint_path.write_text(json.dumps(baseline), encoding="utf-8")

            with patch("run_shadow_intervention.fetch_endpoint_metadata", return_value=raw_endpoint):
                result = load_shadow_preflight(
                    run_dir=run_dir,
                    candidate_path=candidate_path,
                    baseline_endpoint_path=endpoint_path,
                )
            self.assertEqual(result["status"], "preflight_passed")
            manifest["experiment"]["source_hashes"] = {"changed.py": "0" * 64}
            (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "participating sources"):
                load_shadow_preflight(
                    run_dir=run_dir,
                    candidate_path=candidate_path,
                    baseline_endpoint_path=endpoint_path,
                )
            manifest["experiment"]["source_hashes"] = source_hashes
            manifest["experiment"]["endpoint_metadata_sha256"] = "0" * 64
            (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "endpoint"):
                load_shadow_preflight(
                    run_dir=run_dir,
                    candidate_path=candidate_path,
                    baseline_endpoint_path=endpoint_path,
                )
            manifest["experiment"]["endpoint_metadata_sha256"] = endpoint_metadata_sha256(baseline)
            (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            candidate["auditor"]["evidence_event_indices"] = [2]
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "through the candidate decision"):
                load_shadow_preflight(
                    run_dir=run_dir,
                    candidate_path=candidate_path,
                    baseline_endpoint_path=endpoint_path,
                )

    def test_content_dependence_thresholds_are_exact(self) -> None:
        result = assess_content_dependence(
            source_reconstruction_matches=True,
            original_signature='ACT:{"action":1,"data":null}',
            changed_signature='ACT:{"action":2,"data":null}',
            intact_signatures=(
                'ACT:{"action":1,"data":null}',
                'ACT:{"action":1,"data":null}',
                'ACT:{"action":1,"data":null}',
                'ACT:{"action":1,"data":null}',
                'ACT:{"action":2,"data":null}',
            ),
            perturbed_signatures=(
                'ACT:{"action":2,"data":null}',
                'ACT:{"action":2,"data":null}',
                'ACT:{"action":2,"data":null}',
                'ACT:{"action":2,"data":null}',
                'ACT:{"action":1,"data":null}',
            ),
        )

        self.assertEqual(result, "demonstrated")


class SandboxReleaseTests(unittest.TestCase):
    def test_adversarial_suite_passes_and_emits_machine_readable_evidence(self) -> None:
        result = run_sandbox_admissibility_suite(timeout_seconds=5)

        self.assertTrue(result["passed"], json.dumps(result, indent=2))
        self.assertGreaterEqual(len(result["tests"]), 16)
        self.assertTrue(all("expected" in row and "actual" in row for row in result["tests"]))
        self.assertIn("mount_table", result["evidence"])
        self.assertIn("open_file_descriptors", result["evidence"])


if __name__ == "__main__":
    unittest.main()
