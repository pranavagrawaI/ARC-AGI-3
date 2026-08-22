from __future__ import annotations

import json
from pathlib import Path

from tools.epistemic_funnel_core import (
    classify_run,
    default_corpus,
    mine,
    positive_controls,
)

ROOT = Path(__file__).resolve().parents[2]


def test_positive_controls_cover_goal_handoff_and_consolidation() -> None:
    controls = positive_controls(horizon=3)
    assert controls
    assert all(control["pass"] for control in controls)
    assert {control["observed"] for control in controls} >= {
        None,
        "goal_mechanism_failure",
        "handoff_failure",
        "consolidation_failure",
    }


def test_short_trace_is_censored_instead_of_counted_as_goal_null() -> None:
    run = {
        "key": "short",
        "signals": [{"stage": "candidate_objective", "sequence": 0, "evidence_ref": "control"}],
        "max_sequence": 2,
        "decisions": [
            {
                "sequence": sequence,
                "selected_action": None,
                "host_executed": True,
                "actor_spend_usd_cumulative": 0.0,
                "builder_spend_usd_cumulative": 0.0,
                "actor_calls_cumulative": 0,
                "builder_calls_cumulative": 0,
                "evidence_refs": [],
            }
            for sequence in range(3)
        ],
        "terminal_sequences": [],
        "consolidated_sequences": [],
    }
    assert classify_run(run, horizon=3)["status"] == "censored_or_no_supported_failure"


def test_default_corpus_exists() -> None:
    assert all(spec.path.exists() for spec in default_corpus(ROOT))


def test_development_census_is_zero_call_and_preserves_evidence_classes() -> None:
    result = mine(ROOT, horizon=3)
    assert result["execution"] == {"model_calls": 0, "arc_environment_calls": 0}
    assert all(control["pass"] for control in result["positive_controls"])
    assert {run["evidence_class"] for run in result["runs"]} == {
        "legacy_trace",
        "exact_checkpoint",
    }
    assert all(run["input_files"] for run in result["runs"])
    assert all(len(item["sha256"]) == 64 for run in result["runs"] for item in run["input_files"])


def test_development_census_deduplicates_first_failures() -> None:
    result = mine(ROOT, horizon=3)
    episodes = {episode["run_key"]: episode for episode in result["failure_episodes"]}
    assert result["aggregate"]["family_counts"] == {
        "consolidation_failure": 2,
        "goal_mechanism_failure": 2,
        "handoff_failure": 1,
    }
    assert episodes["ar25_goal_gate"]["sequence"] == 3
    assert episodes["ls20_gemma_feasibility"]["sequence"] == 3
    assert episodes["ls20_gemma_reasoning"]["sequence"] == 2
    assert episodes["ar25_symmetry_uptake"]["sequence"] == 16
    assert episodes["ar25_symmetry_replication"]["sequence"] == 19
    sensitivity = {row["goal_obligation_actions"]: row["family_counts"] for row in result["horizon_sensitivity"]}
    assert sensitivity[3]["handoff_failure"] == 1
    assert sensitivity[5]["censored_or_no_supported_failure"] == 1


def test_legacy_multilevel_cost_is_not_backdated_before_terminal() -> None:
    result = mine(ROOT, horizon=3)
    uptake = next(run for run in result["runs"] if run["key"] == "ar25_symmetry_uptake")
    terminal_row = uptake["decisions"][16]
    llm_rows = [
        json.loads(line)
        for line in (ROOT / uptake["path"] / "llm-trace.jsonl").read_text().splitlines()
    ]
    total_actor = sum(float(row.get("cost_usd") or 0.0) for row in llm_rows if row.get("source") == "actor")
    assert terminal_row["actor_spend_usd_cumulative"] < total_actor


def test_top_shadow_is_exact_goal_to_probe_checkpoint() -> None:
    result = mine(ROOT, horizon=3)
    top = result["shadow_candidates"][0]
    assert (top["rank"], top["call_index"], top["sequence"]) == (1, 36, 2)
    assert "goal-to-probe" in top["why"]
