from __future__ import annotations

import json
from dataclasses import replace
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reconstruction import (  # noqa: E402
    CandidateWorld,
    FixtureError,
    Planner,
    Theory,
    compare_held_out,
    load_fixture,
    load_occurrences,
    run_operational,
)
import reconstruction as reconstruction_module  # noqa: E402


FIXTURE = ROOT / "fixtures/ls20-level1-step13-v1.json"


class OperationalTracerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = load_fixture(FIXTURE)
        self.occurrences = load_occurrences(self.fixture)

    def test_closed_occurrences_stop_before_the_held_out_successor(self) -> None:
        self.assertEqual(self.occurrences.frontier, 13)
        self.assertEqual([row.step for row in self.occurrences.observations], list(range(1, 14)))
        self.assertEqual([row.after_step for row in self.occurrences.transitions], list(range(2, 14)))
        self.assertNotIn("game", self.occurrences.observations[0].public)
        self.assertNotIn("action_input", self.occurrences.observations[0].public)

    def test_candidate_world_is_deterministic_and_measures_oracle_referents(self) -> None:
        world = CandidateWorld()
        first = world.measure(self.occurrences, self.fixture["representation"])
        second = world.measure(self.occurrences, self.fixture["representation"])
        self.assertEqual(first.to_bytes(), second.to_bytes())
        refs = {
            (candidate.step, candidate.bbox)
            for candidate in first.candidates
        }
        self.assertIn((13, (15, 20, 34, 39)), refs)
        self.assertIn((1, (10, 15, 34, 39)), refs)
        self.assertTrue(all(":frontier-13:" in row.candidate_id for row in first.candidates))

    def test_replay_keeps_observationally_indistinguishable_competitor_explicit(self) -> None:
        scene = CandidateWorld().measure(self.occurrences, self.fixture["representation"])
        report = Theory().evaluate(
            self.fixture["oracle_proposal"], self.occurrences, scene
        )
        intended = report.by_id("C-cycle")
        competitor = report.by_id("C-cycle-competitor")
        self.assertEqual(intended.status, "unresolved")
        self.assertEqual(competitor.status, "unresolved")
        self.assertEqual(intended.uncovered, (7,))
        self.assertEqual(competitor.uncovered, (7,))
        self.assertEqual(report.discriminators[0].action, 1)
        self.assertEqual(report.discriminators[0].distinguishes, ("C-cycle", "C-cycle-competitor"))
        self.assertEqual(len(report.replay_rows), 12)
        self.assertEqual(report.replay_rows[5].after_step, 7)
        self.assertEqual(report.replay_rows[5].predicted_position, (30, 19))
        self.assertEqual(report.replay_rows[5].actual_position, (30, 19))
        self.assertEqual(report.replay_rows[5].result, "support")
        self.assertEqual(report.final_state.position, (15, 34))
        self.assertEqual(report.final_state.variables["V0"], 0)

    def test_replay_records_every_grounded_candidate_dependency(self) -> None:
        scene = CandidateWorld().measure(self.occurrences, self.fixture["representation"])
        report = Theory().evaluate(
            self.fixture["oracle_proposal"], self.occurrences, scene
        )
        controlled = self.fixture["oracle_proposal"]["groundings"][0]["occurrences"]
        expected = {
            f"R-oracle-lattice-v1:frontier-13:step-{step}:patch-{bbox[0]}-{bbox[2]}"
            for step, bbox in controlled.items()
        }
        expected.update(
            {
                "R-oracle-lattice-v1:frontier-13:step-1:patch-30-19",
                "R-oracle-lattice-v1:frontier-13:step-1:patch-10-34",
            }
        )
        self.assertEqual(set(report.grounded_candidate_ids), expected)
        self.assertEqual(len(report.grounded_candidate_ids), 15)

    def test_planner_emits_one_step_historical_action_certificate(self) -> None:
        scene = CandidateWorld().measure(self.occurrences, self.fixture["representation"])
        report = Theory().evaluate(self.fixture["oracle_proposal"], self.occurrences, scene)
        certificate = Planner().compare(
            report.final_state,
            report,
            self.fixture["oracle_proposal"],
            self.fixture["planning"],
            self.fixture["frozen_projection"],
        )
        self.assertEqual(certificate.actions, (1,))
        self.assertEqual(certificate.objective_conjecture, "O-progress")
        self.assertEqual(certificate.relied_upon_groundings, ("G-controlled", "G-interaction", "G-objective"))
        self.assertEqual(certificate.relied_upon_transitions, ("C-translation", "C-cycle"))
        self.assertEqual(certificate.search_expansions, 1)
        self.assertEqual(certificate.prediction, {"public.levels_completed": 1})
        self.assertEqual(
            certificate.frozen_projection["candidate_ids"],
            list(certificate.relied_upon_candidate_ids),
        )
        self.assertEqual(len(certificate.relied_upon_candidate_ids), 15)

    def test_held_out_comparison_is_revealed_only_after_certificate(self) -> None:
        scene = CandidateWorld().measure(self.occurrences, self.fixture["representation"])
        report = Theory().evaluate(self.fixture["oracle_proposal"], self.occurrences, scene)
        certificate = Planner().compare(
            report.final_state,
            report,
            self.fixture["oracle_proposal"],
            self.fixture["planning"],
            self.fixture["frozen_projection"],
        )
        comparison = compare_held_out(self.fixture, certificate)
        self.assertTrue(comparison.exact_match)
        self.assertEqual(comparison.scope, "historical_action_prediction_fit")
        self.assertEqual(comparison.actual["public.levels_completed"], 1)

    def test_complete_run_commits_certificate_before_first_held_out_read(self) -> None:
        original = reconstruction_module._trace_rows_through
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            held_out_reads = 0

            def observed_read(fixture, last_step_inclusive):
                nonlocal held_out_reads
                if last_step_inclusive == 14:
                    held_out_reads += 1
                    self.assertTrue((output / "plan_certificate.json").is_file())
                    trace = [
                        json.loads(line)
                        for line in (output / "trace.jsonl").read_text().splitlines()
                    ]
                    self.assertEqual(trace[-1]["type"], "plan_certificate")
                return original(fixture, last_step_inclusive)

            reconstruction_module._trace_rows_through = observed_read
            try:
                run_operational(FIXTURE, output)
            finally:
                reconstruction_module._trace_rows_through = original
            self.assertEqual(held_out_reads, 2)

    def test_prediction_is_generated_from_theory_not_copied_from_projection(self) -> None:
        proposal = deepcopy(self.fixture["oracle_proposal"])
        proposal["objectives"][0]["public_effect"]["delta"] = 2
        scene = CandidateWorld().measure(self.occurrences, self.fixture["representation"])
        report = Theory().evaluate(proposal, self.occurrences, scene)
        certificate = Planner().compare(
            report.final_state,
            report,
            proposal,
            self.fixture["planning"],
            self.fixture["frozen_projection"],
        )
        self.assertEqual(certificate.prediction, {"public.levels_completed": 2})
        self.assertFalse(compare_held_out(self.fixture, certificate).exact_match)

    def test_compiler_dispatches_on_closed_kinds_not_fixture_ids(self) -> None:
        proposal = deepcopy(self.fixture["oracle_proposal"])
        renames = {
            "G-controlled": "G0", "G-interaction": "G1", "G-objective": "G2",
            "C-translation": "C0", "C-cycle": "C1", "C-cycle-competitor": "C2",
            "O-progress": "O0",
        }
        for grounding in proposal["groundings"]:
            grounding["id"] = renames[grounding["id"]]
        for transition in proposal["transitions"]:
            transition["id"] = renames[transition["id"]]
        proposal["objectives"][0]["id"] = "O0"
        proposal["competitor"]["id"] = "C2"
        scene = CandidateWorld().measure(self.occurrences, self.fixture["representation"])
        report = Theory().evaluate(proposal, self.occurrences, scene)
        self.assertEqual(report.selected_transition_ids, ("C0", "C1"))
        self.assertEqual(report.objective_id, "O0")
        self.assertEqual(report.discriminators[0].distinguishes, ("C1", "C2"))
        certificate = Planner().compare(
            report.final_state,
            report,
            proposal,
            {**self.fixture["planning"], "objective_id": "O0"},
            self.fixture["frozen_projection"],
        )
        self.assertEqual(
            certificate.unverified_assumptions,
            (
                "C1 and C2 are observationally indistinguishable on the prefix",
                "O0 has no supporting completion event inside the prefix",
            ),
        )

    def test_non_divergent_competitor_does_not_manufacture_a_discriminator(self) -> None:
        proposal = deepcopy(self.fixture["oracle_proposal"])
        proposal["competitor"]["delta"] = 1
        scene = CandidateWorld().measure(self.occurrences, self.fixture["representation"])
        report = Theory().evaluate(proposal, self.occurrences, scene)
        self.assertEqual(report.discriminators, ())

    def test_planner_enforces_expansion_cap_per_examined_successor(self) -> None:
        proposal = deepcopy(self.fixture["oracle_proposal"])
        objective_grounding = next(
            row for row in proposal["groundings"] if row["symbol"] == "S2"
        )
        objective_grounding["bbox"] = [15, 20, 39, 44]
        scene = CandidateWorld().measure(self.occurrences, self.fixture["representation"])
        report = Theory().evaluate(proposal, self.occurrences, scene)
        budget = {**self.fixture["planning"], "max_expansions": 1}
        with self.assertRaisesRegex(FixtureError, "expansion budget"):
            Planner().compare(
                report.final_state,
                report,
                proposal,
                budget,
                self.fixture["frozen_projection"],
            )

    def test_planner_does_not_select_action_excluded_by_public_state(self) -> None:
        scene = CandidateWorld().measure(self.occurrences, self.fixture["representation"])
        report = Theory().evaluate(self.fixture["oracle_proposal"], self.occurrences, scene)
        state = replace(
            report.final_state,
            public={**report.final_state.public, "available_actions": [2, 3, 4]},
        )
        with self.assertRaisesRegex(FixtureError, "UNKNOWN"):
            Planner().compare(
                state,
                report,
                self.fixture["oracle_proposal"],
                self.fixture["planning"],
                self.fixture["frozen_projection"],
            )

    def test_empty_or_non_exact_projection_fails_closed(self) -> None:
        scene = CandidateWorld().measure(self.occurrences, self.fixture["representation"])
        report = Theory().evaluate(self.fixture["oracle_proposal"], self.occurrences, scene)
        certificate = Planner().compare(
            report.final_state,
            report,
            self.fixture["oracle_proposal"],
            self.fixture["planning"],
            self.fixture["frozen_projection"],
        )
        for projection in (
            {**certificate.frozen_projection, "fields": []},
            {**certificate.frozen_projection, "comparison": "not_exact"},
            {**certificate.frozen_projection, "candidate_ids": ["bogus"]},
        ):
            with self.subTest(projection=projection):
                with self.assertRaises(FixtureError):
                    compare_held_out(self.fixture, replace(certificate, frozen_projection=projection))

    def test_complete_run_is_byte_identical_except_run_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = run_operational(FIXTURE, root / "a")
            second = run_operational(FIXTURE, root / "b")
            for name in (
                "closed_occurrences.json",
                "candidate_scene.json",
                "compiled_conjectures.json",
                "replay_report.json",
                "plan_certificate.json",
                "held_out_successor.json",
                "comparison.json",
            ):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes(), name)
                artifact = json.loads((first / name).read_text())
                self.assertEqual(artifact["authority_label"], "oracle_coach", name)
            result = json.loads((first / "result.json").read_text())
            self.assertEqual(result["summary"]["verdict"], "PASS")
            self.assertEqual(result["summary"]["authority_label"], "oracle_coach")
            self.assertEqual(result["summary"]["provider_calls"], 0)
            self.assertEqual(result["summary"]["environment_actions"], 0)

    def test_complete_run_manifest_binds_executed_reconstruction_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = run_operational(FIXTURE, Path(tmp) / "run")
            manifest = json.loads((run / "manifest.json").read_text())
            expected = {
                "reconstruction/__init__.py": sha256(
                    (ROOT / "reconstruction/__init__.py").read_bytes()
                ).hexdigest(),
                "run_tracer.py": sha256((ROOT / "run_tracer.py").read_bytes()).hexdigest(),
            }
            self.assertEqual(manifest["producer_source_sha256"], expected)

    def test_source_digest_mismatch_fails_closed(self) -> None:
        broken = json.loads(FIXTURE.read_text())
        broken["source"]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.json"
            path.write_text(json.dumps(broken))
            with self.assertRaisesRegex(FixtureError, "source digest"):
                load_occurrences(load_fixture(path))

    def test_complete_run_rejects_selected_transition_with_replay_contradictions(self) -> None:
        broken = deepcopy(self.fixture)
        broken["oracle_proposal"]["transitions"][0]["action_deltas"]["3"] = [0, -4]
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            path = root / "fixture.json"
            path.write_text(json.dumps(broken))
            with self.assertRaisesRegex(FixtureError, "selected transition.*contradiction"):
                run_operational(path, root / "run")


if __name__ == "__main__":
    unittest.main()
