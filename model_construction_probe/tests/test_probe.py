from __future__ import annotations

import json
import unittest
from dataclasses import FrozenInstanceError, replace

import fixtures
from construction import NeedEvidence, WorkItem, construct
from kernel import (BranchLimit, ContractError, FrozenModel, ModelMiss, Observation,
                    Prefix, Step, Targets, advance_without_observation, assess,
                    assimilate, initialize, pack, predict, rebuild, score_one_step,
                    score_open_loop)


def obs(**values):
    return Observation.make(values, values)


class PredictionBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.lengths = Targets.make({"length": range(65)})
        self.xs = Targets.make({"x": range(5)})

    def test_observation_does_not_retroactively_repair_prediction(self):
        m = FrozenModel(fixtures.WRONG_BUT_ASSIMILATES)
        b = initialize(m, obs(x=1))
        f = predict(m, b, "right", self.xs)
        actual = obs(x=1)
        verdict = assess(f, actual, self.xs)
        self.assertEqual(verdict.status, "contradicted")
        # The naive posterior comparison WOULD see x=1 and incorrectly pass.
        self.assertEqual(m.call("observe_one", "always_move", {"x": 2}, {"x": 1}), {"x": 1})
        with self.assertRaises(ModelMiss):
            assimilate(m, f, actual, verdict)

    def test_unknown_has_zero_credit_not_a_noop_prediction(self):
        m = FrozenModel(fixtures.UNKNOWN)
        f = predict(m, initialize(m, obs(x=0)), "right", self.xs)
        verdict = assess(f, obs(x=4), self.xs)
        self.assertEqual(verdict.status, "unresolved")
        self.assertEqual((verdict.coverage, verdict.specificity), (0, 0))
        self.assertEqual(json.loads(verdict.fields_json)["x"]["width"], 5)

    def test_full_domain_has_zero_credit(self):
        source = fixtures.UNKNOWN.replace('"claims": {}', '"claims": {"x": [0,1,2,3,4]}')
        m = FrozenModel(source)
        v = assess(predict(m, initialize(m, obs(x=0)), "go", self.xs), obs(x=1), self.xs)
        self.assertEqual(v.status, "unresolved")
        self.assertEqual(v.specificity, 0)

    def test_unknown_branch_cannot_hide_behind_precise_branch(self):
        m = FrozenModel(fixtures.WRONG_BUT_ASSIMILATES)
        f = predict(m, initialize(m, obs(x=0)), "go", self.xs)
        unknown = replace(f.branches[0], claims_json="{}", unresolved=("alternative theory",))
        f = replace(f, branches=f.branches + (unknown,))
        v = assess(f, obs(x=1), self.xs)
        self.assertEqual((v.coverage, v.specificity), (0, 0))

    def test_joint_outcome_not_product_of_marginals(self):
        targets = Targets.make({"x": [0, 1], "meter": range(11)})
        m = FrozenModel(fixtures.CORRELATED_FIELDS)
        f = predict(m, initialize(m, Observation.make({}, {})), "go", targets)
        v = assess(f, obs(x=1, meter=8), targets)
        self.assertTrue(all(x["hit"] for x in json.loads(v.fields_json).values()))
        self.assertEqual(v.status, "contradicted")

    def test_alternating_hypotheses_preserve_two_step_total(self):
        m = FrozenModel(fixtures.ALTERNATING)
        b = initialize(m, obs(length=0))
        f1 = predict(m, b, "click", self.lengths)
        f2 = predict(m, advance_without_observation(f1), "click", self.lengths)
        self.assertEqual({json.loads(x.claims_json)["length"][0] for x in f1.branches}, {1, 2})
        self.assertEqual({json.loads(x.claims_json)["length"][0] for x in f2.branches}, {3})

    def test_independent_outcomes_admit_impossible_alternating_path(self):
        steps = (Step.make("click", obs(length=1)), Step.make("click", obs(length=2)))
        alt = FrozenModel(fixtures.ALTERNATING)
        iid = FrozenModel(fixtures.INDEPENDENT)
        ra = score_open_loop(alt, initialize(alt, obs(length=0)), steps, self.lengths)
        ri = score_open_loop(iid, initialize(iid, obs(length=0)), steps, self.lengths)
        self.assertEqual(ra[-1].status, "contradicted")
        self.assertEqual(ri[-1].status, "consistent")

    def test_assimilation_filters_phase_not_just_visible_length(self):
        m = FrozenModel(fixtures.ALTERNATING)
        f = predict(m, initialize(m, obs(length=0)), "click", self.lengths)
        actual = obs(length=1)
        b = assimilate(m, f, actual, assess(f, actual, self.lengths))
        self.assertEqual(len(b.particles), 1)
        self.assertEqual(json.loads(b.particles[0].state_json)["phase"], 1)
        nxt = predict(m, b, "click", self.lengths)
        self.assertEqual(json.loads(nxt.branches[0].claims_json)["length"], [3])

    def test_dead_candidate_cannot_resurrect_in_rollout(self):
        source = fixtures.ALTERNATING.replace('"phase": 1 - s["phase"]', '"phase": s["phase"]')
        m = FrozenModel(source)
        steps = (Step.make("click", obs(length=1)), Step.make("click", obs(length=4)))
        reports = score_open_loop(m, initialize(m, obs(length=0)), steps, self.lengths)
        self.assertTrue(json.loads(reports[-1].fields_json)["length"]["hit"])
        self.assertEqual(reports[-1].status, "contradicted")

    def test_open_loop_never_calls_observation_assimilation(self):
        source = fixtures.ALTERNATING.replace('def observe_one(candidate, s, o):\n    return s',
                                               'def observe_one(candidate, s, o):\n    raise RuntimeError("unexpected assimilation")')
        m = FrozenModel(source)
        steps = (Step.make("click", obs(length=1)), Step.make("click", obs(length=3)))
        reports = score_open_loop(m, initialize(m, obs(length=0)), steps, self.lengths)
        self.assertEqual([r.status for r in reports], ["consistent", "consistent"])
        with self.assertRaises(RuntimeError):
            score_one_step(m, initialize(m, obs(length=0)), steps, self.lengths)

    def test_covered_terrain_is_retained_not_reencoded(self):
        targets = Targets.make({"terrain": [0, 1, 2]})
        m = FrozenModel(fixtures.OCCLUSION)
        b = initialize(m, obs(terrain=2))
        f = predict(m, b, "cover", targets)
        covered = Observation.make({"covered": True}, {})
        b = assimilate(m, f, covered, assess(f, covered, targets))
        self.assertEqual(json.loads(b.particles[0].state_json)["terrain"], 2)
        f = predict(m, b, "uncover", targets)
        self.assertEqual(assess(f, obs(terrain=2), targets).status, "consistent")

    def test_newly_observed_terrain_updates_knowledge_without_past_credit(self):
        targets = Targets.make({"terrain": [0, 1, 2]})
        m = FrozenModel(fixtures.OCCLUSION)
        b = initialize(m, Observation.make({}, {}))
        f = predict(m, b, "uncover", targets)
        actual = obs(terrain=2)
        v = assess(f, actual, targets)
        self.assertEqual(v.coverage, 0)
        b = assimilate(m, f, actual, v)
        nxt = predict(m, b, "uncover", targets)
        self.assertEqual(assess(nxt, actual, targets).coverage, 1)

    def test_model_version_change_requires_rebuild(self):
        old = FrozenModel(fixtures.ALTERNATING)
        new = FrozenModel(fixtures.ALTERNATING + "\n# revised\n")
        with self.assertRaises(ContractError):
            predict(new, initialize(old, obs(length=0)), "click", self.lengths)
        prefix = Prefix(obs(length=0), (Step.make("click", obs(length=1)),))
        b = rebuild(new, prefix, self.lengths)
        self.assertEqual(b.model_id, new.model_id)

    def test_observation_snapshot_cannot_change_via_original_dict(self):
        raw = {"length": 0}
        o = Observation.make(raw, raw)
        raw["length"] = 50
        self.assertEqual(json.loads(o.raw_json)["length"], 0)
        with self.assertRaises(FrozenInstanceError):
            o.raw_json = "{}"

    def test_forecast_seal_is_stable_after_assimilation(self):
        m = FrozenModel(fixtures.ALTERNATING)
        f = predict(m, initialize(m, obs(length=0)), "click", self.lengths)
        before = f.seal
        actual = obs(length=1)
        assimilate(m, f, actual, assess(f, actual, self.lengths))
        self.assertEqual(before, f.seal)

    def test_assessment_cannot_be_reused_for_another_observation(self):
        m = FrozenModel(fixtures.ALTERNATING)
        f = predict(m, initialize(m, obs(length=0)), "click", self.lengths)
        with self.assertRaises(ContractError):
            assimilate(m, f, obs(length=2), assess(f, obs(length=1), self.lengths))

    def test_module_globals_are_not_hidden_persistent_state(self):
        m = FrozenModel(fixtures.MUTATING_GLOBAL)
        b = initialize(m, obs(x=0))
        self.assertEqual(predict(m, b, "go", self.xs), predict(m, b, "go", self.xs))

    def test_imports_rejected_in_self_contained_prototype(self):
        with self.assertRaises(ContractError):
            FrozenModel("import random\n" + fixtures.ALTERNATING)

    def test_branch_budget_is_explicit_failure_not_silent_selection(self):
        m = FrozenModel(fixtures.INDEPENDENT)
        steps = tuple(Step.make("click", obs(length=i)) for i in range(1, 7))
        with self.assertRaises(BranchLimit):
            score_open_loop(m, initialize(m, obs(length=0)), steps, self.lengths, cap=32)

    def test_empty_outcomes_not_treated_as_no_effect(self):
        source = fixtures.UNKNOWN.replace('return [{"state": {}, "claims": {}, "unresolved": ["transition mechanism"]}]',
                                           'return []')
        m = FrozenModel(source)
        with self.assertRaises(ContractError):
            predict(m, initialize(m, obs(x=0)), "go", self.xs)

    def test_out_of_domain_claim_is_contract_error(self):
        m = FrozenModel(fixtures.WRONG_BUT_ASSIMILATES)
        with self.assertRaises(ContractError):
            predict(m, initialize(m, obs(x=4)), "go", self.xs)

    def test_out_of_domain_measurement_is_contract_error(self):
        m = FrozenModel(fixtures.UNKNOWN)
        f = predict(m, initialize(m, obs(x=0)), "go", self.xs)
        with self.assertRaises(ContractError):
            assess(f, obs(x=100), self.xs)


class ConstructionTests(unittest.TestCase):
    def setUp(self):
        self.targets = Targets.make({"x": range(5)})
        self.prefix = Prefix(Observation.make({"x": 0, "limit": 1}, {"x": 0}),
                             (Step.make("right", obs(x=1)), Step.make("right", obs(x=1))))
        self.work = WorkItem("Predict the next move", "Existing model misses a retained transition",
                             "contrast", "Movement applicability", (0, 1, 2), ("x", "limit"),
                             ((1, ("x",)), (2, ("x",))))
        self.incumbent = FrozenModel(fixtures.WRONG_BUT_ASSIMILATES)

    def test_work_references_cannot_read_past_frontier(self):
        with self.assertRaises(ContractError):
            replace(self.work, evidence=(3,)).validate(self.prefix, self.targets)

    def test_prose_only_completion_rejected(self):
        with self.assertRaises(ContractError):
            replace(self.work, completion_checks=()).validate(self.prefix, self.targets)

    def test_bounded_scripted_repair_returns_candidate_not_prospective_certificate(self):
        proposals = iter((fixtures.WRONG_BUT_ASSIMILATES, fixtures.BOUNDARY))
        seen_frontiers = []
        def builder(work, prefix, incumbent, feedback):
            seen_frontiers.append(prefix.frontier)
            return next(proposals)
        result = construct(self.work, self.prefix, self.incumbent, self.targets, builder)
        self.assertEqual(result.status, "candidate_ready")
        self.assertEqual([a.outcome for a in result.attempts], ["rejected", "prefix_supported"])
        self.assertEqual(seen_frontiers, [2, 2])

    def test_unknown_everywhere_does_not_complete_required_repair(self):
        result = construct(self.work, self.prefix, self.incumbent, self.targets,
                           lambda *args: fixtures.UNKNOWN, max_attempts=1)
        self.assertEqual(result.status, "budget_exhausted")

    def test_need_evidence_is_not_a_model_success(self):
        request = NeedEvidence("boundary geometry", pack("right"), pack({"x": [0, 1]}))
        result = construct(self.work, self.prefix, self.incumbent, self.targets,
                           lambda *args: request)
        self.assertEqual(result.status, "needs_evidence")
        self.assertIsNone(result.model)
        self.assertEqual(len(result.attempts), 0)

    def test_budget_bounds_failed_candidate_repair(self):
        result = construct(self.work, self.prefix, self.incumbent, self.targets,
                           lambda *args: fixtures.WRONG_BUT_ASSIMILATES, max_attempts=2)
        self.assertEqual(result.status, "budget_exhausted")
        self.assertEqual(len(result.attempts), 2)


if __name__ == "__main__":
    unittest.main()
