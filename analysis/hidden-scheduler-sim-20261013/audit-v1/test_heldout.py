"""Behavior checks for leakage, deadline semantics, provenance and token parity."""
import copy
import json
import math
import random
import unittest

import numpy as np

import heldout as h


class FixedHolds:
    def __init__(self, quantum):
        self.quantum = quantum

    def draw(self, gid, hold_num, instance_num, seed):
        return self.quantum


def inst(gid, costs, sources=None, acts=None):
    g = h.sim.GameInst(gid, len(costs), [10] * len(costs), costs, acts or [10] * len(costs))
    g.sources = sources or ["observed_trajectory"] * len(costs)
    return g


class HeldoutContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((h.PARENT / "attempts.json").read_text())
        cls.games = list(next(iter(cls.data["games"].values())))
        cls.folds = h.folds_for(cls.games)

    def test_every_identity_held_out_exactly_once(self):
        flat = sum(self.folds, [])
        self.assertEqual(len(flat), 25)
        self.assertEqual(set(flat), set(self.games))
        self.assertEqual(h.folds_for(list(reversed(self.games))), self.folds)

    def test_heldout_outcomes_cannot_change_fit(self):
        # Detect fitting a policy or hold distribution on the evaluation identities.
        changed = copy.deepcopy(self.data)
        test = set(self.folds[0])
        for a in changed["attempts"]:
            if a["game"] in test:
                a.update(tokens=a["tokens"] * 100 + 1, actions=1, solved=True)
        for cohort in ("c14_only", "all_runs_train_only_scaled"):
            before = h.fit_fold(self.data, self.folds, 0, cohort)
            after = h.fit_fold(changed, self.folds, 0, cohort)
            self.assertEqual(before["train"], after["train"])
            self.assertEqual(before["scale"], after["scale"])
            self.assertEqual(before["holds"].first, after["holds"].first)
            for k in range(5):
                np.testing.assert_array_equal(before["hazard"].hz[k], after["hazard"].hz[k])
                self.assertEqual(before["hazard"].sbar[k], after["hazard"].sbar[k])
            self.assertEqual({a['game'] for a in before['world']['attempts']}, test)

    def test_waiting_is_charged_to_deadline(self):
        # One slot, ten seconds, one token/sec. Game A consumes the first six
        # seconds. B needs five, so cannot finish in its remaining four.
        r = h.simulate([inst("A", [6]), inst("B", [5])], h.sim.fifo,
                       horizon=10, slots=1, rate=1, holds=FixedHolds(6), record=True)
        self.assertEqual(r["levels"], 1)
        self.assertEqual(r["tokens"], 10)
        self.assertEqual(r["solve_times"], [6])
        self.assertEqual(r["max_first_wait_s"], 6)
        self.assertEqual(r["score"], 50)

    def test_no_startup_rate_bonus_and_policy_token_parity(self):
        # Infinite costs saturate every slot: capacity is exactly 2*3*11.
        xs = [inst(str(i), [math.inf], ["unresolved"]) for i in range(8)]
        a = h.simulate(xs, h.sim.fifo, horizon=11, slots=2, rate=3, holds=FixedHolds(4))
        b = h.simulate(xs, lambda g, ctx: g.idx, horizon=11, slots=2, rate=3, holds=FixedHolds(4))
        self.assertAlmostEqual(a["tokens"], 66)
        self.assertAlmostEqual(b["tokens"], 66)
        self.assertEqual(a["levels"], 0)

    def test_provenance_respects_game_score_cap(self):
        g = h.sim.GState(inst=inst("A", [1, 1], ["observed_trajectory", "extrapolated"], [1, 1]),
                        idx=0, k=2, apl=[1, 1])
        score, parts, dependent = h.attributed_score(g)
        self.assertEqual(score, 100)
        self.assertAlmostEqual(parts["observed_trajectory"], 100 / 3)
        self.assertAlmostEqual(parts["extrapolated"], 200 / 3)
        self.assertAlmostEqual(dependent, 200 / 3)
        self.assertAlmostEqual(sum(parts.values()), score)

    def test_empirical_only_never_completes_censored_level(self):
        f = h.fit_fold(self.data, self.folds, 0, "c14_only")
        w = h.TaggedWorld(f["world"], f["train"], empirical_only=True, tail_scale=0)
        for gid in f["test"]:
            for run in w.real[gid]:
                rows = w.real[gid][run]
                done = sum(a["solved"] for a in rows)
                g = w.draw(gid, random.Random(0), run=run)
                self.assertEqual(sum(math.isfinite(c) for c in g.cost), done)
                self.assertTrue(all(s == "observed_trajectory" for s in g.sources[:done]))
                self.assertTrue(all(s == "unresolved" for s in g.sources[done:]))

    def test_tail_zero_never_emits_extrapolated_finite_cost(self):
        f = h.fit_fold(self.data, self.folds, 0, "c14_only")
        w = h.TaggedWorld(f["world"], f["train"], tail_scale=0)
        for rep in range(20):
            for gid in f["test"]:
                g = w.draw(gid, random.Random(rep))
                self.assertNotIn("extrapolated", g.sources)

    def test_hold_tape_independent_of_policy_dispatch_order(self):
        f = h.fit_fold(self.data, self.folds, 0, "c14_only")
        tape = f["holds"]
        a = tape.draw("A", 2, 7, "seed")
        tape.draw("B", 10, 0, "seed")
        self.assertEqual(tape.draw("A", 2, 7, "seed"), a)


if __name__ == "__main__":
    unittest.main()
