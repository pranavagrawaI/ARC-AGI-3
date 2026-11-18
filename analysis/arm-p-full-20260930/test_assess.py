"""Small independent examples protect cutoff, join integrity, and promotion gates."""
import copy
import unittest
from assess import at_time, decision, progression


class AssessmentTests(unittest.TestCase):
    def arms(self, p1=51, p2=52):
        return {k: {'levels': levels, 'issues': [], 'per_game': {'game': {}},
                    'serving': {'busy': {'running': 9}}}
                for k, levels in [('KF', 44), ('C', 42), ('P1', p1), ('P2', p2)]}

    def test_threshold_uses_both_passes_and_includes_boundary(self):
        self.assertEqual(decision(self.arms())['candidate'], 'P')
        self.assertEqual(decision(self.arms(51, 51))['candidate'], 'KF')
        arms = self.arms(60, 43)
        self.assertEqual(decision(arms)['status'], 'PASS')  # No invented each-pass threshold.
        del arms['P2']
        self.assertEqual(decision(arms)['status'], 'PENDING')

    def test_invalidity_and_mechanism_prevent_promotion(self):
        arms = self.arms(70, 70)
        arms['P1']['issues'] = ['wrong horizon']
        self.assertEqual(decision(arms)['status'], 'INVALID')
        arms['P1']['issues'] = []
        arms['P1']['serving']['busy']['running'] = 6
        self.assertEqual(decision(arms)['candidate'], 'KF')

    def test_action_timestamps_are_cumulative_and_cutoff_is_inclusive(self):
        run = {'history': [{'wallclock_seconds': t} for t in [10, 20, 30]], 'levels_completed': 2}
        events = [{'type': 'action', 'action_num': i, 'score': s} for i, s in enumerate([0, 1, 2], 1)]
        points, clears = progression(run, events)
        self.assertEqual(at_time(points, 19), (10, 1, 0))
        self.assertEqual(at_time(points, 20), (20, 2, 1))
        self.assertEqual(clears, [(20, 2, 1), (30, 3, 2)])
        with self.assertRaises(ValueError):
            progression(run, events[:-1])
        bad = copy.deepcopy(events)
        bad[-1]['score'] = 1
        with self.assertRaises(ValueError):
            progression(run, bad)


if __name__ == '__main__':
    unittest.main()
