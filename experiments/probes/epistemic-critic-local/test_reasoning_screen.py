import unittest
from reasoning_screen import cases


class ReasoningPairTests(unittest.TestCase):
    def test_evidence_fixed_within_pairs(self):
        data=cases()
        self.assertEqual(len(data),8)
        for a,b in zip(data[::2],data[1::2]):
            self.assertEqual(a['observations'],b['observations'])
            self.assertEqual(a['provenance'],b['provenance'])
            self.assertEqual({a['expected'],b['expected']},{'concern','no_concern'})
            self.assertNotEqual(a['actor_reasoning'],b['actor_reasoning'])


if __name__=='__main__': unittest.main()
