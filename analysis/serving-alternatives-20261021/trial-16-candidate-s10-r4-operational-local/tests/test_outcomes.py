import unittest
import outcomes
class Outcomes(unittest.TestCase):
 def test_complete_candidate_only_evidence_yields_descriptive_rate_without_ratio(self):
  r=outcomes.classify(dict(integrity_complete=True,identity_bound=True,cleanup_complete=True,within_arm_valid=True,U_resp=123.0,failure=None))
  self.assertEqual(r['status'],'CANDIDATE_ALONE_WORKLOAD_FEASIBLE');self.assertEqual(r['U_resp'],123.0);self.assertNotIn('ratio',r);self.assertNotIn('control',r)
