import copy,json,math,unittest
from pathlib import Path
import outcomes,release_runner
HERE=Path(__file__).resolve().parents[1]
class Policy(unittest.TestCase):
    def test_exact_caps_and_no_control(self):
        c=json.loads((HERE/'packet-spec.json').read_text())['config'];release_runner.validate(c,check_cwd=False)
        for key,value in [('control',{}),('admission',{})]:
            with self.assertRaises(ValueError):release_runner.validate(dict(c,**{key:value}),check_cwd=False)
        c['final_host_policy']='REQUIRE_140_TRUE'
        with self.assertRaises(ValueError):release_runner.validate(c,check_cwd=False)
    def test_complete_negative_and_incomplete_override(self):
        value=dict(integrity_complete=True,identity_bound=True,cleanup_complete=True,failure=dict(kind='OBSERVED_OOM',raw_verified=True))
        self.assertEqual(outcomes.classify(value)['status'],'CLOSE_TESTED_BOUNDED_CONFIGURATION')
        for key in ('integrity_complete','identity_bound','cleanup_complete'):
            self.assertEqual(outcomes.classify(dict(value,**{key:False}))['status'],'INVALID_INCONCLUSIVE')
        value['failure']['kind']='UNKNOWN'
        self.assertEqual(outcomes.classify(value)['status'],'INVALID_INCONCLUSIVE')
    def test_rate_not_exposed_for_invalid_or_nonfinite(self):
        for rate in (-1,math.inf,math.nan,True):
            value=outcomes.classify(dict(integrity_complete=True,identity_bound=True,cleanup_complete=True,within_arm_valid=True,U_resp=rate))
            self.assertEqual(value['status'],'INVALID_INCONCLUSIVE');self.assertIsNone(value['U_resp']);self.assertNotIn('ratio',value)
