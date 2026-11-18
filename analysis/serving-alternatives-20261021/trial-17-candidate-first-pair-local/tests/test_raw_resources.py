import copy,json,unittest
from pathlib import Path
import candidate_evidence as evidence
import pair_evidence
BASE=Path(__file__).resolve().parents[2]
CANDIDATE=BASE/'trial16-three-pass-audit-result-20261024/evidence/serving-candidate-s10-16/output'
NATIVE=BASE/'trial-14-phase-handles-local/remote-readback/archive-v1-kernel140449504/members/serving-post-control-load14/output'
def read(path):return json.loads(path.read_bytes())
class Resources(unittest.TestCase):
 def test_real_candidate_raw_and_false_original140_policy(self):
  root=CANDIDATE;release=read(root/'candidate-release.json');scope=read(root/'candidate-scope.json');sample=release['samples'][-1]
  raw=(root/'query-artifacts'/sample['probes']['gpu']['command']['artifacts']['stdout']['path']).read_text();uuid=raw.split(',')[-1].strip()
  result=evidence.verify_sample(root,sample,scope['binding'],uuid,release['absolute_deadline'])
  self.assertTrue(result['eligible'])
  wrong=copy.deepcopy(sample);wrong['probes']['gpu']['used_mib']+=1
  with self.assertRaisesRegex(ValueError,'raw GPU'):evidence.verify_sample(root,wrong,scope['binding'],uuid,release['absolute_deadline'])
 def test_real_shipping_native_six_probe_raw_rederivation(self):
  root=NATIVE;release=read(root/'release.json');scope=read(root/'release-scope.json');reg=read(root/'control-package/control-frontend-registration.json')
  bound=dict(controller=scope['lease']['created']['controller'],frontend=reg['frontend']['pid'])
  sample=release['samples'][-1];text=(root/'query-artifacts'/sample['probes']['gpu']['command']['artifacts']['stdout']['path']).read_text();uuid=text.split(',')[-1].strip()
  result=evidence.verify_sample(root,sample,bound,uuid,release['absolute_deadline'],native=True)
  self.assertTrue(result['eligible']);self.assertFalse(result['original140'])
  for key in ('graceful_marker','crash_marker','frontend_exited'):
   wrong=copy.deepcopy(sample);wrong['probes']['native'][key]=not wrong['probes']['native'][key]
   with self.subTest(key=key),self.assertRaises(ValueError):evidence.verify_sample(root,wrong,bound,uuid,release['absolute_deadline'],native=True)

class Health(unittest.TestCase):
 def test_retained_candidate_health_raw_matches_actual_ready(self):
  root=CANDIDATE;cfg=read(root.parent/'config.json');events=[json.loads(x) for x in (root/'supervisor.jsonl').read_text().splitlines()]
  self.assertTrue(pair_evidence.health(root,events,cfg,'candidate'))
 def test_retained_shipping_health_raw_matches_actual_ready(self):
  root=NATIVE;cfg=read(root.parent/'config.json');events=[json.loads(x) for x in (root/'supervisor.jsonl').read_text().splitlines()]
  self.assertTrue(pair_evidence.health(root,events,cfg,'control'))

class NativeSignal(unittest.TestCase):
 def test_actual_targeted_frontend_TERM_and_pid_substitution_rejection(self):
  import pair_resources
  release=read(NATIVE/'release.json');reg=read(NATIVE/'control-package/control-frontend-registration.json');bound=dict(frontend=reg['frontend']['pid'])
  self.assertTrue(pair_resources.native_signal(release,bound))
  for field,value in [('pid',bound['frontend']+1),('signal',9),('start',release['start']-1)]:
   changed=copy.deepcopy(release);changed['events'][0][field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):pair_resources.native_signal(changed,bound)
