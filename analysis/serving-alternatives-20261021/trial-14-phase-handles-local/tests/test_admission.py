import copy,time,unittest
import admission
class Admission(unittest.TestCase):
 def sample(self):
  now=time.monotonic();p={name:dict(eligible=True,start=now,end=now) for name in ('gpu','apps','cgroup','port','disk','native')}
  p['cgroup'].update(eligible=False,computed_bytes=10,threshold_bytes=140*1024**3,files_stable_during_read=True)
  return dict(start=now,end=now,process=dict(eligible=True,remaining=[],start=now,end=now),probes=p,probe_worker_alive=False,ipc_error=None,eligible=False)
 def test_host_failure_does_not_change_original_verdict_but_admits_distinct_contract(self):
  s=self.sample();self.assertFalse(s['eligible']);self.assertTrue(admission.nonhost(s,time.monotonic()+1));self.assertFalse(s['probes']['cgroup']['eligible'])
 def test_every_missing_late_unknown_or_bad_nonhost_blocks(self):
  original=self.sample()
  for name in ('gpu','apps','port','disk','native'):
   s=copy.deepcopy(original);s['probes'][name]['eligible']=False;self.assertFalse(admission.nonhost(s,time.monotonic()+1))
  for mutate in (lambda s:s['process'].update(remaining=[1]),lambda s:s.update(probe_worker_alive=True),lambda s:s.update(ipc_error='lost'),lambda s:s['probes'].pop('native'),lambda s:s['probes']['cgroup'].update(error='bad'),lambda s:s['probes']['cgroup'].pop('computed_bytes')):
   s=copy.deepcopy(original);mutate(s);self.assertFalse(admission.nonhost(s,time.monotonic()+1))
  self.assertFalse(admission.nonhost(original,original['end']-.01))
 def test_freshness_and_explicit_ownership_block_admission(self):
  s=self.sample();admission.require_candidate(s,s['end']+1,2,[],[],[])
  for args in [(s,s['end']+3,2,[],[],[]),(s,s['end']+1,2,[1],[],[]),(s,s['end']+1,2,[],[1],[]),(s,s['end']+1,2,[],[],[1])]:
   with self.assertRaises(ValueError):admission.require_candidate(*args)
