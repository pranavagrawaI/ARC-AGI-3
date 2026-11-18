"""Raw public evidence joins from one actual CPU lifecycle, then negative mutations."""
import copy,json,os,shutil,tempfile,unittest
from pathlib import Path
import capacity_evidence as evidence,provenance
import test_capacity_lifecycle as lifecycle
class CapacityEvidence(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)/'output';cls.config=lifecycle.CapacityLifecycle().config();cls.result=lifecycle.CpuRunner(cls.config,cls.root).run()
  if not cls.result['pass_']:raise AssertionError(cls.result)
  cls.events=[json.loads(s) for s in (cls.root/'supervisor.jsonl').read_text().splitlines()]
 @classmethod
 def tearDownClass(cls):
  if os.environ.get('PACKET_PROOF_DIR'):shutil.copytree(cls.root,Path(os.environ['PACKET_PROOF_DIR'])/'cpu-evidence-joins')
  cls.temp.cleanup()
 def read(self,name):return json.loads((self.root/name).read_text())
 def test_actual_raw_release_fresh_joins_and_failure_mutations(self):
  r=self.read('release.json');f=self.read('candidate-admission.json');evidence.release_evidence(r,f,self.events)
  for mutation in (lambda x:x.update(pass_=False,**{'pass':False}),lambda x:x.update(error='failure'),lambda x:x.update(end=x['absolute_deadline']+1)):
   changed=copy.deepcopy(r);mutation(changed)
   with self.assertRaises(ValueError):evidence.release_evidence(changed,f,self.events)
  changed=copy.deepcopy(f);changed['sample']['end']=r['absolute_deadline']+1
  with self.assertRaises(ValueError):evidence.release_evidence(r,changed,self.events)
 def test_actual_final_resource_join_and_late_cleanup_rejected(self):
  r=self.read('candidate-release.json');work=self.events[0]['work_deadline'];evidence.final_resource_evidence(r,self.events,work)
  e=copy.deepcopy(self.events)
  next(v for v in e if v['event']=='cleanup' and v['pgid']==r['server_pgid'])['t']=r['absolute_deadline']+1
  with self.assertRaises(ValueError):evidence.final_resource_evidence(r,e,work)
  changed=copy.deepcopy(r);changed['error']='cleanup failed'
  with self.assertRaises(ValueError):evidence.final_resource_evidence(changed,self.events,work)
 def test_actual_oom_series_and_coherent_counter_increase_rejected(self):
  boundaries=[json.loads(s) for s in (self.root/'host-boundaries.jsonl').read_text().splitlines()];rows=[json.loads(s) for s in (self.root/'host-series.jsonl').read_text().splitlines()];series=self.read('host-series.result.json');evidence.oom_evidence(boundaries,rows,series,self.events,self.root)
  changed=copy.deepcopy(boundaries);changed[-1]['oom_events']['oom_kill']+=1;changed[-1]['raw']['events']=''.join(f'{k} {v}\n' for k,v in changed[-1]['oom_events'].items())
  with self.assertRaisesRegex(ValueError,'OOM'):evidence.oom_evidence(changed,rows,series,self.events,self.root)
 def test_actual_server_identity_join_no_synthetic_legacy_receipt(self):
  for arm in ('control','candidate'):
   binding=self.read(arm+'-server-binding.json');i=binding['identity'];end=next(e['t'] for e in self.events if e['event']=='arm_pass' and e['arm']==arm)
   capture=dict(server_pgid=str(i['pgid']),server_start=str(i['starttime']),boot_id=i['boot_id'],session_id=self.config['run_id'],gpu_uuid='GPU-LOCAL-CPU',end=end)
   self.assertEqual(provenance.server_binding(self.root,self.events,capture,arm,self.config,'GPU-LOCAL-CPU'),binding)
   bad=copy.deepcopy(capture);bad['server_start']='wrong'
   with self.assertRaisesRegex(ValueError,'starttime'):provenance.server_binding(self.root,self.events,bad,arm,self.config,'GPU-LOCAL-CPU')
   legacy=[e for e in self.events if not (e['event']=='capacity_server_bound' and e['arm']==arm)]
   with self.assertRaisesRegex(ValueError,'registered replay'):provenance.server_binding(self.root,legacy,capture,arm,self.config,'GPU-LOCAL-CPU')
