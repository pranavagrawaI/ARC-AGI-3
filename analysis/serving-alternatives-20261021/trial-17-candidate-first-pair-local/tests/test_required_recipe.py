"""Actual retained-shape dependency recipe; no synthetic scientific pass."""
import base64,copy,json,sys,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE/'collection_tools'))
import collector,required_files,primary_bundle,audit_required,space
BASE=HERE.parent
PRIMARY=BASE/'trial-16-candidate-s10-r4-operational-local/remote-readback/current-primary-fallback-local/primary-current/PRIMARY.json'
NATIVE=BASE/'trial-14-phase-handles-local/remote-readback/archive-v1-kernel140449504/members/serving-post-control-load14/output'
class Required(unittest.TestCase):
 def fixture(self):
  bundle=json.loads(PRIMARY.read_bytes());cfg=json.loads(base64.b64decode(bundle['files']['config.json']['base64']));old='/kaggle/working/serving-post-control-load14'
  for key in ('native-lease.json','pre-release.json','release.json'):
   data=(NATIVE/key).read_bytes().replace(old.encode(),cfg['cwd'].encode());bundle['files'][key]=primary_bundle.record(data)
  bundle['files']['control-admission.json']=bundle['files']['candidate-release.json']
  events=(BASE/'trial16-three-pass-audit-result-20261024/evidence/serving-candidate-s10-16/output/candidate/events.jsonl').read_bytes()
  bundle['files']['candidate/events.jsonl']=primary_bundle.record(events)
  bundle['files']['control/events.jsonl']=primary_bundle.record(events.replace((cfg['run_id']+'-candidate-').encode(),(cfg['run_id']+'-control-').encode()))
  payload={name:primary_bundle.record((HERE/name).read_bytes()) for name in ('candidate-plan.json','control-plan.json','supervisor.py')}
  return bundle,payload,cfg
 def test_all_raw_refs_events_sources_and_independent_actual_result(self):
  bundle,payload,cfg=self.fixture();plan=required_files.build(bundle,payload);names={row['name'] for row in plan['rows']};prefix=Path(cfg['cwd']).name
  self.assertIn(prefix+'/output/result.json',names);self.assertNotIn(prefix+'/output/result.json',plan['embedded'])
  self.assertIn(prefix+'/output/host-series.jsonl',names);self.assertIn(prefix+'/supervisor.py',names)
  self.assertFalse(names&set(plan['embedded']));self.assertLess(len(names),20000)
  self.assertTrue(any(row['conditional'] for row in plan['rows']))
  self.assertTrue(any('/query-artifacts/' in name for name in names))
 def test_cross_arm_name_and_path_expansion_reject(self):
  bundle,payload,cfg=self.fixture();raw=base64.b64decode(bundle['files']['control/events.jsonl']['base64']).replace(b'-control-',b'-candidate-');bundle['files']['control/events.jsonl']=primary_bundle.record(raw)
  with self.assertRaises(ValueError):required_files.build(bundle,payload)
  for name in ('../x','/abs','a//b','a/../b','a\\b'):
   with self.subTest(name=name),self.assertRaises(ValueError):collector.safe_name(name)
 def test_actual_rewritten_result_is_retained_and_rejected_not_substituted(self):
  with tempfile.TemporaryDirectory() as td:
   path=Path(td)/'result.json';raw=b'{"qualification":"FEASIBILITY_ONLY","ratio":null,"pass_":false}'
   path.write_bytes(raw);bundle={'files':{'result.json':primary_bundle.record(b'{"qualification":"CLOSE_TESTED_CONFIG","ratio":0.9,"pass_":true}')}}
   with self.assertRaisesRegex(ValueError,'post-publication'):audit_required.independent_result(path,dict(origin='INDEPENDENT_CURRENT_SESSION_DOWNLOAD'),bundle)
   self.assertEqual(path.read_bytes(),raw)
   with self.assertRaisesRegex(ValueError,'primary alias'):audit_required.independent_result(path,dict(origin='EXACT_PRIMARY_EMBEDDED'),bundle)
 def test_space_is_19GiB_simultaneous_and3GiB_derived(self):
  self.assertEqual(space.check(HERE)['required_bytes'],19*1024**3)
  self.assertEqual(space.check(HERE,stage='derived')['required_bytes'],3*1024**3)
  self.assertEqual(space.check(HERE,stage='assembly')['required_bytes'],11*1024**3)
