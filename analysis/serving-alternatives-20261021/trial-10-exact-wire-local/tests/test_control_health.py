"""Real local subprocesses parse exact pinned environ bytes; framework modules are CPU stubs."""
import copy,hashlib,json,os
from pathlib import Path
import sys,tempfile,unittest,venv
from unittest.mock import patch
import control_prepare as C
ROOT=Path(__file__).resolve().parents[1]
RAW={'SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION':'1','SGLANG_HEALTH_CHECK_TIMEOUT':'20'}
ENV_SHA='7431c4e0e679d1983ade2c3d9a8689c0ac78666d166d6743188434f2f25c5ed1'

class ControlHealth(unittest.TestCase):
 def setUp(self):
  tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name);self.out=self.root/'out';self.out.mkdir()
  venv.EnvBuilder(with_pip=False).create(self.root/'venv');self.python=self.root/'venv/bin/python'
  site=next((self.root/'venv/lib').glob('python*/site-packages'))
  pkg=site/'sglang/srt';pkg.mkdir(parents=True)
  (site/'sglang/__init__.py').write_text("__version__='LOCAL_CPU_STUB'\n");(pkg/'__init__.py').write_text('')
  (pkg/'environ.py').write_bytes((ROOT/'evidence/pinned-environ.py').read_bytes());self.native=pkg/'environ.py'
  for name in ('torch','transformers'):(site/(name+'.py')).write_text("__version__='LOCAL_CPU_STUB'\n")
  (self.root/'control_prefetch.py').write_text('def precache(*args,**kwargs): pass\n')
  self.marker=self.root/'SERVER_STARTED'
  argv=[str(self.python),'-c',"from pathlib import Path; import sys; Path(sys.argv[1]).write_text('local fake server started')",str(self.marker)]
  self.state={'python':str(self.python),'argv':argv,'env_overrides':{},'target':'CPU_TARGET','draft':'CPU_DRAFT'}
  (self.out/'control-prepared.json').write_text(json.dumps(self.state))
  plan=copy.deepcopy(C.PLAN);plan['health_policy']={'raw':RAW,'effective':{'generation_enabled':True,'timeout_s':20},'native_environ_sha256':ENV_SHA}
  plan['notebook_extract']['prefetch']['sha256']=hashlib.sha256((self.root/'control_prefetch.py').read_bytes()).hexdigest()
  self.enterContext(patch.object(C,'HERE',self.root));self.enterContext(patch.object(C,'PLAN',plan))
  self.enterContext(patch.dict(os.environ,RAW|{'SCREEN_OUTPUT':str(self.out)}))
 def serve(self):
  with self.assertRaises(SystemExit) as end:C.serve()
  self.assertEqual(end.exception.code,0)
 def test_identity_and_serve_bind_actual_native_parsed_values(self):
  C.identity();self.serve()
  a=json.loads((self.out/'control-health-identity.json').read_text());b=json.loads((self.out/'control-health-serve.json').read_text())
  for r in (a,b):
   self.assertEqual(r['raw'],RAW);self.assertEqual(r['effective'],{'generation_enabled':True,'timeout_s':20});self.assertEqual(r['native_environ_sha256'],ENV_SHA)
  self.assertTrue(self.marker.exists());self.assertEqual(a['runtime_executable'],b['runtime_executable'])
 def test_changed_serve_environment_blocks_before_server(self):
  C.identity()
  with patch.dict(os.environ,{'SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION':'0'}),self.assertRaisesRegex(ValueError,'health'):C.serve()
  self.assertFalse(self.marker.exists())
 def test_missing_explicit_environment_is_not_satisfied_by_native_default(self):
  os.environ.pop('SGLANG_HEALTH_CHECK_TIMEOUT')
  with self.assertRaisesRegex(ValueError,'health'):C.identity()
  self.assertFalse((self.out/'control-health-identity.json').exists())
 def test_prepared_override_collision_blocks(self):
  self.state['env_overrides']={'SGLANG_HEALTH_CHECK_TIMEOUT':'99'};(self.out/'control-prepared.json').write_text(json.dumps(self.state))
  with self.assertRaisesRegex(ValueError,'health'):C.identity()
 def test_identity_receipt_mismatch_blocks_before_server(self):
  C.identity();p=self.out/'control-health-identity.json';r=json.loads(p.read_text());r['effective']['timeout_s']=19;p.write_text(json.dumps(r))
  with self.assertRaisesRegex(ValueError,'health'):C.serve()
  self.assertFalse(self.marker.exists())
 def test_native_source_drift_blocks(self):
  self.native.write_text(self.native.read_text()+'\n# local drift\n')
  with self.assertRaisesRegex(ValueError,'health'):C.identity()
  self.assertFalse((self.out/'control-health-identity.json').exists())
