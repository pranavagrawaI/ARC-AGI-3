import ast,copy,datetime as dt,hashlib,json,os,socket,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import build,packet,transport,launch,reconciliation,release_runner,adapters,capture,registration,phase_tools
from core import resources


class PacketTests(unittest.TestCase):
 def test_exact_runtime_pins_and_source_extractions(self):
  lineage=json.loads((HERE/'SOURCE-LINEAGE.json').read_text())
  for path,h in lineage['references'].items():self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(),h,path)
  prior=HERE.parent/'release-adapters-r5-gate-local-20261023'
  for name,h in json.loads((HERE/'runtime-pins.json').read_text()).items():
   source=prior/name if (prior/name).exists() else HERE.parent/'shutdown-release-prototype-r2-local-20261023'/name
   self.assertEqual((HERE/name).read_bytes(),source.read_bytes(),name)
  original=ast.parse((HERE.parent/'trial-10-exact-wire-local/candidate_prepare.py').read_text())
  current=ast.parse((HERE/'control_support.py').read_text())
  for node in current.body:
   if isinstance(node,ast.FunctionDef):
    old=next(v for v in original.body if isinstance(v,ast.FunctionDef) and v.name==node.name)
    self.assertEqual(ast.dump(node,include_attributes=False),ast.dump(old,include_attributes=False))
  old=ast.parse((HERE.parent/'trial-10-exact-wire-local/supervisor.py').read_text());old=next(n for n in old.body if isinstance(n,ast.ClassDef))
  new=ast.parse((HERE/'phase_tools.py').read_text());new=next(n for n in new.body if isinstance(n,ast.ClassDef))
  for node in new.body:
   prior=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==node.name)
   self.assertEqual(ast.dump(node,include_attributes=False),ast.dump(prior,include_attributes=False),node.name)
 def test_control_identity_delta_no_candidate_or_replay(self):
  spec=json.loads((HERE/'packet-spec.json').read_text());config=spec['config'];release_runner.validate(config,False)
  self.assertEqual(config['purpose'],'CONTROL_POST_LOAD_RELEASE');self.assertEqual(config['arm_order'],['control'])
  self.assertFalse(any('candidate' in name or 'replay' in name or 'corpus' in name for name in spec['files']))
  self.assertEqual(spec['metadata']['dataset_sources'],['dfranzen/pennyroyal-v253']);self.assertEqual(len(spec['metadata']['model_sources']),2)
  self.assertEqual((config['platform_seconds'],config['total_seconds'],config['cleanup_seconds']),(3900,3600,300))
  old=HERE.parent/'trial-10-exact-wire-local'
  for name in ('control_notebook_setup.py','control_prefetch.py','control-argv.reference.json','transport.py','http_once.py'):
   self.assertEqual((HERE/name).read_bytes(),(old/name).read_bytes(),name)
  expected=(HERE.parent/'release-adapters-r6-test-local-20261023/control_prepare_registered.py').read_text().replace('import candidate_prepare as shared','import control_support as shared')
  self.assertEqual((HERE/'control_prepare.py').read_text(),expected)
  plan=json.loads((HERE/'control-plan.json').read_text());oldplan=json.loads((old/'control-plan.json').read_text())
  plan['run_id']=oldplan['run_id'];plan['installed_source_sha256'].pop('sglang/srt/utils/watchdog.py');self.assertEqual(plan,oldplan)
 def test_unassigned_roundtrip_and_exact_payload(self):
  spec=json.loads((HERE/'packet-spec.json').read_text())
  with tempfile.TemporaryDirectory() as td:
   manifest=packet.prepare(Path(td)/'packet');outer=(Path(td)/'packet'/spec['metadata']['code_file']).read_bytes();self.assertLess(len(outer),1000000)
   raw=transport.decode(outer);values=transport.literals(transport.cell_source(json.loads(raw)));self.assertIsNone(values['ASSIGNMENT']);self.assertEqual(values['CONFIG'],spec['config'])
   import base64
   for name,value in values['PAYLOAD'].items():
    path=Path(spec['files'][name]) if name in spec['files'] else HERE/name
    self.assertEqual(base64.b64decode(value['base64']),path.read_bytes())
   with patch.object(Path,'mkdir',side_effect=AssertionError('extraction before authority')),patch('subprocess.Popen',side_effect=AssertionError('process before authority')):
    with self.assertRaisesRegex(RuntimeError,'PREPARATION ONLY'):exec(transport.cell_source(json.loads(outer)),{})
 def test_wrong_config_and_sources_cannot_be_armed(self):
  spec=json.loads((HERE/'packet-spec.json').read_text())
  with tempfile.TemporaryDirectory() as td:
   for values in ({},dict(launch_authorized=False),dict(launch_authorized=True,serialized_assignment='LOCAL_ONLY',config_sha256='wrong',total_seconds=3600,sources_sha256='wrong')):
    with self.assertRaises((ValueError,KeyError)):packet.prepare(Path(td)/'never',values)
 def test_reconciliation_exact_scope_and_baseline(self):
  policy=reconciliation.load_policy();self.assertEqual(policy['quota_used_s'],67694.232)
  self.assertEqual(reconciliation.PREDECESSOR_REF,'pranavagrawai/serving-wire-capacity10-20261023')
  self.assertEqual(policy['predecessor_kernel_id'],140391414)


class CpuRunner(release_runner.Runner):
 def env(self,end):
  env=super().env(end);data=json.loads((self.output/'control-package/control-prepared.json').read_text())
  env.update(LOCAL_CPU_ENTRY=data['argv'][0],LOCAL_CPU_ENTRY_SHA=registration.sha(data['argv'][0]));return env
 def hardware_and_queries(self):self.gpu_uuid='GPU-LOCAL-CPU';self.emit('CPU_ONLY_hardware_substitute')
 def runtime_contract(self):
  out=self.output/'control-package';self.expected=registration.contract(out/'control-prepared.json',json.loads((out/'cpu-sources.json').read_text()),self.config['run_id']);return self.expected
 def make_adapter(self,commands,paths,log,scope,end):
  root=self.output/'cpu-query-inputs';root.mkdir(exist_ok=True)
  (root/'gpu').write_text(resources.GPU_NAME+',0,GPU-LOCAL-CPU\n');(root/'apps').write_text('')
  fixture=HERE.parent/'release-adapters-r6-test-local-20261023/probe_fixture.py'
  commands={name:[sys.executable,str(fixture),'file',str(root/name)] for name in ('gpu','apps')}
  return adapters.Adapters(self.output/'query-artifacts',self.binding,self.gpu_uuid,commands,paths,socket_port(self.config),log,'CPU_JSON_EVENTS',end,scope=scope)

def socket_port(config):return int(config['control']['ready_url'].rsplit(':',1)[1].split('/')[0])

class LifecycleTests(unittest.TestCase):
 def test_cpu_full_wiring_actual200_scoped_release_no_extra_work(self):
  with tempfile.TemporaryDirectory() as td:
   with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
   config=copy.deepcopy(json.loads((HERE/'packet-spec.json').read_text())['config']);config.update(mode='CPU_FIXTURE_ONLY',cwd=str(HERE),run_id='CPU_RELEASE_ONLY',total_seconds=8,cleanup_seconds=2)
   a=config['control'];script=HERE/'tests/cpu_control.py'
   for name in ('prepare','identity','server'):a[name]=dict(argv=[sys.executable,str(script),'serve' if name=='server' else name,str(port)],seconds=2)
   a.update(ready_url=f'http://127.0.0.1:{port}/health',ready_seconds=2,request_seconds=.5,ready_poll_seconds=.02)
   config['release']=dict(seconds=1.5,graceful_seconds=.7,fallback_reserve=.3,probe_seconds=.4,poll=.02);config['verify_seconds']=.5
   result=CpuRunner(config,Path(td)/'output').run();out=Path(td)/'output'
   where=os.environ.get('PACKET_PROOF_DIR')
   if where:
    import shutil
    Path(where).mkdir(parents=True,exist_ok=True);shutil.copytree(out,Path(where)/'cpu-lifecycle',dirs_exist_ok=False)

   events=[json.loads(s) for s in (out/'supervisor.jsonl').read_text().splitlines()]
   self.assertTrue(any(e['event']=='ready' for e in events),result);self.assertTrue((out/'release-scope.json').exists());self.assertTrue((out/'release.json').exists())
   self.assertEqual(result['surviving_native_group'],[]);self.assertEqual(result['surviving_phase_groups'],[]);self.assertEqual(result['retained_queries'],[])
   self.assertFalse(any('candidate' in e.get('phase','') or 'replay' in e.get('phase','') for e in events))
   # Actual local resource thresholds may fail; no synthetic pass is required.
   release=json.loads((out/'release.json').read_text());self.assertEqual(result['pass_'],release['pass'])
