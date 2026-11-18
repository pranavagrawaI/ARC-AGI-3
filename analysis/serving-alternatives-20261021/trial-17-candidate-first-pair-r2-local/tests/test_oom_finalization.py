"""Real CPU candidate lifecycle; deliberate OOM stops before either replay/control."""
import hashlib,json,os,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import pair_runner,candidate_resources,host_observer
from core import shutdown
HERE=Path(__file__).resolve().parents[1]
class OOM(unittest.TestCase):
 def exercise(self,injection):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td)/'run';root.mkdir();cfg=json.loads((HERE/'packet-spec.json').read_text())['config']
   cfg.update(mode='CPU_FIXTURE_ONLY',cwd=str(HERE),run_id='CPU_ONLY_OOM_NO_AUTHORITY',total_seconds=20,cleanup_seconds=3,verify_seconds=2,primary_seconds=2)
   script=str(HERE/'tests/cpu_candidate.py')
   for phase in ('prepare','identity','server','replay','qualify'):cfg['candidate'][phase]=dict(argv=[sys.executable,script,'start' if phase=='server' else phase],seconds=3)
   cfg['candidate'].update(ready_seconds=3,request_seconds=1,ready_poll_seconds=.05)
   cfg['control']['prepare']=dict(argv=[sys.executable,'-c',"raise RuntimeError('FORBIDDEN_SECOND_LOAD_AFTER_OOM')"],seconds=2)
   cfg['corpus_prepare']=dict(argv=[sys.executable,script,'corpus'],seconds=2)
   (root/'config.json').write_text(json.dumps(cfg));(root/'assignment.json').write_text('{}')
   class CPU(pair_runner.Runner):
    def hardware_and_queries(self):self.gpu_uuid='GPU-CPU';self.emit('EXPLICIT_CPU_NO_GPU_SUBSTITUTE')
    def start(self,name,spec,end):
     child=super().start(name,spec,end)
     if name=='observer':
      wanted=hashlib.sha256(b''.join(os.fsencode(v)+b'\0' for v in spec['argv'])).hexdigest();until=time.monotonic()+2
      while shutdown.read_process(child.pid)['identity']['cmdline_sha256']!=wanted:
       if time.monotonic()>until:raise RuntimeError('CPU exec precondition')
       time.sleep(.001)
     return child
   Original=candidate_resources.Adapters
   class CPUQueries(Original):
    def __init__(self,*args,**kwargs):
     kwargs.update(commands=dict(gpu=[sys.executable,'-c',"print('NVIDIA RTX PRO 6000 Blackwell Server Edition, 0, GPU-CPU')"],apps=[sys.executable,'-c',"print('',end='')"]),execution_policy='LOCAL_CPU_ONLY');super().__init__(*args,**kwargs)
   snapshot=host_observer.snapshot;increased=False
   def oom(spec,label):
    nonlocal increased
    row=snapshot(spec,label)
    if label==injection:increased=True
    if increased:
     row['oom_events']['oom']+=1;row['raw']['events']=''.join(f'{k} {v}\n' for k,v in row['oom_events'].items())
    return row
   with patch('host_observer.snapshot',side_effect=oom),patch('candidate_resources.Adapters',CPUQueries),patch('adapters.shutil.disk_usage',return_value=type('Usage',(),dict(free=120*1024**3))()):value=CPU(cfg,root/'output').run()
   self.assertFalse(value['pass_']);self.assertIsNone(value['ratio']);self.assertIn('OOM counter increased',value['error'])
   self.assertEqual(value['cleanup_errors'],[]);self.assertEqual(value['surviving_phase_groups'],[]);self.assertEqual(value['retained_queries'],[])
   output=root/'output';self.assertTrue(json.loads((output/'host-series.result.json').read_text())['pass_'])
   self.assertTrue(json.loads((output/'candidate-release.json').read_text())['pass_'])
   events=[json.loads(x) for x in (output/'supervisor.jsonl').read_text().splitlines()]
   self.assertFalse(any(r['event']=='spawn' and r['phase']=='control.prepare' for r in events))
   if injection=='candidate.replay.before':self.assertFalse(any(r['event']=='spawn' and r['phase'] in ('candidate.replay','candidate.qualify') for r in events))
   self.assertFalse((output/'control-admission.json').exists())
   proof=os.environ.get('PACKET_PROOF_DIR')
   if proof:
    import shutil
    target=Path(proof)/('oom-'+injection);shutil.copytree(root,target)
 def test_oom_before_replay_preserves_finalization_and_stops_workload(self):self.exercise('candidate.replay.before')
 def test_oom_during_cleanup_never_dispatches_second_control(self):self.exercise('candidate.cleaned')
