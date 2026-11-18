"""Real child/group cleanup and actual control.prepare dispatch, CPU only."""
import copy,hashlib,json,os,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import pair_runner,candidate_scope,host_observer
from core import shutdown
HERE=Path(__file__).resolve().parents[1]
class Dispatch(unittest.TestCase):
 def exercise(self,delay):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);cfg=json.loads((HERE/'packet-spec.json').read_text())['config'];cfg.update(mode='CPU_FIXTURE_ONLY',cwd=str(root),total_seconds=20,cleanup_seconds=3)
   runner=pair_runner.Runner(cfg,root/'output')
   try:
    spec=dict(argv=[sys.executable,'-c','import time;time.sleep(60)'],seconds=20)
    observer=runner.start('observer',spec,runner.end);runner.observer=observer
    while shutdown.read_process(observer.pid)['identity']['cmdline_sha256']!=hashlib.sha256(b''.join(s.encode()+b'\0' for s in spec['argv'])).hexdigest():time.sleep(.001)
    runner.observer_identity=shutdown.read_process(observer.pid)['identity']
    server=runner.start('candidate.server',spec,runner.end)
    while shutdown.read_process(server.pid)['identity']['cmdline_sha256']!=runner.observer_identity['cmdline_sha256']:time.sleep(.001)
    src={str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    binding,scope=candidate_scope.bind(server,spec['argv'],cfg['run_id'],src,host_observer.bind(cfg['run_id']))
    runner.server=server;runner.binding=binding;runner.scope=scope
    runner.cleanup([server.pid],deadline=runner.work_end)
    now=time.monotonic();release=dict(start=now-1,end=now,absolute_deadline=runner.work_end,pass_=True,error=None,server_pgid=server.pid)
    (runner.output/'candidate-release.json').write_text(json.dumps(release))
    probes={key:dict(start=now,end=now,eligible=True) for key in ('gpu','apps','port','disk')}
    probes['cgroup']=dict(start=now,end=now,eligible=False,computed_bytes=0,threshold_bytes=140*1024**3,files_stable_during_read=True)
    sample=dict(start=now,end=now,ipc_deadline=runner.work_end,probe_worker_alive=False,ipc_error=None,process=dict(eligible=True,remaining=[]),probes=probes)
    runner.control_admission=dict(sample=sample,admitted_at=now)
    real_open=Path.open
    def delayed(path,*args,**kwargs):
     if path.name=='control.prepare.stderr':time.sleep(delay)
     return real_open(path,*args,**kwargs)
    command=dict(argv=[sys.executable,'-c',"from pathlib import Path;Path('control-dispatched').write_text('yes')"],seconds=5)
    with patch.object(Path,'open',delayed):
     if delay:
      with self.assertRaisesRegex(ValueError,'<=2s'):runner.command('control.prepare',command)
     else:runner.command('control.prepare',command)
    self.assertEqual((root/'control-dispatched').exists(),not bool(delay))
    trace=[json.loads(row) for row in (runner.output/'supervisor.jsonl').read_text().splitlines()]
    spawns=[row for row in trace if row['event']=='spawn' and row.get('phase')=='control.prepare']
    self.assertEqual(len(spawns),0 if delay else 1)
    if spawns:self.assertLessEqual(spawns[0]['t']-now,2)
   finally:
    runner.cleanup(deadline=runner.end);runner.trace.close()
    for handle in runner.handles:handle.close()
 def test_actual_prepare_immediate_and_false_original140_allowed(self):self.exercise(0)
 def test_slow_log_open_prevents_actual_prepare_after_fresh2(self):self.exercise(2.05)
