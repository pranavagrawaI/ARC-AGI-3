import hashlib,json,os,shutil,sys,tempfile,time,unittest
from core import shutdown
from pathlib import Path
from unittest.mock import patch
import candidate_runner,candidate_resources,outcomes,resources
HERE=Path(__file__).resolve().parents[1]
class Pipeline(unittest.TestCase):
    def config(self):
        c=json.loads((HERE/'packet-spec.json').read_text())['config'];c.update(mode='CPU_FIXTURE_ONLY',cwd=str(HERE),run_id='CPU_ONLY_PIPELINE_NO_AUTHORITY',total_seconds=20,cleanup_seconds=3,verify_seconds=2,primary_seconds=2)
        script=str(HERE/'tests/cpu_candidate.py')
        for phase in ('prepare','identity','server','replay','qualify'):
            c['candidate'][phase]=dict(argv=[sys.executable,script,'start' if phase=='server' else phase],seconds=3)
        c['candidate'].update(ready_seconds=3,request_seconds=1,ready_poll_seconds=.05)
        c['corpus_prepare']=dict(argv=[sys.executable,script,'corpus'],seconds=2)
        return c
    def run_fixture(self,root,fail=False,verifier=None):
        config=self.config();(root/'config.json').write_text(json.dumps(config));(root/'assignment.json').write_text(json.dumps(dict(local_only=True,launch_authorized=False)))
        if fail:config['candidate']['prepare']['env']={'CPU_FAIL':'1'}
        class Cpu(candidate_runner.Runner):
            def hardware_and_queries(self):self.gpu_uuid='GPU-CPU';self.emit('EXPLICIT_CPU_HARDWARE_SUBSTITUTE')
            def start(self,name,spec,end):
                child=super().start(name,spec,end)
                if name=='observer':
                    # Fixture precondition only: bound wait for the declared exec,
                    # before the unchanged runtime takes its first identity snapshot.
                    wanted=hashlib.sha256(b''.join(os.fsencode(value)+b'\0' for value in spec['argv'])).hexdigest()
                    ready_end=min(end,time.monotonic()+2)
                    while True:
                        value=shutdown.read_process(child.pid)
                        if value is not None and value['identity']['cmdline_sha256']==wanted:break
                        if child.poll() is not None or time.monotonic()>=ready_end:raise RuntimeError('CPU fixture observer declared exec precondition')
                        time.sleep(.001)
                return child
        original=candidate_resources.Adapters
        class CpuQueries(original):
            def __init__(self,*args,**kwargs):
                kwargs.update(commands=dict(gpu=[sys.executable,'-c',"print('NVIDIA RTX PRO 6000 Blackwell Server Edition, 0, GPU-CPU')"],apps=[sys.executable,'-c',"print('',end='')"]),execution_policy='LOCAL_CPU_ONLY')
                super().__init__(*args,**kwargs)
        # Only final package/1500s outcome is substituted; actual entry/lifecycle,
        # required observer, bounded query containment, port and cleanup run.
        proof=outcomes.classify(dict(integrity_complete=True,identity_bound=True,cleanup_complete=True,within_arm_valid=True,U_resp=1))
        with patch('candidate_runner.candidate_resources.Adapters',CpuQueries),patch('adapters.shutil.disk_usage',return_value=type('Usage',(),dict(free=120*1024**3))()),patch('candidate_runner.evidence.check',side_effect=verifier if verifier is not None else lambda *args,**kwargs:proof):
            value=Cpu(config,root/'output').run()
        if os.environ.get('PACKET_PROOF_DIR'):
            target=Path(os.environ['PACKET_PROOF_DIR'])/('pipeline-failure' if fail else 'pipeline-success');shutil.copytree(root,target)
        return value
    def test_real_candidate_only_process_health_queries_observer_cleanup(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'run';root.mkdir();value=self.run_fixture(root)
            self.assertTrue(value['pass_'],value);self.assertEqual(value['surviving_phase_groups'],[]);self.assertEqual(value['retained_queries'],[])
            events=[json.loads(line) for line in (root/'output/supervisor.jsonl').read_text().splitlines()]
            spawns=[row['phase'] for row in events if row['event']=='spawn']
            self.assertEqual(spawns.count('candidate.prepare'),1);self.assertFalse(any(name.startswith('control') for name in spawns));self.assertTrue(value['candidate_health200'])
            sample=json.loads((root/'output/candidate-release.json').read_text())['samples'][-1]
            self.assertEqual(set(sample['probes']),{'gpu','apps','cgroup','disk','port'});self.assertTrue(sample['process']['eligible']);self.assertFalse(sample['probe_worker_alive'])
            self.assertTrue((Path(td)/'000-serving-candidate16-primary.json').exists())
    def test_preparation_failure_retained_with_no_following_server(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'run';root.mkdir();value=self.run_fixture(root,fail=True)
            self.assertFalse(value['pass_'],value);self.assertIn('exit 9',value['error']);self.assertEqual(value['surviving_phase_groups'],[])
            events=[json.loads(line) for line in (root/'output/supervisor.jsonl').read_text().splitlines()]
            self.assertFalse(any(row['event']=='spawn' and row['phase']=='candidate.server' for row in events));self.assertEqual(value['status'],'INVALID_INCONCLUSIVE')
