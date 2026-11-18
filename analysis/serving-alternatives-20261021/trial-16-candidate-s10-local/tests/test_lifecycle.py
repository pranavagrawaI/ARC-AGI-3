import copy,hashlib,json,os,signal,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import candidate_scope,host_observer,candidate_runner,primary_bundle
from core import shutdown
HERE=Path(__file__).resolve().parents[1]
class Lifecycle(unittest.TestCase):
    def test_actual_candidate_cgroup_binding_rejects_unknown_argv(self):
        argv=[sys.executable,'-c','import time;time.sleep(60)'];child=subprocess.Popen(argv,start_new_session=True);pidfd=os.pidfd_open(child.pid)
        try:
            # Await exec without assuming Popen return has already installed argv.
            end=time.monotonic()+2
            wanted=hashlib.sha256(b''.join(os.fsencode(s)+b'\0' for s in argv)).hexdigest()
            while shutdown.read_process(child.pid)['identity']['cmdline_sha256']!=wanted:
                self.assertLess(time.monotonic(),end);time.sleep(.001)
            source={str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
            binding,scope=candidate_scope.bind(child,argv,'CPU',source,host_observer.bind('CPU'))
            self.assertEqual(binding['pgid'],child.pid);self.assertEqual(scope['observer']['cgroup'],binding['members'][child.pid]['cgroup'])
            candidate_scope.check(scope,binding,{key:Path(value) for key,value in scope['paths'].items()})
            with self.assertRaises(shutdown.UnsafeIdentity):candidate_scope.bind(child,[*argv,'wrong'],'CPU',source,host_observer.bind('CPU'))
        finally:signal.pidfd_send_signal(pidfd,signal.SIGKILL);child.wait(timeout=2);os.close(pidfd)
    def test_late_cleanup_cannot_start_resource_worker_or_qualify(self):
        runner=object.__new__(candidate_runner.Runner);runner.server=type('Server',(),dict(pid=123))();runner.binding={};runner.work_end=time.monotonic()+5
        runner.config=dict(final_resource_seconds=.02);runner.output=Path(tempfile.mkdtemp());runner.scope={};events=[]
        runner.emit=lambda event,**fields:events.append(dict(event=event,**fields))
        def delayed(*args,**kwargs):time.sleep(.03)
        runner.cleanup=delayed
        try:
            with patch('candidate_runner.shutdown.owned',return_value={}),patch('candidate_runner.resource_observe.observe',side_effect=AssertionError('late dispatch')):
                with self.assertRaisesRegex(ValueError,'cleanup late'):runner.finish_candidate()
            receipt=json.loads((runner.output/'candidate-release.json').read_text());self.assertFalse(receipt['pass_']);self.assertEqual(receipt['samples'],[])
        finally:
            import shutil;shutil.rmtree(runner.output)
    def test_primary_contains_failure_and_is_independently_hashed(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'run/output';root.mkdir(parents=True)
            (root/'result.json').write_text(json.dumps(dict(pass_=False,error='ORIGINAL_ERROR',cleanup_errors=['SEPARATE_CLEANUP_ERROR'])))
            config=dict(run_id='CPU');(root.parent/'config.json').write_text(json.dumps(config));(root.parent/'assignment.json').write_text('{}')
            receipt=primary_bundle.write(root,config,time.monotonic()+2);raw=Path(receipt['path']).read_bytes();value=primary_bundle.verify(raw)
            self.assertIn('candidate-scope.json',value['missing']);self.assertFalse(value['capacity_or_rate_qualification'])
            import base64
            actual=json.loads(base64.b64decode(value['files']['result.json']['base64']))
            self.assertEqual(actual['error'],'ORIGINAL_ERROR');self.assertEqual(actual['cleanup_errors'],['SEPARATE_CLEANUP_ERROR'])
            value['files']['result.json']['sha256']='wrong'
            with self.assertRaisesRegex(ValueError,'hash'):primary_bundle.verify(json.dumps(value).encode())
            with self.assertRaises(FileExistsError):primary_bundle.write(root,config,time.monotonic()+2)
