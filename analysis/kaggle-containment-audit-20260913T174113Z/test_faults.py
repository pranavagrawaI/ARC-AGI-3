import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import accelerator
import audit_remote
import probe
import runner

CSV = 'NVIDIA RTX PRO 6000 Blackwell Server Edition, GPU-12345678-1234-1234-1234-123456789abc, 580.1, 97280 MiB\n'

class FaultTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
    def tearDown(self):
        for p in self.base.rglob('*'):
            if p.is_dir() and not p.is_symlink(): p.chmod(0o755)
        self.temp.cleanup()
    def test_gpu_parser_rejects_malformed_no_gpu_multi_gpu_wrong_name(self):
        for text in ('', 'RTX PRO 6000\n', CSV+CSV, CSV.replace('RTX PRO 6000','T4'), CSV.replace('580.1','N/A'), CSV.replace('97280','0'), CSV.replace('GPU-12345678','bad-uuid')):
            with self.subTest(text=text), self.assertRaises(ValueError):
                accelerator.parse_device({'returncode':0,'stdout':text})
        self.assertEqual(accelerator.parse_device({'returncode':0,'stdout':CSV})['device_count'],1)
    def test_gpu_parser_rejects_timeout_output_limit_even_with_valid_csv(self):
        for status in ('timeout','output_limit','unknown'):
            with self.assertRaises(ValueError): accelerator.parse_device({'returncode':0,'status':status,'stdout':CSV})
    def test_loader_refuses_writable_directory_and_never_copies_caller_env(self):
        self.base.chmod(0o777)
        with patch.dict(os.environ, {'LD_LIBRARY_PATH':'/untrusted','SYNTHETIC_SECRET':'canary'}):
            env, facts = accelerator.loader_environment((str(self.base),))
        self.assertNotIn('LD_LIBRARY_PATH',env)
        self.assertNotIn('SYNTHETIC_SECRET',env)
        self.assertFalse(facts[0]['accepted'])
    def test_bound_rejects_invalid_limits(self):
        for kw in ({'timeout':0},{'output_cap':0},{'output_cap':-1}):
            with self.assertRaises(ValueError): probe.bounded(['not-run'],**kw)
    def test_empty_directory_not_whitelisted(self):
        root=self.base/'root'
        manifest=probe.rootfs(root,'def run(): return 1','candidate',[],require_root=False)
        root.chmod(0o755); (root/'unknown').mkdir(mode=0o555)
        with self.assertRaisesRegex(RuntimeError,'DIRECTORY_ALLOWLIST'): probe.validate_tree(root,manifest,require_root=False)
    def test_runner_expected_parent_rejects_setup_race(self):
        r=probe.bounded([sys.executable,'-I','-S','-B',str(probe.RUNNER),'/not-read','2147483647'])
        self.assertNotEqual(r['returncode'],0)
        self.assertIn('PARENT_CHANGED',r['stderr'])
        self.assertFalse(probe.valid_seal(r['stderr']))
    def test_bounded_kills_descendant_after_session_leader_exit(self):
        libc=ctypes.CDLL(None); prior=ctypes.c_int()
        self.assertEqual(libc.prctl(37,ctypes.byref(prior),0,0,0),0)
        self.assertEqual(libc.prctl(36,1,0,0,0),0)
        child=None
        try:
            # One bounded disposable descendant; no resource bomb.
            code='import os,time\npid=os.fork()\nif pid:\n print(pid,flush=True); os._exit(0)\ntime.sleep(20)'
            receipt=probe.bounded([sys.executable,'-c',code],timeout=.15)
            child=int(receipt['stdout'].strip())
            deadline=time.monotonic()+2
            reaped=False
            while time.monotonic()<deadline:
                pid,status=os.waitpid(child,os.WNOHANG)
                if pid:
                    reaped=True
                    self.assertTrue(os.WIFSIGNALED(status)); self.assertEqual(os.WTERMSIG(status),9)
                    child=None; break
                time.sleep(.01)
            self.assertTrue(reaped,'descendant survived exited leader')
        finally:
            if child:
                try: os.kill(child,9)
                except ProcessLookupError: pass
                os.waitpid(child,0)
            libc.prctl(36,prior.value,0,0,0)
    @unittest.skipUnless(os.geteuid()==0,'mapped-root caller-chain test')
    def test_bind_listen_and_socket_acquisition_failures_close_actual_descriptors(self):
        for stage in ('create','bind','listen'):
            acquired=[]; sockets=[]
            original_open=os.open; original_socket=socket.socket
            def tracked_open(path,flags,*a,**k):
                fd=original_open(path,flags,*a,**k)
                if str(path).startswith('/tmp/containment-q-') and flags & os.O_DIRECTORY: acquired.append(fd)
                return fd
            class Listener:
                def __init__(self): self.sock=original_socket(); self.closed=False; sockets.append(self)
                def bind(self,*args):
                    if stage=='bind': raise OSError('INJECTED_BIND')
                    return self.sock.bind(*args)
                def listen(self,*args): raise OSError('INJECTED_LISTEN')
                def close(self): self.sock.close(); self.closed=True
            def make_socket(*a,**k):
                if stage=='create': raise OSError('INJECTED_CREATE')
                return Listener()
            with patch.object(probe,'snapshot',return_value={}), patch.object(probe.os,'open',side_effect=tracked_open), patch.object(socket,'socket',side_effect=make_socket):
                self.assertEqual(probe.main(self.base/stage,local=True),2)
            self.assertTrue(acquired)
            for fd in acquired:
                with self.assertRaises(OSError): os.fstat(fd)
            self.assertTrue(all(s.closed and s.sock.fileno()==-1 for s in sockets))
            result=json.loads((self.base/stage/'result.json').read_text())
            self.assertTrue(result['cleanup_complete'])
            self.assertEqual(result['completed_cases'],0)
    @unittest.skipUnless(os.geteuid()==0,'mapped-root caller-chain test')
    def test_manifest_mutation_between_build_and_admission_blocks_exec(self):
        original=probe.rootfs
        def mutate(*a,**kw):
            manifest=original(*a,**kw)
            p=a[0]/'work/candidate.py'; p.chmod(0o644); p.write_text('def run(): return "mutated"'); p.chmod(0o444)
            return manifest
        with patch.object(probe,'snapshot',return_value={}), patch.object(probe,'rootfs',side_effect=mutate), patch.object(probe,'bounded') as execute:
            self.assertEqual(probe.main(self.base/'mutation',local=True),2)
            execute.assert_not_called()
        result=json.loads((self.base/'mutation/result.json').read_text())
        self.assertIn('HASH_MISMATCH',result['error']); self.assertEqual(result['completed_cases'],0)
    @unittest.skipUnless(os.geteuid()==0,'real sealed child required')
    def test_post_execution_manifest_addition_preserves_failed_receipt(self):
        original=probe.bounded
        def mutate(command,**kw):
            receipt=original(command,**kw)
            root=Path(command[-2]); root.chmod(0o755)
            p=root/'unexpected'; p.write_text('synthetic'); p.chmod(0o444)
            return receipt
        with patch.object(probe,'snapshot',return_value={}), patch.object(probe,'bounded',side_effect=mutate):
            self.assertEqual(probe.main(self.base/'postmutation',local=True),2)
        result=json.loads((self.base/'postmutation/result.json').read_text())
        self.assertEqual(result['completed_cases'],1); self.assertEqual(result['failed_cases'],1)
        self.assertIn('ALLOWLIST',result['cases'][0]['rootfs_validation_error'])
    @unittest.skipUnless(os.geteuid()==0,'mapped-root startup fault')
    def test_empty_cases_not_qualified(self):
        with patch.object(probe,'snapshot',return_value={}), patch.object(probe,'cases',return_value=[]):
            self.assertEqual(probe.main(self.base/'empty',local=True),2)
        result=json.loads((self.base/'empty/result.json').read_text())
        self.assertEqual(result['classification'],'INCOMPLETE_QUALIFICATION')
        self.assertEqual(result['unexercised_cases'],16)
    @unittest.skipUnless(os.geteuid()==0,'fault injection past the real root prerequisite')
    def test_independent_sealing_primitive_faults_abort_before_compile(self):
        # Explicit fault fixtures; positive security execution is tested separately.
        root=self.base/'root'; probe.rootfs(root,'def run(): return 1','candidate',[],require_root=False)
        calls=('qualify','arm_parent_death')
        for function in calls:
            with patch.object(runner,function,side_effect=RuntimeError('INJECTED_'+function)), patch.object(runner,'seccomp') as later:
                with self.assertRaisesRegex(RuntimeError,'INJECTED_'): runner.seal(str(root))
                later.assert_not_called()

class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.base=Path(self.temp.name)
        evidence=Path(os.environ.get('AUDIT_PACKAGE_EVIDENCE',str(probe.HERE/'package-attempt2')))
        shutil.copytree(evidence/'qualification',self.base/'qualification')
        self.directory=self.base/'qualification'
        self.lifecycle=json.loads((evidence/'package-lifecycle.json').read_text())
        self.sources=self.lifecycle['asset_sha256']
        self.identity=self.lifecycle['identity']
    def tearDown(self): self.temp.cleanup()
    def audit(self): return audit_remote.audit_directory(self.directory, identity=self.identity, sources=self.sources, local_fixture=True)
    def mutate_result(self,fn):
        path=self.directory/'result.json'; result=json.loads(path.read_text()); fn(result); path.write_text(json.dumps(result))
    def test_real_packaged_receipts_independently_pass(self): self.assertEqual(self.audit()['passed'],16)
    def test_zero_cases_forged_pass_rejected(self):
        self.mutate_result(lambda r:r.update(cases=[],completed_cases=0,passed_cases=0,failed_cases=0,unexercised_cases=16))
        with self.assertRaisesRegex(ValueError,'INCOMPLETE'): self.audit()
    def test_duplicate_case_and_missing_manifest_rejected(self):
        path=self.directory/'candidate-benign-rootfs.json'; path.unlink()
        with self.assertRaises(FileNotFoundError): self.audit()
    def test_receipt_source_and_unknown_status_tampering_rejected(self):
        for changes in ({'source_sha256':{}},{'classification':'PASS'}):
            original=(self.directory/'result.json').read_text()
            self.mutate_result(lambda r:r.update(changes))
            with self.assertRaises(ValueError): self.audit()
            (self.directory/'result.json').write_text(original)
    def test_forged_seal_and_output_tampering_rejected(self):
        original=(self.directory/'result.json').read_text()
        for key,value in (('stderr','{}'),('stdout','{"qualified":true}'),('rootfs_unchanged',False)):
            self.mutate_result(lambda r:r['cases'][0].update({key:value}))
            with self.assertRaises((ValueError,KeyError)): self.audit()
            (self.directory/'result.json').write_text(original)
    def test_failure_and_success_exact_version_full_audit(self):
        notebook=b'explicit local metadata fixture, not provider evidence'
        build={'kernel':'fixture/test','version':7,'image':'fixture-image','identity':self.identity,'assets':self.sources,'notebook_sha256':hashlib.sha256(notebook).hexdigest()}
        status={'returncode':0,'status':'COMPLETE','stdout':'fixture/test/7 has status "KernelWorkerStatus.COMPLETE"\n'}
        server={'id':'fixture/test','current_version_number':7,'is_private':True,'enable_internet':False,'machine_shape':'NvidiaRtxPro6000','docker_image':'fixture-image',**{k:[] for k in ('dataset_sources','kernel_sources','competition_sources','model_sources')}}
        for failure in (False,True):
            if failure:
                self.mutate_result(lambda r:r.update(classification='UNSUPPORTED_OR_QUALIFICATION_FAILED',error='EXPLICIT_FIXTURE_ABORT',cases=[],completed_cases=0,passed_cases=0,failed_cases=0,unexercised_cases=16))
            report=audit_remote.audit_remote(self.directory,build,status,server,notebook)
            self.assertEqual(report['version'],7); self.assertEqual(report['qualified'],not failure)
            for tamper in ({'returncode':1},{'stdout':'other/kernel/2 has status "KernelWorkerStatus.COMPLETE"\n'},{'status':'RUNNING'}):
                with self.assertRaises(ValueError): audit_remote.audit_remote(self.directory,build,{**status,**tamper},server,notebook)
            with self.assertRaises(ValueError): audit_remote.audit_remote(self.directory,build,status,{**server,'current_version_number':8},notebook)
            with self.assertRaises(ValueError): audit_remote.audit_remote(self.directory,build,status,server,b'drift')

if __name__=='__main__': unittest.main(verbosity=2)
