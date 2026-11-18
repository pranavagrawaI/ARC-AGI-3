import ast
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import probe
import runner

class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
    def tearDown(self):
        for p in [self.base, *self.base.rglob('*')]:
            if p.is_dir() and not p.is_symlink():
                p.chmod(0o755)
        self.tmp.cleanup()
    def build(self, mode='candidate'):
        root = self.base/'root'
        manifest = probe.rootfs(root, 'def run(x): return x', mode, [1], require_root=False)
        return root, manifest
    def test_candidate_exact_allowlist(self):
        root, manifest = self.build()
        self.assertNotIn('work/prefix.json', manifest)
        self.assertNotIn('work/future.json', manifest)
        self.assertEqual(json.loads((root/'work/args.json').read_text()), {'mode':'candidate','args':[1]})
        self.assertFalse(any('site-packages' in p or '/proc/' in p or '/dev/' in p for p in manifest))
        self.assertTrue(all((root/p).is_file() for p in manifest))
    def test_inspection_exact_prefix(self):
        root, manifest = self.build('inspection')
        self.assertEqual(json.loads((root/'work/prefix.json').read_text()), {'visible':['SYNTHETIC_VISIBLE_PREFIX']})
        self.assertIn('work/prefix.json', manifest)
    def test_bad_protocol_modes(self):
        for mode in ('', 'candidate/../inspection', 'other'):
            with self.assertRaises(ValueError):
                probe.rootfs(self.base/'root', '', mode, [], require_root=False)
    def test_bad_protocol_args(self):
        for args in ({}, [float('nan')], ['X'*17000]):
            with self.assertRaises(ValueError):
                probe.rootfs(self.base/'root', '', 'candidate', args, require_root=False)
    def test_traversal_names(self):
        for name in ('/a', '../a', 'a/../b', 'a//b', './a', ''):
            with self.assertRaises(ValueError): probe.validate_relative(name)
    def test_reject_symlink(self):
        root, manifest = self.build(); root.chmod(0o755)
        (root/'link').symlink_to('/etc')
        with self.assertRaisesRegex(RuntimeError, 'SYMLINK'): probe.validate_tree(root, manifest, require_root=False)
    def test_reject_hardlink(self):
        root, manifest = self.build(); root.chmod(0o755)
        os.link(root/'work/args.json', root/'hardlink')
        with self.assertRaisesRegex(RuntimeError, 'HARDLINK'): probe.validate_tree(root, manifest, require_root=False)
    def test_reject_unknown_file(self):
        root, manifest = self.build(); root.chmod(0o755)
        (root/'outside').write_text('synthetic')
        with self.assertRaisesRegex(RuntimeError, 'ALLOWLIST'): probe.validate_tree(root, manifest, require_root=False)
    def test_reject_changed_manifested_file(self):
        root, manifest = self.build()
        p=root/'work/candidate.py'; p.chmod(0o644); p.write_text('def run(): return 2'); p.chmod(0o444)
        with self.assertRaisesRegex(RuntimeError, 'HASH_MISMATCH'): probe.validate_tree(root, manifest, require_root=False)
    def test_reject_writable_tree(self):
        root, manifest = self.build(); (root/'work').chmod(0o777)
        with self.assertRaisesRegex(RuntimeError, 'WRITABLE'): probe.validate_tree(root, manifest, require_root=False)
    def test_reject_special_file(self):
        root, manifest = self.build(); root.chmod(0o755); os.mkfifo(root/'fifo')
        with self.assertRaisesRegex(RuntimeError, 'SPECIAL'): probe.validate_tree(root, manifest, require_root=False)
    def test_unsupported_platform(self):
        with patch.object(runner.platform, 'machine', return_value='aarch64'):
            with self.assertRaisesRegex(RuntimeError, 'UNSUPPORTED'): runner.qualify()
    def test_wrong_or_multiple_gpu_fails_before_rootfs(self):
        for index,acc in enumerate((None, {'returncode':0,'stdout':'NVIDIA T4\n'}, {'returncode':0,'stdout':'RTX PRO 6000\nRTX PRO 6000\n'})):
            out=self.base/str(index)
            with patch.object(probe,'snapshot',return_value={'accelerator':acc}), patch.object(runner,'qualify'), patch.object(probe,'rootfs') as build:
                self.assertEqual(probe.main(out),2)
                build.assert_not_called()
            result=json.loads((out/'result.json').read_text())
            self.assertIn('TARGET_ACCELERATOR_NOT_OBSERVED',result['error'])
    def test_discovery_exact_scrubbed_path_absolute_opt_bin(self):
        real_run = subprocess.run
        fake = self.base/'nvidia-smi'
        fake.write_text('#!/bin/sh\nprintf "RTX PRO 6000, synthetic-uuid, synthetic-driver, 1 MiB\\n"\n')
        fake.chmod(0o755)
        def execute(command, **kwargs):
            self.assertEqual(command[0], '/opt/bin/nvidia-smi')
            self.assertEqual(kwargs['env'], {'PATH':'/usr/local/bin:/usr/bin:/bin','LANG':'C'})
            self.assertEqual(kwargs['stdin'], subprocess.DEVNULL)
            self.assertTrue(kwargs['close_fds'])
            return real_run([str(fake), *command[1:]], **kwargs)
        with patch.dict(os.environ, {'PATH':'/usr/local/bin:/usr/bin:/bin','LANG':'C'}, clear=True), patch.object(Path,'is_file',return_value=True), patch.object(probe.os,'access',return_value=True), patch.object(probe.subprocess,'run',side_effect=execute):
            result=probe.discover_accelerator()
        self.assertEqual(result['path'],'/opt/bin/nvidia-smi')
        self.assertEqual(result['returncode'],0)
        self.assertIn('RTX PRO 6000',result['stdout'])
        self.assertEqual(result['stderr'],'')
    def test_discovery_unavailable_fails_closed(self):
        with patch.object(Path,'is_file',return_value=False), patch.object(probe.subprocess,'run') as run:
            result=probe.discover_accelerator()
            run.assert_not_called()
        self.assertEqual(result['error'],'NVIDIA_SMI_UNAVAILABLE')
        with patch.object(probe,'snapshot',return_value={'accelerator':result}), patch.object(runner,'qualify'), patch.object(probe,'rootfs') as build:
            self.assertEqual(probe.main(self.base/'unavailable'),2)
            build.assert_not_called()
    def test_discovery_nonexecutable_not_run(self):
        with patch.object(Path,'is_file',return_value=True), patch.object(probe.os,'access',return_value=False), patch.object(probe.subprocess,'run') as run:
            self.assertEqual(probe.discover_accelerator()['error'],'NVIDIA_SMI_UNAVAILABLE')
            run.assert_not_called()
    def test_discovery_execution_failures_preserved_and_closed(self):
        for i,value in enumerate((subprocess.CompletedProcess([],9,'RTX PRO 6000\n','driver failure'), OSError('synthetic exec failure'), subprocess.TimeoutExpired('/opt/bin/nvidia-smi',10))):
            kwargs={'side_effect':value} if isinstance(value,Exception) else {'return_value':value}
            with patch.object(Path,'is_file',return_value=True), patch.object(probe.os,'access',return_value=True), patch.object(probe.subprocess,'run',**kwargs) as run:
                result=probe.discover_accelerator()
                self.assertEqual(run.call_count,1)
            self.assertNotEqual(result['returncode'],0)
            if i==0: self.assertEqual(result['stderr'],'driver failure')
            else: self.assertIn(type(value).__name__,result['error'])
            with patch.object(probe,'snapshot',return_value={'accelerator':result}), patch.object(runner,'qualify'), patch.object(probe,'rootfs') as build:
                self.assertEqual(probe.main(self.base/('failure'+str(i))),2)
                build.assert_not_called()
    def test_discovery_wrong_accelerator_actual_execution(self):
        fake=self.base/'nvidia-smi'; fake.write_text('#!/bin/sh\nprintf "NVIDIA T4\\n"\n'); fake.chmod(0o755)
        with patch.object(probe,'NVIDIA_SMI_PATHS',(str(fake),)):
            result=probe.discover_accelerator()
        self.assertEqual(result['stdout'],'NVIDIA T4\n')
        with patch.object(probe,'snapshot',return_value={'accelerator':result}), patch.object(runner,'qualify'), patch.object(probe,'rootfs') as build:
            self.assertEqual(probe.main(self.base/'wrong'),2)
            build.assert_not_called()
    def test_unprivileged_failclosed(self):
        with patch.object(runner.os, 'geteuid', return_value=1000):
            with self.assertRaisesRegex(RuntimeError, 'ROOT_REQUIRED'): runner.qualify()
    def test_seal_failure_precedes_candidate_compile(self):
        root, _ = self.build()
        (root/'work/candidate.py').chmod(0o644)
        (root/'work/candidate.py').write_text('this is invalid syntax !!!!')
        with patch.object(sys, 'argv', ['runner',str(root)]), patch.object(runner, 'seal', side_effect=RuntimeError('SEAL_FAILED')):
            with self.assertRaisesRegex(RuntimeError, 'SEAL_FAILED'): runner.main()
    def test_bad_request_no_candidate_execution(self):
        root, _ = self.build()
        (root/'work/args.json').chmod(0o644)
        (root/'work/args.json').write_text('{"mode":"bad", "args":[]}')
        with patch.object(sys, 'argv', ['runner',str(root)]), patch.object(runner, 'seal') as seal:
            with self.assertRaisesRegex(RuntimeError, 'BAD_PROTOCOL'): runner.main()
            seal.assert_not_called()
    def test_filter_no_dangerous_syscalls(self):
        denied={41,42,56,57,58,59,101,105,106,126,157,161,165,272,308,310,311,317,321,322,425,435,438,0x40000027}
        self.assertFalse(denied & runner.ALLOWED)
        self.assertTrue({0,1,3,60,231,257} <= runner.ALLOWED)
    def test_syscall_filter_real_process(self):
        code = 'import runner,ctypes; runner.seccomp(); c=ctypes.CDLL(None,use_errno=True); r=c.syscall(41,2,1,0); print(r,ctypes.get_errno())'
        p = subprocess.run([sys.executable,'-B','-c',code],cwd=probe.HERE,capture_output=True,text=True,timeout=5)
        self.assertEqual((p.returncode,p.stdout),(0,'-1 1\n'),p.stderr)
    def test_bounded_real_timeout(self):
        r=probe.bounded([sys.executable,'-c','import time; time.sleep(10)'],timeout=.1)
        self.assertEqual((r['status'],r['returncode'],r['reaped']),('timeout',-9,True))
    def test_bounded_real_output(self):
        r=probe.bounded([sys.executable,'-c','print("x"*90000)'],output_cap=1024)
        self.assertEqual((r['status'],r['returncode'],r['reaped']),('output_limit',-9,True))
        self.assertLessEqual(len(r['stdout'])+len(r['stderr']),1025)
    def test_forged_pass_rejected(self):
        receipt={'status':'completed','returncode':0,'reaped':True,'stdout':'PASS\n{"qualified":true}\n','stderr':'{}'}
        self.assertFalse(probe.assess(receipt,{'qualified':True}))
    def test_limit_without_seal_rejected(self):
        for status in ('timeout','output_limit'):
            self.assertFalse(probe.assess({'status':status,'returncode':-9,'reaped':True,'stdout':'','stderr':''},None,status))
    def test_prlimit_self_and_other_pid_denied(self):
        import resource
        victim=subprocess.Popen([sys.executable,'-c','import time; time.sleep(10)'],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        before=resource.prlimit(victim.pid,resource.RLIMIT_NOFILE)
        try:
            code='import runner,ctypes,json; runner.seccomp(); libc=ctypes.CDLL(None,use_errno=True); out=[]\nfor pid in (0,'+str(victim.pid)+'):\n ctypes.set_errno(0); rc=libc.syscall(302,pid,7,0,0); out.append([rc,ctypes.get_errno()])\nprint(json.dumps(out))'
            p=subprocess.run([sys.executable,'-B','-c',code],cwd=probe.HERE,capture_output=True,text=True,timeout=5)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertEqual(json.loads(p.stdout),[[-1,1],[-1,1]])
            self.assertEqual(resource.prlimit(victim.pid,resource.RLIMIT_NOFILE),before)
        finally:
            victim.terminate(); victim.wait(timeout=5)
    def test_missing_seal_rejected(self):
        receipt={'status':'completed','returncode':0,'reaped':True,'stdout':'1','stderr':'{}'}
        self.assertFalse(probe.assess(receipt,1))
    def test_cleanup_failure_blocks_qualification(self):
        result={'classification':'QUALIFIED_SYNTHETIC_PROBE_ONLY','cases':[{'check':True}]*16}
        probe.finalize(result,self.base)
        self.assertEqual(result['classification'],'CLEANUP_FAILED')
        self.assertFalse(result['cleanup_complete'])
    def test_cleanup_scoped_to_owned_path(self):
        result={'classification':'QUALIFIED_SYNTHETIC_PROBE_ONLY','cases':[{'check':True}]*16}
        (self.base/'unrelated').mkdir()
        probe.finalize(result,self.base/'owned-removed')
        self.assertEqual(result['classification'],'QUALIFIED_SYNTHETIC_PROBE_ONLY')
    def test_partial_cases_never_qualified(self):
        result={'classification':'QUALIFIED_SYNTHETIC_PROBE_ONLY','cases':[{'check':True}]*15}
        probe.finalize(result,None)
        self.assertEqual(result['classification'],'INCOMPLETE_QUALIFICATION')
    def test_parent_input_env_fd_not_inherited(self):
        # Real outer process receives synthetic stdin/env/open FD; bounded's
        # fresh exec must discard them BEFORE runner Python starts.
        outer = "import probe,sys; r=probe.bounded([sys.executable,'-c',\"import os,json; print(json.dumps([os.read(0,100).decode(),os.environ.get('SYNTHETIC_ENV')]))\"]); print(r['stdout'],end='')"
        p=subprocess.run([sys.executable,'-B','-c',outer],cwd=probe.HERE,input='STDIN_CANARY',env={**os.environ,'SYNTHETIC_ENV':'ENV_CANARY'},capture_output=True,text=True,timeout=5)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual(json.loads(p.stdout),['',None])
    @unittest.skipUnless(os.geteuid()==0, 'Real chroot needs root; exercised separately with mapped local user namespace')
    def test_real_root_path(self):
        root,_=self.build()
        r=probe.bounded([sys.executable,'-I','-S','-B',str(probe.RUNNER),str(root)])
        self.assertTrue(probe.assess(r,1),r)
    @unittest.skipUnless(os.geteuid()==0, 'Real supervisor-kill test requires mapped root/chroot')
    def test_sealed_child_dies_and_is_reaped_after_supervisor_kill(self):
        import ctypes, signal, time
        root,_=self.build()
        candidate=root/'work/candidate.py'; candidate.chmod(0o644)
        candidate.write_text('def run(*args):\n import time\n time.sleep(20)\n')
        candidate.chmod(0o444)
        libc=ctypes.CDLL(None,use_errno=True)
        prior=ctypes.c_int()
        self.assertEqual(libc.prctl(37,ctypes.byref(prior),0,0,0),0)
        self.assertEqual(libc.prctl(36,1,0,0,0),0)
        code = "import probe,subprocess,sys,json\noriginal=subprocess.Popen\ndef traced(*a,**k):\n p=original(*a,**k)\n seal=p.stderr.readline().decode()\n assert probe.valid_seal(seal),seal\n print(p.pid,flush=True)\n return p\nprobe.subprocess.Popen=traced\nprobe.bounded("+repr([sys.executable,'-I','-S','-B',str(probe.RUNNER),str(root)])+",timeout=25)\n"
        parent=subprocess.Popen([sys.executable,'-B','-c',code],cwd=probe.HERE,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        child=None
        reaped=False
        try:
            import selectors
            with selectors.DefaultSelector() as selector:
                selector.register(parent.stdout,selectors.EVENT_READ)
                self.assertTrue(selector.select(5),'supervisor failed to start')
            line=parent.stdout.readline().strip()
            self.assertTrue(line.isdigit(),line)
            child=int(line)
            parent.kill(); parent.wait(timeout=3)
            deadline=time.monotonic()+2
            while time.monotonic()<deadline:
                got,status=os.waitpid(child,os.WNOHANG)
                if got:
                    reaped=True
                    self.assertTrue(os.WIFSIGNALED(status))
                    self.assertEqual(os.WTERMSIG(status),signal.SIGKILL)
                    break
                time.sleep(.01)
            self.assertTrue(reaped,'sealed child survived supervisor SIGKILL')
        finally:
            if parent.poll() is None: parent.kill(); parent.wait(timeout=3)
            if child and not reaped:
                try: os.kill(child,signal.SIGKILL)
                except ProcessLookupError: pass
                try: os.waitpid(child,0)
                except ChildProcessError: pass
            parent.stdout.close(); parent.stderr.close()
            self.assertEqual(libc.prctl(36,prior.value,0,0,0),0)

if __name__=='__main__': unittest.main(verbosity=2)
