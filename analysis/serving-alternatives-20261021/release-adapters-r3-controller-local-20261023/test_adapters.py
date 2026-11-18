import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest

import adapters,artifacts,capture,registration
from core import ROOT,shutdown

HERE=Path(__file__).resolve().parent


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.owner=capture.owner_identity();self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.raw=self.root/'raw';self.raw.mkdir();self.processes=[];self.bindings=[];self.results=[]

    def tearDown(self):
        for binding in self.bindings:
            shutdown.signal_owned(binding,list(binding['members']),signal.SIGKILL,[],time.monotonic()+1)
        for process in self.processes:process.wait(timeout=2)
        out=os.environ.get('ADAPTER_PROOF_DIR')
        if out:
            p=Path(out)/self._testMethodName;p.mkdir(parents=True)
            import shutil
            shutil.copytree(self.raw,p/'raw')
            (p/'RESULT.json').write_text(json.dumps(self.results,indent=2)+'\n')
        self.tmp.cleanup()

    def command(self,*args,**kwargs):return capture.command(*args,owner=self.owner,**kwargs)

    def start(self):
        alias=self.root/'python';alias.symlink_to(sys.executable)
        entry=self.root/'frontend'
        entry.write_text(f'#!{alias}\nimport sys\nsys.path.insert(0,{str(ROOT)!r})\nimport native_fixture\nnative_fixture.frontend(sys.argv[1])\n')
        entry.chmod(0o700);log=self.root/'native.jsonl'
        prepared=self.root/'prepared.json';prepared.write_text(json.dumps(dict(python=str(alias),argv=[str(entry),str(log)])))
        sources={str(p):registration.sha(p) for p in [entry,ROOT/'native_fixture.py',HERE/'fixture_wrapper.py',HERE/'registration.py']}
        sourcepath=self.root/'sources.json';sourcepath.write_text(json.dumps(sources))
        contract=registration.contract(prepared,sources,'CPU_ADAPTER_ONLY')
        receipt=self.root/'registered.json'
        env=dict(os.environ,LOCAL_CPU_ENTRY=str(entry),LOCAL_CPU_ENTRY_SHA=registration.sha(entry))
        child=subprocess.Popen([sys.executable,str(HERE/'fixture_wrapper.py'),str(prepared),str(sourcepath),str(receipt),'CPU_ADAPTER_ONLY'],start_new_session=True,env=env)
        self.processes.append(child)
        end=time.monotonic()+3
        while not receipt.exists() and time.monotonic()<end:
            if child.poll() is not None:self.fail('registration wrapper exited')
            time.sleep(.01)
        while time.monotonic()<end:
            if log.exists() and any(json.loads(line)['event']=='frontend_ready' for line in log.read_text().splitlines()):break
            time.sleep(.01)
        else:self.fail('CPU readiness barrier did not complete; registration alone is not readiness')
        # Exclusive-file close is the completion boundary, not mere path existence.
        while True:
            try:binding=registration.bind(child,receipt,contract);break
            except json.JSONDecodeError:
                if time.monotonic()>=end:raise
                time.sleep(.01)
        self.bindings.append(copy.deepcopy(binding))
        self.results.append(dict(registration=json.loads(receipt.read_text())))
        return child,binding,contract,receipt,log

    def build(self,binding,log,port):
        old=HERE.parent/'trial-10-exact-wire-local/remote-readback/priority-output/serving-wire-capacity10/output/control-release.json'
        raw=json.loads(old.read_text())['raw'];uuid=raw['gpu'].split(',')[2].strip()
        paths={}
        for key in ('meminfo','maximum','current','stat'):
            paths[key]=self.root/key;paths[key].write_text('0' if key=='current' else raw[key])
        paths['disk']=self.root
        gpu=self.root/'gpu';gpu.write_text(f'NVIDIA RTX PRO 6000 Blackwell Server Edition,0,{uuid}\n')
        apps=self.root/'apps';apps.write_text('')
        commands={name:[sys.executable,str(HERE/'probe_fixture.py'),'file',str(path)] for name,path in [('gpu',gpu),('apps',apps)]}
        return adapters.Adapters(self.raw,binding,uuid,commands,paths,port,log,'CPU_JSON_EVENTS',time.monotonic()+4)

    def test_prepared_registration_and_real_cpu_adapter_release(self):
        child,binding,contract,receipt,log=self.start()
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        adapter=self.build(binding,log,port)
        result=shutdown.release(binding,adapter.probes(),end=time.monotonic()+3,graceful_seconds=2,fallback_reserve=.5,probe_seconds=.8,poll=.1)
        self.results.append(result)
        complete=[s for s in result['samples'] if 'artifact' in s['probes']['native']]
        self.assertTrue(complete)
        last=complete[-1]
        self.assertTrue(last['process']['eligible'])
        for name in ('gpu','apps','cgroup','port','native'):self.assertTrue(last['probes'][name]['eligible'],name)
        # The actual local filesystem may be below the unchanged100GiB gate.
        self.assertEqual(result['pass'],last['probes']['disk']['eligible'])
        for sample in result['samples']:
            for record in sample['probes'].values():
                command=record.get('command')
                if command:
                    self.assertEqual(command['executed_argv'][0],sys.executable)
                    self.assertEqual(command['intended_argv'],capture.GPU_ARGV if record['name']=='gpu' else capture.APPS_ARGV)
                    for ref in command['artifacts'].values():
                        self.assertEqual(artifacts.verify(self.raw,ref,time.monotonic()+1),ref['complete'])
                        if command['ok']:self.assertTrue(ref['complete'])

    def test_loopback_occupied_and_free_port_plumbing(self):
        _,binding,_,_,log=self.start()
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));sock.listen();port=sock.getsockname()[1]
            adapter=self.build(binding,log,port)
            occupied=adapter.port_probe();self.assertFalse(occupied['eligible']);self.results.append(occupied)
        free=adapter.port_probe();self.assertTrue(free['eligible']);self.results.append(free)

    def test_large_stdout_is_exact_referenced_artifact_not_ipc_body(self):
        result=self.command(capture.GPU_ARGV,[sys.executable,str(HERE/'probe_fixture.py'),'big'],self.raw,time.monotonic()+2)
        self.results.append(result);self.assertTrue(result['ok'])
        ref=result['artifacts']['stdout'];self.assertEqual(ref['bytes'],256*1024)
        self.assertEqual(ref['sha256'],hashlib.sha256(b'x'*(256*1024)).hexdigest())
        self.assertTrue(artifacts.verify(self.raw,ref,time.monotonic()+1))
        self.assertLess(len(json.dumps(result)),65536)

    def test_gpu_execution_refused_and_command_error_preserved(self):
        with self.assertRaisesRegex(ValueError,'CPU execution only'):
            self.command(capture.GPU_ARGV,capture.GPU_ARGV,self.raw,time.monotonic()+1)
        from unittest.mock import patch
        with patch.dict(os.environ,LOCAL_ADAPTER_PROOF='1'):
            with self.assertRaisesRegex(ValueError,'forbidden during local proof'):
                self.command(capture.GPU_ARGV,capture.GPU_ARGV,self.raw,time.monotonic()+1,'FUTURE_REVIEWED_GPU_QUERY')
        result=self.command(capture.APPS_ARGV,[sys.executable,str(HERE/'probe_fixture.py'),'error'],self.raw,time.monotonic()+2)
        self.results.append(result);self.assertFalse(result['ok']);self.assertEqual(result['exit_code'],7)
        self.assertEqual((self.raw/result['artifacts']['stderr']['path']).read_text(),'fixture failure\n')

    def test_wrong_registration_contract_and_identity_refused(self):
        child,binding,contract,receipt,log=self.start()
        wrong=copy.deepcopy(contract);wrong['run_id']='wrong'
        with self.assertRaisesRegex(ValueError,'contract mismatch'):registration.bind(child,receipt,wrong)
        bad=json.loads(receipt.read_text());bad['frontend']['starttime']+=1
        alternate=self.root/'bad.json';alternate.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError,'identity drift'):registration.bind(child,alternate,contract)

    def test_command_timeout_retains_partial_bytes_and_reaps_owned_child(self):
        marker=self.root/'pid'
        result=self.command(capture.GPU_ARGV,[sys.executable,str(HERE/'probe_fixture.py'),'slow',str(marker)],self.raw,time.monotonic()+.2)
        self.results.append(result)
        self.assertFalse(result['ok']);self.assertTrue(result['child_reaped'])
        self.assertFalse(result['artifacts']['stdout']['complete'])
        self.assertEqual((self.raw/result['artifacts']['stdout']['path']).read_bytes(),b'partial\n')

    def test_ordinary_grandchild_denied_on_query_timeout(self):
        marker=self.root/'descendant'
        result=self.command(capture.GPU_ARGV,[sys.executable,str(HERE/'probe_fixture.py'),'ordinary-descendant',str(marker)],self.raw,time.monotonic()+.6)
        self.results.append(result)
        outcome=json.loads(marker.read_text());self.results.append(outcome)
        self.assertEqual(outcome['created'],False);self.assertEqual(outcome['errno'],1)
        self.assertFalse(Path(str(marker)+'.grandchild').exists())
        self.assertFalse(result['ok']);self.assertTrue(result['child_reaped'])
        self.assertTrue(result['containment']['verified']);self.assertIsNone(result['cleanup_error'])
        self.assertIn(b'"created": false',(self.raw/result['artifacts']['stdout']['path']).read_bytes())

    def test_normal_threads_complete_under_inherited_contract(self):
        result=self.command(capture.GPU_ARGV,[sys.executable,str(HERE/'probe_fixture.py'),'thread'],self.raw,time.monotonic()+2)
        self.results.append(result);self.assertTrue(result['ok']);self.assertTrue(result['containment']['verified'])
        self.assertEqual((self.raw/result['artifacts']['stdout']['path']).read_bytes(),b'thread-ok\n')

    def test_unknown_or_drifted_containment_identity_rejected(self):
        result=self.command(capture.GPU_ARGV,[sys.executable,str(HERE/'probe_fixture.py'),'thread'],self.raw,time.monotonic()+2)
        self.assertTrue(result['ok']);self.results.append(result)
        guard=json.loads((self.raw/result['containment']['artifact']['path']).read_text())
        key=json.loads((self.raw/result['intent']['path']).read_text())['process_key']
        self.assertTrue(capture.validate_containment(guard,key))
        for name,value in [('pid',key['pid']+1),('starttime',key['starttime']+1),('arch','aarch64'),('filter_sha256','unknown'),('seccomp_mode',0),('no_new_privs',0),('contract','PDEATHSIG_ONLY')]:
            wrong=dict(guard);wrong[name]=value
            with self.assertRaisesRegex(ValueError,'identity/contract'):capture.validate_containment(wrong,key)
        self.results.append(dict(unknown_identity_and_contract_rejected=True))

    def test_process_syscall_paths_denied_and_clone3_falls_back(self):
        result=self.command(capture.GPU_ARGV,[sys.executable,str(HERE/'probe_fixture.py'),'process-syscalls'],self.raw,time.monotonic()+2)
        self.results.append(result);self.assertTrue(result['ok'])
        attempts=json.loads((self.raw/result['artifacts']['stdout']['path']).read_text())
        self.assertEqual(attempts,{'fork':{'result':-1,'errno':1},'clone_process':{'result':-1,'errno':1},'clone3':{'result':-1,'errno':38}})
        self.results.append(attempts)

    def test_runtime_adapter_refuses_unbound_scope(self):
        with self.assertRaisesRegex(ValueError,'runtime cgroup/native fresh-output scope not bound'):
            adapters.Adapters(self.raw,{},'unknown',{}, {},8001,self.root/'old-log','PINNED_SGLANG_TEXT',time.monotonic()+1,'FUTURE_REVIEWED_GPU_QUERY')
        self.results.append(dict(unbound_runtime_adapter_rejected=True))

    def test_thread_exec_clears_pdeathsig_but_controller_cleans_after_worker_death(self):
        import threading
        marker=self.root/'thread-exec'
        worker=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],start_new_session=True)
        worker_fd=os.pidfd_open(worker.pid);witness={}
        def kill_worker():
            end=time.monotonic()+2
            while time.monotonic()<end:
                try:witness.update(json.loads(marker.read_text()));break
                except (FileNotFoundError,json.JSONDecodeError):time.sleep(.005)
            signal.pidfd_send_signal(worker_fd,signal.SIGKILL)
        thread=threading.Thread(target=kill_worker);thread.start()
        try:
            result=self.command(capture.GPU_ARGV,[sys.executable,str(HERE/'probe_fixture.py'),'thread-exec',str(marker)],self.raw,time.monotonic()+2,cancelled=lambda:worker.poll() is not None)
            thread.join(timeout=2);worker.wait(timeout=1)
            self.results.extend([dict(post_exec=witness,worker_exit=worker.returncode),result])
            self.assertEqual(witness['pdeathsig'],0);self.assertEqual(witness['no_new_privs'],1);self.assertEqual(witness['seccomp'],2)
            key=json.loads((self.raw/result['intent']['path']).read_text())['process_key']
            self.assertEqual(witness['pid'],key['pid']);self.assertEqual(witness['starttime'],key['starttime'])
            self.assertFalse(result['ok']);self.assertIn('Cancelled',result['error'])
            self.assertTrue(result['cleanup_verified']);self.assertTrue(result['child_reaped'])
            self.assertLess(result['pidfd_bound_at'],result['execution_authorized_at'])
            self.assertIsNone(shutdown.read_process(witness['pid']))
            self.assertLessEqual(result['cleanup_end'],result['absolute_deadline'])
        finally:
            thread.join(timeout=3)
            if worker.poll() is None:signal.pidfd_send_signal(worker_fd,signal.SIGKILL);worker.wait(timeout=1)
            os.close(worker_fd)

    def test_cancel_before_exec_and_unknown_controller_never_start_query(self):
        marker=self.root/'must-not-run'
        with self.assertRaises(capture.Cancelled):
            self.command(capture.GPU_ARGV,[sys.executable,str(HERE/'probe_fixture.py'),'slow',str(marker)],self.raw,time.monotonic()+1,cancelled=lambda:True)
        bad=dict(self.owner);bad['starttime']+=1
        with self.assertRaisesRegex(shutdown.UnsafeIdentity,'bound long-lived controller'):
            capture.command(capture.GPU_ARGV,[sys.executable,str(HERE/'probe_fixture.py'),'slow',str(marker)],self.raw,time.monotonic()+1,owner=bad)
        self.assertFalse(marker.exists());self.results.append(dict(preexec_cancellation_and_unknown_owner_refused=True))

    def test_receipt_worker_cannot_reuse_controller_query_authority(self):
        import multiprocessing
        context=multiprocessing.get_context('fork');read,write=os.pipe()
        def attempt():
            os.close(read)
            try:self.command(capture.GPU_ARGV,[sys.executable,str(HERE/'probe_fixture.py'),'thread'],self.raw,time.monotonic()+1)
            except shutdown.UnsafeIdentity:os.write(write,b'REFUSED')
            finally:os.close(write)
        worker=context.Process(target=attempt);worker.start();os.close(write)
        try:
            worker.join(timeout=2);self.assertFalse(worker.is_alive());self.assertEqual(os.read(read,32),b'REFUSED')
            self.results.append(dict(forked_worker_owner_refused=True,worker_exit=worker.exitcode))
        finally:
            os.close(read)
            if worker.is_alive():worker.kill();worker.join(timeout=1)
            worker.close()

    def test_invalid_attestation_prevents_exec_and_retains_owned_cleanup(self):
        from unittest.mock import patch
        marker=self.root/'must-not-run'
        with patch('capture.validate_containment',side_effect=ValueError('unknown helper identity')):
            result=self.command(capture.GPU_ARGV,[sys.executable,str(HERE/'probe_fixture.py'),'slow',str(marker)],self.raw,time.monotonic()+1)
        self.results.append(result);self.assertFalse(result['ok']);self.assertIsNone(result['execution_authorized_at'])
        self.assertTrue(result['cleanup_verified']);self.assertFalse(marker.exists())
        self.assertIn('unknown helper identity',result['error'])

    def test_cleanup_failure_cannot_report_clean_or_pass(self):
        from unittest.mock import patch
        original=signal.pidfd_send_signal
        def fail(fd,sig):
            original(fd,sig)
            raise OSError('receipt cleanup failure after signal')
        with patch('capture.signal.pidfd_send_signal',side_effect=fail):
            result=self.command(capture.GPU_ARGV,[sys.executable,str(HERE/'probe_fixture.py'),'slow',str(self.root/'cleanup')],self.raw,time.monotonic()+.5)
        self.results.append(result);self.assertFalse(result['ok']);self.assertFalse(result['cleanup_verified'])
        self.assertIn('cleanup failure',result['cleanup_error'])
        # Popen's child remains ours, no numeric PID signaling; reap after the injected receipt failure.
        try:os.waitpid(result['child_pid'],0)
        except ChildProcessError:pass

    def test_controller_core_delta_is_only_prepare_seam(self):
        old=(ROOT/'shutdown.py').read_text()
        needle="    try:\n        before_deadline(end)\n        worker.start()"
        replacement="    try:\n        # Queries are synchronous controller-owned direct children. Receipt workers never launch them.\n        if hasattr(probes,'prepare_controller'):\n            result['controller_queries']=probes.prepare_controller(probe_end)\n        before_deadline(probe_end)\n        worker.start()"
        self.assertEqual((HERE/'controller_shutdown.py').read_text(),old.replace(needle,replacement))

    def test_artifact_mutation_is_rejected(self):
        ref=artifacts.save(self.raw,'original',b'raw evidence')
        self.assertTrue(artifacts.verify(self.raw,ref,time.monotonic()+1))
        wrong=copy.deepcopy(ref);wrong['path']='tampered.raw';(self.raw/wrong['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'drift'):artifacts.verify(self.raw,wrong,time.monotonic()+1)
        self.results.append(dict(original=ref,tamper_rejected=True))

    def test_actual_pinned_text_crash_is_not_native_release_evidence(self):
        _,binding,_,_,log=self.start()
        adapter=self.build(binding,log,8001)
        adapter.native_path=HERE.parent/'trial-10-exact-wire-local/remote-readback/priority-output/serving-wire-capacity10/output/control.server.stderr'
        adapter.native_format='PINNED_SGLANG_TEXT'
        result=adapter.native();self.results.append(result)
        self.assertFalse(result['eligible']);self.assertTrue(result['crash_marker'])
        self.assertFalse(result['direct_watchdog_order_observed'])

    def test_prepared_source_or_interpreter_mismatch_is_rejected(self):
        child,binding,contract,receipt,log=self.start()
        prepared=Path(contract['prepared_path']);old=prepared.read_bytes()
        try:
            data=json.loads(old);data['argv'].append('--unreviewed');prepared.write_text(json.dumps(data))
            with self.assertRaises(shutdown.UnsafeIdentity):registration.bind(child,receipt,contract)
        finally:prepared.write_bytes(old)
        other=self.root/'other-python';other.write_text('not the registered interpreter')
        data=json.loads(old);data['python']=str(other);wrong=self.root/'wrong-prepared.json';wrong.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'interpreter mismatch'):
            registration.contract(wrong,contract['source_files'],'CPU_ADAPTER_ONLY')

    def test_shipping_wrapper_delta_only_adds_registration(self):
        old=(HERE.parent/'trial-10-exact-wire-local/control_prepare.py').read_text()
        expected=old.replace('import threading\n','import threading\nimport time\nimport registration\n')
        needle="    child=subprocess.Popen(data['argv'],env=data['env'],stdin=subprocess.DEVNULL,start_new_session=False)\n"
        addition="    registration.emit_control(child.pid,data,PLAN,PREFIX,Path(os.environ['SCREEN_OUTPUT']),min(float(os.environ['SCREEN_OVERALL_DEADLINE_MONOTONIC']),time.monotonic()+10))\n"
        self.assertEqual((HERE/'control_prepare_registered.py').read_text(),expected.replace(needle,needle+addition))


if __name__=='__main__':unittest.main()
