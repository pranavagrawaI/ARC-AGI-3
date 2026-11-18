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

import adapters,artifacts,capture,registration,runtime_scope
from core import ROOT,shutdown

HERE=Path(__file__).resolve().parent


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.owner=capture.owner_identity();self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.raw=self.root/'raw';self.raw.mkdir();self.processes=[];self.bindings=[];self.results=[];self.leases={}

    def tearDown(self):
        # Each test is a separate local scenario. Rescue any deliberately or
        # deadline-failed held query by its retained pidfd; preserve its failed
        # receipt, never count this post-scenario rescue as timely cleanup.
        for token,held in list(capture.RETAINED.items()):
            self.assertEqual(held['owner'],os.getpid())
            if held['child'].poll() is None and held['pidfd'] is not None:
                signal.pidfd_send_signal(held['pidfd'],signal.SIGKILL)
            held['child'].wait(timeout=1)
            if held['pidfd'] is not None:os.close(held['pidfd'])
            del capture.RETAINED[token]
            self.results.append(dict(post_scenario_rescue_token=token,original_failed_receipt_unchanged=True))
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

    def start(self,runtime=False,redirect=True):
        alias=self.root/'python';alias.symlink_to(sys.executable)
        entry=self.root/'frontend'
        entry.write_text(f'#!{alias}\nimport sys\nsys.path.insert(0,{str(ROOT)!r})\nimport native_fixture\nnative_fixture.frontend(sys.argv[1])\n')
        entry.chmod(0o700)
        lease=runtime_scope.create_native_output(self.root,'CPU_ADAPTER_ONLY') if runtime else None
        log=Path(lease['created']['log_path']) if runtime else self.root/'native.jsonl'
        prepared=self.root/'prepared.json';prepared.write_text(json.dumps(dict(python=str(alias),argv=[str(entry),str(log)])))
        sources={str(p):registration.sha(p) for p in [entry,ROOT/'native_fixture.py',HERE/'fixture_wrapper.py',HERE/'registration.py']}
        sourcepath=self.root/'sources.json';sourcepath.write_text(json.dumps(sources))
        contract=registration.contract(prepared,sources,'CPU_ADAPTER_ONLY')
        receipt=self.root/'registered.json'
        env=dict(os.environ,LOCAL_CPU_ENTRY=str(entry),LOCAL_CPU_ENTRY_SHA=registration.sha(entry))
        argv=[sys.executable,str(HERE/'fixture_wrapper.py'),str(prepared),str(sourcepath),str(receipt),'CPU_ADAPTER_ONLY']
        if runtime:
            with runtime_scope.open_native_stderr(lease) as native_stderr:
                launched=time.monotonic();child=subprocess.Popen(argv,start_new_session=True,env=env,stderr=native_stderr if redirect else None)
            self.processes.append(child) # retain ownership before any fallible registration
            runtime_scope.record_wrapper(child,lease,launched);self.leases[child.pid]=lease
        else:
            child=subprocess.Popen(argv,start_new_session=True,env=env);self.processes.append(child)
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

    def build(self,binding,log,port,runtime=False):
        old=HERE.parent/'trial-10-exact-wire-local/remote-readback/priority-output/serving-wire-capacity10/output/control-release.json'
        raw=json.loads(old.read_text())['raw'];uuid=raw['gpu'].split(',')[2].strip()
        paths={}
        for key in ('meminfo','maximum','current','stat'):
            paths[key]=self.root/key;paths[key].write_text('0' if key=='current' else raw[key])
        paths['disk']=self.root
        gpu=self.root/'gpu';gpu.write_text(f'NVIDIA RTX PRO 6000 Blackwell Server Edition,0,{uuid}\n')
        apps=self.root/'apps';apps.write_text('')
        commands={name:[sys.executable,str(HERE/'probe_fixture.py'),'file',str(path)] for name,path in [('gpu',gpu),('apps',apps)]}
        scope=None
        if runtime:
            paths,_=runtime_scope.derive(Path('/proc/self/cgroup').read_text(),Path('/proc/self/mountinfo').read_text())
            scope=runtime_scope.bind(binding,paths,self.leases[binding['pgid']],'CPU_ADAPTER_ONLY')
        return adapters.Adapters(self.raw,binding,uuid,commands,paths,port,log,'CPU_JSON_EVENTS',time.monotonic()+4,scope=scope)

    def test_prepared_registration_and_real_cpu_adapter_release(self):
        child,binding,contract,receipt,log=self.start()
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        adapter=self.build(binding,log,port)
        result=shutdown.release(binding,adapter.probes(),end=time.monotonic()+3,graceful_seconds=2,fallback_reserve=.5,probe_seconds=.8,poll=.1)
        self.results.append(result)
        complete=[s for s in result['samples'] if s['process']['eligible']
                  and all(s['probes'][name]['eligible'] for name in ('gpu','apps','cgroup','port','native'))]
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
        expected=old.replace(needle,replacement).replace('    error = None\n    passed = False','    error = None\n    controller_query_failure = None\n    passed = False')
        expected=expected.replace('    except BaseException as failure:\n        error = repr(failure)\n    cleanup_error',"    except BaseException as failure:\n        error = repr(failure)\n        controller_query_failure = getattr(failure,'controller_query_receipt',None)\n    cleanup_error")
        expected=expected.replace('error=error, cleanup_error=cleanup_error, remaining=remaining, events=events, samples=samples,','error=error, cleanup_error=cleanup_error, remaining=remaining, events=events, samples=samples,\n                controller_query_failure=controller_query_failure,')
        expected=expected.replace('            except DeadlineReached:\n                break #',"            except DeadlineReached as failure:\n                controller_query_failure = getattr(failure,'controller_query_receipt',None)\n                break #")
        self.assertEqual((HERE/'controller_shutdown.py').read_text(),expected)

    def test_cgroup_mapping_respects_mount_root_and_rejects_ambiguous_or_outside(self):
        line='42 1 0:30 /tenant /sys/fs/cgroup rw - cgroup2 cgroup rw'
        paths,record=runtime_scope.derive('0::/tenant/job\n',line)
        self.assertEqual(paths['maximum'],Path('/sys/fs/cgroup/job/memory.max'));self.assertEqual(record,line)
        for group,mount in [('0::/other/job\n',line),('0::/tenant/job\n',line+'\n'+line),('0::/tenant/../outside\n',line)]:
            with self.assertRaises(ValueError):runtime_scope.derive(group,mount)
        self.results.append(dict(mount_root_mapping=str(paths['maximum']),ambiguous_outside_parent_paths_rejected=True))

    def test_actual_frontend_cgroup_observer_and_fresh_stderr_binding(self):
        _,binding,_,_,log=self.start(runtime=True)
        adapter=self.build(binding,log,8001,runtime=True)
        positive=adapter.check_scope();self.assertEqual(positive['input_scope'],'LIVE_BOUND_CGROUP_AND_FRESH_STDERR')
        result=shutdown.observe(binding,adapter.probes(),time.monotonic()+2,1.5)
        self.results.extend([dict(scope=adapter.scope),result])
        self.assertFalse(result['eligible']) # frontend still alive; binding proof is not release proof
        self.assertTrue(result['controller_queries']['all_queries_cleaned_before_worker'])
        native=result['probes']['native'];self.assertEqual(native['scope_before']['input_scope'],'LIVE_BOUND_CGROUP_AND_FRESH_STDERR')
        self.assertEqual(native['scope_after']['run_id'],'CPU_ADAPTER_ONLY')
        self.assertNotEqual(native['scope_before']['observer_pid'],os.getpid())
        self.assertEqual(native['scope_before']['cgroup'],binding['members'][binding['frontend']]['cgroup'])
        for command in result['controller_queries']['commands'].values():
            self.assertEqual(command['owner_pid'],os.getpid());self.assertTrue(command['cleanup_verified'])
            self.assertLessEqual(command['cleanup_end'],native['start'])

    def test_saved_cgroup_paths_wrong_run_and_observer_scope_are_rejected(self):
        from unittest.mock import patch
        _,binding,_,_,log=self.start(runtime=True);adapter=self.build(binding,log,8001,runtime=True)
        bad=dict(adapter.paths);bad['current']=self.root/'current'
        with self.assertRaisesRegex(ValueError,'not derived'):runtime_scope.bind(binding,bad,self.leases[binding['pgid']],'CPU_ADAPTER_ONLY')
        with self.assertRaisesRegex(ValueError,'binding mismatch'):runtime_scope.bind(binding,adapter.paths,self.leases[binding['pgid']],'WRONG_RUN')
        seen=runtime_scope.observer();seen['cgroup']='0::/wrong\n'
        with patch('runtime_scope.observer',return_value=seen):
            with self.assertRaisesRegex(ValueError,'scope mismatch'):runtime_scope.bind(binding,adapter.paths,self.leases[binding['pgid']],'CPU_ADAPTER_ONLY')
            with self.assertRaisesRegex(ValueError,'namespace drift'):adapter.check_scope()
        self.results.append(dict(saved_paths_wrong_run_and_observer_rejected=True))

    def test_stale_path_swapped_inode_and_reused_output_rejected(self):
        _,binding,_,_,log=self.start(runtime=True);adapter=self.build(binding,log,8001,runtime=True)
        with self.assertRaises(FileExistsError):runtime_scope.create_native_output(self.root,'CPU_ADAPTER_ONLY')
        old=self.root/'old.stderr';old.write_bytes(log.read_bytes())
        with self.assertRaisesRegex(ValueError,'stale/copied'):runtime_scope.check(adapter.scope,binding,adapter.paths,old)
        saved=log.with_name('original.stderr');log.rename(saved);log.write_bytes(saved.read_bytes())
        try:
            with self.assertRaisesRegex(ValueError,'identity/path drift'):adapter.check_scope()
        finally:log.unlink();saved.rename(log)
        self.assertEqual(adapter.check_scope()['run_id'],'CPU_ADAPTER_ONLY')
        self.results.append(dict(reused_directory_stale_copy_and_inode_swap_rejected=True))

    def test_prefilled_native_log_cannot_be_attached_as_fresh(self):
        lease=runtime_scope.create_native_output(self.root,'OLD_CONTENT')
        Path(lease['created']['log_path']).write_text('copied old startup/shutdown markers')
        with self.assertRaisesRegex(ValueError,'fresh and empty'):runtime_scope.open_native_stderr(lease)
        self.results.append(dict(prefilled_log_refused_before_wrapper=True))

    def test_actual_receipt_worker_sigkill_cannot_orphan_prepared_queries(self):
        _,binding,_,_,log=self.start()
        adapter=self.build(binding,log,8001);probes=adapter.probes()
        probes['gpu']=lambda:signal.raise_signal(signal.SIGKILL)
        result=shutdown.observe(binding,probes,time.monotonic()+2,1.5)
        self.results.append(result);self.assertFalse(result['eligible']);self.assertIsNotNone(result['ipc_error'])
        self.assertFalse(result['probe_worker_alive'])
        for command in result['controller_queries']['commands'].values():
            self.assertTrue(command['cleanup_verified']);self.assertEqual(command['owner_pid'],os.getpid())
            self.assertIsNone(shutdown.read_process(command['child_pid']))
            self.assertLessEqual(command['cleanup_end'],result['probes']['gpu']['start'] or result['end'])

    def test_unresolved_cleanup_retains_controller_handle_and_blocks_further_queries(self):
        from unittest.mock import patch
        with patch('capture.signal.pidfd_send_signal',return_value=None):
            result=self.command(capture.GPU_ARGV,[sys.executable,str(HERE/'probe_fixture.py'),'slow',str(self.root/'retained')],self.raw,time.monotonic()+.4)
        self.results.append(result);self.assertFalse(result['ok']);self.assertFalse(result['cleanup_verified'])
        token=result['retained_query'];self.assertIsNotNone(token);held=capture.RETAINED[token]
        try:
            _,binding,_,_,log=self.start();adapter=self.build(binding,log,8001)
            with patch('capture.command',return_value=result):
                release=shutdown.release(binding,adapter.probes(),end=time.monotonic()+2,graceful_seconds=1,fallback_reserve=.5)
            self.results.append(release);self.assertFalse(release['pass'])
            self.assertEqual(release['controller_query_failure']['commands']['gpu']['retained_query'],token)
            self.assertEqual(len(release['controller_query_failure']['commands']),1) # no second dispatch/worker
            self.assertEqual(release['samples'],[])
            self.assertEqual(held['owner'],os.getpid());self.assertIsNotNone(held['pidfd'])
            with self.assertRaisesRegex(RuntimeError,'unresolved controller query'):
                self.command(capture.GPU_ARGV,[sys.executable,str(HERE/'probe_fixture.py'),'thread'],self.raw,time.monotonic()+1)
        finally:
            # Test-only rescue of deliberately suppressed cleanup. This does not
            # make the original command's expired cleanup receipt successful.
            signal.pidfd_send_signal(held['pidfd'],signal.SIGKILL);held['child'].wait(timeout=1)
            os.close(held['pidfd']);del capture.RETAINED[token]
        self.results.append(dict(injected_unresolved_query_accounted=True,rescue_is_not_qualification=True))

    def test_query_not_admitted_into_reserved_cleanup_time(self):
        marker=self.root/'too-late'
        with self.assertRaises(shutdown.DeadlineReached):
            self.command(capture.GPU_ARGV,[sys.executable,str(HERE/'probe_fixture.py'),'slow',str(marker)],self.raw,time.monotonic()+.05)
        self.assertFalse(marker.exists());self.assertEqual(list(self.raw.iterdir()),[])
        self.results.append(dict(no_query_dispatched_with_only_cleanup_reserve=True))

    def test_fresh_path_and_wrapper_receipt_do_not_replace_actual_frontend_stderr_binding(self):
        _,binding,_,_,log=self.start(runtime=True,redirect=False)
        with self.assertRaisesRegex(ValueError,'frontend stderr is not fresh'):
            self.build(binding,log,8001,runtime=True)
        self.results.append(dict(actual_frontend_fd2_mismatch_rejected=True))

    def test_clean_query_phase_expiry_preserves_owned_fallback(self):
        from unittest.mock import patch
        _,binding,_,_,log=self.start();adapter=self.build(binding,log,8001)
        with patch('capture.command',side_effect=shutdown.DeadlineReached('pre-exec phase deadline')):
            result=shutdown.release(binding,adapter.probes(),end=time.monotonic()+2,graceful_seconds=1,fallback_reserve=.5)
        self.results.append(result);self.assertFalse(result['pass']);self.assertTrue(result['fallback_used'])
        self.assertEqual(result['controller_query_failure']['kind'],'phase_deadline_clean_queries')
        self.assertEqual(result['controller_query_failure']['commands'],{})

    def test_empty_initial_cmdline_is_allowed_only_before_exact_gated_helper_readiness(self):
        from unittest.mock import patch
        original=shutdown.read_process;changed=[]
        def initial_empty(pid):
            value=original(pid)
            if pid!=os.getpid() and value is not None and not changed:
                value=copy.deepcopy(value);value['identity']['cmdline_sha256']=hashlib.sha256(b'').hexdigest();changed.append(pid)
            return value
        with patch('capture.shutdown.read_process',side_effect=initial_empty):
            result=self.command(capture.GPU_ARGV,[sys.executable,str(HERE/'probe_fixture.py'),'thread'],self.raw,time.monotonic()+2)
        self.results.append(result);self.assertTrue(result['ok']);self.assertTrue(changed)
        intent=json.loads((self.raw/result['intent']['path']).read_text())
        self.assertEqual(intent['observed_helper']['identity']['cmdline_sha256'],hashlib.sha256(b'').hexdigest())
        self.assertLess(result['pidfd_bound_at'],result['execution_authorized_at'])

    def test_ready_gate_never_accepts_empty_unknown_argv_or_other_identity_drift(self):
        result=self.command(capture.GPU_ARGV,[sys.executable,str(HERE/'probe_fixture.py'),'thread'],self.raw,time.monotonic()+2)
        self.results.append(result);self.assertTrue(result['ok'])
        intent=json.loads((self.raw/result['intent']['path']).read_text());initial=intent['observed_helper'];argv=intent['helper_argv']
        ready=copy.deepcopy(initial);ready['identity']['cmdline_sha256']=hashlib.sha256(b''.join(os.fsencode(v)+b'\0' for v in argv)).hexdigest()
        self.assertTrue(capture.validate_ready_identity(initial,ready,argv))
        for field,value in [('cmdline_sha256',hashlib.sha256(b'').hexdigest()),('cmdline_sha256','wrong'),('starttime',ready['identity']['starttime']+1),('exe','/unknown'),('cgroup','0::/unknown\n')]:
            wrong=copy.deepcopy(ready);wrong['identity'][field]=value
            with self.assertRaises(shutdown.UnsafeIdentity):capture.validate_ready_identity(initial,wrong,argv)
        bad_initial=copy.deepcopy(initial);bad_initial['identity']['cmdline_sha256']='unexpected'
        with self.assertRaises(shutdown.UnsafeIdentity):capture.validate_ready_identity(bad_initial,ready,argv)
        self.results.append(dict(exact_ready_command_and_other_identity_fields_required=True))

    def test_gate_rechecks_deadline_after_slow_successful_identity_validation(self):
        from unittest.mock import patch
        marker=self.root/'late-gate-query';absolute_end=time.monotonic()+.3;clocks={}
        original=capture.validate_ready_identity
        def delayed(*args):
            value=original(*args);clocks['validated_at']=time.monotonic()
            time.sleep(max(0,absolute_end+.05-time.monotonic()))
            clocks['validation_returned_at']=time.monotonic();return value
        with patch('capture.validate_ready_identity',side_effect=delayed):
            result=self.command(capture.GPU_ARGV,[sys.executable,str(HERE/'probe_fixture.py'),'slow',str(marker)],self.raw,absolute_end)
        self.results.extend([clocks,result])
        self.assertGreater(clocks['validation_returned_at'],absolute_end)
        self.assertIsNone(result['execution_authorized_at'])
        self.assertFalse(result['ok']);self.assertIn('DeadlineReached',result['error'])
        self.assertFalse(marker.exists())

    def test_gate_rechecks_cancellation_after_successful_identity_validation(self):
        from unittest.mock import patch
        marker=self.root/'cancel-gate-query';state=dict(cancelled=False);original=capture.validate_ready_identity
        def cancel_during_validation(*args):
            value=original(*args);state.update(cancelled=True,cancelled_at=time.monotonic());return value
        with patch('capture.validate_ready_identity',side_effect=cancel_during_validation):
            result=self.command(capture.GPU_ARGV,[sys.executable,str(HERE/'probe_fixture.py'),'slow',str(marker)],self.raw,time.monotonic()+2,cancelled=lambda:state['cancelled'])
        self.results.extend([state,result])
        self.assertIsNone(result['execution_authorized_at'])
        self.assertFalse(result['ok']);self.assertIn('Cancelled',result['error'])
        self.assertTrue(result['cleanup_verified']);self.assertFalse(marker.exists())

    def test_gate_ordinary_success_after_recheck(self):
        from unittest.mock import patch
        clocks={};original=capture.validate_ready_identity
        def validated(*args):
            value=original(*args);clocks['validated_at']=time.monotonic();return value
        with patch('capture.validate_ready_identity',side_effect=validated):
            result=self.command(capture.GPU_ARGV,[sys.executable,str(HERE/'probe_fixture.py'),'thread'],self.raw,time.monotonic()+2)
        self.results.extend([clocks,result]);self.assertTrue(result['ok']);self.assertTrue(result['cleanup_verified'])
        self.assertLessEqual(clocks['validated_at'],result['execution_authorized_at'])
        self.assertLessEqual(result['execution_authorized_at'],result['work_deadline'])
        self.assertEqual((self.raw/result['artifacts']['stdout']['path']).read_bytes(),b'thread-ok\n')

    def test_gate_final_deadline_covers_time_spent_inside_checkpoint_callback(self):
        from unittest.mock import patch
        marker=self.root/'callback-late-query';absolute_end=time.monotonic()+.3;state=dict(ready=False);original=capture.validate_ready_identity
        def ready(*args):
            value=original(*args);state['ready']=True;return value
        def delayed_not_cancelled():
            if state['ready']:
                time.sleep(max(0,absolute_end+.05-time.monotonic()))
                state['callback_returned_at']=time.monotonic()
            return False
        with patch('capture.validate_ready_identity',side_effect=ready):
            result=self.command(capture.GPU_ARGV,[sys.executable,str(HERE/'probe_fixture.py'),'slow',str(marker)],self.raw,absolute_end,cancelled=delayed_not_cancelled)
        self.results.extend([state,result]);self.assertGreater(state['callback_returned_at'],absolute_end)
        self.assertIsNone(result['execution_authorized_at']);self.assertFalse(result['ok']);self.assertFalse(marker.exists())
        self.assertIn('DeadlineReached',result['error'])

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
