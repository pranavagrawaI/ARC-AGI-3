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
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
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
                    for ref in command['artifacts'].values():self.assertTrue(artifacts.verify(self.raw,ref,time.monotonic()+1))

    def test_loopback_occupied_and_free_port_plumbing(self):
        _,binding,_,_,log=self.start()
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));sock.listen();port=sock.getsockname()[1]
            adapter=self.build(binding,log,port)
            occupied=adapter.port_probe();self.assertFalse(occupied['eligible']);self.results.append(occupied)
        free=adapter.port_probe();self.assertTrue(free['eligible']);self.results.append(free)

    def test_large_stdout_is_exact_referenced_artifact_not_ipc_body(self):
        result=capture.command(capture.GPU_ARGV,[sys.executable,str(HERE/'probe_fixture.py'),'big'],self.raw,time.monotonic()+2)
        self.results.append(result);self.assertTrue(result['ok'])
        ref=result['artifacts']['stdout'];self.assertEqual(ref['bytes'],256*1024)
        self.assertEqual(ref['sha256'],hashlib.sha256(b'x'*(256*1024)).hexdigest())
        self.assertTrue(artifacts.verify(self.raw,ref,time.monotonic()+1))
        self.assertLess(len(json.dumps(result)),65536)

    def test_gpu_execution_refused_and_command_error_preserved(self):
        with self.assertRaisesRegex(ValueError,'CPU execution only'):
            capture.command(capture.GPU_ARGV,capture.GPU_ARGV,self.raw,time.monotonic()+1)
        from unittest.mock import patch
        with patch.dict(os.environ,LOCAL_ADAPTER_PROOF='1'):
            with self.assertRaisesRegex(ValueError,'forbidden during local proof'):
                capture.command(capture.GPU_ARGV,capture.GPU_ARGV,self.raw,time.monotonic()+1,'FUTURE_REVIEWED_GPU_QUERY')
        result=capture.command(capture.APPS_ARGV,[sys.executable,str(HERE/'probe_fixture.py'),'error'],self.raw,time.monotonic()+2)
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
        result=capture.command(capture.GPU_ARGV,[sys.executable,str(HERE/'probe_fixture.py'),'slow',str(marker)],self.raw,time.monotonic()+.2)
        self.results.append(result)
        self.assertFalse(result['ok']);self.assertTrue(result['child_reaped'])
        self.assertFalse(result['artifacts']['stdout']['complete'])
        self.assertEqual((self.raw/result['artifacts']['stdout']['path']).read_bytes(),b'partial\n')

    def test_probe_worker_death_cannot_leave_query_running(self):
        marker=self.root/'pid'
        program='import capture,sys,time;capture.command(capture.GPU_ARGV,[sys.executable,%r,"slow",%r],%r,time.monotonic()+10)'%(str(HERE/'probe_fixture.py'),str(marker),str(self.raw))
        worker=subprocess.Popen([sys.executable,'-c',program],cwd=HERE,start_new_session=True)
        try:
            end=time.monotonic()+2
            while not marker.exists() and time.monotonic()<end:time.sleep(.01)
            self.assertTrue(marker.exists());pid=int(marker.read_text())
            identity=shutdown.read_process(pid);self.assertIsNotNone(identity)
            fd=os.pidfd_open(worker.pid)
            try:signal.pidfd_send_signal(fd,signal.SIGKILL)
            finally:os.close(fd)
            worker.wait(timeout=1)
            end=time.monotonic()+2
            def gone():
                try:return shutdown.read_process(pid) is None
                except shutdown.UnsafeIdentity:return False
            while not gone() and time.monotonic()<end:time.sleep(.01)
            self.assertIsNone(shutdown.read_process(pid))
            self.results.append(dict(parent_death_child_pid=pid,child_identity=identity,child_observed_gone=True))
        finally:
            if worker.poll() is None:worker.kill();worker.wait(timeout=1)

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
