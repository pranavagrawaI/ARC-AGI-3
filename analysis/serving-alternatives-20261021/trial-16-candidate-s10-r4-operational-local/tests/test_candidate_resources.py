import copy,hashlib,json,os,signal,subprocess,sys,tempfile,time,unittest,shutil
from pathlib import Path
from unittest.mock import patch
import artifacts,candidate_resources,candidate_evidence,capture,resource_observe,runtime_scope
from core import shutdown

class ResourcePaths(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.owner=shutdown.read_process(os.getpid())['identity']
        self.child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
        self.pidfd=os.pidfd_open(self.child.pid)
        identity=shutdown.read_process(self.child.pid)['identity']
        self.binding=dict(controller=self.owner,pgid=self.child.pid,frontend=self.child.pid,members={self.child.pid:identity},source_files={str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        self.paths={name:self.root/name for name in ('meminfo','maximum','current','stat')};self.paths['disk']=self.root
        for key,value in dict(meminfo='MemAvailable: 67108864 kB\n',maximum=str(64*1024**3),current=str(40*1024**3),stat='inactive_file 0\n').items():self.paths[key].write_text(value)
        self.scope=dict(run_id='CPU',observer=runtime_scope.observer(),paths={key:str(value) for key,value in self.paths.items()},binding=self.binding)
        (self.root/'candidate-scope.json').write_text(json.dumps(self.scope))
        signal.pidfd_send_signal(self.pidfd,signal.SIGTERM);self.child.wait(timeout=2)
    def tearDown(self):
        if self.child.poll() is None:signal.pidfd_send_signal(self.pidfd,signal.SIGKILL);self.child.wait(timeout=2)
        proof=os.environ.get('PACKET_PROOF_DIR')
        if proof:
            target=Path(proof)/'resource-cases'/self.id().split('.')[-1];target.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(self.root,target)
            (target/'OWNED-FIXTURE-EXIT.json').write_text(json.dumps(dict(pid=self.child.pid,returncode=self.child.returncode,all_fixture_processes_gone=self.child.poll() is not None)))
        os.close(self.pidfd);self.temp.cleanup()
    def sample(self):
        scope=self.scope
        class Fixture(candidate_resources.Adapters):
            def check_scope(self):return dict(run_id='CPU',observer_pid=os.getpid(),cgroup=scope['observer']['cgroup'],namespaces=scope['observer']['namespaces'],input_scope='CANDIDATE_PROCESS_CGROUP_NO_NATIVE_PROXY')
        commands=dict(gpu=[sys.executable,'-c',"print('NVIDIA RTX PRO 6000 Blackwell Server Edition, 0, GPU-CPU')"],apps=[sys.executable,'-c',"print('',end='')"])
        end=time.monotonic()+5
        adapter=Fixture(self.root/'query-artifacts',self.binding,'GPU-CPU',self.paths,8001,end,self.scope,commands=commands,execution_policy='LOCAL_CPU_ONLY')
        with patch('adapters.shutil.disk_usage',return_value=type('Usage',(),dict(free=120*1024**3))()):
            value=resource_observe.observe(self.binding,adapter.probes(),end,4)
        (self.root/'sample.json').write_text(json.dumps(value,indent=2))
        return value,end
    def test_real_guarded_queries_and_port_hostfalse_permitted(self):
        sample,end=self.sample()
        self.assertFalse(sample['probes']['cgroup']['eligible'])
        self.assertTrue(candidate_resources.accepted(sample,end),sample)
        raw=candidate_evidence.verify_sample(self.root,sample,self.binding,'GPU-CPU',end,allow_cpu=True)
        self.assertTrue(raw['eligible']);self.assertFalse(raw['original140'])
    def test_missing_host_never_becomes_report_only(self):
        sample,end=self.sample();del sample['probes']['cgroup']['artifacts']['stat']
        with self.assertRaises((ValueError,KeyError)):candidate_evidence.verify_sample(self.root,sample,self.binding,'GPU-CPU',end,allow_cpu=True)
    def test_changed_raw_gpu_rejected_even_with_rehashed_artifact(self):
        sample,end=self.sample();ref=sample['probes']['gpu']['command']['artifacts']['stdout'];path=self.root/'query-artifacts'/ref['path'];path.write_text('NVIDIA RTX PRO 6000 Blackwell Server Edition, 3000, GPU-CPU\n')
        ref.update(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest());sample['controller_queries']['commands']['gpu']['artifacts']['stdout'].update(ref)
        with self.assertRaisesRegex(ValueError,'raw GPU'):candidate_evidence.verify_sample(self.root,sample,self.binding,'GPU-CPU',end,allow_cpu=True)
    def test_no_native_proxy_and_malformed_host_rejected(self):
        sample,end=self.sample();sample['probes']['native']={'eligible':True}
        self.assertFalse(candidate_resources.accepted(sample,end))
        with self.assertRaisesRegex(ValueError,'no native proxy'):candidate_evidence.verify_sample(self.root,sample,self.binding,'GPU-CPU',end,allow_cpu=True)
    def test_late_receipt_rejected(self):
        sample,end=self.sample();sample['end']=end+.001
        self.assertFalse(candidate_resources.accepted(sample,end))
        with self.assertRaisesRegex(ValueError,'deadline'):candidate_evidence.verify_sample(self.root,sample,self.binding,'GPU-CPU',end,allow_cpu=True)

    def final_fixture(self):
        sample,end=self.sample();start=sample['start']-.01;deadline=start+180
        events=[dict(event='start',work_deadline=deadline+10),dict(event='candidate_cleanup_start',t=start+.0001,cleanup_start=start,phase_deadline=deadline,server_pgid=self.child.pid),dict(event='cleanup',pgid=self.child.pid,remaining=[],t=start+.001),dict(event='candidate_resources_pass',start=start,end=sample['end'],absolute_deadline=deadline,t=sample['end']+.0001)]
        receipt=dict(start=start,end=sample['end'],absolute_deadline=deadline,server_pgid=self.child.pid,pass_=True,error=None,samples=[sample])
        (self.root/'candidate-release.json').write_text(json.dumps(receipt))
        identity=self.binding['members'][self.child.pid]
        captured=dict(server_pgid=str(self.child.pid),server_start=str(identity['starttime']),boot_id=identity['boot_id'],gpu_uuid='GPU-CPU',end=start-.01)
        config=dict(mode='CPU_FIXTURE_ONLY',final_resource_seconds=180)
        return receipt,events,captured,config
    def test_public_final_phase_hostfalse_valid_and_deadline_mutations_reject(self):
        receipt,events,captured,config=self.final_fixture()
        result=candidate_evidence.final_resources(self.root,config,events,captured)
        self.assertTrue(result['pass_']);self.assertFalse(result['original140']);self.assertFalse(result['native_proxy'])
        receipt['absolute_deadline']+=1;(self.root/'candidate-release.json').write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError,'final180'):candidate_evidence.final_resources(self.root,config,events,captured)
    def test_public_final_replay_starttime_drift_rejected(self):
        receipt,events,captured,config=self.final_fixture();captured['server_start']='reused'
        with self.assertRaisesRegex(ValueError,'replay/server'):candidate_evidence.final_resources(self.root,config,events,captured)
