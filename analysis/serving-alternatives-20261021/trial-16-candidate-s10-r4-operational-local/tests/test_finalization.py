"""Same real CPU finalization/publication triggers as the canonical review."""
import json,os,shutil,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import test_pipeline,host_observer

class Finalization(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)/'run';self.root.mkdir();self.addCleanup(self.temp.cleanup)
    def keep(self):
        target=os.environ.get('PACKET_PROOF_DIR')
        if target:
            destination=Path(target)/'finalization'/self._testMethodName;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copytree(self.temp.name,destination)
    def oom_run(self,mutate=None):
        original=host_observer.snapshot;state={'increased':False}
        def snapshot(spec,label):
            row=original(spec,label)
            if label=='candidate.replay.before':state['increased']=True
            if state['increased']:
                counts=dict(row['oom_events']);counts['oom']+=1;counts['oom_kill']+=1
                row['oom_events']=counts;row['raw']['events']=''.join(f'{key} {value}\n' for key,value in counts.items())
            if mutate is not None:mutate(row,label,state)
            return row
        with patch.dict(os.environ,{},clear=False):
            os.environ.pop('PACKET_PROOF_DIR',None)
            with patch('host_observer.snapshot',side_effect=snapshot):value=test_pipeline.Pipeline().run_fixture(self.root)
        self.keep();return value
    def test_known_oom_stops_workload_but_preserves_resources_and_orderly_series(self):
        value=self.oom_run();out=self.root/'output';release=json.loads((out/'candidate-release.json').read_text())
        self.assertIn('OOM counter increased',value['error']);self.assertFalse(value['pass_']);self.assertIsNone(value['U_resp'])
        self.assertGreater(len(release['samples']),0);self.assertTrue(release['pass_'],release)
        self.assertTrue((out/'observer.stop').exists());series=json.loads((out/'host-series.result.json').read_text());self.assertTrue(series['pass_'])
        self.assertEqual(value['cleanup_errors'],[]);self.assertEqual(value['surviving_phase_groups'],[]);self.assertEqual(value['retained_queries'],[])
        events=list(map(json.loads,(out/'supervisor.jsonl').read_text().splitlines()))
        self.assertFalse(any(row['event']=='spawn' and row['phase'] in ('candidate.replay','candidate.qualify') for row in events))
    def test_late_final_boundary_still_finishes_observer_but_remains_invalid(self):
        def late(row,label,state):
            if label=='final_cleanup':row['end']+=100
        value=self.oom_run(late);out=self.root/'output'
        self.assertEqual(value['status'],'INVALID_INCONCLUSIVE');self.assertTrue(value['cleanup_errors']);self.assertTrue((out/'host-series.result.json').exists())
        self.assertTrue(any('deadline' in error for error in value['cleanup_errors']));self.assertEqual(value['surviving_phase_groups'],[])
    def test_malformed_final_boundary_still_finishes_observer_but_remains_invalid(self):
        def malformed(row,label,state):
            if label=='final_cleanup':row['raw']['stat']='inactive_file invalid\n'
        value=self.oom_run(malformed);out=self.root/'output'
        self.assertEqual(value['status'],'INVALID_INCONCLUSIVE');self.assertTrue(value['cleanup_errors']);self.assertTrue((out/'host-series.result.json').exists())
        self.assertEqual(value['surviving_phase_groups'],[]);self.assertIn('OOM counter increased',value['error'])

class Publication(unittest.TestCase):
    setUp=Finalization.setUp
    keep=Finalization.keep
    # Reuse only fixture setup/artifact persistence, not the OOM test cases.
    def publication_run(self,point):
        import primary_bundle
        fixture=test_pipeline.Pipeline();config=fixture.config();config['primary_seconds']=.2;fixture.config=lambda:config
        original_write=Path.write_text;original_publish=primary_bundle.write;calls=[]
        def write(path,text,*args,**kwargs):
            target='primary-bundle-receipt.json' if point=='receipt' else 'primary-publication-completion.json'
            if point in ('receipt','completion') and path.name==target:
                value=json.loads(text);time.sleep(max(0,value['deadline']-time.monotonic())+.05)
                result=original_write(path,text,*args,**kwargs)
                calls.append(dict(point=point,actual_write_end=time.monotonic(),saved=value));return result
            return original_write(path,text,*args,**kwargs)
        def publish(*args,**kwargs):
            value=original_publish(*args,**kwargs)
            if point=='return':
                time.sleep(max(0,value['deadline']-time.monotonic())+.05);calls.append(dict(point=point,actual_return=time.monotonic(),saved=value))
            return value
        with patch.dict(os.environ,{},clear=False):
            os.environ.pop('PACKET_PROOF_DIR',None)
            with patch.object(Path,'write_text',write),patch.object(primary_bundle,'write',publish):value=fixture.run_fixture(self.root)
        (self.root/'DELAY.json').write_text(json.dumps(calls,indent=2));self.keep();return value,calls,config
    def test_late_receipt_persistence_invalidates_feasible_status(self):
        value,calls,config=self.publication_run('receipt');self.assertEqual(len(calls),1)
        self.assertGreater(calls[0]['actual_write_end'],calls[0]['saved']['deadline'])
        self.assertFalse(value['pass_']);self.assertEqual(value['status'],'INVALID_INCONCLUSIVE');self.assertIsNone(value['U_resp'])
        self.assertIn('primary_bundle_error',value)
    def check_stale_rejected(self,config):
        import primary_bundle
        raw=(self.root.parent/primary_bundle.NAME).read_bytes()
        events=list(map(json.loads,(self.root/'output/supervisor.jsonl').read_text().splitlines()))
        with self.assertRaises((ValueError,FileNotFoundError)):
            primary_bundle.verify_publication(self.root/'output',raw,events[0]['work_deadline'],config['primary_seconds'])
    def test_late_helper_return_is_checked_by_caller(self):
        value,calls,config=self.publication_run('return');self.assertEqual(len(calls),1)
        self.assertFalse(value['pass_']);self.assertEqual(value['status'],'INVALID_INCONCLUSIVE');self.assertIsNone(value['U_resp']);self.assertIn('returned late',value['primary_bundle_error'])
        self.check_stale_rejected(config)
    def test_late_caller_completion_write_cannot_leave_stale_primary_qualified(self):
        value,calls,config=self.publication_run('completion');self.assertEqual(len(calls),1)
        self.assertFalse(value['pass_']);self.assertEqual(value['status'],'INVALID_INCONCLUSIVE');self.assertIsNone(value['U_resp']);self.assertIn('persistence returned late',value['primary_bundle_error'])
        self.check_stale_rejected(config)
    def test_timely_publication_has_independent_after_persistence_and_return_join(self):
        import primary_bundle
        value,calls,config=self.publication_run('normal');self.assertTrue(value['pass_'],value);self.assertEqual(calls,[])
        raw=(self.root.parent/primary_bundle.NAME).read_bytes();events=list(map(json.loads,(self.root/'output/supervisor.jsonl').read_text().splitlines()))
        proof=primary_bundle.verify_publication(self.root/'output',raw,events[0]['work_deadline'],config['primary_seconds'])
        self.assertLessEqual(proof['helper_persisted_end'],proof['returned_at']);self.assertLessEqual(proof['returned_at'],proof['deadline'])
        path=self.root/'output/primary-publication-completion.json';receipt=json.loads(path.read_text());receipt['helper_persisted_end']=receipt['deadline']+.01;path.write_text(json.dumps(receipt));self.check_stale_rejected(config)

class OomOutcome(unittest.TestCase):
    setUp=Finalization.setUp
    keep=Finalization.keep
    def bound_run(self,mode='complete'):
        import candidate_runner,candidate_evidence,resources,sys
        from phase_tools import PhaseTools
        original_snapshot=host_observer.snapshot;original_start=PhaseTools.start;original_emit=PhaseTools.emit
        flag=self.root/'injected-oom';original_check=candidate_evidence.check
        def snapshot(spec,label):
            row=original_snapshot(spec,label)
            if label=='candidate.replay.before':flag.write_text('EXPLICIT_CPU_INPUT_ONLY')
            if flag.exists():
                counts=dict(row['oom_events']);counts['oom']+=1;counts['oom_kill']+=1
                row['oom_events']=counts;row['raw']['events']=''.join(f'{key} {value}\n' for key,value in counts.items())
            return row
        def start(runner,name,spec,end):
            if name=='observer':spec['argv']=[sys.executable,str(Path(__file__).with_name('oom_observer.py')),str(flag),mode,*spec['argv'][1:]]
            return original_start(runner,name,spec,end)
        def emit(runner,event,**fields):
            original_emit(runner,event,**fields)
            if event=='EXPLICIT_CPU_HARDWARE_SUBSTITUTE':original_emit(runner,'hardware_pass',name=resources.GPU_NAME,memory_mib=90000,driver='EXPLICIT_CPU_FIXTURE',uuid='GPU-CPU')
        def verifier(*args,**kwargs):
            # Only model/package identity is substituted; actual candidate binding,
            # raw query resources, final180, host/OOM and interpretation run unchanged.
            with patch.object(candidate_evidence,'package',return_value=True):return original_check(*args,**kwargs)
        with patch.dict(os.environ,{},clear=False):
            os.environ.pop('PACKET_PROOF_DIR',None)
            with patch('host_observer.snapshot',side_effect=snapshot),patch.object(candidate_runner.Runner,'start',start),patch.object(candidate_runner.Runner,'emit',emit):
                value=test_pipeline.Pipeline().run_fixture(self.root,verifier=verifier)
        self.keep();return value,verifier
    def test_real_reducer_closes_complete_oom_configuration_without_any_replay(self):
        value,verify=self.bound_run();out=self.root/'output'
        self.assertEqual(value['status'],'CLOSE_TESTED_BOUNDED_CONFIGURATION',value);self.assertFalse(value['pass_']);self.assertIsNone(value['U_resp']);self.assertIn('OOM counter increased',value['error'])
        self.assertEqual(value['cleanup_errors'],[]);self.assertEqual(value['surviving_phase_groups'],[]);self.assertEqual(value['retained_queries'],[])
        self.assertTrue(value['host_evidence']['oom_increased']);self.assertEqual(value['failure_evidence']['kind'],'OBSERVED_OOM')
        config=json.loads((self.root/'config.json').read_text());proof=verify(out,config,value['error'],value['failed_stage'])
        self.assertTrue(all(value[key]==result for key,result in proof.items()))
        events=list(map(json.loads,(out/'supervisor.jsonl').read_text().splitlines()))
        self.assertFalse(any(row['event']=='spawn' and row['phase'] in ('candidate.replay','candidate.qualify') for row in events))
    def test_malformed_sampled_oom_finishes_series_but_stays_invalid(self):
        value,verify=self.bound_run('malformed');out=self.root/'output'
        self.assertEqual(value['status'],'INVALID_INCONCLUSIVE');self.assertFalse(value['pass_']);self.assertIsNone(value['U_resp']);self.assertTrue(value['cleanup_errors'])
        self.assertTrue((out/'host-series.result.json').exists());self.assertEqual(value['surviving_phase_groups'],[])
