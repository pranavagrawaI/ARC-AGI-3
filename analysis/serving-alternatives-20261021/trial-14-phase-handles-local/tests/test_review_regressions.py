"""Public boundary and full archive-reducer regressions; CPU/read-only fixtures."""
import contextlib,copy,importlib.util,io,json,os,shutil,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import host_observer,release_runner
import test_lifecycle as lifecycle_fixture
HERE=Path(__file__).resolve().parents[1]
REVIEW=HERE.parent/'coordination-20261023/trial13-binding-review-evidence'

class BoundaryEvidence(unittest.TestCase):
 def exercise(self,kind):
  with tempfile.TemporaryDirectory() as td:
   out=Path(td)/'output';runner=release_runner.Runner(lifecycle_fixture.Lifecycle().config(),out)
   scope=host_observer.bind('CPU_BOUNDARY');row=host_observer.snapshot(scope,'candidate.prepare.after')
   runner.initial_events=dict(row['oom_events']);runner.config['observer']['boundary_timeout_seconds']=.01
   if kind=='oom':
    row['oom_events']['oom_kill']+=1
    row['raw']['events']=''.join(f'{k} {v}\n' for k,v in row['oom_events'].items())
   def returned(*args):
    row['start']=time.monotonic()
    if kind=='late':time.sleep(.03)
    row['end']=time.monotonic();return row
   try:
    with patch.object(host_observer,'snapshot',side_effect=returned):
     with self.assertRaisesRegex(ValueError,'boundary deadline' if kind=='late' else 'OOM counter'):runner.boundary('candidate.prepare.after')
    path=out/'host-boundaries.jsonl';self.assertTrue(path.exists(),'available invalid snapshot lost')
    saved=json.loads(path.read_text());self.assertFalse(saved['valid']);self.assertTrue(saved['invalidity']);self.assertEqual(saved['raw'],row['raw'])
    events=[json.loads(s) for s in (out/'supervisor.jsonl').read_text().splitlines()]
    self.assertTrue(any(e['event']=='host_boundary' and not e['valid'] for e in events))
   finally:
    runner.trace.close()
    if os.environ.get('PACKET_PROOF_DIR'):shutil.copytree(out,Path(os.environ['PACKET_PROOF_DIR'])/('boundary-'+kind))
 def test_returned_oom_snapshot_is_persisted_before_rejection(self):self.exercise('oom')
 def test_returned_late_snapshot_is_persisted_before_rejection(self):self.exercise('late')
 def test_boundary_persistence_failure_still_raises(self):
  with tempfile.TemporaryDirectory() as td:
   runner=release_runner.Runner(lifecycle_fixture.Lifecycle().config(),Path(td)/'output');scope=host_observer.bind('CPU_PERSIST');row=host_observer.snapshot(scope,'fixture')
   original=Path.open
   def failing(path,*args,**kwargs):
    if path.name=='host-boundaries.jsonl':raise OSError('injected boundary persistence failure')
    return original(path,*args,**kwargs)
   try:
    with patch.object(host_observer,'snapshot',return_value=row),patch.object(Path,'open',failing),self.assertRaisesRegex(OSError,'persistence failure'):runner.boundary('fixture')
   finally:runner.trace.close()

class ArchiveReducer(unittest.TestCase):
 def run_case(self,case):
  spec=importlib.util.spec_from_file_location('reviewed_archive_reduce',HERE/'collection_tools/reduce.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  with tempfile.TemporaryDirectory() as td:
   out=Path(td)/'derived';fixture=REVIEW/case
   with patch.object(module,'HERE',REVIEW/'envelope'),patch.object(sys,'argv',['reduce.py',str(fixture/'archive'),str(fixture/'readback'),str(out)]),contextlib.redirect_stdout(io.StringIO()):module.main()
   value=json.loads((out/'REDUCTION.json').read_text())
   if os.environ.get('PACKET_PROOF_DIR'):shutil.copytree(out,Path(os.environ['PACKET_PROOF_DIR'])/('reducer-'+case))
   return value
 def test_complete_captured_cpu_baseline_remains_qualified(self):self.assertEqual(self.run_case('baseline')['status'],'POST_CONTROL_CANDIDATE_LOAD_TO_HEALTH_ONLY')
 def test_failed_release_never_qualifies(self):self.assertEqual(self.run_case('release_false')['status'],'NOT_QUALIFIED')
 def test_fresh_sample_after_original_release_deadline_never_qualifies(self):self.assertEqual(self.run_case('fresh_after_original_release_deadline')['status'],'NOT_QUALIFIED')
 def test_boundary_oom_increase_never_qualifies(self):self.assertEqual(self.run_case('boundary_oom')['status'],'NOT_QUALIFIED')
 def test_nonterminal_readback_does_not_claim_terminal_quota_or_slot_release(self):
  v=self.run_case('nonterminal_readback');self.assertFalse(v['terminal_collection_complete']);self.assertIsNone(v['terminal_typed_quota']);self.assertTrue(v['collection_gaps']);self.assertFalse(v['slot_release_claim'])
