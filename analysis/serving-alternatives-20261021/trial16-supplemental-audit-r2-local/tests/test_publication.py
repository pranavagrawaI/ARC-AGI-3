"""Output-finalization fixtures only; never fake a complete scientific checker."""
import ast,hashlib,json,os,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
import publication

PROOF=dict(status='CANDIDATE_ALONE_WORKLOAD_FEASIBLE',U_resp=1.)
def initial():return dict(status='INVALID_INCONCLUSIVE',U_resp=None,supplemental_complete=False,issues=[])
def write(path,value):path.write_text(json.dumps(value)+'\n')

class PublicationTests(unittest.TestCase):
    def test_origins_write_error_cannot_keep_positive_verdict(self):
        def fail_origins(path,value):
            if path.name=='ORIGINS.json':raise OSError('fixture origins write failed')
            write(path,value)
        with tempfile.TemporaryDirectory() as d:
            result=publication.publish(Path(d),initial(),PROOF,{'file':{'sha256':'fixture'}},fail_origins)
            self.assertEqual(result['status'],'INVALID_INCONCLUSIVE');self.assertIsNone(result['U_resp']);self.assertFalse(result['supplemental_complete'])

    def test_success_requires_origins_report_and_hash_bound_completion(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);result=publication.publish(root,initial(),PROOF,{'file':{'sha256':'fixture'}},publication.durable_write)
            self.assertTrue(result['supplemental_complete']);self.assertEqual(result['U_resp'],1.)
            self.assertEqual(publication.read_completed(root),result)
            completed=json.loads((root/'AUDIT-COMPLETION.json').read_text())
            self.assertEqual(completed['origins_sha256'],hashlib.sha256((root/'ORIGINS.json').read_bytes()).hexdigest())

    def test_report_write_error_resets_returned_and_read_verdict(self):
        def fail_report(path,value):
            if path.name=='SUPPLEMENTAL-AUDIT.json':raise OSError('fixture final report write failed')
            publication.durable_write(path,value)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);result=publication.publish(root,initial(),PROOF,{},fail_report)
            self.assertFalse(result['supplemental_complete']);self.assertIsNone(result['U_resp'])
            self.assertEqual(publication.read_completed(root)['status'],'INVALID_INCONCLUSIVE')
            self.assertFalse((root/'AUDIT-COMPLETION.json').exists())

    def test_report_bytes_written_then_flush_error_are_not_accepted(self):
        def fail_flush(path,value):
            write(path,value)
            if path.name=='SUPPLEMENTAL-AUDIT.json':raise OSError('fixture fsync failed after positive bytes')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);result=publication.publish(root,initial(),PROOF,{},fail_flush)
            self.assertEqual(json.loads((root/'SUPPLEMENTAL-AUDIT.json').read_text())['U_resp'],1.)
            self.assertFalse(result['supplemental_complete']);self.assertIsNone(result['U_resp'])
            self.assertFalse(publication.read_completed(root)['supplemental_complete'])

    def test_completion_pending_write_error_cannot_publish_completion(self):
        def fail_pending(path,value):
            write(path,value)
            if path.name=='AUDIT-COMPLETION.pending.json':raise OSError('fixture completion flush failure')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);result=publication.publish(root,initial(),PROOF,{},fail_pending)
            self.assertFalse(result['supplemental_complete']);self.assertFalse((root/'AUDIT-COMPLETION.json').exists())
            self.assertFalse(publication.read_completed(root)['supplemental_complete'])

    def test_report_tampering_after_completion_is_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);publication.publish(root,initial(),PROOF,{},publication.durable_write)
            (root/'SUPPLEMENTAL-AUDIT.json').write_text('{"supplemental_complete":true,"U_resp":999}')
            self.assertIsNone(publication.read_completed(root)['U_resp'])

    def test_failure_receipt_persistence_error_still_returns_invalid(self):
        def fail_all(path,value):raise OSError('fixture disk unavailable')
        with tempfile.TemporaryDirectory() as d:
            result=publication.publish(Path(d),initial(),PROOF,{},fail_all)
            self.assertEqual(result['status'],'INVALID_INCONCLUSIVE');self.assertIsNone(result['U_resp'])
            self.assertIn('failure_receipt_persistence_error',result)

    def test_scientific_functions_and_checker_calls_unchanged(self):
        here=Path(__file__).resolve().parents[1]
        old=ast.parse((here/'evidence/approved-audit.py.txt').read_text());new=ast.parse((here/'audit.py').read_text())
        olddefs={x.name:x for x in old.body if isinstance(x,ast.FunctionDef)};newdefs={x.name:x for x in new.body if isinstance(x,ast.FunctionDef)}
        for name in olddefs.keys()-{'supplemental'}:
            self.assertEqual(ast.dump(olddefs[name]),ast.dump(newdefs[name]),name)
        for call in ('candidate_evidence.check','primary_bundle.verify_publication','primary.verify'):
            original=[ast.dump(x) for x in ast.walk(olddefs['supplemental']) if isinstance(x,ast.Call) and ast.unparse(x.func)==call]
            corrected=[ast.dump(x) for x in ast.walk(newdefs['supplemental']) if isinstance(x,ast.Call) and ast.unparse(x.func)==call]
            self.assertEqual(original,corrected,call)

    def test_real_report_fsync_error_is_not_accepted(self):
        original=os.fsync
        def fault(fd):
            if Path(os.readlink(f'/proc/self/fd/{fd}')).name=='SUPPLEMENTAL-AUDIT.json':raise OSError('fixture report fsync error')
            return original(fd)
        with tempfile.TemporaryDirectory() as d,patch('publication.os.fsync',side_effect=fault):
            root=Path(d);result=publication.publish(root,initial(),PROOF,{},publication.durable_write)
            self.assertFalse(result['supplemental_complete']);self.assertIsNone(publication.read_completed(root)['U_resp'])

    def test_final_directory_flush_failure_quarantines_completion(self):
        original=os.fsync;directory_calls=[]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            def fault(fd):
                if Path(os.readlink(f'/proc/self/fd/{fd}'))==root:
                    directory_calls.append(fd)
                    if len(directory_calls)==4:raise OSError('fixture final directory fsync error')
                return original(fd)
            with patch('publication.os.fsync',side_effect=fault):result=publication.publish(root,initial(),PROOF,{},publication.durable_write)
            self.assertFalse(result['supplemental_complete']);self.assertFalse((root/'AUDIT-COMPLETION.json').exists())
            self.assertTrue((root/'AUDIT-COMPLETION.failed.json').exists());self.assertIsNone(publication.read_completed(root)['U_resp'])

if __name__=='__main__':unittest.main()
