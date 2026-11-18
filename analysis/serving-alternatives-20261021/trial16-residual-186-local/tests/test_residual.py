import copy,hashlib,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bridge,context,run_residual

class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.a={n:{'conditional':n=='waived'} for n in ('a','failed','new','waived')}
    def test_two_pass_counts(self):
        got=bridge.partition(self.a,{'a'},['a','failed'],{'waived'},{'failed','new'},['failed','new'],'failed')
        self.assertEqual(got,dict(successful_files=3,waived_files=1,total_attempts=4,repeated_names=['failed']))
    def test_reject_redownload_complete(self):
        with self.assertRaises(ValueError):bridge.partition(self.a,{'a'},['a','failed'],{'waived'},{'a','failed','new'},['a','failed','new'],'failed')
    def test_reject_unapproved_repeat(self):
        with self.assertRaises(ValueError):bridge.partition(self.a,{'a'},['a','failed','new'],{'waived'},{'failed','new'},['failed','new'],'failed')
    def test_reject_third_attempt(self):
        with self.assertRaises(ValueError):bridge.partition(self.a,{'a'},['a','failed'],{'waived'},{'failed','new'},['failed','failed','new'],'failed')
    def test_reject_missing_dependency(self):
        with self.assertRaises(ValueError):bridge.partition(self.a,{'a'},['a','failed'],{'waived'},{'failed'},['failed'],'failed')
    def test_reject_primary_result_waiver(self):
        with self.assertRaises(ValueError):bridge.partition(self.a,{'a'},['a','failed'],{'waived','new'},{'failed'},['failed'],'failed')

class DriverTests(unittest.TestCase):
    def test_600_deadline_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);state={'complete':True};values=iter([10.1,10.2,10.3])
            got=run_residual.persist(p,state,10,clock=lambda:next(values))
            self.assertTrue(got['complete']);c=json.loads((p/'DRIVER-COMPLETION.json').read_bytes())
            self.assertEqual(c['deadline'],610);self.assertEqual(c['schema'],'RESIDUAL_DRIVER_COMPLETION_V1')
            self.assertEqual(c['receipt_sha256'],context.sha(p/'COLLECTION.json'))
    def test_late_persistence_invalidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);values=iter([10.1,10.2,610.01]);state=run_residual.persist(p,{'complete':True},10,clock=lambda:next(values))
            self.assertFalse(state['complete']);self.assertIn('deadline',state['error']['reason'])
    def test_local_preparation_not_remote_authority(self):
        with self.assertRaises(ValueError):run_residual.approval_fields({'remote_authorized':False})
    def test_exact_authority_rejects_expanded_bounds(self):
        a=dict(decision=run_residual.DECISION,remote_authorized=True,kernel_id=140493086,version=1,bounds=context.CAPS|{'downloads':187},repeat_connection_name=context.REPEAT)
        with self.assertRaises(ValueError):run_residual.approval_fields(a)

class RealRetainedTests(unittest.TestCase):
    def test_actual_first_pass_and_waivers(self):
        plan=bridge.prepare();self.assertEqual(len(plan['remaining']),186);self.assertEqual(len(plan['waivers']),10)
        self.assertEqual(len(plan['first_files']),1400);self.assertEqual(len(plan['first_attempts']),1401)
        self.assertFalse(plan['first']['complete']);self.assertNotIn('after',plan['first'])
        self.assertIn(context.ACTUAL_RESULT,{r['name'] for r in plan['remaining']})
        self.assertTrue(all(r['conditional'] is False for r in plan['remaining']))
    def test_first_receipt_exact_pin(self):
        self.assertEqual(context.sha(context.FIRST/'COLLECTION.json'),context.FIRST_SHA)
    def test_raw_waiver_requires_no_body(self):
        from collector import pending_body_required
        row=json.loads(context.REVIEW.read_bytes())['conditional_waivers'][0]
        h=json.loads((context.FIRST/'members'/row['http_name']).read_bytes());h['status']=200;h['headers']=[['Content-Length','0']]
        self.assertTrue(pending_body_required(h,row['name']))
    def test_approved_functions_byte_identical(self):
        import ast
        old=ast.parse((context.AUDIT_R2/'audit.py').read_text());new=ast.parse((context.HERE/'audit_two_pass.py').read_text())
        for name in ('independent_actual_result','merge_inputs'):
            a=next(x for x in old.body if isinstance(x,ast.FunctionDef) and x.name==name)
            b=next(x for x in new.body if isinstance(x,ast.FunctionDef) and x.name==name)
            self.assertEqual(ast.dump(a),ast.dump(b))
        a=(context.AUDIT_R2/'audit.py').read_text();b=(context.HERE/'audit_two_pass.py').read_text()
        self.assertEqual(a[a.index("        prefix='serving"):a.index('        completed_proof=')],b[b.index("        prefix='serving"):b.index('        output_publication.durable_write')])

if __name__=='__main__':unittest.main()
