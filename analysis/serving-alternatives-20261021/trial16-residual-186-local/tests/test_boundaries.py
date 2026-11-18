"""Changed adapter/ledger seams only. No scientific outcome simulation."""
import copy,hashlib,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import context,bridge,run_residual
from collector import collect
spec=importlib.util.spec_from_file_location('approved_collector_fixtures',context.APPROVED/'tests/test_collector.py')
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)

class Boundaries(unittest.TestCase):
    def make(self,directory):
        api=fixtures.API({'a':b'abc','retained/not-requested':b'ignore'})
        state=collect([fixtures.row('a')],directory,api,limits=context.CAPS)
        state['driver_elapsed_seconds']=state['elapsed_seconds']
        return state,api
    def test_filter_and_smaller_limits_real_collector(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'new';state,api=self.make(p)
            files,names=bridge.ledger(p,state,{'a':fixtures.row('a')},context.CAPS,True)
            self.assertEqual(names,['a']);self.assertEqual(api.downloads,['a']);self.assertEqual(len(api.identities),2)
            self.assertEqual(files['a']['sha256'],hashlib.sha256(b'abc').hexdigest())
    def test_first_error_stops_no_after_no_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            api=fixtures.API({'a':TimeoutError('fixture'),'b':b'b'})
            state=collect([fixtures.row('a'),fixtures.row('b')],Path(tmp)/'new',api,limits=context.CAPS)
            self.assertFalse(state['complete']);self.assertEqual(api.downloads,['a']);self.assertEqual(len(api.identities),1)
    def test_smaller_byte_cap_rejects_before_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            state=collect([fixtures.row('a')],Path(tmp)/'new',fixtures.API({'a':b'abc'}),limits=context.CAPS|{'bytes':2})
            self.assertFalse(state['complete']);self.assertEqual(state['downloaded_bytes'],0)
    def test_residual_187_rows_rejected_before_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            api=fixtures.API({});state=collect([fixtures.row(str(i)) for i in range(187)],Path(tmp)/'new',api,limits=context.CAPS)
            self.assertFalse(state['complete']);self.assertEqual(api.identities,[])
    def test_ledger_rejects_hash_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'new';state,_=self.make(p);(p/'members/a').write_bytes(b'bad')
            with self.assertRaises(ValueError):bridge.ledger(p,state,{'a':fixtures.row('a')},context.CAPS,True)
    def test_ledger_rejects_wrong_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'new';state,_=self.make(p);state['files'][0]['origin']='PRIMARY_EMBEDDED'
            with self.assertRaises(ValueError):bridge.ledger(p,state,{'a':fixtures.row('a')},context.CAPS,True)
    def test_ledger_rejects_budget_overrun(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'new';state,_=self.make(p);state['driver_elapsed_seconds']=600
            with self.assertRaises(ValueError):bridge.ledger(p,state,{'a':fixtures.row('a')},context.CAPS,True)
    def test_ledger_rejects_duplicate_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'new';state,_=self.make(p)
            with (p/'attempts.jsonl').open('a') as stream:stream.write(json.dumps({'name':'a','attempt':1})+'\n')
            with self.assertRaises(ValueError):bridge.ledger(p,state,{'a':fixtures.row('a')},context.CAPS,True)
    def test_new_driver_rejects_inherited_1800_clock(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'COLLECTION.json').write_text('{}')
            (p/'DRIVER-COMPLETION.json').write_text(json.dumps(dict(schema='RESIDUAL_DRIVER_COMPLETION_V1',started=1,deadline=1801,receipt_persisted_end=2,receipt_sha256=context.sha(p/'COLLECTION.json'))))
            with self.assertRaises(ValueError):bridge.driver(p,600,'RESIDUAL_DRIVER_COMPLETION_V1')
    def test_approval_exact_source_binding(self):
        a=dict(decision=run_residual.DECISION,remote_authorized=True,kernel_id=140493086,version=1,bounds=context.CAPS,repeat_connection_name=context.REPEAT,first_collection_sha256=context.FIRST_SHA)
        run_residual.approval_fields(a)
        for key,value in [('first_collection_sha256','changed'),('repeat_connection_name','other'),('version',2)]:
            b=a|{key:value}
            with self.subTest(key=key),self.assertRaises(ValueError):run_residual.approval_fields(b)
    def test_saved_readback_drift_rejected(self):
        state=context.read(context.FIRST/'COLLECTION.json');state['before']['version']=2
        with self.assertRaises(ValueError):bridge.readback(context.FIRST,state,'before')
    def test_actual_result_rewrite_still_retained_and_rejected(self):
        import audit_two_pass,base64
        actual=json.dumps({'status':'INVALID_INCONCLUSIVE','primary_bundle_error':'late'}).encode()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'new';state=collect([fixtures.row(context.ACTUAL_RESULT)],p,fixtures.API({context.ACTUAL_RESULT:actual}),limits=context.CAPS)
            self.assertTrue(state['complete'])
            with self.assertRaisesRegex(ValueError,'actual result differs'):
                audit_two_pass.independent_actual_result(p,{'files':{'result.json':{'base64':base64.b64encode(b'{"status":"FEASIBLE"}').decode()}}})
            self.assertEqual((p/'members'/context.ACTUAL_RESULT).read_bytes(),actual)
    def test_second_origin_preserves_first_and_pass_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);second=base/'second';(second/'members').mkdir(parents=True)
            (second/'COLLECTION.json').write_text('{}');(second/'members/b').write_bytes(b'b')
            evidence=base/'evidence';evidence.mkdir();(evidence/'a').write_bytes(b'a')
            origins={'a':[{'collection_pass':1,'collection_receipt_sha256':context.FIRST_SHA}]}
            row=dict(name='b',bytes=1,sha256=hashlib.sha256(b'b').hexdigest(),origin='INDEPENDENT_CURRENT_SESSION_DOWNLOAD',complete=True,attempt=1)
            bridge.append_second(second,evidence,origins,{'b':row})
            self.assertEqual((evidence/'a').read_bytes(),b'a');self.assertEqual((evidence/'b').read_bytes(),b'b')
            self.assertEqual(origins['a'][0]['collection_pass'],1);self.assertEqual(origins['b'][0]['collection_pass'],2)
            self.assertEqual(origins['b'][0]['collection_receipt_sha256'],context.sha(second/'COLLECTION.json'))
    def test_second_origin_rejects_overwriting_retained_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);second=base/'second';(second/'members').mkdir(parents=True)
            (second/'COLLECTION.json').write_text('{}');(second/'members/a').write_bytes(b'new')
            evidence=base/'evidence';evidence.mkdir();(evidence/'a').write_bytes(b'old')
            with self.assertRaises(ValueError):bridge.append_second(second,evidence,{}, {'a':{'sha256':hashlib.sha256(b'new').hexdigest()}})
            self.assertEqual((evidence/'a').read_bytes(),b'old')
    def test_audit_cannot_run_from_local_preparation(self):
        import audit_two_pass
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'new'
            with self.assertRaises(ValueError):audit_two_pass.supplemental(context.REVIEW,context.FIRST,p)
            self.assertFalse(p.exists())
