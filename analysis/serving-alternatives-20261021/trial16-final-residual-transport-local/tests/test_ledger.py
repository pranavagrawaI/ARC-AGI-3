import ast,copy,hashlib,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import settings,ledger,run_final
from retry_transport import RetryRemote
from test_retry import Base,Clock,Response
from collector import collect
import requests

class LedgerTests(unittest.TestCase):
    def test_actual_retained_ledgers_and_exact167(self):
        p=ledger.prepare();self.assertEqual(len(p['retained_files']),1419);self.assertEqual(len(p['remaining']),167)
        self.assertEqual(len(p['waivers']),10);self.assertFalse(p['first']['first']['complete']);self.assertFalse(p['second']['complete'])
        self.assertNotIn('after',p['first']['first']);self.assertNotIn('after',p['second'])
        self.assertNotIn(settings.REPEAT75,{r['name'] for r in p['remaining']});self.assertIn(settings.ACTUAL_RESULT,{r['name'] for r in p['remaining']})
        c=ledger.lifetime_counts(p,__import__('collections').Counter({r['name']:2 for r in p['remaining']}))
        self.assertEqual(c[settings.FAILED],3);self.assertEqual(c[settings.REPEAT75],2);self.assertEqual(sum(c.values()),1755)
    def test_no_retained_redownload(self):
        p=ledger.prepare();counts=__import__('collections').Counter({r['name']:1 for r in p['remaining']});counts[settings.REPEAT75]=1
        with self.assertRaises(ValueError):ledger.lifetime_counts(p,counts)
    def test_third_connection_in_same_pass_rejected(self):
        p=ledger.prepare();counts=__import__('collections').Counter({r['name']:1 for r in p['remaining']});counts[settings.FAILED]=3
        with self.assertRaises(ValueError):ledger.lifetime_counts(p,counts)
    def test_local_receipt_not_remote_authority(self):
        with self.assertRaises(ValueError):run_final.approval_fields(settings.read(settings.REVIEW))
    def test_exact_policy_gate(self):
        a=dict(decision=run_final.DECISION,remote_authorized=True,kernel_id=140493086,version=1,bounds=settings.POLICY,retry_policy=run_final.RETRY_POLICY,first_collection_sha256=settings.FIRST_SHA,second_collection_sha256=settings.SECOND_SHA)
        run_final.approval_fields(a)
        for change in ({'bounds':settings.POLICY|{'new_download_attempts_max':335}},{'retry_policy':run_final.RETRY_POLICY|{'additional_attempts':2}},{'second_collection_sha256':'other'}):
            with self.subTest(change=change),self.assertRaises(ValueError):run_final.approval_fields(a|change)
    def journal_fixture(self,p):
        c=Clock();b=Base([requests.ConnectTimeout(),Response()]);a=RetryRemote(b,{'a'},p,900,clock=c,sleep=c.sleep);a.list_page('',100)
        with a.download('https://fixture.invalid/a?secret=HIDDEN',900) as response:
            (p/'members').mkdir();(p/'members/a').write_bytes(b''.join(response.iter_content(chunk_size=65536)))
        return dict(transport_attempts=2,transport_journal_sha256=settings.sha(p/'transport-attempts.jsonl')),{'a':{'bytes':1}}
    def mutate_journal(self,p,state,edit):
        r=[json.loads(x) for x in (p/'transport-attempts.jsonl').read_bytes().splitlines()];edit(r)
        (p/'transport-attempts.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in r));state['transport_journal_sha256']=settings.sha(p/'transport-attempts.jsonl')
    def test_real_adapter_journal_validated(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);s,f=self.journal_fixture(p);counts,retries=ledger.transport_journal(p,s,f)
            self.assertEqual(counts,{'a':2});self.assertEqual(retries,1)
    def test_journal_response_present_retry_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);s,f=self.journal_fixture(p);self.mutate_journal(p,s,lambda r:r[1].update(response_absent=False))
            with self.assertRaises(ValueError):ledger.transport_journal(p,s,f)
    def test_journal_short_delay_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);s,f=self.journal_fixture(p);self.mutate_journal(p,s,lambda r:r[2].update(at=.99))
            with self.assertRaises(ValueError):ledger.transport_journal(p,s,f)
    def test_journal_partial_retry_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);s,f=self.journal_fixture(p);self.mutate_journal(p,s,lambda r:r[1].update(persisted_bytes=1))
            with self.assertRaises(ValueError):ledger.transport_journal(p,s,f)
    def test_journal_cap_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);s,f=self.journal_fixture(p);s['transport_attempts']=335
            with self.assertRaises(ValueError):ledger.transport_journal(p,s,f)
    def test_900_driver_lateness_invalidates(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);clock=iter([.1,.2,900.1]);r=run_final.persist(p,{'complete':True},0,clock=lambda:next(clock))
            self.assertFalse(r['complete']);self.assertEqual(settings.read(p/'DRIVER-COMPLETION.json')['deadline'],900)
    def test_third_origin_retains_prior_origins(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);third=p/'third';(third/'members').mkdir(parents=True);(third/'members/c').write_bytes(b'c');(third/'COLLECTION.json').write_text('{}')
            evidence=p/'evidence';evidence.mkdir();(evidence/'a').write_bytes(b'a');origins={'a':[{'collection_pass':1}]}
            row=dict(name='c',bytes=1,sha256=hashlib.sha256(b'c').hexdigest(),origin='INDEPENDENT_CURRENT_SESSION_DOWNLOAD',complete=True,attempt=1)
            ledger.append_third(third,evidence,origins,{'c':row})
            self.assertEqual(origins['a'][0]['collection_pass'],1);self.assertEqual(origins['c'][0]['collection_pass'],3)
            self.assertEqual(origins['c'][0]['collection_receipt_sha256'],settings.sha(third/'COLLECTION.json'))
            self.assertEqual((evidence/'a').read_bytes(),b'a')
    def test_local_decision_cannot_run_audit(self):
        import audit_three_pass
        with tempfile.TemporaryDirectory() as t:
            output=Path(t)/'new'
            with self.assertRaises(ValueError):audit_three_pass.supplemental(settings.REVIEW,settings.SECOND,output)
            self.assertFalse(output.exists())
    def test_scientific_body_unchanged(self):
        old=(settings.AUDIT_R2/'audit.py').read_text();new=(settings.HERE/'audit_three_pass.py').read_text()
        self.assertEqual(old[old.index("        prefix='serving"):old.index('        completed_proof=')],new[new.index("        prefix='serving"):new.index('        output_publication.durable_write')])
        a=ast.parse(old);b=ast.parse(new)
        for name in ('independent_actual_result','merge_inputs'):
            select=lambda tree:ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name))
            self.assertEqual(select(a),select(b))

class CollectorIntegrationTests(unittest.TestCase):
    def collect(self,p,values,byte_cap=512*1024**2):
        class API(Base):
            def readback(self,dest,remaining):
                from collector import IDENTITY
                dest.mkdir();return dict(IDENTITY)
        c=Clock();b=API(values);a=RetryRemote(b,{'a'},p,900,clock=c,sleep=c.sleep)
        result=collect([dict(name='a',conditional=False,expected_sha256=None,expected_bytes=None)],p,a,limits=settings.CAPS|{'bytes':byte_cap},clock=c)
        return result,a,b
    def test_unchanged_collector_with_one_safe_retry(self):
        with tempfile.TemporaryDirectory() as t:
            r,a,b=self.collect(Path(t)/'new',[requests.ConnectTimeout(),Response()]);self.assertTrue(r['complete']);self.assertEqual(r['download_attempts'],1);self.assertEqual(a.attempts,2);self.assertEqual(r['downloaded_bytes'],1)
    def test_size_failure_no_retry(self):
        with tempfile.TemporaryDirectory() as t:
            response=Response();response.headers['content-length']='2';r,a,b=self.collect(Path(t)/'new',[response,Response()],byte_cap=1)
            self.assertFalse(r['complete']);self.assertEqual(b.calls,1);self.assertEqual(r['downloaded_bytes'],0)
    def test_hash_failure_no_retry_actual_retained(self):
        class API(Base):
            def readback(self,dest,remaining):
                from collector import IDENTITY
                dest.mkdir();return dict(IDENTITY)
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'new';c=Clock();b=API([Response(),Response()]);a=RetryRemote(b,{'a'},p,900,clock=c,sleep=c.sleep)
            r=collect([dict(name='a',conditional=False,expected_sha256='wrong',expected_bytes=None)],p,a,limits=settings.CAPS,clock=c)
            self.assertFalse(r['complete']);self.assertEqual(b.calls,1);self.assertEqual((p/'members/a').read_bytes(),b'x')
if __name__=='__main__':unittest.main()
