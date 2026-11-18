import base64,hashlib,io,json,shutil,tempfile,unittest
from pathlib import Path
import collector

IDENTITY=dict(kernel_id=140493086,version=1,source_exact=True,status='COMPLETE',inner_sha256='fdab81dc2113e336eaf38d46ccff1ed9e8b4b65efbe7811d219d88933c3559f4')

class Stream:
    def __init__(self,data,status=200,length=None):
        self.data=data;self.status_code=status;self.headers={'content-length':str(len(data) if length is None else length)}
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def raise_for_status(self):
        if self.status_code>=400:raise RuntimeError('HTTP failure')
    def iter_content(self,chunk_size):yield self.data

class API:
    def __init__(self,files,pages=None):
        self.files=files;self.pages=pages or [[(n,'https://fixture.invalid/'+n+'?secret=DO_NOT_SAVE') for n in files]]
        self.downloads=[];self.lists=[];self.identities=[]
    def readback(self,destination,remaining):
        destination.mkdir();self.identities.append(str(destination));return dict(IDENTITY)
    def list_page(self,token,page_size):
        index=int(token or 0);self.lists.append((token,page_size));return self.pages[index],str(index+1) if index+1<len(self.pages) else ''
    def download(self,url,remaining):
        name=url.split('fixture.invalid/',1)[1].split('?',1)[0];self.downloads.append(name)
        value=self.files[name]
        if isinstance(value,Exception):raise value
        return value if isinstance(value,Stream) else Stream(value)

def row(name,conditional=False,**kwargs):return dict(name=name,conditional=conditional,expected_sha256=None,expected_bytes=None,**kwargs)

class CollectorTests(unittest.TestCase):
    def test_only_exact_missing_names_are_downloaded(self):
        api=API({'needed/raw.bin':b'abc','needed/raw.bin.extra':b'ignore','already/recovered.bin':b'ignore'})
        with tempfile.TemporaryDirectory() as d:
            result=collector.collect([row('needed/raw.bin')],Path(d)/'new',api)
            self.assertTrue(result['complete']);self.assertEqual(api.downloads,['needed/raw.bin'])
            self.assertEqual((Path(d)/'new/members/needed/raw.bin').read_bytes(),b'abc')
            self.assertNotIn('DO_NOT_SAVE',json.dumps(result));self.assertNotIn('https://',json.dumps(result))

    def test_actual_late_invalid_result_is_retained_and_rejected(self):
        import audit
        actual=json.dumps({'status':'INVALID_INCONCLUSIVE','primary_bundle_error':'late receipt'}).encode()
        bundled=json.dumps({'status':'CANDIDATE_ALONE_WORKLOAD_FEASIBLE'}).encode()
        api=API({audit.ACTUAL_RESULT:actual})
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'new';result=collector.collect([row(audit.ACTUAL_RESULT)],target,api)
            self.assertTrue(result['complete']);self.assertEqual((target/'members'/audit.ACTUAL_RESULT).read_bytes(),actual)
            bundle={'files':{'result.json':{'base64':base64.b64encode(bundled).decode()}}}
            with self.assertRaisesRegex(ValueError,'actual result differs'):
                audit.independent_actual_result(target,bundle)

    def test_pending_body_waits_for_http200_evidence(self):
        name='candidate/rid/body.bin';h='candidate/rid/http.json'
        http=json.dumps(dict(rid='rid',status=200,received_bytes=3,persisted_bytes=3,headers=[['content-length','5']],complete=False)).encode()
        api=API({name:b'abc',h:http})
        with tempfile.TemporaryDirectory() as d:
            result=collector.collect([row(name,True),row(h)],Path(d)/'new',api)
            self.assertTrue(result['complete']);self.assertEqual(api.downloads,[h,name])

    def test_no_headers_and_zero_bytes_skips_conditional_body(self):
        name='candidate/rid/body.bin';h='candidate/rid/http.json'
        http=json.dumps(dict(rid='rid',status=None,received_bytes=0,persisted_bytes=0,complete=False)).encode()
        api=API({h:http})
        with tempfile.TemporaryDirectory() as d:
            result=collector.collect([row(name,True),row(h)],Path(d)/'new',api)
            self.assertTrue(result['complete']);self.assertEqual(api.downloads,[h]);self.assertEqual(result['skipped_conditionals'][0]['name'],name)

    def test_malformed_pending_receipt_stops_before_body(self):
        name='candidate/rid/body.bin';h='candidate/rid/http.json';api=API({h:b'{"rid":"other","status":200}',name:b'x'})
        with tempfile.TemporaryDirectory() as d:
            result=collector.collect([row(name,True),row(h)],Path(d)/'new',api)
            self.assertFalse(result['complete']);self.assertEqual(api.downloads,[h]);self.assertEqual(len(api.identities),1)

    def test_unsafe_manifest_has_no_api_or_download(self):
        for name in ('../escape','/absolute','a/../b','a\\b','a//b'):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as d:
                api=API({});result=collector.collect([row(name)],Path(d)/'new',api)
                self.assertFalse(result['complete']);self.assertEqual(api.lists,[]);self.assertEqual(api.identities,[])

    def test_repeated_pagination_token_stops(self):
        class Repeated(API):
            def list_page(self,token,page_size):self.lists.append((token,page_size));return [],'same'
        api=Repeated({})
        with tempfile.TemporaryDirectory() as d:
            result=collector.collect([row('missing')],Path(d)/'new',api)
            self.assertFalse(result['complete']);self.assertEqual(len(api.lists),2);self.assertEqual(api.downloads,[])

    def test_page_limit_stops_before_next_request(self):
        api=API({'a':b'a'},pages=[[],[('a','https://fixture.invalid/a')]])
        with tempfile.TemporaryDirectory() as d:
            result=collector.collect([row('a')],Path(d)/'new',api,limits={'list_pages':1})
            self.assertFalse(result['complete']);self.assertEqual(len(api.lists),1);self.assertEqual(api.downloads,[])

    def test_one_failed_file_stops_without_retry_or_post_readback(self):
        api=API({'a':RuntimeError('https://fixture.invalid/a?secret=DO_NOT_SAVE'),'b':b'b'})
        with tempfile.TemporaryDirectory() as d:
            result=collector.collect([row('a'),row('b')],Path(d)/'new',api)
            self.assertFalse(result['complete']);self.assertEqual(api.downloads,['a']);self.assertEqual(len(api.identities),1)
            self.assertNotIn('DO_NOT_SAVE',json.dumps(result))

    def test_byte_cap_refuses_body_before_stream_consumption(self):
        api=API({'a':b'abcd'})
        with tempfile.TemporaryDirectory() as d:
            result=collector.collect([row('a')],Path(d)/'new',api,limits={'bytes':3})
            self.assertFalse(result['complete']);self.assertEqual(result['downloaded_bytes'],0)
            self.assertFalse((Path(d)/'new/members/a').exists())

    def test_hash_mismatch_retains_actual_bytes_then_stops(self):
        api=API({'a':b'actual','b':b'b'})
        with tempfile.TemporaryDirectory() as d:
            item=row('a');item['expected_sha256']='0'*64
            result=collector.collect([item,row('b')],Path(d)/'new',api)
            self.assertFalse(result['complete']);self.assertEqual((Path(d)/'new/members/a').read_bytes(),b'actual');self.assertEqual(api.downloads,['a'])

    def test_after_identity_drift_invalidates_completed_downloads(self):
        class Drift(API):
            def readback(self,destination,remaining):
                value=super().readback(destination,remaining)
                if len(self.identities)==2:value['version']=2
                return value
        api=Drift({'a':b'a'})
        with tempfile.TemporaryDirectory() as d:
            result=collector.collect([row('a')],Path(d)/'new',api)
            self.assertFalse(result['complete']);self.assertEqual(api.downloads,['a']);self.assertIn('after identity',result['error']['reason'])

    def test_wall_lateness_stops_after_inflight_list_and_no_download(self):
        class Clock:
            now=0
            def __call__(self):return self.now
        clock=Clock()
        class Late(API):
            def list_page(self,*args):
                result=super().list_page(*args);clock.now=11;return result
        api=Late({'a':b'a'})
        with tempfile.TemporaryDirectory() as d:
            result=collector.collect([row('a')],Path(d)/'new',api,limits={'wall_seconds':10},clock=clock)
            self.assertFalse(result['complete']);self.assertEqual(api.downloads,[])

    def test_exclusive_destination_refuses_overwrite(self):
        api=API({})
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileExistsError):collector.collect([],Path(d),api)
            self.assertEqual(api.identities,[])

    def test_embedded_result_alias_cannot_be_actual_result(self):
        import audit
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'COLLECTION.json').write_text(json.dumps({'files':[dict(name=audit.ACTUAL_RESULT,origin='VERIFIED_CURRENT_PRIMARY_EMBEDDED',complete=True)]}))
            with self.assertRaisesRegex(ValueError,'independent fetched actual-result origin'):
                audit.independent_actual_result(p,{'files':{}})

    def test_actual_installed_sdk_request_omits_version_label(self):
        import types,remote
        from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
        captured=[]
        class Client:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def list_kernel_session_output(self,request):
                self_request=ApiListKernelSessionOutputRequest.to_dict(request);captured.append(self_request)
                return types.SimpleNamespace(files=[],next_page_token='')
            @property
            def kernels(self):return types.SimpleNamespace(kernels_api_client=self)
        client=Client();api=remote.Remote();api.api=types.SimpleNamespace(build_kaggle_client=lambda:client)
        self.assertEqual(api.list_page('',100),([],''));self.assertEqual(api.list_page('next-page',100),([],''))
        self.assertEqual(captured[0],dict(userName='pranavagrawai',kernelSlug='serving-candidate-s10-16-20261024',pageSize=100))
        self.assertEqual(captured[1]['pageToken'],'next-page');self.assertNotIn('versionLabel',captured[1])

    def test_duplicate_requested_name_has_no_download(self):
        api=API({'a':b'a'},pages=[[('a','https://fixture.invalid/a')],[('a','https://fixture.invalid/a')]])
        with tempfile.TemporaryDirectory() as d:
            result=collector.collect([row('a')],Path(d)/'new',api)
            self.assertFalse(result['complete']);self.assertEqual(api.downloads,[])

    def test_expanded_manifest_stops_before_any_request(self):
        api=API({'a':b'a','b':b'b'})
        with tempfile.TemporaryDirectory() as d:
            result=collector.collect([row('a'),row('b')],Path(d)/'new',api,limits={'downloads':1})
            self.assertFalse(result['complete']);self.assertEqual(api.identities,[])

    def test_missing_content_length_is_invalid_without_writing(self):
        stream=Stream(b'abc');stream.headers={};api=API({'a':stream})
        with tempfile.TemporaryDirectory() as d:
            result=collector.collect([row('a')],Path(d)/'new',api)
            self.assertFalse(result['complete']);self.assertEqual(result['downloaded_bytes'],0)

    def test_symlink_actual_result_cannot_alias_embedded_file(self):
        import audit
        actual=b'{}'
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);target=p/'members'/audit.ACTUAL_RESULT;target.parent.mkdir(parents=True)
            embedded=p/'embedded.json';embedded.write_bytes(actual);target.symlink_to(embedded)
            (p/'COLLECTION.json').write_text(json.dumps({'files':[dict(name=audit.ACTUAL_RESULT,origin='INDEPENDENT_CURRENT_SESSION_DOWNLOAD',complete=True,bytes=2,sha256=hashlib.sha256(actual).hexdigest())]}))
            with self.assertRaisesRegex(ValueError,'alias'):audit.independent_actual_result(p,{'files':{}})

    def test_error_capture_never_reads_unbuffered_body_or_persists_url(self):
        import remote
        class Response:
            status_code=403
            headers={'content-type':'text/plain','Authorization':'DO_NOT_SAVE'}
            @property
            def content(self):raise AssertionError('must not read error stream')
        error=RuntimeError('https://signed.invalid/file?token=DO_NOT_SAVE');error.response=Response()
        value=remote.Remote.error_record(error)
        self.assertEqual(value['status'],403);self.assertNotIn('DO_NOT_SAVE',json.dumps(value));self.assertNotIn('signed.invalid',json.dumps(value))

    def test_oversized_chunk_fails_closed_and_retains_only_bounded_prefix(self):
        api=API({'a':Stream(b'oversize',length=1)})
        with tempfile.TemporaryDirectory() as d:
            result=collector.collect([row('a')],Path(d)/'new',api)
            self.assertFalse(result['complete']);self.assertEqual((Path(d)/'new/members/a').stat().st_size,0)

    def test_local_preparation_receipt_cannot_authorize_remote_work(self):
        import run
        from binding import HERE
        with self.assertRaisesRegex(ValueError,'no exact remote authorization'):
            run.authorize(HERE.parent/'coordination-20261024/trial16-missing-raw-local-preparation.json')

    def test_late_driver_receipt_completion_cannot_remain_complete(self):
        import run
        ticks=iter([1.,2.,1800.05])
        with tempfile.TemporaryDirectory() as d:
            result=run.persist_driver_result(Path(d),{'complete':True},0.,clock=lambda:next(ticks))
            self.assertFalse(result['complete']);self.assertFalse(json.loads((Path(d)/'COLLECTION.json').read_text())['complete'])

    def test_timely_driver_persistence_has_hash_bound_completion(self):
        import run
        ticks=iter([1.,2.,3.])
        with tempfile.TemporaryDirectory() as d:
            result=run.persist_driver_result(Path(d),{'complete':True},0.,clock=lambda:next(ticks))
            receipt=Path(d)/'COLLECTION.json';completion=json.loads((Path(d)/'DRIVER-COMPLETION.json').read_text())
            self.assertTrue(result['complete']);self.assertEqual(completion['receipt_sha256'],hashlib.sha256(receipt.read_bytes()).hexdigest())

    def test_download_ledger_rederives_exact_names_counts_and_bytes(self):
        import audit,run
        api=API({'missing/a':b'abc'})
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'new';manifest=[row('missing/a')];result=collector.collect(manifest,target,api)
            result['driver_elapsed_seconds']=1.
            files,waived=audit.verify_download_ledger(result,target,{x['name']:x for x in manifest})
            self.assertEqual(list(files),['missing/a']);self.assertEqual(waived,set())
            result['downloaded_bytes']=2
            with self.assertRaisesRegex(ValueError,'byte/time'):audit.verify_download_ledger(result,target,{x['name']:x for x in manifest})

    def test_provenance_merge_preserves_originals_and_checks_every_hash(self):
        import audit
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);base=root/'base';(base/'evidence/retained').mkdir(parents=True);(base/'evidence/retained/a').write_bytes(b'original')
            (base/'ORIGINS.json').write_text(json.dumps({'retained/a':[dict(bytes=8,sha256=hashlib.sha256(b'original').hexdigest())]}))
            target=root/'new';result=collector.collect([row('missing/b')],target,API({'missing/b':b'new'}))
            destination=root/'audit';destination.mkdir();files={x['name']:x for x in result['files']};allowed={'missing/b':row('missing/b')}
            evidence,origins=audit.merge_inputs(target,destination,files,allowed,base)
            self.assertEqual((evidence/'retained/a').read_bytes(),b'original');self.assertEqual((evidence/'missing/b').read_bytes(),b'new')
            self.assertEqual(origins['missing/b'][0]['origin'],'INDEPENDENT_CURRENT_SESSION_DOWNLOAD')
            (base/'evidence/retained/a').write_bytes(b'changed!');dest2=root/'audit2';dest2.mkdir()
            with self.assertRaisesRegex(ValueError,'retained evidence origin/hash'):audit.merge_inputs(target,dest2,files,allowed,base)

    def test_actual_saved_decimal_readback_source_identity_offline(self):
        import datetime as dt,remote
        from binding import ROOT
        directory=ROOT/'remote-readback/current-primary-fallback-local/fresh-readback'
        rb=json.loads((directory/'READBACK.json').read_text())
        value=remote.validate_readback(directory,now=dt.datetime.fromisoformat(rb['typed_quota']['utc']))
        self.assertEqual(value['kernel_id'],140493086);self.assertEqual(value['version'],1);self.assertEqual(value['wire_seconds']['used_s'],'2714.079')

    def test_bound_before_after_saved_source_and_driver_receipts(self):
        import datetime as dt,remote,run,audit
        from binding import ROOT
        source=ROOT/'remote-readback/current-primary-fallback-local/fresh-readback'
        rb=json.loads((source/'READBACK.json').read_text())
        checked=remote.validate_readback(source,now=dt.datetime.fromisoformat(rb['typed_quota']['utc']))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for label in ('before','after'):shutil.copytree(source,root/label)
            ticks=iter([1.,2.,3.]);receipt=run.persist_driver_result(root,dict(complete=True,before=checked,after=checked),0.,clock=lambda:next(ticks))
            self.assertEqual(audit.verify_bound_readbacks(receipt,root)['kernel_id'],140493086)
            metadata=json.loads((root/'after/metadata.json').read_text());metadata['currentVersionNumber']=2
            (root/'after/metadata.json').write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError,'metadata/current version'):audit.verify_bound_readbacks(receipt,root)

    def test_download_ledger_rejects_attempt_name_substitution(self):
        import audit
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'new';item=row('a');result=collector.collect([item],root,API({'a':b'a'}));result['driver_elapsed_seconds']=1.
            (root/'attempts.jsonl').write_text('{"name":"unlisted","attempt":1}\n')
            with self.assertRaisesRegex(ValueError,'attempt download ledger'):audit.verify_download_ledger(result,root,{'a':item})

if __name__=='__main__':unittest.main()
