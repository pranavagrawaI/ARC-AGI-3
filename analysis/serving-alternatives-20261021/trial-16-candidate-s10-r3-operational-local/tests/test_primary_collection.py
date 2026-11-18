import base64,contextlib,datetime as dt,hashlib,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import packet,transport,primary_bundle
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'collection_tools'))
import primary
try:import kagglesdk
except ImportError:kagglesdk=None

class PrimaryCollection(unittest.TestCase):
    @unittest.skipIf(kagglesdk is None,'SDK-specific test runs under installed Kaggle interpreter')
    def test_actual_version_request_one_primary_hash_bound_no_archive(self):
        with tempfile.TemporaryDirectory() as td:
            temp=Path(td);manifest=packet.prepare(temp/'unassigned');spec=json.loads((HERE/'packet-spec.json').read_text())
            assignment=dict(launch_authorized=True,serialized_assignment='LOCAL_ONLY_PRIMARY_FIXTURE',local_only=True,
                confirmed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),total_seconds=5100,platform_hard_timeout_s=5400,kernel_ref=spec['metadata']['id'],sources_sha256=manifest['sources_sha256'],config_sha256=manifest['config_sha256'])
            packet.prepare(temp/'outer',assignment);outer=(temp/'outer'/spec['metadata']['code_file']).read_bytes();inner=transport.decode(outer);values=transport.literals(transport.cell_source(json.loads(inner)))
            run=temp/'run';out=run/'output';out.mkdir(parents=True)
            for name,item in values['PAYLOAD'].items():(run/name).write_bytes(base64.b64decode(item['base64']))
            (run/'config.json').write_text(json.dumps(spec['config']));(run/'assignment.json').write_text(json.dumps(assignment))
            result=dict(pass_=False,status='INVALID_INCONCLUSIVE',error='EXACT_PRIMARY_FAILURE',surviving_phase_groups=[],retained_queries=[])
            (out/'result.json').write_text(json.dumps(result));(out/'supervisor.jsonl').write_text(json.dumps(dict(seq=0,t=1,event='terminal',**result))+'\n')
            import time
            with patch.object(primary_bundle,'__file__',str(run/'primary_bundle.py')):receipt=primary_bundle.write(out,spec['config'],time.monotonic()+2)
            raw=Path(receipt['path']).read_bytes();readback=temp/'readback';readback.mkdir()
            rb=dict(kernel_ref=spec['metadata']['id'],kernel_id=123,version=1,source_exact=True,status={'status':'ERROR'})
            (readback/'READBACK.json').write_text(json.dumps(rb));(readback/'metadata.json').write_text(json.dumps(dict(id=123,currentVersionNumber=1,ref=spec['metadata']['id'])));(readback/'source.ipynb').write_bytes(outer)
            (temp/'launch-attempt.json').write_text(json.dumps({'inner_notebook_sha':transport.sha(inner)}));(temp/'packet-spec.json').write_text(json.dumps(spec))
            calls=[];gets=[]
            class Endpoint:
                def list_kernel_session_output(self,request):
                    calls.append(json.loads(str(request)))
                    return SimpleNamespace(files=[SimpleNamespace(file_name=primary_bundle.NAME,url='LOCAL_FAKE_URL')])
            @contextlib.contextmanager
            def client():yield SimpleNamespace(kernels=SimpleNamespace(kernels_api_client=Endpoint()))
            class Stream:
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def raise_for_status(self):pass
                def iter_content(self,chunk_size):yield raw
            def get(url,**kwargs):gets.append((url,kwargs));return Stream()
            with patch.object(primary,'HERE',temp),patch.object(primary,'accepted',return_value=(spec['metadata']['id'],123,1)),patch.object(primary.space,'check',return_value={'eligible':True}):
                proof=primary.fetch(SimpleNamespace(build_kaggle_client=client),get,temp/'retrieved',readback)
                self.assertEqual(proof['result'],result);self.assertFalse(proof['full_raw_qualification']);self.assertEqual(len(calls),1);self.assertEqual(len(gets),1)
                self.assertEqual(calls[0]['versionLabel'],'1');self.assertEqual(calls[0]['pageSize'],100)
                changed=json.loads(raw);changed['files']['assignment.json']['sha256']='changed'
                with self.assertRaisesRegex(ValueError,'hash'):primary.verify(json.dumps(changed).encode(),readback)
                with self.assertRaises(FileExistsError):primary.fetch(SimpleNamespace(build_kaggle_client=client),get,temp/'retrieved',readback)
