"""Full local archive->raw API->terminal classification; package stub is explicit."""
import base64,contextlib,hashlib,importlib.util,io,json,sys,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parents[1];sys.path.append(str(HERE.parent/'trial-10-exact-wire-local/tests'));sys.path.insert(0,str(HERE/'collection_tools'))
import catalog,reduce_api,transport,test_reducer
spec=importlib.util.spec_from_file_location('offline_collection_reducer',HERE/'collection_tools/reduce.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class CollectionReducer(unittest.TestCase):
 def case(self,kind):
  with tempfile.TemporaryDirectory() as td:
   base=Path(td);env=base/'envelope';env.mkdir();arc=base/'archive';arc.mkdir();readback=base/'readback';readback.mkdir();parent=arc/'members/CPU-COLLECTION';root=parent/'output';root.mkdir(parents=True)
   cat,_=test_reducer.fixture(root,'control',0,[100,700,1000,1300]);test_reducer.fixture(root,'candidate',2000,[100,700,800,1000,1100,1300,1400])
   with patch.object(catalog,'load',return_value=cat),patch.object(reduce_api.provenance,'check',return_value={'CPU_PACKAGE_PROVENANCE_SUBSTITUTION':True}):pair=reduce_api.compare(root)
   self.assertEqual(pair['ratio'],2.0)
   result=dict(pass_=True,qualification=pair['verdict'],ratio=pair['ratio'],surviving_phase_groups=[],surviving_native_group=[],retained_queries=[],query_cleanup=[])
   if kind=='cleanup':result.update(pass_=False,cleanup_errors=['UnsafeIdentity CPU fixture'])
   (root/'result.json').write_text(json.dumps(result));(root/'supervisor.jsonl').write_text(json.dumps(dict(seq=0,t=4000,event='terminal',**result))+'\n')
   saved=dict(pair)
   if kind=='saved_pair':saved['ratio']=1.8
   (root/'paired-result.json').write_text(json.dumps(saved))
   config=dict(cwd='/kaggle/working/CPU-COLLECTION');assignment=dict(local_only=True,CPU_FIXTURE_NOT_AUTHORITY=True);(parent/'config.json').write_text(json.dumps(config));(parent/'assignment.json').write_text(json.dumps(assignment));payload=b'CPU_ONLY'
   (parent/'example.py').write_bytes(b'DRIFT' if kind=='payload' else payload);ph=hashlib.sha256(payload).hexdigest()
   source='PAYLOAD = '+repr({'example.py':{'base64':base64.b64encode(payload).decode(),'sha256':ph}})+'\nCONFIG = '+repr(config)+'\nASSIGNMENT = '+repr(assignment)+'\n'
   inner=json.dumps(dict(nbformat=4,nbformat_minor=5,metadata={},cells=[dict(cell_type='code',source=source.splitlines(True))])).encode();(readback/'source.ipynb').write_bytes(transport.wrap(inner))
   response=dict(kernelId=1,versionNumber=1);(env/'launch-response.json').write_text(json.dumps(response));(env/'launch-attempt.json').write_text(json.dumps(dict(inner_notebook_sha=transport.sha(inner))))
   (env/'freeze.json').write_text(json.dumps(dict(files={},references={})));(env/'packet-spec.json').write_text(json.dumps(dict(config=config,metadata={'id':'CPU/FIXTURE'})));(env/'review-packet.json').write_text(json.dumps(dict(files={'example.py':ph})))
   (readback/'metadata.json').write_text(json.dumps(dict(id=1,currentVersionNumber=2 if kind=='metadata' else 1,ref='CPU/FIXTURE')))
   (readback/'READBACK.json').write_text(json.dumps(dict(source_exact=True,kernel_id=1,version=1,status={'status':'RUNNING' if kind=='nonterminal' else 'COMPLETE'},typed_quota=dict(used_s=71670.895,reserved_s=0))))
   with zipfile.ZipFile(arc/'output.zip','w') as z:
    for path in sorted((arc/'members').rglob('*')):
     if path.is_file():z.write(path,str(path.relative_to(arc/'members')))
   digest=hashlib.sha256((arc/'output.zip').read_bytes()).hexdigest();(arc/'ARCHIVE.json').write_text(json.dumps(dict(complete=True,CRC_verified=True,archive_sha256=digest,kernel_id=1,version=1,kernel_ref='CPU/FIXTURE')))
   out=base/'derived'
   with patch.object(module,'HERE',env),patch.object(module.space,'check',return_value={'eligible':True,'CPU_ONLY':True}),patch.object(catalog,'load',return_value=cat),patch.object(reduce_api.provenance,'check',return_value={'CPU_PACKAGE_PROVENANCE_SUBSTITUTION':True}),patch.object(sys,'argv',['reduce.py',str(arc),str(readback),str(out)]),contextlib.redirect_stdout(io.StringIO()):module.main()
   return json.loads((out/'REDUCTION.json').read_text())
 def test_complete_cpu_archive_classifies_analytical_pair(self):self.assertEqual(self.case('positive')['status'],'QUALITY_INVESTIGATION_ONLY')
 def test_later_cleanup_error_invalidates_saved_online_ratio(self):
  value=self.case('cleanup');self.assertEqual(value['status'],'FEASIBILITY_ONLY');self.assertIsNone(value['ratio']);self.assertIsNone(value['independent_pair']['ratio'])
 def test_nonterminal_quota_cannot_qualify_saved_ratio(self):
  value=self.case('nonterminal');self.assertEqual(value['status'],'FEASIBILITY_ONLY');self.assertIsNone(value['ratio']);self.assertFalse(value['terminal_collection_complete'])
 def test_source_metadata_and_saved_pair_drift_rejected(self):
  for kind in ('payload','metadata','saved_pair'):
   with self.subTest(kind=kind):self.assertEqual(self.case(kind)['status'],'FEASIBILITY_ONLY')
