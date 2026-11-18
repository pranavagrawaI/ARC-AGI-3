"""Bounded read-only source/corpus audit; no model/GPU/SDK or external network."""
import argparse,collections,hashlib,importlib,io,json,os,signal,sys,tempfile,time,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(exist_ok=False);signal.alarm(600);started=time.monotonic()
 bindings=json.loads((HERE/'SOURCE-BINDINGS.json').read_text());sources=bindings['references']
 for name,h in sources.items():assert sha(name)==h,name
 cap=Path(bindings['prior_exact_wire_capacity']);working=Path(bindings['prior_working_release']);sys.path.insert(0,str(cap));sys.path.insert(1,str(cap/'tests'))
 os.environ['PYTHONPATH']=str(working/'guard');sys.path.insert(0,str(working/'guard'));import sitecustomize
 import replay,catalog,corpus_base,accounting,load_contract,reduce_api,provenance
 for name in bindings['actual_imports']:
  module=sys.modules[Path(name).stem];assert Path(module.__file__).resolve()==Path(name).resolve(),module.__file__
 cat=catalog.load();records=[];kwargs=collections.Counter();max_context=[]
 with tempfile.TemporaryDirectory() as td:
  directory=Path(td)/'bodies';index=replay.prepare(bindings['corpus'],directory)
  for key,rec in cat['records'].items():
   raw=(directory/index[key]['file']).read_bytes();body=catalog.checked_body(raw,rec)
   assert index[key]['sha256']==rec['prepared_body_sha256']==catalog.digest(raw)
   assert body['max_tokens']==rec['cap'] and body['model']=='flashnext'
   assert catalog.image_observations(body)==rec['images'] and body.get('truncate_prompt_tokens') is None
   kwargs[json.dumps(body.get('chat_template_kwargs'),sort_keys=True)]+=1
   total=rec['local_prompt']+rec['cap'];max_context.append((total,key));records.append(dict(key=key,prepared_body_sha256=catalog.digest(raw),wire_sha256=rec['wire_sha'],source_png_sha256=rec['images'],expected_api_image_tokens=len(rec['images'])*400,expected_api_prompt_tokens=rec['local_prompt'],cap=rec['cap'],body_bytes=len(raw),context_total=total))
  index_sha=catalog.digest((directory/'index.json').read_bytes())
 assert len(records)==1369 and len(cat['streams'])==25 and len(cat['initial_holders'])==10
 assert max(max_context)[0]<=139264,'source-known context overflow'
 proposal=json.loads((HERE/'proposal.json').read_text());caps=proposal['caps'];assert sum(proposal['phase_ceilings_seconds'].values())==9590 and caps['work_seconds']-9590==310
 assert caps['cooperative_seconds']==caps['work_seconds']+caps['cleanup_seconds'] and caps['platform_seconds']==caps['cooperative_seconds']+300
 assert proposal['observer']['max_samples']>=int(caps['cooperative_seconds']/.5)+1
 suite=unittest.TestSuite();names=['test_accounting','test_source_semantics','test_body_binding','test_deadline_race','test_persistence','test_protocol_cutoff'];os.environ['LOCAL_EVIDENCE']=str(out/'loopback-witnesses')
 for name in names:suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(importlib.import_module(name)))
 stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite);(out/'tests.log').write_text(stream.getvalue());assert result.wasSuccessful(),stream.getvalue()
 for name,h in sources.items():assert sha(name)==h,name
 (out/'BODY-INVENTORY.json').write_text(json.dumps(records,indent=2)+'\n')
 report=dict(status='LOCAL_READINESS_AUDIT_ONLY_NOT_EXECUTABLE_OR_LAUNCHED',references_checked=len(sources),source_bindings_sha256=sha(HERE/'SOURCE-BINDINGS.json'),proposal_sha256=sha(HERE/'proposal.json'),seconds=time.monotonic()-started,records=len(records),streams=25,initial_holders=10,all_exact_body_png_cap_hashes_match=True,catalog_sha256=sha(cap/'catalog.json'),prepared_index_sha256=index_sha,production_chat_template_kwargs=dict(kwargs),max_context_total=max(max_context)[0],max_context_key=max(max_context)[1],max_images=max(len(r['source_png_sha256']) for r in records),body_bytes_one_full_corpus=sum(r['body_bytes'] for r in records),selected_regressions=result.testsRun,selected_regressions_pass=True,inherited_runtime_suite='NOT_RERUN; accepted64/65 retained',native_running_stream_metric='UNAVAILABLE; HTTP pending never substituted',network_calls=0,gpu_work=False,model_loads=0,actual_import_dependencies=bindings['actual_imports'],runtime_unknowns=['Candidate production-thinking/full-image/full-reply S10/API counts not established by local body/render proof or health200','Post-control-replay nonhost release+fresh and candidate allocation under changed cache/host history unproved','Fixed-order temporal stability/interference assumption; no engine-only causal identification','Bounded observer sequential/sampled/sharedRSS/cumulativepeak visibility and delayedEOF limitation remain'],remaining_integration=proposal['unimplemented_required_joins'],launchable=False)
 report['inventory']={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file()};(out/'AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='inventory'},indent=2));signal.alarm(0)
if __name__=='__main__':main()
