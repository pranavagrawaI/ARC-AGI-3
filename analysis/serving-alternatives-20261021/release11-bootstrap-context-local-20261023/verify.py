"""Bounded CPU-only source/context diagnosis; no remote API or packet creation."""
import argparse,ast,hashlib,importlib.util,json,os,signal,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();out=a.output.resolve();out.mkdir(exist_ok=False)
 signal.alarm(600);started=time.monotonic()
 inputs=json.loads((HERE/'INPUTS.json').read_text());failed=Path(inputs['failed_directory'])
 for p,h in inputs['references'].items():assert sha(p)==h,p
 manifest=json.loads((failed/'freeze.json').read_text())
 for p,h in manifest['files'].items():assert sha(failed/p)==h,p
 for p,h in manifest['references'].items():assert sha(p)==h,p
 spec=importlib.util.spec_from_file_location('frozen_transport',failed/'transport.py');transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
 outer=(failed/'kernel/serving-release11-20261023.ipynb').read_bytes();inner=transport.decode(outer);source=transport.cell_source(json.loads(inner))
 values=transport.literals(source)
 marker=json.loads((failed/'launch-attempt.json').read_text());assert hashlib.sha256(inner).hexdigest()==marker['inner_notebook_sha']
 import bootstrap_patch
 corrected=bootstrap_patch.apply(source)
 cases=[('baseline_outer_clean',transport.cell_source(json.loads(outer)),False,None,'missing'),('baseline_inner_clean',source,False,None,'missing'),
 ('corrected_clean',corrected,False,None,'cpu'),('baseline_preloaded_entry',source,False,'release_runner','wrong'),
 ('corrected_preloaded_entry',corrected,False,'release_runner','origin'),('corrected_preloaded_dependency',corrected,False,'adapters','origin'),('corrected_foreign_shadow',corrected,True,None,'cpu')]
 rows=[]
 for name,code,shadow,poison,expect in cases:
  case=out/name;case.mkdir();(case/'bootstrap.py').write_text(code)
  data=dict(name=name,shadow=shadow,poison=poison,confirmed_utc=values['ASSIGNMENT']['confirmed_utc'])
  (case/'case.json').write_text(json.dumps(data,indent=2)+'\n')
  run=subprocess.run([sys.executable,'-I','-B',str(HERE/'replay_child.py'),str(case)],cwd=case,capture_output=True,text=True,timeout=20)
  (case/'stdout').write_text(run.stdout);(case/'stderr').write_text(run.stderr)
  assert run.returncode==0,(name,run.stderr)
  row=json.loads((case/'RESULT.json').read_text());rows.append(row)
  # Persist every replay before assertions. All forbidden dispatch counts must be0.
  assert row['forbidden_attempts']==[],row
  if expect=='missing':
   assert row['error']==dict(type='ModuleNotFoundError',message="No module named 'release_runner'"),row
   assert not row['boundaries'] and row['result'] is None,row
  if expect=='wrong':assert row['wrong_module_executed'] and not row['boundaries'],row
  if expect=='origin':
   assert row['error']['type']=='RuntimeError' and 'package module origin mismatch:' in row['error']['message'],row
   assert not row['wrong_module_executed'] and not row['boundaries'] and row['result'] is None,row
  if expect=='cpu':
   assert len(row['boundaries'])==1 and not row['wrong_module_executed'],row
   assert row['boundaries'][0]['source_sha256']==values['PAYLOAD']['release_runner.py']['sha256'],row
   assert row['result']['qualification']=='NOT_QUALIFIED' and 'CPU_ONLY_STOP_BEFORE_HARDWARE_QUERY_MODEL' in row['result']['error'],row
   assert not row['result']['surviving_phase_groups'] and not row['result']['surviving_native_group'] and not row['result']['retained_queries'],row
  for n,record in values['PAYLOAD'].items():
   assert sha(case/'extracted'/n)==record['sha256'],(name,n)
 for p,h in inputs['references'].items():assert sha(p)==h,p
 report=dict(status='LOCAL_BOOTSTRAP_CONTEXT_CORRECTION_ONLY_NOT_RUNTIME_RELEASE',cases=len(rows),seconds=time.monotonic()-started,cpu_cap_s=600,
  original_inner_notebook_sha256=hashlib.sha256(inner).hexdigest(),original_inner_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
  corrected_inner_source_sha256=hashlib.sha256(corrected.encode()).hexdigest(),
  original_inputs_and166freeze_bindings_exact=True,only_bootstrap_guard_added=True,payload_config_assignment_bytes_unchanged=True,
  cases_summary=[dict(case=r['case'],error=r['error'],phase_boundary=len(r['boundaries']),wrong_module_executed=r['wrong_module_executed'],forbidden_attempts=r['forbidden_attempts']) for r in rows],
  limits=['Clock replay and temporary root mapping are explicit local environment fixtures, not fresh authority','CPU phase boundary stops before actual hardware/query/model/release','No GPU/network/API/new packet or assignment','Pre/post import origin guard addresses ordinary path shadowing/preloaded payload modules, not hostile Python import hooks'],
  sources={p.name:sha(p) for p in HERE.glob('*.py')},inventory={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
 with (out/'PROOF.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ('inventory','cases_summary')},indent=2));signal.alarm(0)
if __name__=='__main__':main()
