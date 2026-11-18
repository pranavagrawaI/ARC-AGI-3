import contextlib,hashlib,importlib.util,io,json,sys,zipfile
from pathlib import Path
root=Path(__file__).resolve().parent
p=Path('/home/pranav/projects/arc-agi-3/analysis/serving-alternatives-20261021/trial-13-post-control-load-local')
sys.path.insert(0,str(p))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sp=importlib.util.spec_from_file_location('trial13_reduce_review',p/'collection_tools/reduce.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);m.HERE=root/'envelope'
summary={'scope':'OFFLINE_CPU_ARCHIVE_FIXTURE; synthetic accepted identity; exact original reducer; all original source hashes checked','cases':{}}
for case in ('baseline','release_false','fresh_after_original_release_deadline','boundary_oom','nonterminal_readback'):
 d=root/case;arc=d/'archive';out=next((arc/'members').glob('*/output'))
 # CPU lifecycle intentionally skips hardware; populate compatibility with its actual captured CPU query commands, explicitly marked.
 pre=json.loads((out/'pre-release.json').read_text());compat={n:pre['probes'][n]['command'] for n in ('gpu','apps')};compat['fixture']='CPU_ONLY_COMPATIBILITY_ENVELOPE_FROM_CAPTURED_PRE_RELEASE_COMMANDS'
 (out/'query-compatibility.json').write_text(json.dumps(compat))
 if case=='fresh_after_original_release_deadline':
  release=json.loads((out/'release.json').read_text());rows=[json.loads(x) for x in (out/'supervisor.jsonl').read_text().splitlines()]
  for row in rows:
   if row['event']=='release_start':row['deadline']=release['absolute_deadline']
   if row['event']=='release_result':row['absolute_deadline']=release['absolute_deadline']
  (out/'supervisor.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
 with zipfile.ZipFile(arc/'output.zip','w',zipfile.ZIP_DEFLATED) as z:
  for file in (arc/'members').rglob('*'):
   if file.is_file():z.write(file,file.relative_to(arc/'members'))
 r=json.loads((arc/'ARCHIVE.json').read_text());r['archive_sha256']=sha(arc/'output.zip');(arc/'ARCHIVE.json').write_text(json.dumps(r))
 dest=d/'derived-complete-fixture'
 sys.argv=['reduce.py',str(arc),str(d/'readback'),str(dest)]
 with contextlib.redirect_stdout(io.StringIO()):m.main()
 result=json.loads((dest/'REDUCTION.json').read_text());summary['cases'][case]={k:result[k] for k in ('status','issues','release_pass')}
(root/'COMPLETE-FIXTURE-SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
