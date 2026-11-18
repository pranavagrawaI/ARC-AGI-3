"""Replay saved CPU archive fixtures against unchanged reducer; offline only."""
import contextlib,datetime,importlib.util,io,json,sys
from pathlib import Path
root=Path(__file__).resolve().parent
p=Path('/home/pranav/projects/arc-agi-3/analysis/serving-alternatives-20261021/trial-13-post-control-load-local')
sys.path.insert(0,str(p))
sp=importlib.util.spec_from_file_location('trial13_reduce_review',p/'collection_tools/reduce.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);m.HERE=root/'envelope'
output=root/'replays'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');output.mkdir(parents=True,exist_ok=False)
results={}
for case in ('baseline','release_false','fresh_after_original_release_deadline','boundary_oom','nonterminal_readback'):
 d=root/case;sys.argv=['reduce.py',str(d/'archive'),str(d/'readback'),str(output/case)]
 with contextlib.redirect_stdout(io.StringIO()):m.main()
 r=json.loads((output/case/'REDUCTION.json').read_text());results[case]={k:r[k] for k in ('status','issues','release_pass')}
(output/'SUMMARY.json').write_text(json.dumps(results,indent=2)+'\n');print(output);print(json.dumps(results,indent=2))
