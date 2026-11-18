"""Deterministic allowlisted notebook build. No donor imports or remote I/O."""
from pathlib import Path
import ast,base64,hashlib,io,json,zipfile
HERE=Path(__file__).resolve().parent

def payloads():
    contract=json.loads((HERE/'runtime/CHASSIS.json').read_text())
    z=io.BytesIO()
    with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as out:
        for name in sorted([*contract['files'],'CHASSIS.json']):
            item=zipfile.ZipInfo(name,(2026,9,17,0,0,0));item.compress_type=zipfile.ZIP_DEFLATED
            out.writestr(item,(HERE/'runtime'/name).read_bytes())
    files={'nooa-join.zip':z.getvalue()}
    for name in ('budget.py','entry.py','install.py','official-hashes.json'):
        files[name]=(HERE/name).read_bytes()
    for dest,src in {'submission_play.py':'play.py','my_agent.py':'runtime/adapter.py',
        'shared_local_solver.py':'runtime/shared_local_solver.py','wrapper_local_solver.py':'runtime/local_solver.py',
        'official_launcher_runtime.py':'runtime/exact_request_outer_runtime.py'}.items():
        files[dest]=(HERE/src).read_bytes()
    for name in ('PROTOTYPE-qwen3_8_r8_l3b_probe.py','PROTOTYPE-qwen3_8_fp8-manifest.json','public-environment-manifest.json'):
        files[name]=(HERE/'assets'/name).read_bytes()
    return files

def notebook():
    files=payloads()
    s='''import os,time,sys,base64,hashlib
from pathlib import Path
os.environ['ARC_COMPETITION_STARTED']=str(time.monotonic())
fixture=os.environ.get('ARC_LOCAL_FIXTURE')=='1'
root=Path(os.environ['ARC_LOCAL_ROOT']) if fixture else Path('/tmp')
root.mkdir(parents=True,exist_ok=True)
rerun=os.environ.get('KAGGLE_IS_COMPETITION_RERUN','').lower() in ('true','1')
if rerun:
'''
    for name,data in sorted(files.items()):
        s+=f"    data=base64.b64decode({base64.b64encode(data).decode()!r})\n    assert hashlib.sha256(data).hexdigest()=={hashlib.sha256(data).hexdigest()!r}\n    (root/{name!r}).write_bytes(data)\n"
    s+='''    sys.path.insert(0,str(root))
    import entry
    entry.main(root)
else:
    import pandas as pd
    work=root/'working' if fixture else Path('/kaggle/working')
    work.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([['1_0','1',True,1]],columns=['row_id','game_id','end_of_game','score']).to_parquet(work/'submission.parquet',index=False)
'''
    ast.parse(s)
    return dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'name':'python3','language':'python','display_name':'Python 3'}},cells=[dict(cell_type='markdown',id='identity',metadata={},source='# ARC3 Actor + critic competition v1\nFeasibility only. Independent review and authorization required.'),dict(cell_type='code',id='entry',metadata={},execution_count=None,outputs=[],source=s)])

if __name__=='__main__':
    (HERE/'submission.ipynb').write_text(json.dumps(notebook(),indent=1)+'\n')
    (HERE/'ASSETS.json').write_text(json.dumps({k:hashlib.sha256(v).hexdigest() for k,v in payloads().items()},indent=2)+'\n')
