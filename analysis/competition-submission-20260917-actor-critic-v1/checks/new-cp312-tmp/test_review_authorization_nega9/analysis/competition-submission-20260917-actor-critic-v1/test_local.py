import importlib.util,json,os,subprocess,sys
from pathlib import Path
import pytest
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[1];P=ROOT/'deployment/kaggle/submission-20260917-actor-critic-competition-v1'
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_budget():
    b=load('budget',P/'budget.py');v=b.validate()
    assert v['games']==110 and v['provider_seconds']==32400 and v['admission_seconds']>0
    assert v['total_model_requests']==1761
    assert b.GAMES*(v['admission_seconds']+2*b.REQUEST+b.TOOL_MARGIN+b.GAME_CLEANUP)+180==pytest.approx(b.GAMEPLAY)

@pytest.mark.parametrize('args',[(10,110,90),(28020,0,90),(float('nan'),110,90),(28020,110,-1),(28020,110.5,90)])
def test_impossible_budget(args):
    with pytest.raises(ValueError):load('budget',P/'budget.py').admission(*args)

def test_commit_exact_notebook(tmp_path):
    n=json.loads((P/'submission.ipynb').read_text());source='\n'.join(''.join(c['source']) for c in n['cells'] if c['cell_type']=='code')
    script=tmp_path/'notebook.py';script.write_text(source)
    env=dict(os.environ,ARC_LOCAL_FIXTURE='1',ARC_LOCAL_ROOT=str(tmp_path/'materialized'),KAGGLE_IS_COMPETITION_RERUN='false',PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable,str(script)],cwd=tmp_path,env=env,check=True,timeout=30)
    import pandas as pd
    frame=pd.read_parquet(tmp_path/'materialized/working/submission.parquet')
    assert frame.to_dict('records')==[dict(row_id='1_0',game_id='1',end_of_game=True,score=1)]
    assert list((tmp_path/'materialized').iterdir())==[tmp_path/'materialized/working']

def test_payloads():
    import base64,ast,hashlib
    b=load('build_local',P/'build.py');n=json.loads((P/'submission.ipynb').read_text())
    assert n==b.notebook()
    assert {k:hashlib.sha256(v).hexdigest() for k,v in b.payloads().items()}==json.loads((P/'ASSETS.json').read_text())
    source=''.join(n['cells'][-1]['source'])
    assert "fixture=os.environ.get('ARC_LOCAL_FIXTURE')=='1'" in source
    assert 'competition rerun is prohibited' not in source
