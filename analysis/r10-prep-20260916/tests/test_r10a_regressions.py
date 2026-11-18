"""R10a real sealed/HTTP subprocess regressions; no model or remote claims."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import pytest

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parents[1]
PACKAGE=HERE/'r10a-package'
R9=ROOT/'deployment/kaggle/actor-critic-construction-kaggle-r9-cd82'
EVIDENCE=ROOT/'analysis/actor-critic-construction-kaggle-r9-s5i5-20260915T183130Z/runtime-evidence-exact-request/model-runtime/solver-workspaces/arm-B/s5i5-18d95033-xmnhchcq'

def isolated(package,code,payload):
    p=subprocess.run([sys.executable,'-I','-B','-c',f'import sys,json;sys.path.insert(0,{str(package)!r});\n'+code],input=json.dumps(payload),text=True,capture_output=True,timeout=50)
    assert p.returncode==0,p.stderr
    return json.loads(p.stdout)


def test_s5i5_preserved_source_reproduces_sealed_failure_and_repair(tmp_path):
    event=json.loads((EVIDENCE/'tool-events.jsonl').read_text().splitlines()[0])
    source=json.loads(event['call']['function']['arguments'])['candidate_source']
    row=json.loads((EVIDENCE/'ipc/transitions.jsonl').read_text().splitlines()[0])
    # Exact preserved bytes, not a fabricated approximation of the failure.
    code='''from prefix import ActorPrefixCase
from evaluate_visible import evaluate_visible_abi
p=json.load(sys.stdin)
print(json.dumps(evaluate_visible_abi(p['source'],ActorPrefixCase((p['row'],),1))))
'''
    bad=isolated(R9,code,dict(source=source,row=row))
    assert bad['visible']['status']=='failed' and 'sealed invocation failed closed' in bad['visible']['error']
    assert source.endswith('\"}')
    repaired=source[:-2].replace('\\n','\n') # Fixture correction of THIS known source only; production never does this.
    good=isolated(PACKAGE,code,dict(source=repaired,row=row))
    assert good['visible']['status']=='passed',good
    (tmp_path/'S5I5-REPRODUCTION.json').write_text(json.dumps({'preserved_source_sha256':hashlib.sha256(source.encode()).hexdigest(),'source_event':str(EVIDENCE/'tool-events.jsonl'),'r9':bad,'corrected_fixture_r10':good},indent=2))


def test_literal_backslash_n_inside_python_string_survives_sealed_execution():
    source=r'''
def initialize(o): return {"grid":o["grid"],"literal":r"left\nright"}
def predict(s,a): return [{"state":s,"commitments":{"literal":s["literal"]}}]
def assimilate(s,o,v): return {"grid":o["grid"],"literal":s["literal"]}
'''
    result=isolated(PACKAGE,'''from integration_boundary import CandidateRPC,SealedBoundary
p=json.load(sys.stdin); rpc=CandidateRPC(SealedBoundary(),p['source'])
s=rpc.initialize({'grid':[[1]]}); print(json.dumps(rpc.predict(s,'CLICK 31 41')))
''',{'source':source})
    assert result[0]['commitments']['literal']==r'left\nright'


def test_constructor_entry_budget_and_cell_deferral(tmp_path):
    result=isolated(PACKAGE,'''import headless,os
from construction_service import _limits
from live_agent import LiveConstructionAgent
from pathlib import Path
p=json.load(sys.stdin);headless.apply_live_argv([])
a=object.__new__(LiveConstructionAgent);a.ipc_dir=Path(p['path']);a.run_dir=a.ipc_dir
(a.ipc_dir/'cell_execution.json').write_text(json.dumps({'finished_at_monotonic':None}))
print(json.dumps({'wall':_limits().wall_seconds,'requests':_limits().max_model_requests,'hard':os.environ['ARC_CONSTRUCTOR_HARD_SECONDS'],'job':str(headless.construction_service.JOB),'deferred':json.loads(a.request_construction())}))
''',{'path':str(tmp_path)})
    assert result['wall']==240 and result['requests']==6 and result['hard']=='245'
    assert result['job']==str(PACKAGE/'persist_job.py')
    assert result['deferred']['construction_status']=='SCHEDULED_HOST_CONTEXT'
    assert not (tmp_path/'constructor-job').exists()


def test_typed_scoring_remains_measured_not_model_certification(tmp_path):
    result=isolated(PACKAGE,'''from pathlib import Path
sys.path.insert(0,str(Path(sys.path[0])/'nooa-join/chassis'))
from decision_packet import score_experiments
p=json.load(sys.stdin);ipc=Path(p['ipc'])
e={'id':'probe','turn':0,'step':0,'action':'CLICK 31 41','hypotheses':['change','still'],'outcomes':{'metric':'grid_changed','predictions':{'change':True,'still':False},'updates':{'change':'proposal','still':'alternative'},'otherwise':'unknown'}}
(ipc/'decision_experiments.jsonl').write_text(json.dumps(e)+'\\n')
rows=[{'turn':0,'step':1,'action':'CLICK 31 41','before':{'grid':[[0]]},'after':{'grid':[[1]]}}]
a=score_experiments(ipc,rows);b=score_experiments(ipc,rows)
print(json.dumps({'scores':a,'again':b}))
''',{'ipc':str(tmp_path)})
    assert result['scores']==result['again'] and len(result['scores'])==1
    score=result['scores'][0]
    assert score['observed'] is True and score['matched_prediction_label']=='change'
    assert score['authority']=='measured_prediction_only' and score['transition_step']==1


def test_r9_click_stale_state_reproduced_and_r10_repairs_through_live_overlay(tmp_path):
    code='''import headless,hashlib
from pathlib import Path
from live_agent import LiveConstructionAgent
from solver_agent import MdArcSolverAgent
p=json.load(sys.stdin);root=Path(p['root']);root.mkdir();ipc=root/'ipc';ipc.mkdir()
source='def initialize(o): return {"grid":o["grid"]}\\ndef predict(s,a): return [{"state":s,"commitments":{"before":s["grid"][0][0]}}]\\ndef assimilate(s,o,v): return {"grid":o["grid"]}\\n'
(root/'FROZEN.json').write_text(json.dumps({'candidate_source':source,'source_digest':hashlib.sha256(source.encode()).hexdigest()}))
a=object.__new__(LiveConstructionAgent);a.run_dir=root;a.ipc_dir=ipc;a._construction_attempted=True
state={'step':1,'turn':1,'available_actions':['UP','CLICK','DOWN'],'grid_rows':['0']}
a._latest_state=lambda:dict(state)
type.__setattr__(MdArcSolverAgent,'format_game_context',lambda self:'fixture context')
def submit(self,actions,*args,**kw):
    old=int(state['grid_rows'][0]); state['step']+=1;state['turn']+=1;state['grid_rows']=[str(old+1)]
    row={'step':state['step'],'action':actions[0],'before':{'grid':[[old]],'levels_completed':0},'after':{'grid':[[old+1]],'levels_completed':0}}
    with (ipc/'transitions.jsonl').open('a') as f:f.write(json.dumps(row)+'\\n')
    return 'submitted'
type.__setattr__(MdArcSolverAgent,'submit_actions',submit)
for action in ['UP','CLICK 31 41','DOWN']:
    a.submit_actions([action]);a.format_game_context()
print(json.dumps([json.loads(x) for x in (root/'host-events.jsonl').read_text().splitlines()]))
'''
    old=isolated(R9,code,{'root':str(tmp_path/'r9')})
    new=isolated(PACKAGE,code,{'root':str(tmp_path/'r10')})
    oldq=[e for e in old if e['event']=='query_frozen_predict']
    newq=[e for e in new if e['event']=='query_frozen_predict']
    assert [e['action'] for e in oldq]==['UP','DOWN']
    assert json.loads(oldq[-1]['tool_result'])['commitments']['before']==1
    assert [e['action'] for e in newq]==['UP','CLICK 31 41','DOWN']
    assert json.loads(newq[-1]['tool_result'])['commitments']['before']==2
    assert len([e for e in new if e['event']=='assimilate_frozen' and e['status']=='COMPLETED'])==3
    (tmp_path/'STALE-STATE-REPRODUCTION.json').write_text(json.dumps({'scope':'scripted state writer; actual live overlay and sealed RPC','r9':old,'r10':new},indent=2))
