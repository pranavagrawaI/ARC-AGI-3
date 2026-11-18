"""R10 repair acceptance through real admission, isolated jobs and sealed RPC."""
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent/'r10a-package'
WORK={'decision':'next','obstacle':'unknown','operation':'test','claim':'successor','evidence_refs':[1],'dependencies':['grid'],'completion_checks':[1],'remaining_scope':'unseen','reasoning_need':'needs_computation'}

def run_probe(tmp_path, source, delayed=False):
    ipc=tmp_path/'ipc';ipc.mkdir()
    (ipc/'transitions.jsonl').write_text(json.dumps({'step':1,'turn':0,'action':'UP','before':{'grid':[[0]],'levels_completed':0},'after':{'grid':[[1]],'levels_completed':0}})+'\n')
    responses=[]
    for i,(name,args) in enumerate([('execute_python',{'code':'import json\nprint(json.load(open("/evidence/case.json")))'}),('submit_candidate',{'work_item':WORK,'candidate_source':source})]):
        responses.append({'choices':[{'message':{'role':'assistant','content':None,'tool_calls':[{'id':str(i),'type':'function','function':{'name':name,'arguments':json.dumps(args)}}]}}],'usage':{'prompt_tokens':100,'completion_tokens':30}})
    fixture=tmp_path/'responses.json';fixture.write_text(json.dumps(responses))
    env=dict(os.environ,ARC_CONSTRUCTOR_TRANSPORT='fixture',ARC_CONSTRUCTOR_FIXTURE=str(fixture),PYTHONDONTWRITEBYTECODE='1',ARC_CONSTRUCTOR_HARD_SECONDS='5' if delayed else '245')
    if delayed:
        job=tmp_path/'delayed_job.py'
        job.write_text('import sys,time\nsys.path.insert(0,'+repr(str(PACKAGE))+')\nfrom persist_job import main\nrc=main()\nassert rc==0\ntime.sleep(30)\n')
    script='''import sys,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import headless
import construction_service as service
from audit import freeze_record
root=Path(sys.argv[2])
if sys.argv[3]=='True': service.JOB=root/'delayed_job.py'
r=service.construct_from_run(root,root/'ipc')
from gameplay_play import session_directories
summaries=session_directories(root.parent)
out={'summary':next(x for x in summaries if x['session_dir']==root.name),'service':r,'root_frozen':(root/'FROZEN.json').exists(),'job_frozen':(root/'constructor-job/FROZEN.json').exists(),'audit_freeze':freeze_record(root)}
if sys.argv[3]!='True':
 from live_agent import LiveConstructionAgent
 from solver_agent import MdArcSolverAgent
 a=object.__new__(LiveConstructionAgent);a.run_dir=root;a.ipc_dir=root/'ipc';a._construction_attempted=True
 state={'step':1,'turn':1,'available_actions':['UP','CLICK','DOWN'],'grid_rows':['1']}
 a._latest_state=lambda:dict(state)
 type.__setattr__(MdArcSolverAgent,'format_game_context',lambda self:'fixture context')
 def submit(self,actions,*args,**kw):
  if actions==['CLICK 0 0']: return 'REJECTED fixture without execution'
  old=int(state['grid_rows'][0]);state['step']+=1;state['turn']+=1;state['grid_rows']=[str(old+1)]
  row={'step':state['step'],'action':actions[0],'before':{'grid':[[old]],'levels_completed':0},'after':{'grid':[[old+1]],'levels_completed':0}}
  with (root/'ipc/transitions.jsonl').open('a') as f:f.write(json.dumps(row)+'\\n')
  return 'submitted'
 type.__setattr__(MdArcSolverAgent,'submit_actions',submit)
 out['unexecuted_submit']=a.submit_actions(['CLICK 0 0'])
 out['repeat_query']=a.query_frozen_predict('DOWN')
 out['wrong_assimilation']=a._frozen_client.assimilate_frozen({'grid':[[2]]},executed_action='CLICK 1 1',executed_step=2)
 out['pending_after_rejection']=a._frozen_client._pending_key
 out['first_submit']=a.submit_actions(['CLICK 31 41'])
 out['wrong_step']=a._frozen_client.assimilate_frozen({'grid':[[2]]},executed_action='CLICK 31 41',executed_step=3)
 out['after_context']=a.format_game_context()
 try:out['next_submit']=a.submit_actions(['DOWN'])
 except Exception as e:out['next_error']=type(e).__name__+': '+str(e)
 out['events']=[json.loads(s) for s in (root/'host-events.jsonl').read_text().splitlines()]
print(json.dumps(out))
'''
    proc=subprocess.run([sys.executable,'-B','-c',script,str(PACKAGE),str(tmp_path),str(delayed)],env=env,capture_output=True,text=True,timeout=35)
    (tmp_path/'probe.stderr').write_text(proc.stderr)
    assert proc.returncode==0,proc.stderr
    out=json.loads(proc.stdout);(tmp_path/'probe.json').write_text(json.dumps(out,indent=2)+'\n');return out

def test_admitted_null_state_next_submit_and_unexecuted_binding(tmp_path):
    source='def initialize(o): return {}\ndef predict(s,a): return [{"state":None,"commitments":{}}]\ndef assimilate(s,o,v):\n assert s is None\n return {}\n'
    out=run_probe(tmp_path,source)
    assert out['service']['frozen'] is True and out['summary']['frozen'] is True
    assert out['service']['inspections']==1 and out['service']['model_requests']==2
    assert out['unexecuted_submit'].startswith('REJECTED')
    assert out['repeat_query']=='PREDICT_WITHOUT_ASSIMILATE'
    assert out['wrong_assimilation']=='REJECT_MISMATCH'
    assert out['pending_after_rejection']==['CLICK 0 0',1]
    assert out['wrong_step']=='REJECT_MISMATCH'
    assert out['first_submit']=='submitted' and out['next_submit']=='submitted'
    assert 'assimilated' in out['after_context'] and 'next_error' not in out
    cancellations=[x for x in out['events'] if x['event']=='prediction_cancelled']
    assert len(cancellations)==1 and cancellations[0]['action']=='CLICK 0 0' and cancellations[0]['step']==1
    assims=[x for x in out['events'] if x['event']=='assimilate_frozen' and x['status']=='COMPLETED']
    assert len(assims)==1 and assims[0]['action']=='CLICK 31 41' and assims[0]['step']==2


def test_post_child_write_hard_kill_has_forensics_but_no_live_freeze(tmp_path):
    source='def initialize(o): return {}\ndef predict(s,a): return [{"state":{},"commitments":{}}]\ndef assimilate(s,o,v): return {}\n'
    out=run_probe(tmp_path,source,True)
    assert out['service']['construction_status']=='FAILED'
    assert out['service']['worker_alive'] is False
    assert out['root_frozen'] is False and out['job_frozen'] is True
    assert out['audit_freeze'] is None and out['summary']['frozen'] is False
    assert out['summary']['child_freeze_artifacts']==[tmp_path.name+'/constructor-job/FROZEN.json']
    assert 'constructor-job/FROZEN.json' in '\n'.join(out['summary']['constructor_files'])
    assert json.loads((tmp_path/'constructor-job/RESULT.json').read_text())['frozen'] is True
    assert json.loads((tmp_path/'CONSTRUCTION.json').read_text())['construction_status']=='FAILED'
