"""Local manufactured integration fixtures, not model or gameplay evidence."""
import copy,json,sys
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(HERE),str(HERE/'engine')]
from evidence import prepare,read,put,raw_json,digest,select
from pipeline import construct,predict_once,score_bound,FixtureHTTP
from prediction_contract import score
from fixtures import IDENTITY,source_for,Server,replies

def trace(tmp_path,target=None,current=None):
 current=current or [[0]*10 for _ in range(10)]
 target=target if target is not None else [[7]+[0]*9]+[[0]*10 for _ in range(9)]
 def state(g):return {'grid':g,'state':'NOT_FINISHED','levels_completed':0}
 rows=[{'step':1,'turn':0,'action':'RIGHT','before':state(current),'after':state(current)}, {'step':2,'turn':1,'action':'DOWN','before':state(current),'after':state(target)}]
 source=tmp_path/'trace.jsonl';source.write_text(''.join(json.dumps(x)+'\n' for x in rows))
 prepare(source,tmp_path/'data','fixture-session',HERE/'EXPERIMENT-CARD.md')
 return tmp_path/'data',rows

def built(tmp_path,source=IDENTITY,target=None,current=None,inspection=None):
 data,rows=trace(tmp_path,target,current)
 with Server(replies(source,inspection) if inspection else replies(source)) as server:
  result=construct(data/'prefix.jsonl',tmp_path/'constructor',server.url)
 assert result.status=='CANDIDATE_READY'
 return data,tmp_path/'constructor/FROZEN.json',server.requests

def predict(data,frozen,tmp_path):
 output=tmp_path/'prediction'
 sha=predict_once(frozen,data/'prefix.jsonl',data/'query.json',output,registry=tmp_path/'registry')
 return output,sha

def scoring(data,frozen,output,sha):return score_bound(output,frozen,data/'prefix.jsonl',data/'query.json',data/'heldout/outcome.json',sha)

@pytest.mark.parametrize('kind',['identity','exact','abstain'])
def test_http_constructor_sealed_freeze_prediction_then_score(tmp_path,kind):
 Y=[[7]+[0]*9]+[[0]*10 for _ in range(9)]
 source=IDENTITY if kind=='identity' else source_for(repr({'type':'grid','predicted_grid':Y}) if kind=='exact' else repr({'type':'abstain','reason':'not enough evidence'}))
 data,frozen,requests=built(tmp_path,source,Y)
 output,sha=predict(data,frozen,tmp_path)
 assert not (output/'SCORE.json').exists()
 assert [x['method'] for x in read(output/'RAW-PREDICTION.json')['rpc_methods']]==['predict']
 result=scoring(data,frozen,output,sha);m=result['metrics']
 assert m['baseline']['cell_accuracy']==.99 and not m['baseline']['exact_match'] and m['changed_cells']==1
 if kind=='identity':assert m['prediction']['cell_accuracy']==.99 and m['prediction']['changed_cell_accuracy']==0 and m['paired_cell_accuracy_difference']==0 and not m['predictive_success']
 if kind=='exact':assert m['prediction']['exact_match'] and m['prediction']['changed_cell_accuracy']==1 and m['paired_cell_accuracy_difference']==pytest.approx(.01)
 if kind=='abstain':assert m['coverage']==0 and m['prediction'] is None and m['paired_cell_accuracy_difference'] is None and not m['predictive_success']
 assert result['heldout_assimilations']==0 and result['target_feedback_to_constructor'] is False
 assert len(requests)==2
 assert all('DOWN' not in json.dumps(r) for r in requests)
 assert list(output.glob('sealed-rpc.jsonl'))

@pytest.mark.parametrize('value',[{}, {'type':'grid'}, {'type':'grid','predicted_grid':[[0]]}, {'type':'grid','predicted_grid':[[False]*10]*10}, {'type':'grid','predicted_grid':[[0.0]*10]*10}, {'type':'grid','predicted_grid':[[0]*10]*10,'reason':'mixed'}, {'type':'abstain','reason':' '}, {'predicted_grid':[[0]*10]*10}, 'a grid', {'type':'abstain','reason':'unknown','predicted_grid':[[0]*10]*10}])
def test_malformed_heldout_commitment_never_opens_target(tmp_path,monkeypatch,value):
 data,frozen,_=built(tmp_path,source_for(repr(value)))
 original=Path.read_bytes
 def guarded(path):
  if path==data/'heldout/outcome.json':raise AssertionError('target leaked before binding')
  return original(path)
 monkeypatch.setattr(Path,'read_bytes',guarded)
 with pytest.raises((ValueError,RuntimeError)):predict(data,frozen,tmp_path)
 assert (tmp_path/'prediction/FAILURE.json').exists() and not (tmp_path/'prediction/BOUND.json').exists()


def test_target_file_environment_and_action_not_in_constructor_tools(tmp_path):
 target=[[987654321]+[0]*9]+[[0]*10 for _ in range(9)]
 # A tool actively probes the real target path; sealed chroot must deny it.
 path=tmp_path/'data/heldout/outcome.json'
 code='''import json,os
v=json.load(open("/evidence/case.json"))
assert v["frontier"]==1
assert len(v["visible_transitions"])==1
assert v["visible_transitions"][0]["action"]=="RIGHT"
assert set(os.environ)=={"LANG","PYTHONDONTWRITEBYTECODE"}
try:
 open('''+repr(str(path))+''').read()
 raise AssertionError("target opened")
except FileNotFoundError:pass
print("prefix only; target denied")'''
 data,frozen,requests=built(tmp_path,IDENTITY,target,inspection=code)
 assert '987654321' not in json.dumps(requests) and '"action": "DOWN"' not in json.dumps(requests)
 events=[json.loads(x) for x in (tmp_path/'constructor/child/tool-events.jsonl').read_text().splitlines()]
 assert 'prefix only; target denied' in events[0]['rendered_result_message']['content']
 # Predictor gets only state/action; a source probe of /evidence must also fail.
 output,sha=predict(data,frozen,tmp_path);assert scoring(data,frozen,output,sha)['metrics']['coverage']==1


def test_read_target_guard_and_immutable_binding(tmp_path,monkeypatch):
 data,frozen,_=built(tmp_path);out,sha=predict(data,frozen,tmp_path)
 original=Path.read_bytes;reads=[]
 def guard(path):
  if path==data/'heldout/outcome.json':
   assert digest(original(out/'BOUND.json'))==sha
   assert read(out/'SCORING-CONSUMED.json')['binding_sha256']==sha
   reads.append(str(path))
  return original(path)
 monkeypatch.setattr(Path,'read_bytes',guard)
 scoring(data,frozen,out,sha);assert len(reads)==1
 with pytest.raises(FileExistsError):scoring(data,frozen,out,sha)
 assert len(reads)==1
 with pytest.raises(FileExistsError):predict_once(frozen,data/'prefix.jsonl',data/'query.json',tmp_path/'retry-new-output',registry=tmp_path/'registry')

@pytest.mark.parametrize('changed',['prediction','source','bound','prefix','query'])
def test_drift_stops_before_target_read(tmp_path,monkeypatch,changed):
 data,frozen,_=built(tmp_path);out,sha=predict(data,frozen,tmp_path)
 f={'prediction':out/'PREDICTION.json','source':frozen,'bound':out/'BOUND.json','prefix':data/'prefix.jsonl','query':data/'query.json'}[changed]
 f.chmod(0o644);f.write_bytes(f.read_bytes()+b' ')
 original=Path.read_bytes
 def guard(path):
  if path==data/'heldout/outcome.json':raise AssertionError('target read after drift')
  return original(path)
 monkeypatch.setattr(Path,'read_bytes',guard)
 with pytest.raises(ValueError):scoring(data,frozen,out,sha)

def test_score_without_prediction_cannot_read_target(tmp_path,monkeypatch):
 data,frozen,_=built(tmp_path);out=tmp_path/'prediction';out.mkdir()
 original=Path.read_bytes
 def guard(path):
  if path==data/'heldout/outcome.json':raise AssertionError('early target read')
  return original(path)
 monkeypatch.setattr(Path,'read_bytes',guard)
 with pytest.raises(FileNotFoundError):scoring(data,frozen,out,'missing')

@pytest.mark.parametrize('mutation',['reset','episode','level','gap','terminal','discontinuity','absent'])
def test_recording_boundaries_fail_closed(tmp_path,mutation):
 data,rows=trace(tmp_path)
 if mutation=='reset':rows[1]['action']='RESET'
 if mutation=='episode':rows[1]['session_id']='other'
 if mutation=='level':rows[1]['after']['levels_completed']=1
 if mutation=='gap':rows[1]['step']=3
 if mutation=='terminal':rows[0]['after']['state']='WIN'
 if mutation=='discontinuity':rows[1]['before']['grid']=[[2]*10]*10
 if mutation=='absent':rows=rows[:1]
 with pytest.raises(ValueError):select(rows,'fixture-session')

@pytest.mark.parametrize('target',[[[1]],[[False]*10]*10,[[1.0]*10]*10,[[float('nan')]*10]*10,[[float('inf')]*10]*10])
def test_scoring_rejects_shape_type_nonfinite(target):
 with pytest.raises(ValueError):score([[0]*10]*10,target,{'type':'grid','predicted_grid':[[0]*10]*10})

def test_unchanged_identity_and_undefined_diagnostics():
 m=score([[0]],[[0]],{'type':'grid','predicted_grid':[[0]]})
 assert m['prediction']['exact_match'] and m['baseline']['exact_match'] and m['paired_cell_accuracy_difference']==0
 assert m['changed_cells']==0 and m['prediction']['changed_cell_accuracy'] is None
 m=score([[0]],[[1]],{'type':'grid','predicted_grid':[[1]]})
 assert m['prediction']['false_change_rate'] is None
 m=score([[0,0]],[[1,0]],{'type':'grid','predicted_grid':[[0,2]]})
 assert m['prediction']['false_changes']==1 and m['prediction']['false_change_rate']==1 and m['paired_cell_accuracy_difference']==-.5

def test_no_remote_transport_or_auth():
 with pytest.raises(ValueError):FixtureHTTP('https://openrouter.ai/api/v1/chat/completions',Path('/unused'))

def test_prediction_sandbox_cannot_read_target_prefix_or_environment(tmp_path):
 path=tmp_path/'data/heldout/outcome.json'
 probe='''
 import os
 assert set(os.environ)=={"LANG","PYTHONDONTWRITEBYTECODE"}
 for path in ["/evidence/case.json",'''+repr(str(path))+''']:
  try:
   open(path).read()
   raise ValueError("LEAK")
  except FileNotFoundError:pass
'''
 source=IDENTITY.replace('def predict(state,action):','def predict(state,action):'+probe)
 data,frozen,_=built(tmp_path,source);out,sha=predict(data,frozen,tmp_path)
 assert scoring(data,frozen,out,sha)['metrics']['coverage']==1


def test_malformed_typed_visible_commitment_gets_constructor_feedback(tmp_path):
 from fixtures import reply,WORK
 data,_=trace(tmp_path)
 bad=IDENTITY.replace('{"type":"grid","predicted_grid":state["grid"]}','{}')
 responses=replies(bad)+[reply('submit_candidate',{'work_item':WORK,'candidate_source':IDENTITY.splitlines()},3)]
 with Server(responses) as server:result=construct(data/'prefix.jsonl',tmp_path/'constructor',server.url)
 assert result.status=='CANDIDATE_READY' and result.submissions==2
 feedback=json.loads([m for m in server.requests[2]['messages'] if m['role']=='tool'][-1]['content'])
 assert feedback['visible_evaluation']['visible']['status']=='failed'
 assert 'commitments must be exactly' in feedback['visible_evaluation']['visible']['error']
 assert not (tmp_path/'prediction').exists()


def test_import_provided_artifact_rechecks_typed_contract(tmp_path):
 from pipeline import freeze_artifact
 data,_,_=built(tmp_path)
 artifact=tmp_path/'provided.json';put(artifact,{'candidate_source':IDENTITY,'source_digest':digest(IDENTITY.encode()),'fixture_constructor':True})
 freeze_artifact(artifact,data/'prefix.jsonl',tmp_path/'imported')
 out,sha=predict(data,tmp_path/'imported/FROZEN.json',tmp_path)
 assert scoring(data,tmp_path/'imported/FROZEN.json',out,sha)['fixture_constructor'] is True


def test_target_wrong_type_retained_without_retry(tmp_path):
 data,frozen,_=built(tmp_path)
 # Manufactured malformed target has matching declared hash; scorer must still reject shape.
 target=read(data/'heldout/outcome.json');target['grid']=[[False]*10]*10
 f=data/'heldout/outcome.json';f.chmod(0o644);f.write_bytes(raw_json(target))
 q=read(data/'query.json');q['outcome_sha256']=digest(f.read_bytes());g=data/'query.json';g.chmod(0o644);g.write_bytes(raw_json(q))
 out,sha=predict(data,frozen,tmp_path)
 with pytest.raises(ValueError):scoring(data,frozen,out,sha)
 assert (out/'SCORING-FAILURE.json').exists()
 with pytest.raises(FileExistsError):scoring(data,frozen,out,sha)

def test_child_write_then_hard_kill_never_promotes_freeze(tmp_path,monkeypatch):
 import pipeline
 data,_=trace(tmp_path);original=pipeline.cb.bounded
 def delayed_worker(command,**kwargs):
  child=tmp_path/'constructor/child'
  code='from pathlib import Path;import time;p=Path('+repr(str(child))+');p.mkdir();(p/"FROZEN.json").write_text("forensic child bytes");time.sleep(30)'
  return original([sys.executable,'-B','-c',code],timeout=.15,output_cap=10000)
 monkeypatch.setattr(pipeline.cb,'bounded',delayed_worker)
 with pytest.raises(RuntimeError):construct(data/'prefix.jsonl',tmp_path/'constructor','http://127.0.0.1:1/fixture',hard_seconds=.15)
 assert (tmp_path/'constructor/child/FROZEN.json').exists()
 assert not (tmp_path/'constructor/FROZEN.json').exists()
 assert read(tmp_path/'constructor/SUPERVISION.json')['status']=='timeout'


def test_raw_prediction_drift_is_detected(tmp_path,monkeypatch):
 data,frozen,_=built(tmp_path);out,sha=predict(data,frozen,tmp_path)
 f=out/'RAW-PREDICTION.json';f.chmod(0o644);f.write_bytes(f.read_bytes()+b' ')
 original=Path.read_bytes
 def guard(path):
  if path==data/'heldout/outcome.json':raise AssertionError('target read after raw prediction drift')
  return original(path)
 monkeypatch.setattr(Path,'read_bytes',guard)
 with pytest.raises(ValueError):scoring(data,frozen,out,sha)

@pytest.mark.parametrize('kind',['nan','inf','multiple_branches'])
def test_nonfinite_or_multiple_frozen_predictions_never_score(tmp_path,kind):
 if kind in ('nan','inf'):
  source=source_for('{"type":"grid","predicted_grid":[[float('+repr(kind)+')]*10]*10}')
 else:
  source=IDENTITY.replace('def predict(state,action):','def predict(state,action):\n if action=="DOWN": return [{"state":state,"commitments":{"type":"grid","predicted_grid":state["grid"]}}]*2')
 data,frozen,_=built(tmp_path,source)
 with pytest.raises((ValueError,RuntimeError)):predict(data,frozen,tmp_path)
 assert not (tmp_path/'prediction/BOUND.json').exists()
 assert (tmp_path/'prediction/FAILURE.json').exists()
