import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import probe
import score

def test_budget_caps_before_spend():
    b=probe.Budget()
    for _ in range(24): b.reserve(1000)
    with pytest.raises(RuntimeError):b.reserve(1000)
    b.arm_calls=0
    with pytest.raises(RuntimeError):b.reserve(200001)
    b.reserved=.99
    with pytest.raises(RuntimeError):b.reserve(1000)

def test_seed_excludes_held_out_steps_and_final_notes(tmp_path):
    ipc=probe.seed(tmp_path/'replay')
    assert max(r['step'] for r in probe.rows(ipc/'states.jsonl'))==6
    assert max(r['step'] for r in probe.rows(ipc/'transitions.jsonl'))==6
    assert not list((tmp_path/'replay').rglob('notes*'))

def test_captured_wrong_band_prediction_fails_then_fixed_passes():
    path=probe.CAPTURE/'team_nemo/shared/helpers/worldmodel.py'
    spec=importlib.util.spec_from_file_location('seed_model',path)
    model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)
    future=[r for r in probe.rows(probe.CAPTURE/'ipc/transitions.jsonl') if r['step']>6]
    assert score.evaluate(model,future)['band_correct']==0
    original=model.predict
    def fixed(z,a):
        result=original(z,a);result['band_right']=z['band_right'];return result
    assert score.evaluate(SimpleNamespace(encode=model.encode,predict=fixed),future)['band_correct']==4

def test_worker_environment_does_not_carry_credentials(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY','secret')
    monkeypatch.setenv('OTHER_TOKEN','secret')
    env=probe.worker_env(probe.BASELINE,'http://127.0.0.1:1/v1')
    assert env['ARC_LLM_API_KEY']=='local-relay'
    assert 'OPENROUTER_API_KEY' not in env and 'OTHER_TOKEN' not in env

@pytest.mark.parametrize('candidate',[probe.BASELINE,probe.CANDIDATE])
def test_real_codeact_worker_submits_from_replay_without_network(tmp_path,candidate):
    import subprocess
    run=tmp_path/'replay'
    probe.seed(run)
    proc=subprocess.run([sys.executable,str(HERE/'worker.py'),'--run-dir',str(run),'--offline-smoke'],
        env=probe.worker_env(candidate,'http://127.0.0.1:1/v1'),capture_output=True,text=True,timeout=30)
    assert proc.returncode==0,proc.stdout+proc.stderr
    assert (run/'REPLAY-COMPLETE.json').exists()
    assert len(probe.rows(run/'ipc/actions.jsonl'))==1
    if candidate==probe.CANDIDATE:
        feedback=probe.rows(run/'ipc/model_feedback.jsonl')
        assert any(r['status']=='mismatch' for r in feedback),feedback

def test_exact_observed_coroutine_allocation_loop_now_fails_promptly():
    import ast
    candidate=(probe.CANDIDATE/'chassis/solver_agent.py').read_text()
    node=next(n for n in ast.parse(candidate).body if isinstance(n,ast.ClassDef) and n.name=='_StateAccess')
    namespace={}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<state-access>','exec'),namespace)
    actor=SimpleNamespace(game_states=namespace['_StateAccess']())
    states=[]
    with pytest.raises(RuntimeError,match='latest_state'):
        while True:
            item=actor.game_states.get(timeout=1.0)
            states.append(item)
    assert states==[]

def test_completed_usage_credits_only_unspent_reservation():
    b=probe.Budget();before=b.reserved
    amount=b.reserve(10000)
    charge=b.reconcile(amount,{'prompt_tokens':1000,'completion_tokens':100,'cost':.0005})
    assert charge==pytest.approx(.00072)
    assert b.reserved==pytest.approx(before+charge)
    amount=b.reserve(10000);before=b.reserved
    assert b.reconcile(amount,None)==amount
    assert b.reserved==before
    with pytest.raises(RuntimeError,match='exceeds'):
        b.reconcile(.0001,{'prompt_tokens':1000,'completion_tokens':100})

def test_newly_added_false_band_left_is_not_silently_ignored():
    path=probe.CAPTURE/'team_nemo/shared/helpers/worldmodel.py'
    spec=importlib.util.spec_from_file_location('seed_model_extra',path)
    model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)
    original=model.predict
    def encode(g):
        z=model.encode(g);z['band_left']=int(score.np.where(g==11)[1].min());return z
    def predict(z,a):
        r=original(z,a);r['band_right']=z['band_right'];return r
    future=[r for r in probe.rows(probe.CAPTURE/'ipc/transitions.jsonl') if r['step']>6]
    result=score.evaluate(SimpleNamespace(encode=encode,predict=predict),future)
    assert result['band_correct']==4
    assert result['all_fields_correct']==0
    assert all(not r['field_correct']['band_left'] for r in result['transitions'])

def test_scorer_uses_committed_source_not_later_helper(tmp_path):
    import hashlib,json,subprocess
    arm=tmp_path/'candidate-a'
    ipc=probe.seed(arm)
    source=(arm/'team_nemo/shared/helpers/worldmodel.py').read_text().replace('max(12, z["band_right"] - 2)','z["band_right"]')
    (ipc/'model_predictions.jsonl').write_text(json.dumps({'step':6,'status':'committed','source':source,'model_sha256':hashlib.sha256(source.encode()).hexdigest()})+'\n')
    (arm/'REPLAY-COMPLETE.json').write_text('{}')
    proc=subprocess.run([sys.executable,str(HERE/'score.py'),str(tmp_path)],capture_output=True,text=True,timeout=30,
        env=probe.worker_env(probe.CANDIDATE,'http://127.0.0.1:1/v1'))
    assert proc.returncode==0,proc.stdout+proc.stderr
    result=json.loads((tmp_path/'SCORE.json').read_text())[0]
    assert result['source_selection']=='exact_action_committed_source'
    assert result['band_correct']==4
