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
    for _ in range(12): b.reserve(1000)
    with pytest.raises(RuntimeError):b.reserve(1000)
    b.arm_calls=0
    with pytest.raises(RuntimeError):b.reserve(80001)
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
