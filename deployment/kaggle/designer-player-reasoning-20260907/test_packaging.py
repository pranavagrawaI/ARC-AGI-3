import importlib.util
import json
from pathlib import Path
import sys
import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import assess_candidate
import prepare_candidate


def test_notebook_arms_preserve_inputs_and_isolate_outputs():
    nb=prepare_candidate.build()
    a,b=[''.join(c['source']) for c in nb['cells'][-2:]]
    assert a.replace('designer-player-loop','designer-player-mindset').replace("ARM']='loop'", "ARM']='mindset'")==b
    assert a.index('subprocess.run(command, check=True)') < a.index('assess_candidate.py')
    assert 'speculative-algorithm' not in a
    assert 'submission.parquet' not in a
    assert "TYCHO_PUBLIC_GAME_IDS'] = 'ls20'" in a
    assert '"enable_thinking":true' in a
    assert '"reasoning":{"enabled":true}' in a
    assert "ARC_LLM_MAX_TOKENS'] = '32768'" in a
    assert "ARC_LLM_CONTEXT_WINDOW'] = '262144'" in a
    assert "ARC_LLM_TIMEOUT_SECONDS'] = '1200'" in a


def test_assessor_rejects_missing_and_empty_evidence(tmp_path):
    assert assess_candidate.assess(tmp_path,'mindset')['status']=='INVALID'


def test_exact_shutdown_line_only():
    spec=importlib.util.spec_from_file_location('candidate_runtime',HERE/'exact_request_outer_runtime.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    accepted=b'[2026-09-06 23:17:39] INFO: Waiting for background tasks to complete. (CTRL+C to force quit)\n'
    good=module.classify_teardown_log(accepted)
    assert all(e['allowed'] for e in good['entries'])
    bad=module.classify_teardown_log(accepted.replace(b'complete.',b'complete with unknown failure.'))
    assert any(not e['allowed'] for e in bad['entries'])


def test_extracted_actual_worker(monkeypatch,tmp_path):
    import test_worker_path
    extracted=prepare_candidate.OUT/'extracted'
    assert (extracted/'CHASSIS.json').exists(),'Run prepare_candidate.py first'
    monkeypatch.setattr(test_worker_path,'HERE',extracted)
    test_worker_path.test_packaged_path_has_exact_identity_and_continuity(monkeypatch,tmp_path)


@pytest.mark.parametrize('failure', ['length', 'outer_signal'])
def test_extracted_failure_paths(monkeypatch, tmp_path, failure):
    import test_lifecycle
    monkeypatch.setattr(test_lifecycle, 'HERE', prepare_candidate.OUT/'extracted')
    test_lifecycle.test_real_worker_failure_retires_before_cleanup(monkeypatch, tmp_path, failure)


def test_extracted_evidence_and_preflight_tools(monkeypatch, tmp_path):
    import test_reasoning
    monkeypatch.setattr(test_reasoning, 'HERE', prepare_candidate.OUT/'extracted')
    test_reasoning.test_actual_worker_records_reasoning_and_commits_tool_action(monkeypatch, tmp_path)
