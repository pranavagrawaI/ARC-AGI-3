"""The assessor must accept a valid reasoning-off run and reject a dropped arm.

The 116-test suite never ran assess_candidate.py over evidence shaped like the
run it gates, so two inherited constants (reasoning demanded, action limit
pinned to 30) would have failed both arms of a correct notebook. These cases
build that evidence directly.
"""
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import assess_candidate
import prepare_candidate

ACTIONS = int(prepare_candidate.ACTIONS_PER_ARM)
ADMISSION = int(prepare_candidate.ADMISSION_SECONDS)


OFF_BODY = '{"chat_template_kwargs":{"enable_thinking":false},"reasoning":{"enabled":false}}'
ON_BODY = '{"chat_template_kwargs":{"enable_thinking":true},"reasoning":{"enabled":true}}'


def evidence(root, *, contingent_arm, reasoning_tokens=0, reasoning_present=False,
             action_limit=ACTIONS, recorded_arm=..., extra_body=OFF_BODY):
    """One session's worth of a clean run, parameterised where a run can differ."""
    manifest = json.loads((HERE / 'CHASSIS.json').read_text())
    root.mkdir(parents=True, exist_ok=True)
    (root / 'result.json').write_text(json.dumps({'status': 'passed', 'lifecycle': {
        'containment_passed': True, 'quiet_window_valid': True,
        'term_only': True, 'log_classification_allowed': True}}))
    (root / 'request-count-facts.json').write_text(
        json.dumps({'whole_model_request_count_verified': True}))
    session = root / 'solver-workspaces' / 'ls20-fixture'
    ipc = session / 'ipc'
    ipc.mkdir(parents=True)
    (session / 'launch.json').write_text(json.dumps({
        'contract_sha256': manifest['contract_sha256'], 'solver_arm': 'loop',
        'discovery_arm': 'a',
        'model_config': {'ARC_LLM_EXTRA_BODY': extra_body},
        'contingent_arm': contingent_arm if recorded_arm is ... else recorded_arm}))
    (ipc / 'model_calls.jsonl').write_text(json.dumps(
        {'reasoning_tokens': reasoning_tokens, 'reasoning_present': reasoning_present}) + '\n')
    (ipc / 'transitions.jsonl').write_text(json.dumps(
        {'turn': 0, 'step': 1, 'after': {'levels_completed': 0}}) + '\n')
    (ipc / 'actions.jsonl').write_text(json.dumps({'turn': 0}) + '\n')
    (ipc / 'integrity.jsonl').write_text(json.dumps({'ok': True}) + '\n')
    (root / 'solver-workspaces' / 'game-stops.jsonl').write_text(json.dumps({
        'action_limit': action_limit, 'admission_horizon_seconds': ADMISSION,
        'executed_solver_actions': 1, 'reason': 'step_budget'}) + '\n')
    return root


def assess(root, **kw):
    kw.setdefault('action_limit', ACTIONS)
    kw.setdefault('admission_seconds', ADMISSION)
    return assess_candidate.assess(root, 'loop', **kw)


@pytest.mark.parametrize('arm', prepare_candidate.ARMS)
def test_a_clean_reasoning_off_arm_passes(tmp_path, arm):
    result = assess(evidence(tmp_path, contingent_arm=arm), reasoning='off', contingent_arm=arm)
    assert result['errors'] == []
    assert result['status'] == 'PASS'


def test_a_dropped_factor_fails_the_off_arm(tmp_path):
    # The env-strip failure: the cell asked for off, the worker recorded on.
    result = assess(evidence(tmp_path, contingent_arm='off', recorded_arm='on'),
                    reasoning='off', contingent_arm='off')
    assert result['status'] == 'INVALID'
    assert any('contingent arm not recorded as off' in e for e in result['errors'])


def test_an_absent_record_fails_rather_than_defaulting(tmp_path):
    result = assess(evidence(tmp_path, contingent_arm='on', recorded_arm=None),
                    reasoning='off', contingent_arm='on')
    assert result['status'] == 'INVALID'


def test_the_declared_reasoning_state_is_what_is_gated(tmp_path):
    # A reasoning-on body under an arm that declares off is a real contradiction.
    result = assess(evidence(tmp_path, contingent_arm='on', extra_body=ON_BODY),
                    reasoning='off', contingent_arm='on')
    assert result['status'] == 'INVALID'
    assert any('reasoning-off state not established' in e for e in result['errors'])
    # The same body is what a declared reasoning-on run requires.
    assert assess(evidence(tmp_path / 'on', contingent_arm='on', extra_body=ON_BODY),
                  reasoning='on', contingent_arm='on')['errors'] == []


def test_an_empty_think_block_does_not_abort_a_reasoning_off_run(tmp_path):
    # A thinking-off Qwen3 turn can emit '<think>\n\n</think>', whose whitespace is
    # truthy in reasoning_present and may be counted in reasoning_tokens. The card's
    # stop clause does not authorise aborting on that, so it must be measured, not gated.
    result = assess(evidence(tmp_path, contingent_arm='off', reasoning_tokens=3,
                             reasoning_present=True), reasoning='off', contingent_arm='off')
    assert result['errors'] == []
    assert result['status'] == 'PASS'
    assert result['sessions'][0]['reasoning_tokens'] == 3
    assert result['sessions'][0]['calls_reporting_reasoning'] == 1


def test_the_configured_action_limit_is_what_is_checked(tmp_path):
    # The inherited constant was 30; a ten-action run must not be called wrong.
    assert ACTIONS != 30
    result = assess(evidence(tmp_path, contingent_arm='on', action_limit=30),
                    reasoning='off', contingent_arm='on')
    assert result['status'] == 'INVALID'
    assert any('wrong budgets' in e for e in result['errors'])


def test_the_notebook_hands_the_assessor_its_own_budgets():
    nb = prepare_candidate.build()
    for arm, cell in zip(prepare_candidate.ARMS, nb['cells'][-len(prepare_candidate.ARMS):]):
        source = ''.join(cell['source'])
        assert "'--contingent-arm',os.environ['ARC_CONTINGENT_ARM']" in source
        assert "'--action-limit',os.environ['TYCHO_KAGGLE_MAX_ACTIONS']" in source
        assert "'--admission-seconds',os.environ['TYCHO_GAMEPLAY_ADMISSION_SECONDS']" in source
        assert "'--reasoning','off'" in source
