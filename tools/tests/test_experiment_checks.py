import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.experiment_checks import Evidence, batch_gate, batching, coverage, flash, preserve


ROOT = Path(__file__).resolve().parents[2]
FLASH = ROOT / 'analysis/deployment-probes/20260909-flash-next-feasibility'
ARMS = ROOT / 'analysis/deployment-probes/20260909-contingent-arm-qualification/v3-output'
CARD = ROOT / 'deployment/kaggle/flash-next-feasibility-20260909/EXPERIMENT-CARD.md'


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_evidence_cannot_pass(tmp_path):
    (tmp_path / 'EXPERIMENT-CARD.md').write_text(CARD.read_text())
    result = flash(tmp_path)
    assert result['status'] == 'unresolved'
    assert any('vllm-openai-server.log' in item for item in result['unresolved'])


def test_complete_marker_cannot_override_recovered_oom(tmp_path):
    (tmp_path / 'EXPERIMENT-CARD.md').write_text(CARD.read_text())
    (tmp_path / 'verdict.json').write_text('{"feasibility":"pass"}')
    (tmp_path / 'vllm-openai-server.log').write_text(
        'progress\rprogress\rallocation failed with OOM\nRecovered; server ready\nCOMPLETE\n')
    result = flash(tmp_path)
    assert result['status'] == 'invalid'
    assert result['failures'][0]['condition'] == 'OOM'
    assert result['failures'][0]['evidence'][0]['source'].endswith(':1')


def test_no_oom_mention_is_not_an_oom_event(tmp_path):
    (tmp_path / 'EXPERIMENT-CARD.md').write_text(CARD.read_text())
    (tmp_path / 'vllm-openai-server.log').write_text('no OOM occurred\n')
    assert not flash(tmp_path)['observations']['oom']


def test_different_card_requires_interpretation(tmp_path):
    (tmp_path / 'EXPERIMENT-CARD.md').write_text('Invalid if: fatal OOM only.')
    assert flash(tmp_path)['status'] == 'unresolved'


def test_bad_jsonl_does_not_disappear(tmp_path):
    path = tmp_path / 'rows.jsonl'
    path.write_text('{}\nnot json\n')
    evidence = Evidence()
    assert len(evidence.rows(path)) == 1
    assert evidence.report('test', [])['status'] == 'unresolved'


@pytest.mark.parametrize('filename,value', [
    ('vllm-setup-provenance.json', {'runtime': None}),
    ('vllm-server-identity.json', {'ready_epoch': []}),
    ('results.jsonl', {'name': []}),
    ('resources.jsonl', {'gpu': 'GPU, 1 MiB, 2 MiB, 0 %', 'epoch': None}),
    ('resources.jsonl', {'gpu': 'GPU, , 2 MiB, 0 %', 'epoch': 10}),
])
def test_malformed_metadata_cannot_erase_oom(tmp_path, filename, value):
    (tmp_path / 'EXPERIMENT-CARD.md').write_text(CARD.read_text())
    (tmp_path / 'vllm-openai-server.log').write_text('CUDA out of memory\n')
    (tmp_path / 'vllm-server-identity.json').write_text('{"ready_epoch":10}')
    (tmp_path / filename).write_text(json.dumps(value) + '\n')
    result = flash(tmp_path)
    assert result['status'] == 'invalid'
    assert result['observations']['oom']
    assert result['unresolved']


@pytest.mark.parametrize('value', ['null', 'NaN', '[]'])
def test_non_object_json_is_unresolved(tmp_path, value):
    path = tmp_path / 'object.json'
    path.write_text(value)
    evidence = Evidence()
    assert evidence.obj(path) is None
    assert evidence.report('test', [])['status'] == 'unresolved'


@pytest.mark.parametrize('change,condition', [('terminal', 'missing terminal stream/usage'),
                                            ('cleanup', 'failed cleanup'),
                                            ('request', 'request accounting hash mismatch')])
def test_completed_response_does_not_hide_invalidity(tmp_path, change, condition):
    import hashlib
    (tmp_path / 'EXPERIMENT-CARD.md').write_text(CARD.read_text())
    (tmp_path / 'vllm-openai-server.log').write_text('server ready\n')
    (tmp_path / 'cleanup.json').write_text(json.dumps(dict(stopped=change != 'cleanup',
        survivors=[], port_open=False, identity_matches=True)))
    request = b'{}'
    (tmp_path / 'text.request.json').write_bytes(request)
    usage = dict(prompt_tokens=1, completion_tokens=1)
    row = dict(name='text', usage=usage, finish_reason='stop', done=True,
               request_sha256='wrong' if change == 'request' else hashlib.sha256(request).hexdigest())
    write_rows(tmp_path / 'results.jsonl', [row])
    (tmp_path / 'text.response.json').write_text(json.dumps(row))
    stream = 'data: ' + json.dumps(dict(usage=usage, choices=[dict(finish_reason='stop')])) + '\n'
    if change != 'terminal':
        stream += 'data: [DONE]\n'
    (tmp_path / 'text.sse').write_text(stream)
    result = flash(tmp_path)
    assert result['status'] == 'invalid'
    assert condition in [f['condition'] for f in result['failures']]


@pytest.mark.parametrize('status,allowed,proposal_hash,feedback_hash,expected', [
    ('committed', True, 'a', 'a', True),
    ('committed', False, 'a', 'a', False),
    ('committed', True, 'b', 'a', False),
    ('unavailable', True, 'a', 'a', False),
    ('committed', None, 'a', 'a', None),
    ('committed', True, None, 'a', None),
])
def test_all_batch_prerequisites(status, allowed, proposal_hash, feedback_hash, expected):
    proposal = dict(status=status, model_sha256=proposal_hash)
    feedback = dict(batch_allowed=allowed, model_sha256=feedback_hash)
    assert batch_gate(proposal, feedback) is expected
    if expected is not None:
        module = load_module('prediction_feedback_checks', ROOT /
            'deployment/kaggle/contingent-controller-20260909/chassis/prediction_feedback.py')
        assert (module.PredictionFeedback.batch_error(None, proposal, feedback) is None) is expected


def test_false_permission_never_becomes_declined_opportunities(tmp_path):
    write_rows(tmp_path / 'model_feedback.jsonl', [dict(batch_allowed=False)] * 4)
    for name in ['model_predictions', 'actions', 'batch_stops']:
        write_rows(tmp_path / f'{name}.jsonl', [])
    result = batching(tmp_path)
    assert result['observations']['permission_true'] == []
    assert result['observations']['declined_opportunities'] is None


@pytest.mark.parametrize('actions', ['UP', None, {}, 2, []])
def test_malformed_actions_are_not_counted_as_batches(tmp_path, actions):
    write_rows(tmp_path / 'model_feedback.jsonl', [dict(batch_allowed=False)])
    for name in ['model_predictions', 'batch_stops']:
        write_rows(tmp_path / f'{name}.jsonl', [])
    write_rows(tmp_path / 'actions.jsonl', [dict(turn=0, step=0, actions=actions)])
    result = batching(tmp_path)
    assert result['status'] == 'unresolved'
    assert result['observations']['batch_submissions'] == 0
    assert result['observations']['malformed_submission_rows'] == 1


def test_future_feedback_does_not_license_past_submission(tmp_path):
    write_rows(tmp_path / 'model_feedback.jsonl', [dict(batch_allowed=True, turn=2, model_sha256='a')])
    write_rows(tmp_path / 'model_predictions.jsonl', [dict(turn=1, step=1, actions=['UP'],
                                                        status='committed', model_sha256='a')])
    write_rows(tmp_path / 'actions.jsonl', [dict(turn=1, step=1, actions=['UP'])])
    write_rows(tmp_path / 'batch_stops.jsonl', [])
    result = batching(tmp_path)['observations']
    assert result['submissions'][0]['eligibility_at_submission'].startswith('unresolved')


def test_coverage_tail_and_output_reservation(tmp_path):
    write_rows(tmp_path / 'results.jsonl', [dict(name='actor-a', usage=dict(prompt_tokens=21000))])
    calls = tmp_path / 'calls.jsonl'
    write_rows(calls, [dict(request_purpose='actor', prompt_tokens=n) for n in [21000, 32000, 38644]])
    result = coverage(tmp_path, [calls], 32768, 1024)
    assert result['status'] == 'invalid'
    assert result['observations']['prompt_only_exceeds'] == 1
    assert len(result['observations']['prompt_plus_output_exceeds']) == 2
    assert 'tokenizer' in result['unresolved'][0]


@pytest.mark.parametrize('usage', [None, {}, {'prompt_tokens': True}, {'prompt_tokens': -1},
                                 {'prompt_tokens': '10000'}])
def test_malformed_fixture_counts_remain_visible(tmp_path, usage):
    write_rows(tmp_path / 'results.jsonl', [dict(name='actor-good', usage=dict(prompt_tokens=200)),
                                          dict(name='actor-bad', usage=usage)])
    calls = tmp_path / 'calls.jsonl'
    write_rows(calls, [dict(request_purpose='actor', prompt_tokens=100)])
    result = coverage(tmp_path, [calls], 32768, 1024)
    assert result['status'] == 'unresolved'
    assert any('results.jsonl:2' in item for item in result['unresolved'])


def test_empty_inventory_is_unresolved(tmp_path):
    for name in ['model_feedback', 'model_predictions', 'actions', 'batch_stops']:
        write_rows(tmp_path / f'{name}.jsonl', [])
    assert batching(tmp_path)['status'] == 'unresolved'


@pytest.mark.parametrize('status', ['unmeasured', 'unavailable'])
def test_runtime_no_model_feedback_needs_no_helper_identity(tmp_path, status):
    write_rows(tmp_path / 'model_feedback.jsonl', [dict(status=status, batch_allowed=False)])
    for name in ['model_predictions', 'actions', 'batch_stops']:
        write_rows(tmp_path / f'{name}.jsonl', [])
    result = batching(tmp_path)
    assert result['status'] == 'supported'
    assert result['observations']['permission_true'] == []


def test_preservation_refuses_overwrite_and_hashes_content(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'server.log').write_text('original OOM evidence\n')
    dest = tmp_path / 'attempt-1'
    preserve(source, dest)
    manifest = json.loads((dest / 'EVIDENCE-MANIFEST.json').read_text())
    assert 'server.log' in manifest['files']
    (source / 'server.log').write_text('later version\n')
    with pytest.raises(FileExistsError):
        preserve(source, dest)
    assert (dest / 'server.log').read_text() == 'original OOM evidence\n'
    with pytest.raises(ValueError):
        preserve(source, source / 'nested')


def test_preservation_rejects_special_entries_before_copy(tmp_path):
    import os
    source = tmp_path / 'source'
    source.mkdir()
    os.mkfifo(source / 'server.log')
    destination = tmp_path / 'snapshot'
    with pytest.raises(ValueError, match='regular files'):
        preserve(source, destination)
    assert not destination.exists()


@pytest.mark.parametrize('content,reasoning', [('Useful prose', ''), ('Useful prose', '\n\n')])
def test_actual_probe_accepts_actor_prose_and_whitespace(tmp_path, monkeypatch, content, reasoning):
    probe = load_module('flash_probe_checks', ROOT /
                        'deployment/kaggle/flash-next-feasibility-20260909/probe.py')
    probe.ROOT = tmp_path
    events = [dict(choices=[dict(delta=dict(content=content, reasoning_content=reasoning),
                                 finish_reason='stop')]),
              dict(choices=[], usage=dict(prompt_tokens=10, completion_tokens=2))]

    class Stream:
        def __enter__(self):
            return iter([('data: ' + json.dumps(e) + '\n').encode() for e in events] + [b'data: [DONE]\n'])

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(probe, 'urlopen', lambda *args, **kwargs: Stream())
    probe.generate('actor-case-test', dict(messages=[], tool_choice='auto'))
    row = json.loads((tmp_path / 'results.jsonl').read_text())
    assert row['actor_returned_tool'] is False
    assert not (tmp_path / 'actor-case-test.failure.json').exists()


@pytest.mark.parametrize('tool,arguments,accepted', [
    ('offered', '{"value":4}', True),
    ('not_offered', '{"value":4}', False),
    ('offered', '{"value":"wrong type"}', False),
    ('offered', 'not json', False),
])
def test_actual_probe_tool_contract(tmp_path, monkeypatch, tool, arguments, accepted):
    probe = load_module('flash_probe_tool_checks', ROOT /
                        'deployment/kaggle/flash-next-feasibility-20260909/probe.py')
    probe.ROOT = tmp_path
    events = [dict(choices=[dict(delta=dict(tool_calls=[dict(index=0,
                 function=dict(name=tool, arguments=arguments))]), finish_reason='tool_calls')]),
              dict(choices=[], usage=dict(prompt_tokens=10, completion_tokens=2))]

    class Stream:
        def __enter__(self):
            return iter([('data: ' + json.dumps(e) + '\n').encode() for e in events] + [b'data: [DONE]\n'])

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(probe, 'urlopen', lambda *args, **kwargs: Stream())
    payload = dict(messages=[], tool_choice='auto', tools=[dict(type='function', function=dict(
        name='offered', parameters=dict(type='object', properties=dict(value=dict(type='integer')),
                                        required=['value'])))])
    if accepted:
        probe.generate('actor-case-test', payload)
        assert json.loads((tmp_path / 'results.jsonl').read_text())['actor_returned_tool'] is True
    else:
        with pytest.raises(Exception):
            probe.generate('actor-case-test', payload)
        assert (tmp_path / 'actor-case-test.failure.json').exists()


@pytest.mark.skipif(not (FLASH / 'raw-v3').exists(), reason='historical evidence not present')
def test_historical_flash_rejects_pass_and_separates_measurements():
    for version in ['raw', 'raw-v3']:
        result = flash(FLASH / version)
        assert result['status'] == 'invalid'
        assert result['observations']['oom']
        assert result['observations']['response_count'] == (3 if version == 'raw' else 8)
        assert result['observations']['issued_request_count'] == (3 if version == 'raw' else 8)
    obs = result['observations']
    assert obs['setup_seconds'] == pytest.approx(514.106841395)
    assert [r['free_mib'] for r in obs['post_readiness_samples']] == [5187, 3027, 3027]
    assert obs['all_run_sampled_min_free_mib'] == 1259


@pytest.mark.skipif(not ARMS.exists(), reason='historical evidence not present')
def test_historical_arms_and_context():
    paths = [next((ARMS / f'contingent-{arm}').rglob('ipc/model_feedback.jsonl')).parent
             for arm in ['off', 'on']]
    reports = [batching(path) for path in paths]
    assert all(report['status'] == 'supported' for report in reports)
    off, on = [report['observations'] for report in reports]
    assert off['feedback_rows'] == 13 and on['feedback_rows'] == 19
    assert [r['start_step'] for r in off['permission_true']] == [6, 14]
    assert on['permission_true'] == []
    assert (off['batch_submissions'], on['batch_submissions']) == (2, 0)
    stopped = [s for r in off['submissions'] for s in r['stops'] if s.get('skipped_actions') == 1]
    assert stopped and stopped[0]['reason'] == 'prediction_mismatch'
    context = coverage(FLASH / 'raw-v3', [p / 'model_calls.jsonl' for p in paths], 32768, 1024)
    assert context['status'] == 'invalid'
    assert context['observations']['live']['count'] == 136
    assert context['observations']['prompt_only_exceeds'] == 8


def test_cli_missing_evidence_exits_unresolved(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / 'tools/experiment_checks.py'),
                             'flash', str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 2
    assert json.loads(result.stdout)['status'] == 'unresolved'
