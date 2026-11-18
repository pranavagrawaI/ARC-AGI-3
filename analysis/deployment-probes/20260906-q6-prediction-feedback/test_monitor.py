import json
from pathlib import Path

import pytest
import monitor_long_run as monitor


@pytest.fixture
def harness(monkeypatch, tmp_path):
    helpers = tmp_path/'helpers'
    helpers.mkdir()
    for name in ('replay_final_feedback.py','summarize_long_run.py'):
        (helpers/name).write_text('# frozen analysis helper\n')
    monkeypatch.setattr(monitor, 'HERE', helpers)
    def snapshot(candidate, destination):
        destination.mkdir()
        return {'kernel':'owner/q6-probe','contract_sha256':'frozen'}
    monkeypatch.setattr(monitor, 'snapshot', snapshot)
    return tmp_path/'output'


@pytest.mark.parametrize('terminal', ['COMPLETE','ERROR','CANCELED'])
def test_one_download_and_analysis_for_terminal_kernel(harness, terminal):
    calls = []
    statuses = iter(['RUNNING', terminal])
    def run(argv, timeout):
        argv = [str(x) for x in argv]
        calls.append(argv)
        result = {'argv':argv,'returncode':0,'stdout':'','stderr':''}
        if argv[1:3] == ['kernels','status']:
            result['stdout'] = f'owner/q6-probe has status "KernelWorkerStatus.{next(statuses)}"'
        elif argv[1:3] == ['kernels','output']:
            (Path(argv[-1])/'nooa-join-public3/solver-workspaces/ls20-test').mkdir(parents=True)
        return result
    sleeps = []
    result = monitor.monitor('owner/q6-probe', harness, run=run, sleep=sleeps.append)
    assert result['download_attempts'] == 1 and sleeps == [60]
    assert sum(c[1:3] == ['kernels','output'] for c in calls) == 1
    assert not any('push' in c or 'submit' in c for c in calls)
    assert any('summarize_long_run.py' in c[1] for c in calls)
    assert any('model-feedback-final-replay-ls20-test.json' in c[-1] for c in calls)
    assert result['status'] == ('HOST_PASS_REQUIRES_BEHAVIORAL_REVIEW' if terminal=='COMPLETE' else 'FAILED')
    assert json.loads((harness/'MONITOR.json').read_text())['kernel_status'] == terminal
    with pytest.raises(FileExistsError):
        monitor.monitor('owner/q6-probe', harness, run=run)


def test_download_failure_is_not_retried(harness):
    calls = []
    def run(argv, timeout):
        argv = [str(x) for x in argv]
        calls.append(argv)
        return {'argv':argv,'returncode':int('output' in argv),
                'stdout':'KernelWorkerStatus.COMPLETE' if 'status' in argv else '', 'stderr':''}
    result = monitor.monitor('owner/q6-probe', harness, run=run)
    assert result['status'] == 'FAILED' and result['download_attempts'] == 1
    assert len(calls) == 2 and (harness/'download.json').exists()


def test_unknown_status_does_not_download(harness):
    def run(argv, timeout):
        return {'argv':[str(x) for x in argv], 'returncode':0,'stdout':'bad response','stderr':''}
    result = monitor.monitor('owner/q6-probe', harness, run=run)
    assert result['status']=='FAILED' and result['download_attempts']==0
