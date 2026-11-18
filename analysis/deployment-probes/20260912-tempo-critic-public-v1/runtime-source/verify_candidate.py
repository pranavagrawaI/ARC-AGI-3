"""Capture the installed Kaggle CLI request without network or upload.

Every assertion below describes THIS package: the two-arm contingent-controller
contrast. It is the last gate before a push, so it must fail whenever the
notebook, the isolated factor, the budgets or the launch identity drift.
"""
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from kaggle.api.kaggle_api_extended import KaggleApi

HERE = Path(__file__).resolve().parent


def main():
    launch = json.loads((HERE / 'LAUNCH.json').read_text())
    metadata = json.loads((HERE / 'kernel-metadata.json').read_text())
    captured = []

    @contextmanager
    def client(_):
        yield SimpleNamespace(kernels=SimpleNamespace(kernels_api_client=SimpleNamespace(
            save_kernel=lambda r: captured.append(r))))

    with patch.object(KaggleApi, 'build_kaggle_client', client):
        KaggleApi().kernels_push(str(HERE), timeout=launch['session_timeout_seconds'],
                                 acc='NvidiaRtxPro6000')
    assert len(captured) == 1
    request = captured[0]

    # Launch identity: the request must be the one LAUNCH.json authorizes, and
    # the slug must not be a donor's, whose runs are already written up.
    assert request.slug == launch['kernel'] == metadata['id']
    assert request.slug != 'pranavagrawai/structural-self-test-deep-20260909'
    assert request.session_timeout_seconds == launch['session_timeout_seconds'] == 10800
    assert request.machine_shape == 'NvidiaRtxPro6000'
    assert request.is_private and request.enable_gpu and not request.enable_internet
    assert launch['launch_ready'] and not launch['promotion']
    assert launch['factor'] == 'ARC_CONTINGENT_ARM' and launch['arms'] == ['off', 'on']

    nb = json.loads(request.text)
    code = '\n'.join(''.join(c['source']) for c in nb['cells'])

    # The isolated factor: exactly one cell per arm, both discovery arm A.
    for arm in launch['arms']:
        assert code.count(f"os.environ['ARC_CONTINGENT_ARM']={arm!r}") == 1
    assert code.count("os.environ['ARC_DISCOVERY_ARM']='a'") == len(launch['arms'])
    assert "os.environ['ARC_SOLVER_ARM']='loop'" in code
    assert 'mindset' not in code
    # The inherited discovery arms are gone; a leftover would confound the factor.
    assert not any(f"ARC_DISCOVERY_ARM']={a!r}" in code for a in ('b', 'c', 'd'))

    # Reasoning stays off in both arms: the v2 blocker was a forced-on default.
    assert code.count('"enable_thinking":false') == len(launch['arms'])
    assert code.count('"reasoning":{"enabled":false}') == len(launch['arms'])
    assert '"enable_thinking":true' not in code
    assert '"reasoning":{"enabled":true}' not in code

    # Matched budgets, and no submission or retry.
    per_arm = len(launch['arms'])
    assert code.count(f"'--launcher-timeout-s', '{launch['launcher_timeout_seconds_per_arm']}'") == per_arm
    assert code.count(f"TYCHO_GAMEPLAY_ADMISSION_SECONDS'] = '{launch['admission_seconds_per_arm']}'") == per_arm
    assert code.count(f"TYCHO_KAGGLE_MAX_ACTIONS'] = '{launch['actions_per_arm']}'") == per_arm
    assert code.count(f"ARC_LLM_MAX_TOKENS'] = '{launch['max_output_tokens_per_request']}'") == per_arm
    assert code.count(f"ARC_LLM_CONTEXT_WINDOW'] = '{launch['model_context_tokens']}'") == per_arm
    assert code.count("ARC_LLM_RETRIES'] = '0'") == per_arm
    assert 'kaggle competitions submit' not in code
    assert metadata['competition_sources'] == ['arc-prize-2026-arc-agi-3']

    print(json.dumps(dict(status='PASS', network_calls=0, kernel=request.slug,
                          factor=launch['factor'], arms=launch['arms'],
                          competition=metadata['competition_sources'],
                          actions_per_arm=launch['actions_per_arm'],
                          max_output_tokens=launch['max_output_tokens_per_request'],
                          model_context_tokens=launch['model_context_tokens'])))


if __name__ == '__main__':
    main()
