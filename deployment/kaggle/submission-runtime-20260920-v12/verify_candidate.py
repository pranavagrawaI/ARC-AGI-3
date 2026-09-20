"""Capture the installed Kaggle CLI request without network or upload.

Every assertion below describes THIS package: v12 transcript-eviction on
sb26/su15, sequential episodes, one pinned arm. It is the last gate before a
push, so it must fail whenever the notebook, the games, the budgets or the
launch identity drift.
"""
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from kaggle.api.kaggle_api_extended import KaggleApi
from slugify import slugify

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

    assert request.slug == launch['kernel'] == metadata['id']
    assert slugify(metadata['title']) == metadata['id'].split('/')[1]
    assert request.slug == 'pranavagrawai/tycho-nooa-v12-sb26-su15'
    assert request.slug != 'pranavagrawai/contingent-controller-20260909'
    assert request.session_timeout_seconds == launch['session_timeout_seconds'] == 10800
    assert request.machine_shape == 'NvidiaRtxPro6000'
    assert request.is_private and request.enable_gpu and not request.enable_internet
    assert launch['launch_ready'] and not launch['promotion']
    assert launch['factor'] == 'v12-transcript-eviction'
    assert launch['games'] == ['sb26', 'su15']
    assert launch['passes'] == 2 and launch['episodes'] == 4
    assert launch['transcript_keep_turns'] == 4
    assert launch['arms'] == ['on']

    nb = json.loads(request.text)
    code = '\n'.join(''.join(c['source']) for c in nb['cells'])

    assert code.count("os.environ['ARC_CONTINGENT_ARM']='on'") == 1
    assert code.count("os.environ['ARC_DISCOVERY_ARM']='a'") == 1
    assert "os.environ['ARC_SOLVER_ARM']='loop'" in code
    assert "os.environ['ARC_TRANSCRIPT_KEEP_TURNS']='4'" in code
    assert "os.environ['TYCHO_PUBLIC_GAME_PASSES']='2'" in code
    assert "TYCHO_PUBLIC_GAME_IDS'] = 'sb26,su15'" in code
    assert 'mindset' not in code
    assert "ARC_DISCOVERY_ARM']='b'" not in code

    assert code.count('"enable_thinking":false') == 1
    assert code.count('"reasoning":{"enabled":false}') == 1
    assert '"enable_thinking":true' not in code
    assert '"reasoning":{"enabled":true}' not in code

    assert f"'--launcher-timeout-s', '{launch['launcher_timeout_seconds_per_arm']}'" in code
    assert f"TYCHO_GAMEPLAY_ADMISSION_SECONDS'] = '{launch['admission_seconds_per_arm']}'" in code
    assert f"TYCHO_KAGGLE_MAX_ACTIONS'] = '{launch['actions_per_arm']}'" in code
    assert f"ARC_LLM_MAX_TOKENS'] = '{launch['max_output_tokens_per_request']}'" in code
    assert f"ARC_LLM_CONTEXT_WINDOW'] = '{launch['model_context_tokens']}'" in code
    assert "ARC_LLM_RETRIES'] = '0'" in code
    assert "'--episodes','4'" in code
    assert 'kaggle competitions submit' not in code
    assert metadata['competition_sources'] == ['arc-prize-2026-arc-agi-3']

    print(json.dumps(dict(
        status='PASS', network_calls=0, kernel=request.slug,
        factor=launch['factor'], games=launch['games'],
        episodes=launch['episodes'],
        competition=metadata['competition_sources'],
        actions_per_arm=launch['actions_per_arm'],
        admission_seconds=launch['admission_seconds_per_arm'],
        max_output_tokens=launch['max_output_tokens_per_request'],
        model_context_tokens=launch['model_context_tokens'],
    )))


if __name__ == '__main__':
    main()
