"""Capture the installed Kaggle CLI request without network or upload.

This package is v12 Duck batch on sb26: letter boards, tool clamp, 32k trim,
open batching, critic off, contingent off, two sequential passes.
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
    assert request.slug == 'pranavagrawai/tycho-nooa-v12-sb26-duck-batch'
    assert request.slug != 'pranavagrawai/tycho-nooa-v12-sb26-critic-off'
    assert request.session_timeout_seconds == launch['session_timeout_seconds'] == 10800
    assert request.machine_shape == 'NvidiaRtxPro6000'
    assert request.is_private and request.enable_gpu and not request.enable_internet
    assert launch['launch_ready'] and not launch['promotion']
    assert launch['factor'] == 'v12-duck-batch'
    assert launch['duck_batch'] == '1'
    assert launch['games'] == ['sb26']
    assert launch['passes'] == 2 and launch['episodes'] == 2
    assert launch['transcript_keep_turns'] == 4
    assert launch['arms'] == ['off']
    assert launch['epistemic_critic'] == '0'

    nb = json.loads(request.text)
    code = '\n'.join(''.join(c['source']) for c in nb['cells'])

    assert code.count("os.environ['ARC_CONTINGENT_ARM']='off'") == 1
    assert "os.environ['ARC_EPISTEMIC_CRITIC']='0'" in code
    assert "os.environ['ARC_DUCK_BATCH']='1'" in code
    assert code.count("os.environ['ARC_DISCOVERY_ARM']='a'") == 1
    assert "os.environ['ARC_SOLVER_ARM']='loop'" in code
    assert "os.environ['ARC_TRANSCRIPT_KEEP_TURNS']='4'" in code
    assert "os.environ['TYCHO_PUBLIC_GAME_PASSES']='2'" in code
    assert "TYCHO_PUBLIC_GAME_IDS'] = 'sb26'" in code
    assert "TYCHO_PUBLIC_GAME_IDS'] = 'sb26,su15'" not in code
    assert 'mindset' not in code
    assert "ARC_DISCOVERY_ARM']='b'" not in code
    assert "subprocess.run(command, check=True)" not in code
    assert 'launcher = subprocess.run(command, check=False)' in code
    assert 'assess = subprocess.run(' in code
    assert 'check=False)' in code

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
    assert "'--epistemic-critic',os.environ['ARC_EPISTEMIC_CRITIC']" in code
    assert 'kaggle competitions submit' not in code
    assert metadata['competition_sources'] == ['arc-prize-2026-arc-agi-3']

    print(json.dumps(dict(
        status='PASS', network_calls=0, kernel=request.slug,
        factor=launch['factor'], games=launch['games'],
        episodes=launch['episodes'], epistemic_critic=launch['epistemic_critic'],
        competition=metadata['competition_sources'],
        actions_per_arm=launch['actions_per_arm'],
        admission_seconds=launch['admission_seconds_per_arm'],
        max_output_tokens=launch['max_output_tokens_per_request'],
        model_context_tokens=launch['model_context_tokens'],
    )))


if __name__ == '__main__':
    main()
