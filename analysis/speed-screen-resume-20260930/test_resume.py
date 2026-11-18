"""Offline checks for validation builds and interrupted-launch containment."""
import ast
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import resume as worker


def test_validation_keeps_gameplay_weights_and_horizon():
    base = json.loads((worker.FLASH / 'prune/p1' / worker.NB).read_text())
    overrides = {'TAAF_VLLM_MTP_INDEX_SHARE_FOR_ITERATION': '1', 'TAAF_VLLM_ENABLE_PREFIX_CACHING': '1',
                 'TAAF_VLLM_MTP_TOKENS': '4', 'TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE': '60'}
    with tempfile.TemporaryDirectory() as tmp, patch.object(worker, 'HERE', Path(tmp)):
        worker.build_validation(overrides)
        for arm in ('pplus1', 'pplus2'):
            folder = Path(tmp) / arm
            nb = json.loads((folder / worker.NB).read_text())
            assert all(c == base['cells'][i] for i, c in enumerate(nb['cells']) if i != 3)
            nodes = ast.parse(''.join(nb['cells'][3]['source'])).body
            profile = None
            update = None
            for node in nodes:
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'PUBLIC25_VLLM_PROFILE_ENV':
                    profile = ast.literal_eval(node.value)
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    call = node.value
                    if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) and call.func.value.id == 'PUBLIC25_VLLM_PROFILE_ENV':
                        update = ast.literal_eval(call.args[0])
            assert update == overrides
            profile.update(update)
            assert profile['TAAF_VLLM_KV_CACHE_MEMORY_BYTES'] == '12884901888'
            assert profile['TAAF_VLLM_MAX_NUM_BATCHED_TOKENS'] == '8192'
            assert profile['TAAF_VLLM_MAX_NUM_SEQS'] == '12'
            assert (folder / 'EXPERIMENT-CARD.md').exists()
            assert json.loads((folder / 'kernel-metadata.json').read_text())['machine_shape'] == 'NvidiaRtxPro6000'


def test_uncertain_first_push_never_repeats():
    with tempfile.TemporaryDirectory() as tmp, patch.object(worker, 'HERE', Path(tmp)):
        worker.build_validation({'TAAF_VLLM_MTP_INDEX_SHARE_FOR_ITERATION': '1'})
        calls = []

        def failed(args, log, timeout):
            calls.append(args)
            assert args[-2:] == ['--timeout', '10800']
            assert (Path(tmp) / 'pplus1/launch-attempt.json').exists()
            raise TimeoutError('receipt unknown')

        with patch.object(worker, 'frozen_check'), patch.object(worker, 'command', failed):
            for attempt in (0, 1):
                try:
                    worker.launch_pair()
                except (TimeoutError, RuntimeError):
                    pass
                else:
                    raise AssertionError('Uncertain launch was allowed')
        assert len(calls) == 1
        assert not (Path(tmp) / 'pplus2/launch-attempt.json').exists()


if __name__ == '__main__':
    test_validation_keeps_gameplay_weights_and_horizon()
    print('PASS unchanged gameplay, weights, horizon and resource configuration')
    test_uncertain_first_push_never_repeats()
    print('PASS uncertain launch is contained and never repeated')
