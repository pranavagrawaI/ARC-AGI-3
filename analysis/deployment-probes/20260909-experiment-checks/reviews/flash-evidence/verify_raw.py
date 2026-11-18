"""Independent offline raw-stream reconciliation; prints findings, writes nothing."""
import ast
import hashlib
import json
from pathlib import Path

import jsonschema

BASE = Path(__file__).resolve().parents[3] / '20260909-flash-next-feasibility'
for version in ('raw', 'raw-v3'):
    root = BASE / version
    rows = [json.loads(line) for line in (root / 'results.jsonl').read_text().splitlines()]
    print(version, 'results_rows', len(rows))
    for path in sorted(root.glob('*.request.json')):
        name = path.name.removesuffix('.request.json')
        request = json.loads(path.read_text())
        response = json.loads((root / f'{name}.response.json').read_text())
        content, reasoning, calls, finishes, usages, ids = '', '', {}, [], [], set()
        done = 0
        for line in (root / f'{name}.sse').read_text().splitlines():
            if not line.startswith('data: '):
                continue
            if line[6:] == '[DONE]':
                done += 1
                continue
            event = json.loads(line[6:])
            assert not event.get('error'), event
            ids.add(event['id'])
            if event.get('usage'):
                usages.append(event['usage'])
            for choice in event.get('choices', []):
                assert choice['index'] == 0
                delta = choice['delta']
                content += delta.get('content') or ''
                reasoning += delta.get('reasoning_content') or delta.get('reasoning') or ''
                if choice.get('finish_reason'):
                    finishes.append(choice['finish_reason'])
                for call in delta.get('tool_calls') or []:
                    acc = calls.setdefault(str(call['index']), {'name': '', 'arguments': ''})
                    for key in acc:
                        acc[key] += (call.get('function') or {}).get(key) or ''
        assert done == 1 and len(ids) == 1 and len(finishes) == 1 and len(usages) == 1
        assert response['done'] and response['finish_reason'] == finishes[0]
        assert response['content'] == content and response['reasoning'] == reasoning
        assert response['calls'] == calls and response['usage'] == usages[0]
        assert response['request_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert response['seconds'] < 120 and request['max_tokens'] == 1024
        assert 0 < usages[0]['completion_tokens'] <= 1024
        assert usages[0]['total_tokens'] == usages[0]['prompt_tokens'] + usages[0]['completion_tokens']
        offered = {t['function']['name']: t['function']['parameters'] for t in request.get('tools', [])}
        for call in calls.values():
            jsonschema.validate(json.loads(call['arguments']), offered[call['name']])
        exact = None
        if name == 'text':
            exact = content.strip() == '4'
        elif name == 'json':
            exact = json.loads(content) == {'value': 4}
        elif name == 'tool':
            exact = len(calls) == 1 and calls['0']['name'] == 'submit_number' and json.loads(calls['0']['arguments']) == {'value': 4}
        matching = [r for r in rows if r['name'] == name]
        assert len(matching) <= 1
        if matching:
            for key, value in response.items():
                assert matching[0][key] == value, (name, key)
        print(name, 'seconds', round(response['seconds'], 6), 'exact', exact, 'tools', [c['name'] for c in calls.values()], 'in_results', bool(matching))
    provenance = json.loads((root / 'vllm-setup-provenance.json').read_text())
    identity = json.loads((root / 'vllm-server-identity.json').read_text())
    constants = {}
    for node in ast.parse((root / 'donor-serving_setup.py').read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    for field, pin in [('manifest_sha256', 'MODEL_MANIFEST_SHA256'), ('config_sha256', 'MODEL_CONFIG_SHA256'), ('quant_config_sha256', 'MODEL_QUANT_CONFIG_SHA256'), ('file_count', 'MODEL_FILE_COUNT'), ('total_bytes', 'MODEL_TOTAL_BYTES')]:
        assert provenance['model'][field] == constants[pin], field
    for field, pin in [('index_digest', 'VLLM_INDEX_DIGEST'), ('amd64_manifest_digest', 'VLLM_AMD64_MANIFEST_DIGEST'), ('manifest_sha256', 'VLLM_RUNTIME_MANIFEST_SHA256')]:
        assert provenance['runtime'][field] == constants[pin], field
    assert provenance['ple_patch']['patched_sha256'] == provenance['ple_patch']['patch_identity']['patched_target_sha256']
    assert 'NVIDIA RTX PRO 6000 Blackwell Server Edition' in provenance['inventory']['gpu_rows'][0]
    assert provenance['server_identity']['argv'] == identity['argv']
    assert provenance['model_hf_repo'] == 'RadixArk/Qwen3.8-Flash-Next-NVFP4'
    assert provenance['model_hf_revision'] == '7b719225242aacd3dbd3f9407468c2ee9a9d2594'
    assert provenance['vllm_version'] == '0.1.dev20073+g8e685d198'
    assert provenance['setup_elapsed_seconds'] < 1800
    log = json.loads((root / 'flash-next-feasibility-20260909.log').read_text())
    print('setup_seconds', provenance['setup_elapsed_seconds'], 'last_platform_log_seconds', max(r['time'] for r in log))
