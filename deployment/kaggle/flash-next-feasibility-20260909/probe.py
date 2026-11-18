"""Eight-request probe, using the donor server unchanged; never executes tools."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
from urllib.request import Request, urlopen

ROOT = Path('/kaggle/working')
MODEL = 'Qwen/Qwen3.8-Flash-Next-NVFP4'
BASE = 'http://127.0.0.1:1234'

def save(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2) + '\n')

def deadline(signum, frame):
    raise TimeoutError('Frozen probe deadline reached')

def metrics(name):
    with urlopen(BASE + '/metrics', timeout=10) as response:
        (ROOT / name).write_bytes(response.read())

def sample_resources(stop):
    with (ROOT / 'resources.jsonl').open('a', buffering=1) as output:
        while not stop.is_set():
            row = {'epoch': time.time(), 'meminfo': Path('/proc/meminfo').read_text()}
            try:
                row['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader'], text=True, timeout=5)
            except Exception as exc:
                row['error'] = repr(exc)
            output.write(json.dumps(row) + '\n')
            stop.wait(5)

def generate(name, payload):
    payload = copy.deepcopy(payload)
    # Preserve Actor messages/tools, normalize provider-specific transport only.
    payload.pop('reasoning', None)
    payload.update(model=MODEL, stream=True, stream_options={'include_usage': True}, max_tokens=1024, temperature=1.0, top_p=0.95, top_k=20)
    payload['chat_template_kwargs'] = {'enable_thinking': False}
    data = json.dumps(payload).encode()
    (ROOT / f'{name}.request.json').write_bytes(data)
    started = time.monotonic()
    first = None
    content, reasoning, calls = '', '', {}
    usage, finish, done = None, None, False
    try:
        with urlopen(Request(BASE + '/v1/chat/completions', data=data, headers={'Content-Type': 'application/json'}), timeout=120) as response, (ROOT / f'{name}.sse').open('wb') as raw:
            for line in response:
                raw.write(line)
                raw.flush()
                if not line.startswith(b'data: '):
                    continue
                body = line[6:].strip()
                if body == b'[DONE]':
                    done = True
                    break
                event = json.loads(body)
                if event.get('error'):
                    raise RuntimeError(event['error'])
                usage = event.get('usage') or usage
                for choice in event.get('choices', []):
                    delta = choice.get('delta', {})
                    if first is None and any(delta.get(k) for k in ['content', 'reasoning_content', 'reasoning', 'tool_calls']):
                        first = time.monotonic() - started
                    content += delta.get('content') or ''
                    reasoning += delta.get('reasoning_content') or delta.get('reasoning') or ''
                    for call in delta.get('tool_calls') or []:
                        acc = calls.setdefault(call['index'], {'name': '', 'arguments': ''})
                        for key in acc:
                            acc[key] += (call.get('function') or {}).get(key) or ''
                    finish = choice.get('finish_reason') or finish
        elapsed = time.monotonic() - started
        row = {'name': name, 'seconds': elapsed, 'ttft_seconds': first, 'usage': usage, 'finish_reason': finish, 'done': done, 'content': content, 'reasoning': reasoning, 'calls': calls, 'request_sha256': hashlib.sha256(data).hexdigest()}
        save(f'{name}.response.json', row)
        assert done and usage and finish, 'Missing terminal stream or usage'
        assert content or calls, 'Empty answer'
        assert not reasoning.strip(), 'Reasoning-off contract violated'
        offered = {t['function']['name']: t['function']['parameters'] for t in payload.get('tools', [])}
        for call in calls.values():
            assert call['name'] in offered, f'Unoffered tool: {call}'
            args = json.loads(call['arguments'])
            import jsonschema
            jsonschema.validate(args, offered[call['name']])
        if name == 'text':
            assert content.strip() == '4'
        elif name == 'tool':
            assert len(calls) == 1 and finish == 'tool_calls'
            call = next(iter(calls.values()))
            assert call['name'] == 'submit_number' and json.loads(call['arguments']) == {'value': 4}
        elif name == 'json':
            assert json.loads(content) == {'value': 4}
        elif name.startswith('actor-'):
            # tool_choice is 'auto' on every retained case, so prose is a permitted
            # response and a finding about Actor compatibility, not a stop.
            row['actor_returned_tool'] = bool(calls) and finish == 'tool_calls'
        row['effective_output_tokens_per_second'] = usage['completion_tokens'] / elapsed
        with (ROOT / 'results.jsonl').open('a') as output:
            output.write(json.dumps(row) + '\n')
        print(json.dumps({'name': name, 'seconds': elapsed, 'ttft': first, 'usage': usage}), flush=True)
    except BaseException:
        save(f'{name}.failure.json', {'traceback': traceback.format_exc()})
        raise

def main():
    ROOT.mkdir(exist_ok=True)
    bundle = next(Path('/kaggle/input').rglob('taaf-kaggle-bundle.json')).parent
    setup_path = bundle / 'serving_setup.py'
    assert hashlib.sha256(setup_path.read_bytes()).digest() == hashlib.sha256((ROOT / 'donor-serving_setup.py').read_bytes()).digest()
    runtime_ref = 'keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1'
    runtime = next(p for p in [Path('/kaggle/input') / runtime_ref.split('/')[1], Path('/kaggle/input/datasets') / runtime_ref] if p.exists())
    env = {'TAAF_KAGGLE_BUNDLE_DIR': str(bundle), 'TAAF_KAGGLE_WORKING_DIR': str(ROOT), 'TAAF_KAGGLE_SETUP_ENV': str(ROOT / 'taaf_setup_env.json'), 'TAAF_KAGGLE_INPUT_PATHS': json.dumps({'keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1': str(bundle), runtime_ref: str(runtime)}), 'TAAF_KAGGLE_FAST_START': '1', 'TAAF_VLLM_ENABLE_PREFIX_CACHING': '0', 'TAAF_VLLM_KV_CACHE_DTYPE': 'auto', 'TAAF_VLLM_KV_CACHE_MEMORY_BYTES': '5368709120', 'TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE': '32', 'TAAF_VLLM_MAX_NUM_BATCHED_TOKENS': '8192', 'TAAF_VLLM_MAX_NUM_SEQS': '8', 'TAAF_VLLM_MTP_TOKENS': '3', 'TAAF_VLLM_OMP_THREADS': '1'}
    os.environ.update(env)
    os.environ['LIBRARY_PATH'] = '/usr/local/nvidia/lib64' + os.pathsep + os.environ.get('LIBRARY_PATH', '')
    save('taaf_setup_env.json', {})
    save('probe-environment.json', env)
    spec = importlib.util.spec_from_file_location('donor_setup', setup_path)
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    stop = threading.Event()
    monitor = threading.Thread(target=sample_resources, args=(stop,), daemon=True)
    monitor.start()
    signal.signal(signal.SIGALRM, deadline)
    end = time.monotonic() + 2500
    signal.alarm(2500)
    try:
        with (ROOT / 'setup.log').open('w') as log:
            subprocess.run([sys.executable, str(setup_path)], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1830)
        metrics('metrics-before.prom')
        records = json.loads((ROOT / 'actor-cases.json').read_text())
        simple = lambda text: {'messages': [{'role': 'user', 'content': text}]}
        tool = simple('Call submit_number with value 4 now. Do not answer in prose.')
        tool.update(tools=[{'type': 'function', 'function': {'name': 'submit_number', 'parameters': {'type': 'object', 'properties': {'value': {'type': 'integer'}}, 'required': ['value'], 'additionalProperties': False}}}], tool_choice='required')
        structured = simple('Return a JSON object whose only key is named "value" and whose value is the integer 4.')
        structured['response_format'] = {'type': 'json_object'}
        cases = [('text', simple('Answer with only: 4')), ('tool', tool), ('json', structured)] + [('actor-' + record['name'], record['request']) for record in records]
        assert len(cases) == 8
        for name, payload in cases:
            signal.alarm(max(1, min(120, int(end - time.monotonic()))))
            generate(name, payload)
            signal.alarm(max(1, int(end - time.monotonic())))
        save('verdict.json', {'feasibility': 'pass', 'promotion': False, 'long_context_tested': False, 'prefix_cache_enabled': False})
    except BaseException:
        save('failure.json', {'traceback': traceback.format_exc(), 'promotion': False})
        raise
    finally:
        signal.alarm(0)
        try:
            metrics('metrics-after.prom')
        except Exception as exc:
            save('metrics-failure.json', {'error': repr(exc)})
        cleanup = setup.stop_owned_server()
        save('cleanup.json', cleanup)
        stop.set()
        monitor.join(timeout=6)
        assert cleanup.get('stopped'), cleanup

if __name__ == '__main__':
    main()
