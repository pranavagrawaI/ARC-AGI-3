"""Finite-output and concurrency qualification only; no performance comparison."""
import concurrent.futures
import json
import os
import time
import urllib.request
from pathlib import Path

OUT = Path(os.environ.get('PRUNING_OUTPUT', '/kaggle/working/pruning'))
BASE = os.environ.get('PRUNING_SERVER', 'http://127.0.0.1:8001')


def http(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(BASE + path, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def main():
    target = json.loads((OUT / 'target.json').read_text())
    draft = json.loads((OUT / 'draft.json').read_text())
    if not target['complete'] or target['target_experts'] != 448 or draft['draft_experts'] != 512:
        raise ValueError('Pruning loader receipts failed')
    info = http('/get_server_info')
    (OUT / 'server-info.json').write_text(json.dumps(info, indent=2))
    if info['max_running_requests'] != 18:
        raise ValueError('Requested capacity differs')
    windows = []
    for concurrency in [14, 18]:
        start = time.monotonic()

        def request_one(i):
            payload = {'model': 'flashnext', 'messages': [{'role': 'user', 'content': f'Problem {i}: derive a formula for the sum of the first n odd integers, explain why it is true, and check n=17.'}],
                       'temperature': 0, 'max_tokens': 512, 'stream': False}
            (OUT / f'c{concurrency}-{i}-request.json').write_text(json.dumps(payload))
            began = time.monotonic()
            timing = {'completed': False}
            try:
                response = http('/v1/chat/completions', payload)
                (OUT / f'c{concurrency}-{i}-response.json').write_text(json.dumps(response, allow_nan=False))
                timing['completed'] = True
            except Exception as error:
                timing['error'] = repr(error)
                raise
            finally:
                timing['elapsed_s'] = time.monotonic() - began
                (OUT / f'c{concurrency}-{i}-timing.json').write_text(json.dumps(timing))
            choice = response['choices'][0]
            message = choice['message']
            if not (message.get('content') or message.get('reasoning_content')) or response['usage']['completion_tokens'] <= 0:
                raise ValueError('Empty completion')
            return {'finish_reason': choice['finish_reason'], 'completion_tokens': response['usage']['completion_tokens']}

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            responses = list(pool.map(request_one, range(concurrency)))
        windows.append({'concurrency': concurrency, 'elapsed_s': time.monotonic() - start, 'responses': responses})
    (OUT / 'SMOKE.json').write_text(json.dumps({'passed': True, 'windows': windows,
        'limits': ['Short requests do not establish long-context capacity or quality.', 'No comparative throughput inference.']}, indent=2) + '\n')


if __name__ == '__main__':
    main()
