"""Explicitly authorized six-call OpenRouter screen; preserves failures, never retries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / 'analysis/local-probes/20260911-critic-trace-screen-v1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--call-cap', type=int, choices=(6, 8), default=6)
    args = parser.parse_args()
    out = args.output.resolve()
    if ROOT not in out.parents:
        raise ValueError('output must be inside the repository')
    paths = sorted(out.glob('*.request.json'))
    assert len(paths) == args.call_cap
    reservations = []
    for p in paths:
        body = json.loads(p.read_text())
        assert body['model'] == 'qwen/qwen3.8-27b' and body['max_tokens'] == 1024
        assert body['provider']['only'] == ['coreweave/fp8'] and body['provider']['allow_fallbacks'] is False
        # Reserve conservatively at one input token per UTF-8 byte plus overhead.
        reservations.append((len(p.read_bytes()) + 4096) * .42 / 1e6 + 1024 * 3 / 1e6)
    assert sum(reservations) < .25
    if not args.execute:
        print(json.dumps({'paid_calls': 0, 'reserved_usd': sum(reservations), 'requests': len(paths)})); return
    card = out / 'EXPERIMENT-CARD.md'
    assert card.exists(), 'Freeze the card before execution'
    key = os.environ.get('OPENROUTER_API_KEY')
    if not key:
        for p in [ROOT / '.env', Path('/home/pranav/projects/arc-agi-3/.env')]:
            if p.exists():
                for line in p.read_text().splitlines():
                    if line.startswith('OPENROUTER_API_KEY='):
                        key = line.split('=', 1)[1].strip().strip('\"\'')
                        break
            if key: break
    assert key, 'No credential available'
    started = {'time': time.time(), 'reserved_usd': sum(reservations),
               'card_sha256': hashlib.sha256(card.read_bytes()).hexdigest(),
               'request_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    with (out / 'STARTED.json').open('x') as f: json.dump(started, f, indent=2)
    result = {'status': 'completed', 'attempted': 0, 'actual_cost_usd': 0.0}
    try:
        for p in paths:
            t = time.monotonic()
            result['attempted'] += 1
            req = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions', p.read_bytes(),
                 {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
            with urllib.request.urlopen(req, timeout=45) as response: raw = response.read()
            p.with_name(p.name.replace('.request.', '.response.')).write_bytes(raw)
            r = json.loads(raw)
            choice = r['choices'][0]
            usage = r.get('usage', {})
            msg = choice['message']
            details = {'seconds': time.monotonic() - t, 'usage': usage, 'provider': r.get('provider'),
                       'model': r.get('model'), 'finish_reason': choice.get('finish_reason')}
            p.with_name(p.name.replace('.request.', '.timing.')).write_text(json.dumps(details, indent=2))
            assert r.get('provider') == 'CoreWeave', 'unexpected provider'
            assert choice.get('finish_reason') == 'stop', 'unfinished completion'
            assert not (msg.get('reasoning') or msg.get('reasoning_content')), 'reasoning present'
            assert usage.get('completion_tokens_details', {}).get('reasoning_tokens', 0) == 0
            assert isinstance(usage.get('cost'), (int, float)), 'missing usage cost'
            result['actual_cost_usd'] += usage['cost']
            assert result['actual_cost_usd'] < .25
    except Exception as e:
        result.update(status='aborted', error=f'{type(e).__name__}: {e}')
        if hasattr(e, 'read'):
                (out / 'error-body.txt').write_bytes(e.read())
    (out / 'EXECUTION.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if result['status'] != 'completed': raise SystemExit(1)


if __name__ == '__main__': main()
