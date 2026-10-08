"""Recompute S10 instrumentation qualification from raw receipts; no launch path."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'mtp-depth-20261006'))
from probe import RUN_ID, PLAN, accounting, capacity, parse_metrics, plateau, verify_trace


def verify_metrics(directory, events):
    rows = []
    retractions = []
    seen_retractions = False
    for event in events:
        if event['kind'] not in ('metrics_before', 'metrics', 'metrics_after'):
            continue
        path = directory / event['raw_metrics']
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != event['raw_sha256']:
            raise ValueError('raw metric hash mismatch: ' + str(path))
        computed = parse_metrics(raw)
        projected = {k:v for k,v in event.items() if k not in ('t','wall','kind','pending','raw_metrics','raw_sha256')}
        if computed != projected:
            raise ValueError('raw metric projection mismatch: ' + str(path))
        if 'retractions' in computed:
            seen_retractions = True
            retractions.append(computed['retractions'])
        elif seen_retractions:
            raise ValueError('retraction counter disappeared')
        if event['kind'] == 'metrics':
            rows.append(event)
    if any(not math.isfinite(x) or x != 0 for x in retractions):
        raise ValueError('retraction counter nonzero/nonfinite in isolated qualification')
    return rows


def verify_block(root, label, n):
    directory = root / label
    row = json.loads((directory / 'summary.json').read_text())
    events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
    measured = plateau(verify_metrics(directory, events), n)
    if row.get('status') != 'complete' or row.get('n') != n or row.get('plateau') != measured:
        raise ValueError('incomplete or inconsistent block: ' + label)
    request_paths = sorted(directory.glob('*-request.json'))
    if len(request_paths) != n or len(row.get('responses', [])) != n:
        raise ValueError('request/response count mismatch: ' + label)
    total = 0
    for path in request_paths:
        req = json.loads(path.read_text())
        resp = json.loads(path.with_name(path.name.replace('-request', '-response')).read_text())
        usage = resp['usage']
        expected = 32 if label == 'WARMUP' else 4096
        if (req.get('max_tokens') != expected or usage['completion_tokens'] != expected or
            req.get('ignore_eos') is not True or req.get('seed') != 20261008 or
            req.get('temperature') != .7 or req.get('top_p') != .95 or req.get('top_k') != 20):
            raise ValueError('wire parameter or length mismatch: ' + label)
        cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0)
        if label != 'WARMUP' and cached / usage['prompt_tokens'] < .9:
            raise ValueError('cache gate failed: ' + label)
        total += usage['completion_tokens']
    if label != 'WARMUP' and (not measured['valid'] or row.get('retractions') != 0):
        raise ValueError('decode qualification failed: ' + label)
    before = [e for e in events if e['kind'] == 'metrics_before']
    after = [e for e in events if e['kind'] == 'metrics_after']
    if len(before) != 1 or len(after) != 1:
        raise ValueError('missing/duplicated accounting endpoints: ' + label)
    reconciled = accounting(before[0], [e for e in events if e['kind'] == 'metrics'], after[0], total, n)
    if row.get('accounting') != reconciled or row.get('retractions') != reconciled['retractions']:
        raise ValueError('raw accounting differs from summary: ' + label)
    receipt = {'label':label, 'n':n, 'plateau':measured, 'generated_tokens':total}
    receipt['accounting'] = reconciled
    if label.startswith('PROFILE'):
        traces = verify_trace(label, root)
        if row.get('profile_attempted') is not True or row.get('traces') != traces:
            raise ValueError('profile receipt mismatch: ' + label)
        if sum(e['kind'] == 'profile_request' for e in events) != 1:
            raise ValueError('profile must be requested exactly once: ' + label)
        responses = [e for e in events if e['kind'] == 'profile_response']
        if len(responses) != 1 or responses[0].get('raw', '').strip() != 'Start profiling.':
            raise ValueError('profile HTTP response missing/invalid: ' + label)
        receipt['traces'] = traces
    return receipt


def audit(raw):
    root = raw / 'headroom'
    result = {'run_id':RUN_ID, 'qualified':False, 'blocks':[],
              'comparison':'S10 instrumentation qualification only; 16-request advantage remains untested.'}
    try:
        hashes = json.loads((root / 'output-hashes.json').read_text())
        if not hashes:
            raise ValueError('empty output manifest')
        for name, expected in hashes.items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
                raise ValueError('output hash mismatch: ' + name)
        result['verified_hashes'] = len(hashes)
        hardware = json.loads((root / 'hardware.json').read_text())
        if hardware.get('valid') is not True:
            raise ValueError('hardware guard failed')
        state = json.loads((root / 'state.json').read_text())
        if state.get('run_id') != RUN_ID or state.get('stop') or [b['label'] for b in state['blocks']] != [x[0] for x in PLAN]:
            raise ValueError('incomplete run state')
        result['capacity'] = capacity(json.loads((root / 'server-info-before.json').read_text()),
                                      json.loads((root / 'cohort.json').read_text()))
        expected_native = json.loads((HERE / 'native-contract.json').read_text())['files']
        native_receipt = json.loads((root / 'runtime-source/identity.json').read_text())['files']
        for name, expected in expected_native.items():
            if native_receipt.get(name) != expected or hashlib.sha256((root / 'runtime-source' / name).read_bytes()).hexdigest() != expected:
                raise ValueError('native source mismatch: ' + name)
        for label,n in [('WARMUP',10), *PLAN]:
            result['blocks'].append(verify_block(root,label,n))
        if any(result['blocks'][i]['plateau'].get('CV') is None for i in (3,4)):
            raise ValueError('insufficient unprofiled variation windows')
        result['qualified'] = True
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, ZeroDivisionError, EOFError) as error:
        result['error'] = repr(error)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('raw',type=Path); parser.add_argument('--output',type=Path)
    args = parser.parse_args(); result = audit(args.raw)
    if args.output: args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['qualified'] else 1)
