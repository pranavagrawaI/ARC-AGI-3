"""Freeze exact HTTP-200 captured bodies; never infer successful Actor behavior."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path('analysis/deployment-probes/20260906-actor-capture/raw/actor-capture-20260906')
DESTINATION = Path(__file__).with_name('corpus.json')

def sha(data):
    return hashlib.sha256(data).hexdigest()

def build(*, check=False):
    events = [json.loads(line) for line in (ROOT / SOURCE / 'events.jsonl').read_text().splitlines()]
    def capture(index):
        request_path = SOURCE / f'{index:04d}-request.json'
        response_path = SOURCE / f'{index:04d}-response.json'
        request_bytes = (ROOT / request_path).read_bytes()
        response_bytes = (ROOT / response_path).read_bytes()
        recorded = next(e for e in events if e.get('attempt') == index and e['event'] == 'recorded_before_forward')
        completed = next(e for e in events if e.get('attempt') == index and e['event'] == 'upstream_response_complete')
        assert recorded['sha256'] == sha(request_bytes)
        assert completed['sha256'] == sha(response_bytes) and completed['status'] == 200
        request = json.loads(request_bytes)
        response = json.loads(response_bytes)
        returned_tools = [t.get('function', {}).get('name') for t in response['choices'][0].get('message', {}).get('tool_calls', [])]
        offered_tools = {t.get('function', {}).get('name') for t in request.get('tools', [])}
        assert not {'api_key', 'authorization', 'headers'} & request.keys()
        return {'name': f'case-{index}', 'request': request, 'captured_usage': response['usage'],
            'captured_quality': {'finish_reason': response['choices'][0]['finish_reason'],
                'returned_tool_names': returned_tools,
                'unoffered_tool_names': [name for name in returned_tools if name not in offered_tools],
                'http_success_does_not_establish_actor_success': True}, 'provenance': {
            'source_run': '20260906-actor-capture', 'attempt': index,
            'request_path': str(request_path), 'request_sha256': sha(request_bytes),
            'response_path': str(response_path), 'response_sha256': sha(response_bytes),
            'capture_events_path': str(SOURCE / 'events.jsonl'),
            'recorded_before_forward': True, 'response_status': completed['status'],
            'body_preserved_exactly_as_json': True}}
    pairs = []
    for name, a, b in [('normal_short_0_1', 0, 1), ('normal_long_12_13', 12, 13)]:
        pairs.append({'name': name, 'kind': 'normal_successor', 'seed': capture(a), 'successor': capture(b)})
    donor_path = Path('deployment/kaggle/actor-mtp-20260906/corpus.json')
    donor_bytes = (ROOT / donor_path).read_bytes()
    donor = json.loads(donor_bytes)
    corpus = {'schema': 'prefix-cache-corpus-v1', 'pairs': pairs,
        'batch_records': [capture(i) for i in (0, 5, 12, 13)],
        'generation_config': donor['generation_config'],
        'generation_config_provenance': {'path': str(donor_path), 'sha256': sha(donor_bytes)},
        'coverage': {'normal_successor': True, 'summary_boundary': False,
            'missing_summary_boundary': 'The saved q6-summary-boundary run contains responses and input hashes but no complete captured chat request bodies. Do not synthesize or label a normal successor as a summary boundary.'},
        'scope': 'Two actual consecutive HTTP-200 request pairs from a capture run that ended in error, approximately 11k and 21k prompt tokens. Attempts 12 and 13 returned unoffered submit_actions tools. Transport success is not valid Actor behavior; not representative of 100k context or summarization.'}
    expected = (json.dumps(corpus, indent=2) + '\n').encode()
    if check:
        if not DESTINATION.exists() or DESTINATION.read_bytes() != expected:
            raise SystemExit('Frozen corpus is missing or differs from verified sources; --check did not modify it.')
    else:
        DESTINATION.write_bytes(expected)
    return corpus

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify frozen corpus without writing any file.')
    args = parser.parse_args()
    corpus = build(check=args.check)
    print(json.dumps({'path': str(DESTINATION), 'pairs': len(corpus['pairs']), 'sha256': sha(DESTINATION.read_bytes())}))
