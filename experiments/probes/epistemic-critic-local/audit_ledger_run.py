"""Audit preserved v3 responses and independently count disputed raw-frame cells."""
import argparse
import hashlib
import json
from pathlib import Path

from ledger_screen import BASELINE, ROOT, digest, read_prefix


def audit(out):
    prepared = json.loads((out / 'PREPARED.json').read_text())
    started = json.loads((out / 'STARTED.json').read_text())
    assert hashlib.sha256((out / 'EXPERIMENT-CARD.md').read_bytes()).hexdigest() == started['card_sha256']
    summary = {'mechanical_status': 'pass', 'calls': 0, 'seconds': 0, 'cost_usd': 0,
               'prompt_tokens': 0, 'completion_tokens': 0, 'reasoning_tokens': 0,
               'format_findings': []}
    for entry in prepared['cases']:
        file = entry['request']
        body = json.loads((out / file).read_text())
        assert hashlib.sha256((out / file).read_bytes()).hexdigest() == started['request_sha256'][file]
        assert hashlib.sha256(body['messages'][0]['content'].encode()).hexdigest() == prepared['system_sha256']
        user = json.loads(body['messages'][1]['content'])
        old_body = json.loads((BASELINE / file).read_text())
        old_user = json.loads(old_body['messages'][1]['content'])
        assert user['actor_claim'] == old_user['actor_claim']
        assert user['actor_proposal'] == old_user['actor_proposal']
        assert {k:v for k,v in body.items() if k!='messages'} == {k:v for k,v in old_body.items() if k!='messages'}
        source = read_prefix(ROOT / entry['source'], entry['cutoff'])
        assert digest(source) == entry['prefix_sha256']
        frame = user['evidence_view']['current_frame']
        decoded = [[int(c, 16) for c in row] for row in frame['grid_rows']]
        assert decoded == source[-1]['after']['grid']
        response = json.loads((out / file.replace('.request.', '.response.')).read_text())
        timing = json.loads((out / file.replace('.request.', '.timing.')).read_text())
        choice = response['choices'][0]
        assert response['provider'] == 'CoreWeave' and choice['finish_reason'] == 'stop'
        assert not choice['message'].get('reasoning') and not choice['message'].get('reasoning_content')
        usage = response['usage']
        assert usage['completion_tokens_details']['reasoning_tokens'] == 0
        parsed = json.loads(choice['message']['content'])
        refs = {r['ref'] for r in user['evidence_view']['ledger']} | {frame['ref']}
        assert set(parsed) == {'literal_check', 'status', 'issues', 'decision_effect', 'decision_reason'}
        assert parsed['status'] in {'issues', 'no_issue_found', 'cannot_assess'}
        assert len(parsed['issues']) <= 2
        assert parsed['status'] == 'issues' or not parsed['issues']
        for item in [parsed['literal_check']] + parsed['issues']:
            assert set(item['evidence_refs']) <= refs
        for item in parsed['issues']:
            if len(item['claim'].split()) > 25 or len((item['error']+' '+item['correction']).split()) > 60:
                summary['format_findings'].append({'case': entry['case'], 'issue': 'word limit'})
        summary['calls'] += 1
        summary['seconds'] += timing['seconds']
        summary['cost_usd'] += usage['cost']
        summary['prompt_tokens'] += usage['prompt_tokens']
        summary['completion_tokens'] += usage['completion_tokens']
        if entry['case'] == 'cd_actual':
            red = [(y,x) for y,row in enumerate(decoded) for x,c in enumerate(row) if c==2]
            summary['cd82_red_cells'] = {'count': len(red),
                'row_bounds': [min(y for y,x in red), max(y for y,x in red)],
                'column_bounds': [min(x for y,x in red), max(x for y,x in red)]}
            assert len(red) == 30
        if entry['case'] == 'ls_actual':
            summary['ls20_current_band_region'] = [row[34:39] for row in decoded[40:50]]
    execution = json.loads((out / 'EXECUTION.json').read_text())
    assert execution['status'] == 'completed' and summary['calls'] == 6
    assert abs(execution['actual_cost_usd'] - summary['cost_usd']) < 1e-10
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.output)
    (args.output / 'AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
