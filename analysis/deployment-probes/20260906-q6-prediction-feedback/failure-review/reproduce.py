"""Check the failed summarizer request without network calls or raw mutations."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
raw = root / 'long-run/raw'
ipc = raw / 'nooa-join-public3/solver-workspaces/ls20-w_gx27z4/ipc'
rows = [json.loads(line) for line in (ipc / 'integrity.jsonl').read_text().splitlines()]
failure = next(row for row in reversed(rows) if row['event'] == 'INTEGRITY_FAILED')
assert failure['reason'] == 'visible note event differs at final request'
messages = failure['messages']
assert any('TokenBudgetSummarizer' in m['content'] for m in messages)
content = next(m['content'] for m in messages if 'history_markdown = ' in m['content'])
literal = content.split('history_markdown = ', 1)[1].rsplit('\ntarget_chars = ', 1)[0]
history = ast.literal_eval(literal)
expected = failure['expected']
assert expected not in content
assert expected in history
prior = [r for r in rows if r['event'] == 'NOTE_RENDER' and r.get('expected_event') == expected]
assert prior and all(r['stored'] in expected for r in prior)
deployed = raw / 'nooa-join/chassis'
frozen = root / 'long-run/candidate/chassis'
hashes = {}
for name in ('integrity_guard.py', 'solver_agent.py'):
    assert (deployed / name).read_bytes() == (frozen / name).read_bytes()
    hashes[name] = hashlib.sha256((deployed / name).read_bytes()).hexdigest()
print(json.dumps({
    'status': 'REPRODUCED',
    'request_kind': 'TokenBudgetSummarizer',
    'raw_request_contains_expected_event': False,
    'decoded_history_contains_exact_expected_event': True,
    'expected_event_characters': len(expected),
    'decoded_history_characters': len(history),
    'earlier_successful_render_checks_for_event': len(prior),
    'deployed_matches_frozen': hashes,
    'scope': 'Saved-request representation failure; no model call and no code fix.',
}, indent=2))
