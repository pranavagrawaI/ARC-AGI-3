"""Verify report links, cited file identities, and selected preserved run counts.

This is an offline report check, not a gameplay assessor. Output is written only
to a new file named on the command line; existing evidence cannot be overwritten.
"""
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = Path(sys.argv[1]).resolve()

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def identity(path):
    return {'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

links = []
missing = []
cited = {}
for document in [HERE / 'REPORT.md', *sorted((HERE / 'appendices').glob('*.md'))]:
    for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', document.read_text()):
        target = target.strip('<>')
        if target.startswith(('http:', 'https:', '#', 'mailto:')):
            continue
        path_text = unquote(target.split('#', 1)[0])
        path_text = re.sub(r':\d+(?:-\d+)?$', '', path_text)
        path = (document.parent / path_text).resolve()
        if path == OUTPUT:
            links.append({'document': str(document.relative_to(ROOT)), 'target': target,
                          'generated_by_this_check': True})
            continue
        record = {'document': str(document.relative_to(ROOT)), 'target': target, 'exists': path.exists()}
        links.append(record)
        if not path.exists():
            missing.append(record)
        elif path.is_file() and path.is_relative_to(ROOT):
            cited[str(path.relative_to(ROOT))] = identity(path)

snap = {x['path']: x for x in json.loads((HERE / 'sources/source-documents.json').read_text())}
updates_path = HERE / 'sources/reviewed-source-updates.json'
updates = json.loads(updates_path.read_text())['files'] if updates_path.exists() else []
for update in updates:
    assert hashlib.sha256(update['reviewed_text'].encode()).hexdigest() == update['sha256']
    snap[update['path']] = update
changed = [p for p, r in cited.items() if p in snap and snap[p]['sha256'] != r['sha256']]
base = ROOT / 'analysis/deployment-probes'
selected = {}
for label, path in {
    'q6-summary-boundary': base / '20260906-q6-summary-boundary/raw/nooa-join-public3',
    'controller-off': base / '20260909-contingent-arm-qualification/v3-output/contingent-off/nooa-join-public3',
    'controller-on': base / '20260909-contingent-arm-qualification/v3-output/contingent-on/nooa-join-public3',
}.items():
    result_path = path / 'official-launcher-result.json'
    result = json.loads(result_path.read_text())
    game = result['public_progress']['games']['ls20']
    calls_path, = (path / 'solver-workspaces').glob('*/ipc/model_calls.jsonl')
    calls = rows(calls_path)
    selected[label] = {
        'sources': [identity(result_path), identity(calls_path)],
        'launcher_returncode': result['launcher_returncode'],
        'actions': game['recorded_action_results'],
        'levels': game['final_levels_completed'],
        'launcher_grid_hash_changes': game['grid_hash_changes'],
        'model_calls': len(calls),
        'summarizer_calls': sum(bool(c.get('summarizer')) for c in calls),
        'actor_calls': sum(not c.get('summarizer') for c in calls),
        'reasoning_tokens': sum(c.get('reasoning_tokens', 0) or 0 for c in calls),
    }
assert (selected['q6-summary-boundary']['actions'], selected['q6-summary-boundary']['levels'],
        selected['q6-summary-boundary']['actor_calls'], selected['q6-summary-boundary']['summarizer_calls']) == (100, 0, 153, 2)
assert (selected['controller-off']['actions'], selected['controller-off']['levels']) == (20, 0)
assert (selected['controller-on']['actions'], selected['controller-on']['levels']) == (20, 1)
old = ROOT / 'experiments/hypothesis/0003-temporal-game-judgment/runs/0003-live-route-ceiling-e4b-v1-20260807T213148Z/result.json'
ceiling = json.loads(old.read_text())
assert (ceiling['environment_actions'], ceiling['levels_completed'], ceiling['status']) == (13, 1, 'PASS')
selected['supplied-route-ceiling'] = {'source': identity(old), 'result': ceiling,
    'limit': 'Supplied certified route, not autonomous route discovery'}
report = {'link_count': len(links), 'missing_links': missing,
          'cited_files_changed_since_inventory': changed,
          'accepted_source_updates': [{'path': u['path'], 'sha256': u['sha256']} for u in updates],
          'cited_file_identities': list(cited.values()), 'selected_raw_checks': selected,
          'limits': 'No paid calls or reruns; selected recorded outcomes only. Link/hash checks do not certify prose interpretation. Historical test counts are reported, not rerun.'}
with Path(sys.argv[1]).open('x') as stream:
    json.dump(report, stream, indent=2)
    stream.write('\n')
print(json.dumps({'links': len(links), 'missing': missing, 'changed_sources': changed, 'selected_outcomes': {k: {key: v[key] for key in ['actions', 'levels', 'model_calls'] if key in v} for k,v in selected.items()}}))
raise SystemExit(bool(missing or changed))
