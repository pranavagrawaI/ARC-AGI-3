"""Prepare prefix-limited host measurements for the saved-decision critic screen."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASELINE = ROOT / 'analysis/local-probes/20260911-critic-trace-screen-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_prefix(path, cutoff):
    if cutoff < 1:
        raise ValueError('cutoff must be positive')
    records = []
    with path.open() as stream:
        for expected in range(1, cutoff + 1):
            row = json.loads(next(stream))
            if row['step'] != expected:
                raise ValueError('non-contiguous prefix')
            records.append(row)
    return records


def measure(records):
    ledger = []
    for t in records:
        a, b = t['before']['grid'], t['after']['grid']
        same = len(a) == len(b) and all(len(x) == len(y) for x, y in zip(a, b))
        changes = [[y, x, u, v] for y, (r, s) in enumerate(zip(a, b))
                   for x, (u, v) in enumerate(zip(r, s)) if u != v] if same else None
        ledger.append({'ref': f'transition:{t["step"]}', 'action': t['action'],
                       'origin': 'host comparison of raw grids',
                       'comparable_shapes': same,
                       'changed_cells': len(changes) if changes is not None else None,
                       'changed_rows': sorted({c[0] for c in changes}) if changes is not None else None,
                       'changes_y_x_before_after': changes,
                       'levels_before': t['before']['levels_completed'],
                       'levels_after': t['after']['levels_completed']})
    grid = records[-1]['after']['grid']
    # Lossless hex rows; no pipe prefix or column offset. No semantic object labels.
    if not all(isinstance(v, int) and 0 <= v <= 15 for row in grid for v in row):
        raise ValueError('grid cannot be losslessly encoded as hex colors')
    frame = {'ref': f'frame:{records[-1]["step"]}', 'origin': 'raw host successor',
             'coordinates': 'zero-based row=y, column=x; row strings have no prefixes',
             'encoding': 'one hexadecimal color digit per cell',
             'grid_rows': [''.join(format(v, 'x') for v in row) for row in grid],
             'grid_sha256': digest(grid)}
    return {'ledger': ledger, 'current_frame': frame,
            'coverage': {'first_step': records[0]['step'], 'last_step': records[-1]['step'],
                         'later_events': 'excluded', 'absolute_frames': 'current successor only',
                         'earlier_absolute_frames': 'not supplied; do not invent earlier positions',
                         'object_identity': 'not measured; color regions are not certified objects'}}


def system_prompt():
    original = (HERE / 'narrow-v2/system.md').read_text()
    rules = original.split('Return JSON only:', 1)[0]
    return rules + '''
The host evidence_view supplies exact counts and the current full frame. First
compare literal claims to the ledger, including variation across ALL listed counts.
Do not recalculate a ledger fact from Actor prose. Position claims require the
relevant frame; a color or bounding region need not be a single object. Do not
describe unobserved earlier positions as measured. Claim scope may exceed what
the latest frame proves. An actual frame conflict is distinct from missing evidence.

Return JSON only with exactly these keys:
{"literal_check":{"claim":"exact numeric/position claim or empty if absent",
"evidence_refs":[],"observed":"specific host facts or evidence limitation",
"relation":"contradicted|consistent|not_assessable|no_literal_claim"},
"status":"issues|no_issue_found|cannot_assess","issues":[
{"claim":"exact excerpt","type":"contradicted|unsupported|unresolved_attribution",
"evidence_refs":[],"error":"exact inference error",
"correction":"smallest warranted claim"}],
"decision_effect":"retain_action|revise_commitment|needs_observation",
"decision_reason":"one short sentence tying correction to the proposal"}

At most two issues. Check and report an explicit numeric contradiction first when
one exists. Each claim excerpt: at most 25 words. Error plus correction: at most
60 words per issue. No_issue_found and cannot_assess require empty issues.
An unsupported claim may be corrected while one exploratory action remains
reasonable. Do not invent action effects, objectives, positions, or counterfactual
outcomes. No_issue_found is limited to the supplied evidence, not model certification.
'''


def prepare(out):
    cases = json.loads((BASELINE / 'cases.json').read_text())
    packets, manifest = [], []
    for i, c in enumerate(cases):
        path = ROOT / c['provenance']['source']
        if path.is_dir():
            path = path / 'ipc/transitions.jsonl'
        records = read_prefix(path, c['provenance']['cutoff_step'])
        evidence = measure(records)
        # Independently verify retained differences against the old fixture.
        for old, new in zip(c['evidence'], evidence['ledger'], strict=True):
            for k in ('action', 'changes_y_x_before_after', 'levels_before', 'levels_after'):
                if old[k] != new[k]:
                    raise ValueError(f'original fixture mismatch: {c["id"]}/{k}')
        packet = {'evidence_view': evidence, 'actor_claim': c['actor_claim'],
                  'actor_proposal': c['actor_proposal']}
        body = json.loads((BASELINE / f'{i:02d}.request.json').read_text())
        body['messages'] = [{'role': 'system', 'content': system_prompt()},
                            {'role': 'user', 'content': json.dumps(packet)}]
        packets.append((f'{i:02d}.request.json', body))
        manifest.append({'request': f'{i:02d}.request.json', 'case': c['id'],
                         'source': str(path.relative_to(ROOT)),
                         'prefix_sha256': digest(records), 'cutoff': records[-1]['step'],
                         'actor_content_sha256': digest([c['actor_claim'], c['actor_proposal']])})
    out.mkdir(parents=True, exist_ok=False)
    for name, body in packets:
        (out / name).write_text(json.dumps(body, indent=2) + '\n')
    (out / 'PREPARED.json').write_text(json.dumps({'status': 'prepared_only', 'paid_calls': 0,
        'system_sha256': hashlib.sha256(system_prompt().encode()).hexdigest(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'cases': manifest}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output)
    print('Prepared six measured-prefix packets; no inference.')
