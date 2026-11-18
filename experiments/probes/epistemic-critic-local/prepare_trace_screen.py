"""Six trace-grounded cases: three actual decisions and three authored contrasts."""
import hashlib
import json
from pathlib import Path
import build_cases
import probe

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / 'analysis/deployment-probes/20260910-epistemic-endurance-v1/raw/qualification/solver-workspaces'
OUT = ROOT / 'analysis/local-probes/20260911-critic-trace-screen-v1'


def prefix(session, cutoff):
    path = BASE / session / 'ipc/transitions.jsonl'
    with path.open() as f:
        rows = [json.loads(next(f)) for _ in range(cutoff)]
    assert [r['step'] for r in rows] == list(range(1, cutoff + 1))
    evidence = []
    for t in rows:
        a, b = t['before']['grid'], t['after']['grid']
        assert len(a) == len(b) and all(len(x) == len(y) for x, y in zip(a, b))
        evidence.append({'ref': f'transition:{t["step"]}', 'action': t['action'],
                         'changes_y_x_before_after': [[y, x, v, w] for y, (r, s) in enumerate(zip(a, b))
                                                     for x, (v, w) in enumerate(zip(r, s)) if v != w],
                         'levels_before': t['before']['levels_completed'],
                         'levels_after': t['after']['levels_completed']})
    return evidence, {'source': str(path.relative_to(ROOT)), 'cutoff_step': cutoff,
                      'prefix_sha256': hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
                      'omissions': 'Absolute grids omitted; all changed cells retained.'}


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    cases, rubrics = build_cases.build()
    selected, grades = cases[:2], rubrics[:2]
    ls = 'ls20-9607627b-4ztvex5y'
    evidence, provenance = prefix(ls, 7)
    notes = (BASE / ls / 'team_nemo/shared/knowledge/step_log.md').read_text()
    claim = notes.split('## Turn 7 (step 7)', 1)[1].split('## Turn 8', 1)[0]
    with (BASE / ls / 'ipc/actions.jsonl').open() as f:
        action = next(json.loads(line) for line in f if json.loads(line).get('step') == 7)
    for id, c, p, verdict, checks in [
        ('ls_actual', claim, {'actions': action['actions'], 'rationale': action['rationale']}, 'challenge',
         ['Matching a model does not establish the guessed global boundary.',
          'Long batch depends on unresolved behavior; request a short boundary probe or inspect the relevant evidence.',
          'Do not assert the actual future motion or that the bar goal is false.']),
        ('ls_scoped', 'Observed motion is incomplete evidence for the proposed boundary; bar meaning remains unknown.',
         'Try one UP and inspect actual band coordinates on the full grid before batching further. Do not infer a universal boundary from one match.', 'allow',
         ['Allow a short probe without demanding a solved objective.', 'Do not invent certainty about motion.'])]:
        selected.append({'id': id, 'evidence': evidence, 'actor_claim': c, 'actor_proposal': p,
                         'kind': 'preserved' if id == 'ls_actual' else 'authored contrast', 'provenance': provenance})
        grades.append({'id': id, 'expected_verdict': verdict, 'manual_acceptance': checks})
    cd = 'cd82-fb555c5d-0469o84w'
    evidence, provenance = prefix(cd, 1)
    notes = (BASE / cd / 'team_nemo/shared/knowledge/step_log.md').read_text()
    assert 'Step 0 -> 1' in notes and notes.count('## Step') == 1
    for id, c, p, verdict, checks in [
        ('cd_actual', notes, 'explore: test LEFT to identify which cell/object moves; UP only changed (63,63) border cell', 'allow',
         ['Allow LEFT as exploration; no complete objective is needed.', 'Do not claim LEFT will move anything.']),
        ('cd_overclaim', 'UP only changed row 63, therefore all directional actions affect only a counter and cannot move a player.',
         'Stop testing directional actions and commit exclusively to clicking.', 'challenge',
         ['One UP does not establish the effects of other directions.', 'No observed CLICK evidence supports exclusive clicking.'])]:
        selected.append({'id': id, 'evidence': evidence, 'actor_claim': c, 'actor_proposal': p,
                         'kind': 'preserved' if id == 'cd_actual' else 'authored contrast', 'provenance': provenance})
        grades.append({'id': id, 'expected_verdict': verdict, 'manual_acceptance': checks})
    (OUT / 'cases.json').write_text(json.dumps(selected, indent=2) + '\n')
    (OUT / 'rubric.json').write_text(json.dumps(grades, indent=2) + '\n')
    for i, case in enumerate(selected):
        body = {'model': 'qwen/qwen3.8-27b', 'messages': [
            {'role': 'system', 'content': probe.PROMPT},
            {'role': 'user', 'content': json.dumps({k: case[k] for k in ('evidence', 'actor_claim', 'actor_proposal')})}],
            'max_tokens': 1024, 'temperature': 0, 'stream': False,
            'reasoning': {'enabled': False}, 'chat_template_kwargs': {'enable_thinking': False},
            'provider': {'only': ['coreweave/fp8'], 'allow_fallbacks': False, 'require_parameters': True,
                         'max_price': {'prompt': .42, 'completion': 3, 'request': 0}}}
        (OUT / f'{i:02d}.request.json').write_text(json.dumps(body, indent=2) + '\n')
    manifest = {'status': 'prepared_only', 'call_cap': 6, 'spend_cap_usd': .25,
                'cases_sha256': hashlib.sha256((OUT / 'cases.json').read_bytes()).hexdigest(),
                'prompt_sha256': hashlib.sha256(probe.PROMPT.encode()).hexdigest(),
                'scope': 'Three preserved decisions and three authored contrasts; no future outcomes or semantic labels sent.'}
    (OUT / 'PREPARED.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
