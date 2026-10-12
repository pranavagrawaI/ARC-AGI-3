"""Score-first retention from immutable paired raw outputs."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / 'model-serving-20261011/wave-1/e2-c14/output/benchmark.json'
sys.path.insert(0, str(HERE.parent / 'model-serving-20261011'))
from analyze import serve_stats


def decision(control, candidate):
    positive = (candidate['levels'] >= (control['levels'] * 110 + 99) // 100
                and candidate['score'] >= control['score']
                and candidate['throughput'] >= 1.10 * control['throughput']
                and candidate['retractions'] == 0)
    negative = candidate['levels'] <= control['levels'] - 10 or candidate['score'] <= .90 * control['score']
    return 'RETAIN_CANDIDATE' if positive else 'REJECT_CANDIDATE' if negative else 'INCONCLUSIVE_RETAIN_CONTROL'


def arm(name, expected_experts, streams):
    root = HERE / name
    collection = json.loads((root / 'collection.json').read_text())
    if collection['status'] != 'COMPLETE':
        raise ValueError(name + ': kernel not COMPLETE')
    output = root / 'output'
    target = json.loads((output / 'pruning/target.json').read_text())
    draft = json.loads((output / 'pruning/draft.json').read_text())
    if not target['complete'] or target['target_experts'] != expected_experts or draft['draft_experts'] != 512:
        raise ValueError(name + ': wrong loaded expert banks')
    info = json.loads((output / 'pruning/server-info-start.json').read_text())
    if info['max_running_requests'] != streams or info['context_length'] != 102400:
        raise ValueError(name + ': configuration differs')
    games = json.loads((output / 'benchmark.json').read_text())['game_runs']
    ids = {g['game_id'] for g in json.loads(REFERENCE.read_text())['game_runs']}
    if len(games) != 25 or {g['game_id'] for g in games} != ids or any(g['state'] not in {'won', 'gave_up'} for g in games):
        raise ValueError(name + ': incomplete or erroneous gameplay')
    text = (output / 'kernel.log').read_text() if (output / 'kernel.log').exists() else ''
    if f'priority gate active: {streams} concurrent streams' not in text:
        # Some API versions supply the notebook log as an ordinary output file instead.
        text += '\n'.join(p.read_text(errors='replace') for p in output.glob('*.log') if p.name not in {'serve.log', 'kernel.log'})
    if f'priority gate active: {streams} concurrent streams' not in text:
        raise ValueError(name + ': missing active-stream marker')
    stats = serve_stats(output)
    if stats.get('window_s', 0) < 1800 or not stats.get('tw_gen_tok_s'):
        raise ValueError(name + ': insufficient throughput evidence')
    return {'levels': sum(int(g['levels_completed']) for g in games),
            'score': sum(float(g['final_score'] or 0) for g in games) / 25,
            'throughput': stats['tw_gen_tok_s'], 'retractions': stats['retract_lines'],
            'serve': stats, 'games': {g['game_id']: g['levels_completed'] for g in games}}


def evaluate(candidate_run='candidate'):
    result = {'status': 'INVALID_RETAIN_CONTROL', 'retained': 'c14', 'fallback': 'c14'}
    try:
        control = arm('control', 512, 14); candidate = arm(candidate_run, 448, 18)
        status = decision(control, candidate)
        result.update(status=status, candidate_run=candidate_run, control=control, candidate=candidate,
                      retained='REAP448+c18' if status == 'RETAIN_CANDIDATE' else 'c14',
                      limitations=['One paired run; no hidden-set generalization claim.', 'Bundled pruning plus capacity; not isolated pruning efficacy.'])
    except Exception as error:
        result['error'] = str(error)
    (HERE / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    identity = json.loads((HERE / 'runs.json').read_text())
    retained = {'status': result['status'], 'retained': result['retained'], 'fallback': 'c14',
                'runs': identity, 'evidence': str(HERE / 'RESULT.json'), 'competition_submission': False}
    (HERE / 'retention.json').write_text(json.dumps(retained, indent=2) + '\n')
    return result


if __name__ == '__main__':
    print(json.dumps(evaluate(), indent=2))
