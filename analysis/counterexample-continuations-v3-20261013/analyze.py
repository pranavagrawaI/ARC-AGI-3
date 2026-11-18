"""Read collected receipts; keep missing/invalid episodes out of efficacy claims."""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = ['control', 'reminder', 'counterexample']
VALID = {'solved', 'token_budget', 'request_budget', 'episode_deadline', 'gave_up'}


def read(path):
    return json.loads(path.read_text())


def audit_episode(directory, expected):
    path = directory / 'result.json'
    if not path.exists():
        return {**expected, 'valid': False, 'status': 'not collected', 'issues': ['missing result']}
    result = read(path)
    issues = []
    for key in ['checkpoint', 'seed', 'arm']:
        if result.get(key) != expected[key]:
            issues.append(f'{key} differs from matrix')
    for filename in ['checkpoint.json', 'restoration.json', 'wire.jsonl', 'actions.json']:
        if not (directory / filename).exists():
            issues.append(f'missing {filename}')
    if issues:
        return {**expected, **result, 'valid': False, 'issues': issues}
    checkpoint = read(directory / 'checkpoint.json')
    restoration = read(directory / 'restoration.json')
    events = [json.loads(line) for line in (directory / 'wire.jsonl').read_text().splitlines()]
    requests = [e for e in events if e['event'] == 'request']
    responses = [e for e in events if e['event'] == 'response']
    if not checkpoint.get('qualified') or not checkpoint.get('actions_match') or not restoration.get('actions_match'):
        issues.append('restoration not qualified')
    if not all(c.get('visible_content_identical') and c.get('tools_identical') for c in checkpoint['comparisons']):
        issues.append('historical request divergence')
    if result.get('unresolved') or result['status'] not in VALID:
        issues.append('incomplete or failed transport/episode')
    tokens = 0
    if not requests or len(requests) != len(responses) or len(requests) != result.get('new_model_calls'):
        issues.append('request/response coverage mismatch')
    for request, response in zip(requests, responses):
        usage = response.get('body', {}).get('usage', {}).get('completion_tokens')
        if response['call'] != request['call'] or response['status'] != 200:
            issues.append('unmatched or unsuccessful response')
        if type(usage) is not int or not 0 <= usage <= request['body']['max_tokens']:
            issues.append('invalid usage receipt')
        else:
            tokens += usage
    if tokens != result.get('generated_tokens') or tokens > 32768 or len(requests) > 32:
        issues.append('budget or token accounting mismatch')
    actions = read(directory / 'actions.json')
    if actions[:len(checkpoint['prefix_actions'])] != checkpoint['prefix_actions']:
        issues.append('action prefix differs')
    solved = result.get('levels_completed', 0) >= 2
    if solved != (result['status'] == 'solved'):
        issues.append('completion/status disagreement')
    return {**expected, **result, 'valid': not issues, 'issues': issues,
            'new_actions': len(actions) - len(checkpoint['prefix_actions']),
            'receipted_tokens': tokens, 'solved': solved}


def summarize(rows):
    by_key = {(r['checkpoint'], r['seed'], r['arm']): r for r in rows}
    pairs = {}
    for arm in ARMS[1:]:
        values = []
        for checkpoint in ['b1', 'control']:
            for seed in [1701, 2903]:
                baseline, treatment = [by_key[checkpoint, seed, a] for a in ['control', arm]]
                complete = baseline['valid'] and treatment['valid']
                values.append({'checkpoint': checkpoint, 'seed': seed, 'valid': complete,
                    'completion_delta': int(treatment['solved']) - int(baseline['solved']) if complete else None,
                    'score_delta': treatment['score'] - baseline['score'] if complete else None})
        gains = sum(p['completion_delta'] == 1 for p in values)
        losses = sum(p['completion_delta'] == -1 for p in values)
        coverage = sum(p['valid'] for p in values)
        pairs[arm] = {'pairs': values, 'coverage': coverage, 'gains': gains, 'losses': losses,
                      'screen_gate': 'INCOMPLETE_OR_INVALID' if coverage < 4 else
                      ('PASS' if gains >= 2 and losses == 0 else 'NOT_MET')}
    return {'episodes': rows, 'valid_episodes': sum(r['valid'] for r in rows),
            'expected_episodes': 12, 'comparisons': pairs,
            'scope': 'Two selected checkpoints, two seeds; completion screen, not general efficacy.'}


def plot(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(10, 5.5))
    rows = summary['episodes']
    blocks = [(c, s) for c in ['b1', 'control'] for s in [1701, 2903]]
    for r in rows:
        x, y = ARMS.index(r['arm']), blocks.index((r['checkpoint'], r['seed']))
        color = '#d4eee7' if r.get('solved') and r['valid'] else '#f1e7d2' if r['valid'] else '#eeeeee'
        ax.add_patch(Rectangle((x-.48, y-.46), .96, .92, facecolor=color, edgecolor='white'))
        if r['valid']:
            label = ('SOLVED L2' if r['solved'] else 'L2 UNSOLVED') + f"\nscore {r['score']:.3f}\n{r['receipted_tokens']:,} tokens / {r['new_actions']} new actions"
        else:
            label = r['status'].replace('_', ' ').upper() + '\nexcluded from comparison'
        ax.text(x, y, label, ha='center', va='center', fontsize=10)
    ax.set(xlim=(-.55, 2.55), ylim=(3.6, -.65), xticks=range(3), yticks=range(4),
           xticklabels=['Ordinary continuation', 'Observation reminder', 'Specific counterexample'],
           yticklabels=[f'{c.upper()} / seed {s}' for c, s in blocks])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.suptitle('Does correcting the model buy a completed level?', fontsize=16, y=.98)
    fig.text(.02, .02, f"Receipt-qualified episodes: {summary['valid_episodes']}/12. Gray means missing or invalid, never a measured zero.\nMatched continuation budgets: 32,768 new tokens, 32 requests, 15 minutes. Selected cases only.", fontsize=10)
    fig.tight_layout(rect=(0, .10, 1, .94))
    fig.savefig(output / 'paired-results.png', dpi=160)
    fig.savefig(output / 'paired-results.svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=HERE / 'analysis')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    rows = [audit_episode(args.episodes / e['id'], e) for e in read(HERE / 'matrix.json')]
    summary = summarize(rows)
    args.output.mkdir(exist_ok=True, parents=True)
    (args.output / 'paired-results.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k != 'episodes'}, indent=2))
    if args.plot:
        plot(summary, args.output)


if __name__ == '__main__':
    main()
