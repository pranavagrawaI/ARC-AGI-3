"""Read out a screen against its pre-registered rules (its EXPERIMENT-CARD.md). CPU only.
python3 -B analyze.py [screen-1]        -> prints and writes screen-1/RESULT.json
python3 -B analyze.py screen-2 f1x     -> adds tempo and the screen-2 rewrite rule"""
import ast, json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRELOADED = {'bisect', 'collections', 'copy', 'functools', 'heapq', 'itertools', 'json', 'math', 'statistics'}
OLD_CAP, NEW_CAP, NO_HARM = 12 * 1024, 16 * 1024, 7
NAME_ERROR = re.compile(r"NameError: name '(\w+)' is not defined")


def as_dict(v):
    return v if isinstance(v, dict) else ast.literal_eval(v) if isinstance(v, str) and v.startswith('{') else {}


def arm(out):
    bench = json.loads((out / 'benchmark.json').read_text())
    games = bench['game_runs']
    m = {'games': len(games), 'levels': sum(g['levels_completed'] for g in games),
         'states': sorted({g['state'] for g in games}), 'responses': 0, 'completion_tokens': 0,
         'length_cutoffs': 0, 'cutoff_tokens': [], 'tool_calls_between_caps': 0, 'preload_name_errors': 0,
         'name_errors': 0}
    for f in sorted(out.glob('*_requests.jsonl')):
        seen = set()
        for line in f.open():
            r = json.loads(line)
            if r.get('event') == 'response':
                usage = as_dict(r.get('usage'))
                n = int(usage.get('completion_tokens') or 0)
                m['responses'] += 1; m['completion_tokens'] += n
                if r.get('finish_reason') == 'length':
                    m['length_cutoffs'] += 1; m['cutoff_tokens'].append(n)
                elif r.get('finish_reason') == 'tool_calls' and OLD_CAP < n <= NEW_CAP:
                    m['tool_calls_between_caps'] += 1
                continue
            for msg in r.get('messages') or []:
                if not isinstance(msg, dict) or msg.get('role') != 'tool':
                    continue
                text = msg['content'] if isinstance(msg['content'], str) else json.dumps(msg['content'])
                if text in seen:  # every later request repeats earlier tool results
                    continue
                seen.add(text)
                for name in NAME_ERROR.findall(text):
                    m['name_errors'] += 1
                    m['preload_name_errors'] += name in PRELOADED
    m['cutoff_tokens'] = sorted(set(m['cutoff_tokens']))
    m['tempo'] = tempo(games)
    return m


def tempo(games, minutes=(30, 45, 60)):
    """Cumulative levels/actions at each minute; history wallclock_seconds is cumulative per game."""
    t = {f'{k}@{m}': 0 for m in minutes for k in ('levels', 'actions')}
    tok30 = 0
    for g in games:
        ts = [float(x['wallclock_seconds']) for x in g['history']]
        for x, s in zip(g['history'], ts):
            tok30 += s <= 1800 and int(float(x['generated_tokens']))
            for m in minutes:
                t[f'actions@{m}'] += s <= m * 60
        cum = 0
        for n in g['actions_per_level'][:g['levels_completed']]:
            cum += n
            for m in minutes:
                t[f'levels@{m}'] += bool(ts) and ts[min(cum, len(ts)) - 1] <= m * 60
    t['tokens_per_action@30'] = round(tok30 / max(t['actions@30'], 1))
    return t


def main():
    screen = HERE / (sys.argv[1] if len(sys.argv) > 1 else 'screen-1')
    treat = sys.argv[2] if len(sys.argv) > 2 else 'f1'
    f1, ctl = arm(screen / treat / 'output'), arm(screen / 'b1ctl' / 'output')
    crashed = any(s not in {'won', 'gave_up', 'cancelled', 'in_progress'} for a in (f1, ctl) for s in a['states'])
    mechanism = {
        'preload_zero_in_f1': f1['preload_name_errors'] == 0,
        'out16k_applied': f1['tool_calls_between_caps'] > 0 or (bool(f1['cutoff_tokens']) and
                                                                 set(f1['cutoff_tokens']) == {NEW_CAP}),
        'no_crashed_games': not crashed and f1['games'] == ctl['games'] == 25,
    }
    no_harm = f1['levels'] >= ctl['levels'] - NO_HARM
    verdict = ('PASS' if all(mechanism.values()) and no_harm else
               'NO_HARM_FAIL: split once (preload | out16k)' if all(mechanism.values()) else
               'MECHANISM_FAIL: inspect before any rerun')
    result = {'verdict': verdict, 'no_harm': {f'{treat}_levels': f1['levels'], 'b1ctl_levels': ctl['levels'],
              'threshold': ctl['levels'] - NO_HARM, 'pass': no_harm}, 'mechanism': mechanism,
              'arms': {treat: f1, 'b1ctl': ctl}}
    if treat == 'f1x':  # screen-2 rule (screen-2/EXPERIMENT-CARD.md)
        r = f1['tempo']['actions@30'] / ctl['tempo']['actions@30']
        rewrite = 'REWRITE_SLOWS' if r >= 0.95 else 'NOT_REWRITE' if r < 0.90 else 'AMBIGUOUS'
        candidate = None if not no_harm else 'F1' if rewrite == 'NOT_REWRITE' else 'F1x'
        result['rewrite'] = {'R_actions@30': round(r, 3), 'call': rewrite, 'draw_candidate': candidate}
        result['verdict'] = verdict if no_harm else 'NO_HARM_FAIL: no F1 variant drawn; inspect'
        result['verdict'] = result['verdict'].split(':')[0] + f' / {rewrite} -> {candidate}'
    keys = ('verdict', 'no_harm', 'mechanism') + (('rewrite',) if 'rewrite' in result else ())
    (screen / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in keys}, indent=2))


if __name__ == '__main__':
    main()
