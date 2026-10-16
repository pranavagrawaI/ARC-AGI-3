"""Read out screen-1 against its pre-registered rules (EXPERIMENT-CARD.md). CPU only.
python3 -B analyze.py [screen-1]   -> prints and writes screen-1/RESULT.json"""
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
    return m


def main():
    screen = HERE / (sys.argv[1] if len(sys.argv) > 1 else 'screen-1')
    f1, ctl = arm(screen / 'f1' / 'output'), arm(screen / 'b1ctl' / 'output')
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
    result = {'verdict': verdict, 'no_harm': {'f1_levels': f1['levels'], 'b1ctl_levels': ctl['levels'],
              'threshold': ctl['levels'] - NO_HARM, 'pass': no_harm}, 'mechanism': mechanism,
              'arms': {'f1': f1, 'b1ctl': ctl}}
    (screen / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('verdict', 'no_harm', 'mechanism')}, indent=2))


if __name__ == '__main__':
    main()
