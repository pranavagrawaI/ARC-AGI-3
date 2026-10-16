"""Read out the lexicon screen against its pre-registered rules (screen-1/EXPERIMENT-CARD.md). CPU only.
python3 -B analyze.py [screen-1]   -> prints and writes screen-1/RESULT.json"""
import json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NO_HARM, SIGNAL = 7, 7  # levels: 1 sd of a paired 60-min difference


def arm(out):
    games = json.loads((out / 'benchmark.json').read_text())['game_runs']
    logs = '\n'.join(p.read_text(errors='replace') for p in out.glob('*.log'))
    final = re.findall(r'ARCAID3_LEXICON_FINAL (.*)', logs)
    return {'games': len(games), 'levels': sum(g['levels_completed'] for g in games),
            'states': sorted({g['state'] for g in games}),
            'ready': 'ARCAID3_LEXICON_READY' in logs, 'off': 'ARCAID3_LEXICON_OFF' in logs,
            'final_counters': final[-1][:600] if final else None}


def main():
    screen = HERE / (sys.argv[1] if len(sys.argv) > 1 else 'screen-1')
    lex, ctl = arm(screen / 'lex' / 'output'), arm(screen / 'rawctl' / 'output')
    crashed = any(s not in {'won', 'gave_up', 'cancelled', 'in_progress'} for a in (lex, ctl) for s in a['states'])
    valid = lex['ready'] and not lex['off'] and not crashed and lex['games'] == ctl['games'] == 25
    diff = lex['levels'] - ctl['levels']
    verdict = ('INVALID: lexicon not installed or games crashed' if not valid else
               'SIGNAL: queue hidden draw; port onto F1 next' if diff >= SIGNAL else
               'NO_HARM: queue hidden draw (free); port only if the draw beats 30' if diff >= -NO_HARM else
               'HARM: drop')
    result = {'verdict': verdict, 'levels': {'lex': lex['levels'], 'rawctl': ctl['levels'], 'diff': diff},
              'arms': {'lex': lex, 'rawctl': ctl}}
    (screen / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
