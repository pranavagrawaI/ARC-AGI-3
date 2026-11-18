"""Inspect historical score headroom using action receipts and benchmark clocks.

This is game-selection evidence, never a replacement for a fresh matched control.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / 'analysis/m2-winners-20261001/runs'
OUT = Path(__file__).resolve().parent / 'evidence/score-headroom.json'
GAMES = {'ar25', 'sb26', 'ls20', 're86', 'vc33', 'sc25', 'r11l', 'dc22'}


def main():
    # Use the actual inherited scorer, rather than a new approximation.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
    from taaf.game import GameRun
    rows = []
    sources = {}
    for name in ('out-control', 'out-control-r2', 'out-s1'):
        directory = RUNS / name
        benchmark_path = directory / 'benchmark.json'
        benchmark = json.loads(benchmark_path.read_text())
        sources[str(benchmark_path.relative_to(ROOT))] = hashlib.sha256(benchmark_path.read_bytes()).hexdigest()
        for game in benchmark['game_runs']:
            if game['game_id'][:4] not in GAMES:
                continue
            event_paths = list((directory / 'artifacts').glob(game['game_id'] + '*_events.jsonl'))
            assert len(event_paths) == 1, (name, game['game_id'], 'ambiguous events')
            event_path = event_paths[0]
            sources[str(event_path.relative_to(ROOT))] = hashlib.sha256(event_path.read_bytes()).hexdigest()
            actions = []
            for line in event_path.open():
                event = json.loads(line)
                if event['type'] == 'action':
                    actions.append(event)
            assert [e['action_num'] for e in actions] == list(range(1, len(actions) + 1))
            assert len(actions) == len(game['history'])
            assert actions[-1]['score'] == game['levels_completed']
            run = GameRun.from_json_dict(game)
            assert abs(run._compute_final_score() - game['final_score']) < 1e-8
            samples = []
            for cutoff in (2400, 3600):
                reached = [event for event, receipt in zip(actions, game['history'])
                           if receipt['wallclock_seconds'] <= cutoff]
                # Completed levels have fixed action totals in the final ledger.
                # Use the last observed progress, not future final completion.
                run.levels_completed = int(reached[-1]['score']) if reached else 0
                samples.append({'game_clock_seconds': cutoff,
                                'observed_actions': len(reached),
                                'levels': run.levels_completed,
                                'score': run._compute_final_score()})
            rows.append({'run': name, 'game': game['game_id'],
                         'levels_total': game['number_of_levels'],
                         'final_score': game['final_score'],
                         'final_levels': game['levels_completed'], 'samples': samples})
    report = {'purpose': 'Historical game-selection evidence only; no matched efficacy inference.',
              'clock': 'Benchmark history wallclock_seconds, including admission/readiness waits. '
                       'This is not notebook allocated GPU time.',
              'limitations': 'Historical 25-game scheduling differs from planned three-game scheduling. '
                             'S1 is a prompt treatment, not a third identical control. '
                             'Completed-level action totals come from final ledgers; partial current-level actions score zero.',
              'sources_sha256': sources, 'rows': rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + '\n')
    print('game  control@60m  control-r2@60m  S1@60m  final scores [control, r2, S1]')
    for game in sorted(GAMES):
        subset = [r for r in rows if r['game'].startswith(game)]
        print(game, ' '.join(f"{r['samples'][1]['score']:.2f}" for r in subset),
              [round(r['final_score'], 2) for r in subset])


if __name__ == '__main__':
    main()
