"""Queue the merge for its hidden draw once screen-1 PASSes. Idempotent; never overwrites another day's entry.
python3 -B queue_candidate.py   (run_screen.sh calls it after analyze.py)

The entry carries requires [D11, D12]: daily_draw.py submits it only after D11 (REAP448+c18, 2026-10-20) and D12
(the F1x fallback, 2026-10-21) are both COMPLETE with a score above 0, so a zero can still be attributed."""
import datetime as dt, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
QUEUE = HERE.parents[2] / 'analysis/hidden-draws-20261010/queue.json'
EARLIEST = dt.date(2026, 10, 22)
REQUIRES = ['D11', 'D12']


def main():
    result = json.loads((HERE / 'screen-1/RESULT.json').read_text())
    if not result['verdict'].startswith('PASS'):
        print(f"{result['verdict']}; nothing queued"); return
    resp = json.loads((HERE / 'screen-1/merge/launch-response.json').read_text())
    queue = json.loads(QUEUE.read_text()) if QUEUE.exists() else {}
    kernel = resp['ref'].split('/code/')[-1]
    if any(e['kernel'] == kernel for e in queue.values()):
        print(f'{kernel} already queued; leaving it'); return
    day = max(EARLIEST, dt.datetime.now(dt.timezone.utc).date() + dt.timedelta(days=1))
    while day.isoformat() in queue:
        day += dt.timedelta(days=1)
    s, n = result['serving'], result['no_harm']
    queue[day.isoformat()] = {
        'kernel': kernel, 'version': resp['versionNumber'], 'requires': REQUIRES,
        'message': f"(candidate, F1x+REAP448): F1x harness (B1 + stdlib preload + 16K output) on REAP448 serving "
                   f"(448 target experts, 512-expert drafter) at 14 streams, 116K input. Screen-1 {n['merge_levels']} levels "
                   f"at 60 min vs B1-lineage pool mean {n['floor'] + 7:.0f}; {s['agg_tok_s_0_60']:.0f} vs 697 tok/s, KV p90 "
                   f"{s['kv_usage_p90']:.2f}. Drawn after D11 (REAP) and D12 (F1x) read valid. Exact screened v1."}
    t = QUEUE.with_suffix('.tmp'); t.write_text(json.dumps(queue, indent=1) + '\n'); t.replace(QUEUE)
    print(f'queued for {day}: {queue[day.isoformat()]}')


if __name__ == '__main__':
    main()
