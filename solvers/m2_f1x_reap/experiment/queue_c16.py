"""Queue c16 v1 as an extra hidden-draw ticket once screen-2 PASSes. Idempotent; never overwrites another day.
python3 -B queue_c16.py   (run_screen2.sh calls it after analyze_c16.py)"""
import datetime as dt, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
QUEUE = HERE.parents[2] / 'analysis/hidden-draws-20261010/queue.json'
EARLIEST = dt.date(2026, 10, 26)


def main():
    result = json.loads((HERE / 'screen-2/RESULT.json').read_text())
    if result['verdict'] != 'PASS':
        print(f"{result['verdict']}; nothing queued"); return
    resp = json.loads((HERE / 'screen-2/c16/launch-response.json').read_text())
    queue = json.loads(QUEUE.read_text()) if QUEUE.exists() else {}
    kernel = resp['ref'].split('/code/')[-1]
    if any(e['kernel'] == kernel for e in queue.values()):
        print(f'{kernel} already queued; leaving it'); return
    day = max(EARLIEST, dt.datetime.now(dt.timezone.utc).date() + dt.timedelta(days=1))
    while day.isoformat() in queue:
        day += dt.timedelta(days=1)
    s, n = result['serving'], result['no_harm']
    queue[day.isoformat()] = {
        'kernel': kernel, 'version': resp['versionNumber'],
        'message': f"(candidate, F1x+REAP448 c16): the D16 stack at 16 streams (116K input). Screen-2 {n['levels']} levels at "
                   f"60 min vs c14's 80; {s['agg_tok_s_0_60']:.0f} vs 823 tok/s, KV p90 {s['kv_usage_p90']:.2f}, "
                   f"{s['retractions']} retractions. Extra ticket, not a replacement. Exact screened v1."}
    t = QUEUE.with_suffix('.tmp'); t.write_text(json.dumps(queue, indent=1) + '\n'); t.replace(QUEUE)
    print(f'queued for {day}: {queue[day.isoformat()]}')


if __name__ == '__main__':
    main()
