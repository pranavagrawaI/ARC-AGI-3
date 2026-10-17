"""Queue screen-2's pre-registered draw candidate for the next UTC day's hidden draw. Idempotent; never overwrites.
python3 -B queue_candidate.py   (run_screen2.sh calls it after analyze.py)"""
import datetime as dt, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
QUEUE = HERE.parents[2] / 'analysis/hidden-draws-20261010/queue.json'
# The screen edit sets soft_end only when not TRUE_SUBMISSION, so a screen kernel version submits full-length.
KERNELS = {'F1': ('screen-1/f1', 'B1-fixed (frame-API text repaired) + stdlib preload + 16K output cap'),
           'F1x': ('screen-2/f1x', 'B1 (frame text untouched) + stdlib preload + 16K output cap')}


def main():
    result = json.loads((HERE / 'screen-2/RESULT.json').read_text())
    cand = result['rewrite']['draw_candidate']
    if cand is None:
        print('no candidate (no-harm failed); nothing queued'); return
    run, what = KERNELS[cand]
    resp = json.loads((HERE / run / 'launch-response.json').read_text())
    day = (dt.datetime.now(dt.timezone.utc).date() + dt.timedelta(days=1)).isoformat()
    queue = json.loads(QUEUE.read_text()) if QUEUE.exists() else {}
    if day in queue:
        print(f'{day} already queued: {queue[day]["kernel"]}; leaving it'); return
    s1 = json.loads((HERE / 'screen-1/RESULT.json').read_text())['no_harm']
    s2 = result['no_harm']
    queue[day] = {'kernel': resp['ref'].split('/code/')[-1], 'version': resp['versionNumber'],
                  'message': f"(candidate, {cand}): {what}. Screen-1 f1 {s1['f1_levels']} vs b1ctl {s1['b1ctl_levels']}; "
                             f"screen-2 f1x {s2['f1x_levels']} vs b1ctl {s2['b1ctl_levels']}, "
                             f"rewrite call {result['rewrite']['call']} (R={result['rewrite']['R_actions@30']}) at 60 min public-25."}
    t = QUEUE.with_suffix('.tmp'); t.write_text(json.dumps(queue, indent=1) + '\n'); t.replace(QUEUE)
    print(f'queued {cand} for {day}: {queue[day]}')


if __name__ == '__main__':
    main()
