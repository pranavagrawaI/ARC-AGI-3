"""Poll a launched run until terminal, then download its output once and record session seconds in budget.json.
~/.local/share/uv/tools/kaggle/bin/python -B collect.py wave-1/e1-swift wave-1/e2-c14"""
import datetime as dt, json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'mtp-depth-20261006'))
import launch_watch as lw
lw.EVID = HERE
from launch import LEDGER, save, now

def main():
    for run in sys.argv[1:]:
        d = HERE / run.rstrip('/')
        resp = json.loads((d / 'launch-response.json').read_text())
        ref = resp['ref'].split('/code/')[-1] if 'ref' in resp else json.loads((d / 'kernel/kernel-metadata.json').read_text())['id']
        hist = d / 'status-history.jsonl'
        while True:
            lw.refresh_token()
            api = lw.KaggleApi(); api.authenticate()
            try:
                st = api.kernels_status(ref)
                status, msg = str(st.status), getattr(st, 'failure_message', None)
            except Exception as e:
                status, msg = f'unknown ({type(e).__name__})', str(e)[:200]
            with hist.open('a') as f:
                f.write(json.dumps({'utc': now(), 'status': status, 'failure': msg}) + '\n')
            if any(s in status for s in ('COMPLETE', 'ERROR', 'CANCEL')):
                break
            time.sleep(120)
        out = d / 'output'; out.mkdir(exist_ok=True)
        try:
            api.kernels_output(ref, path=str(out), force=True, quiet=True)
        except TypeError:
            api.kernels_output(ref, path=str(out))
        rows = [json.loads(l) for l in hist.read_text().splitlines()]
        start = json.loads((d / 'launch-attempt.json').read_text())['utc']
        wall = (dt.datetime.fromisoformat(rows[-1]['utc']) - dt.datetime.fromisoformat(start)).total_seconds()
        ledger = json.loads(LEDGER.read_text())
        for r in ledger['runs']:
            if r['run'] == run.rstrip('/'):
                # launch-to-poll wall overstates when polling stalls; the session cannot outlive its hard cap
                r['actual_s'] = round(min(wall, r['hard_cap_s'])); r['collector_wall_s'] = round(wall); r['final_status'] = status
        save(LEDGER, ledger)
        print(f'{now()} {run}: {status}; ~{wall/3600:.2f} h charged; output in {out}')

if __name__ == '__main__':
    main()
