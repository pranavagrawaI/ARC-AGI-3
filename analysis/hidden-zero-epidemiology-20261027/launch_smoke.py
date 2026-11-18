"""One-attempt push of the F1x+keepalive kernel (its Save & Run is the smoke and the submittable version).
Admission: notebook = build_keepalive(F1x screen-2 v1 parent), at most one other active GPU session, weekly quota.
launch-attempt.json (exclusive create) means never push again.
~/.local/share/uv/tools/kaggle/bin/python -B launch_smoke.py [--dry]"""
import datetime as dt, hashlib, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'mtp-depth-20261006'))
sys.path.insert(0, str(HERE))
import launch_watch as lw
import build_keepalive
lw.EVID = HERE
KDIR = HERE / 'kernel-f1x-ka'
PARENT = HERE.parents[1] / 'solvers/m2_f1/experiment/screen-2/f1x/kernel/m2-f1-screen2-f1x.ipynb'
PARENT_SHA = '826dfdc2a483a786bdfe888ad9b9422630bc4c615a187b34bdc9762975fb7df9'
CAP_S = 9000  # F1x screen v1 ran 72 min end to end; allow a 60+ min load


def now(): return dt.datetime.now(dt.timezone.utc).isoformat()


def main():
    dry = '--dry' in sys.argv
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == PARENT_SHA
    nb = KDIR / 'm2-f1x-keepalive.ipynb'
    expect = json.dumps(build_keepalive.build(json.loads(PARENT.read_text())), indent=1) + '\n'
    assert nb.read_text() == expect, 'kernel notebook is not the builder output'
    m = json.loads((KDIR / 'kernel-metadata.json').read_text())
    assert m['id'] == 'pranavagrawai/m2-f1x-keepalive' and m['machine_shape'] == 'NvidiaRtxPro6000' and m['is_private']
    if (HERE / 'launch-attempt.json').exists():
        raise SystemExit('launch already attempted; do not retry')
    lw.refresh_token()
    api = lw.KaggleApi(); api.authenticate()
    lw.NOTEBOOK_DIR = KDIR
    active = lw.active_sessions(api)
    used, q = lw.quota_used_s(api)
    reasons = []
    if any(str(s).startswith('unknown') for _, s in active) or len(active) >= 1:
        reasons.append(f'slots: active {active} (a 2nd concurrent GPU kernel has landed on a T4)')
    if used + CAP_S > 30 * 3600:
        reasons.append(f'weekly quota: used {used/3600:.2f} h')
    print(now(), f'quota used {used/3600:.2f} h, active {active}:', 'READY' if not reasons else '; '.join(reasons))
    if dry or reasons:
        return
    attempt = {'utc': now(), 'state': 'attempting', 'notebook_sha256': hashlib.sha256(nb.read_bytes()).hexdigest(),
               'metadata': m, 'cap_s': CAP_S, 'quota_before': q, 'active_before': active}
    with (HERE / 'launch-attempt.json').open('x') as f:
        json.dump(attempt, f, indent=2)
    try:
        result = json.loads(str(api.kernels_push(str(KDIR), timeout=str(CAP_S))))
        (HERE / 'launch-response.json').write_text(json.dumps(result, indent=2))
        if result.get('error') or not result.get('versionNumber'):
            raise RuntimeError(f'push not accepted: {result}')
        attempt.update(state='accepted', response=result)
        print(now(), 'LAUNCHED', result)
    except Exception as e:
        attempt.update(state='acceptance unresolved; inspect, do not retry', error=repr(e)[:500])
        print(now(), 'PUSH FAILED', repr(e))
    (HERE / 'launch-attempt.json').write_text(json.dumps(attempt, indent=2))


if __name__ == '__main__':
    main()
