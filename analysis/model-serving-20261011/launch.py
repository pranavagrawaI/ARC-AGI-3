"""One-attempt launcher for this experiment's runs. Admission: frozen notebook hashes, enough free GPU slots,
weekly quota, and the experiment ceiling (5.5 GPU-h counting actual time of finished runs and hard caps of
unfinished ones). A launch-attempt.json in a run directory means never push that run again.
~/.local/share/uv/tools/kaggle/bin/python -B launch.py wave-1 [--dry]"""
import datetime as dt, hashlib, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'mtp-depth-20261006'))
import launch_watch as lw  # quota_used_s, active_sessions, refresh_token (shared, already field-tested)
lw.EVID = HERE  # keep this experiment's log lines out of mtp-depth's evidence

CEILING_S = 8.0 * 3600  # raised 2026-10-11 for wave-2 (B1+c14 qualifying pass), user-approved next step
WEEKLY_CAP_S = 30 * 3600
GROUPS = {'wave-1': (['wave-1/e1-swift', 'wave-1/e2-c14'], 9000), 'wave-2': (['wave-2/b1-c14'], 9000)}
RUNS_FILE = {'wave-1': 'runs.json', 'wave-2': 'runs-wave2.json'}
LEDGER = HERE / 'budget.json'

def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def save(p, v):
    t = p.with_suffix(p.suffix + '.tmp'); t.write_text(json.dumps(v, indent=2) + '\n'); t.replace(p)

def committed_s(ledger):
    return sum(r.get('actual_s') if r.get('actual_s') is not None else r['hard_cap_s'] for r in ledger['runs'])

def verify(run):
    frozen = {r['run']: r for r in json.loads((HERE / RUNS_FILE[run.split('/')[0]]).read_text())['runs']}[run]
    d = HERE / run / 'kernel'
    nb = d / f"{frozen['slug']}.ipynb"
    assert hashlib.sha256(nb.read_bytes()).hexdigest() == frozen['notebook_sha256'], 'notebook drifted from runs.json'
    m = json.loads((d / 'kernel-metadata.json').read_text())
    assert m == frozen['metadata'] and m['is_private'] and m['machine_shape'] == 'NvidiaRtxPro6000'
    assert m['competition_sources'] == ['arc-prize-2026-arc-agi-3']
    if (HERE / run / 'launch-attempt.json').exists():
        raise SystemExit(f'{run}: launch already attempted; do not retry')
    card = HERE / run.split('/')[0] / 'EXPERIMENT-CARD.md'
    return {**frozen, 'card_sha256': hashlib.sha256(card.read_bytes()).hexdigest()}

def main():
    group, dry = sys.argv[1], '--dry' in sys.argv
    runs, cap = GROUPS[group]
    idents = {r: verify(r) for r in runs}
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {'ceiling_s': CEILING_S, 'runs': []}
    lw.refresh_token()
    api = lw.KaggleApi(); api.authenticate()
    lw.NOTEBOOK_DIR = HERE / runs[0] / 'kernel'
    active = lw.active_sessions(api)
    used, q = lw.quota_used_s(api)
    reasons = []
    if any(str(s).startswith('unknown') for _, s in active) or len(active) + len(runs) > 2:
        reasons.append(f'slots: active {active}, need {len(runs)}')
    if used + cap * len(runs) > WEEKLY_CAP_S:
        reasons.append(f'weekly quota: used {used/3600:.2f} h + {cap*len(runs)/3600:.2f} h > 30 h')
    if committed_s(ledger) + cap * len(runs) > CEILING_S:
        reasons.append(f'experiment ceiling: committed {committed_s(ledger)/3600:.2f} h + {cap*len(runs)/3600:.2f} h > 8.0 h')
    print(f'{now()} {group}: quota used {used/3600:.2f} h, active {active}, committed {committed_s(ledger)/3600:.2f} h:',
          'READY' if not reasons else '; '.join(reasons))
    if dry or reasons:
        return
    for run in runs:
        d = HERE / run
        attempt = {'utc': now(), 'state': 'attempting', 'identity': idents[run], 'hard_cap_s': cap,
                   'quota_before': q, 'active_before': active}
        with (d / 'launch-attempt.json').open('x') as f:
            json.dump(attempt, f, indent=2)
        ledger['runs'].append({'run': run, 'hard_cap_s': cap, 'actual_s': None, 'launched_utc': now()})
        save(LEDGER, ledger)  # reserve before pushing: a lost response still counts
        try:
            result = json.loads(str(api.kernels_push(str(d / 'kernel'), timeout=str(cap))))
            save(d / 'launch-response.json', result)
            if result.get('error') or not result.get('versionNumber'):
                raise RuntimeError(f'push not accepted: {result}')
            attempt.update(state='accepted', response=result)
            print(f'{now()} LAUNCHED {run} {result}')
        except Exception as e:
            attempt.update(state='acceptance unresolved; inspect, do not retry', error=repr(e)[:500])
            print(f'{now()} PUSH FAILED {run} {e!r}')
        save(d / 'launch-attempt.json', attempt)

if __name__ == '__main__':
    main()
