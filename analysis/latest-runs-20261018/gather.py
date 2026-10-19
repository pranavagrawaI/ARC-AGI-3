"""Read-only latest-run collection. Does not launch, submit, or edit prior artifacts."""
import concurrent.futures, csv, datetime as dt, hashlib, io, json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RAW = OUT / 'live'
RAW.mkdir(exist_ok=True)
SPECS = [
    ('F1', 'solvers/m2_f1/experiment/screen-1', 'f1', 'b1ctl', 'm2-f1-screen1'),
    ('Lexicon', 'solvers/m2_lexicon/experiment/screen-1', 'lex', 'rawctl', 'm2-lex-screen1'),
    ('F1x', 'solvers/m2_f1/experiment/screen-2', 'f1x', 'b1ctl', 'm2-f1-screen2'),
]
commands = {
    'submissions': ['kaggle', 'competitions', 'submissions', 'arc-prize-2026-arc-agi-3', '--csv'],
    'kernels': ['kaggle', 'kernels', 'list', '--mine', '--page-size', '15', '--sort-by', 'dateRun', '--csv'],
    'timer': ['systemctl', '--user', 'show', 'arc-hidden-draw-daily.timer', '-p', 'ActiveState', '-p', 'UnitFileState', '-p', 'NextElapseUSecRealtime'],
}
for _, _, treat, ctl, prefix in SPECS:
    for arm in [treat, ctl]:
        kernel = prefix + '-' + arm
        commands[kernel] = ['kaggle', 'kernels', 'status', 'pranavagrawai/' + kernel]

def run(item):
    key, command = item
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        r = subprocess.run(command, capture_output=True, text=True, timeout=90)
        result = dict(command=command, utc=started, returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)
    except subprocess.TimeoutExpired as e:
        result = dict(command=command, utc=started, error=str(e), returncode=None)
    (RAW / (key + '.json')).write_text(json.dumps(result, indent=2))
    return key, result

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    live = dict(pool.map(run, commands.items()))
assert live['submissions']['returncode'] == 0, live['submissions']
submissions = list(csv.DictReader(io.StringIO(live['submissions']['stdout'])))
source_paths = []
def read(path):
    p = ROOT / path
    source_paths.append(p)
    return json.loads(p.read_text())

screens = []
all_ids = None
for label, base, treat, ctl, prefix in SPECS:
    result = read(base + '/RESULT.json')
    arms = {}
    for name in [treat, ctl]:
        bench = read(base + '/' + name + '/output/benchmark.json')
        games = bench['game_runs']
        ids = {g['game_id'] for g in games}
        assert len(games) == len(ids) == 25
        if all_ids is None: all_ids = ids
        assert ids == all_ids
        assert all(0 <= g['levels_completed'] <= g['number_of_levels'] for g in games)
        levels = sum(g['levels_completed'] for g in games)
        assert levels == result['arms'][name]['levels']
        state_counts = {state: sum(g['state'] == state for g in games) for state in sorted({g['state'] for g in games})}
        assert set(state_counts) <= {'won', 'cancelled'}
        arms[name] = dict(levels=levels, mean_score=sum(g['final_score'] for g in games)/25,
            start_time=bench['start_time'], end_time=bench['end_time'], states=state_counts,
            per_game={g['game_id']: dict(levels=g['levels_completed'], total=g['number_of_levels'], score=g['final_score']) for g in games},
            kernel='pranavagrawai/' + prefix + '-' + name,
            live_status=live[prefix + '-' + name], metrics=result['arms'][name])
    differences = {gid: arms[treat]['per_game'][gid]['levels'] - arms[ctl]['per_game'][gid]['levels'] for gid in sorted(all_ids)}
    delta = arms[treat]['levels'] - arms[ctl]['levels']
    assert sum(differences.values()) == delta
    screens.append(dict(label=label, base=base, treatment=treat, control=ctl, arms=arms, delta=delta,
        relative_delta=delta/arms[ctl]['levels'], per_game_delta=differences,
        games_ahead=sum(v>0 for v in differences.values()), games_tied=sum(v==0 for v in differences.values()), games_behind=sum(v<0 for v in differences.values()),
        source_verdict=result['verdict'], result=result))

hidden = []
names = {'D1':'Evidence', 'D2':'Control', 'D3':'B1 + c14', 'D4':'Control', 'D5':'B1', 'D7':'Control', 'D8':'Control', 'D9':'F1'}
for s in submissions:
    draw = s['description'].split(' ')[2] if s['description'].startswith('Hidden draw ') else None
    if draw not in names and s['ref'] != '56747536': continue
    hidden.append(dict(draw=draw or 'Base', arm=names.get(draw, 'Control'), ref=s['ref'], date=s['date'],
        status=s['status'].split('.')[-1], score=float(s['publicScore']) if s['publicScore'] else None,
        private_score=float(s['privateScore']) if s['privateScore'] else None,
        description=s['description']))
hidden.sort(key=lambda s:s['date'])
controls = [h['score'] for h in hidden if h['arm'] == 'Control' and h['score'] is not None]
queue = read('analysis/hidden-draws-20261010/queue.json')
for rel in ['analysis/hidden-draws-20261010/receipt-d9.json',
    'analysis/hidden-draws-20261010/receipt-d8.json',
    'analysis/hidden-draws-20261010/d9-f1/EXPERIMENT-CARD.md',
    'solvers/m2_f1/experiment/screen-2/EXPERIMENT-CARD.md']:
    source_paths.append(ROOT / rel)
manifest = {str(p.relative_to(ROOT)): dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size) for p in source_paths}
data = dict(as_of_utc=live['submissions']['utc'], scope='Six latest public screening kernels (2026-10-17) and hidden draw series through D9; hidden publicScore is an aggregate leaderboard score, not public-25 levels.',
    screens=screens, hidden=hidden, control_summary=dict(n=len(controls), mean=sum(controls)/len(controls), minimum=min(controls), maximum=max(controls)),
    queue=queue, live=live, source_manifest=manifest,
    caveats=['Public screens: 25 identical game IDs per arm, one pass, nominal 60-minute soft stop; cancelled games are expected at the stop.',
      'Hidden scores: independent draws, uncontrolled variation; no hidden per-game outcomes exposed.',
      'Screen-2 REWRITE_SLOWS automated label was voided in the experiment-card addendum; effect on speed is unresolved.',
      'Mechanism counters are read from saved RESULT.json; benchmark levels/game alignment are independently recomputed here.'])
(OUT/'data.json').write_text(json.dumps(data, indent=2))
with (OUT/'per-game.csv').open('w') as f:
    w=csv.writer(f); w.writerow(['game_id','screen','treatment_levels','control_levels','difference'])
    for s in screens:
        for gid,delta in s['per_game_delta'].items():
            w.writerow([gid,s['label'],s['arms'][s['treatment']]['per_game'][gid]['levels'],s['arms'][s['control']]['per_game'][gid]['levels'],delta])
print(json.dumps(dict(as_of=data['as_of_utc'],hidden=hidden[-3:],controls=data['control_summary'],screens=[{k:s[k] for k in ['label','delta','games_ahead','games_tied','games_behind']} for s in screens],live={k:v.get('stdout',v.get('error','')) for k,v in live.items() if k not in {'submissions','kernels'}}),indent=2))
