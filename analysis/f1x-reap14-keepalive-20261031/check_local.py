"""Offline checks of the actual inserted cell, source and budget branches."""
import ast
import datetime as dt
import json
import sys
import types
from prepare import expected, PARENT, HERE, SLUG
from run import observation_bounds

nb = expected()
parent = json.loads(PARENT.read_text())
assert json.loads((HERE / 'kernel' / (SLUG + '.ipynb')).read_text()) == nb
cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
ka = next(c for c in cells if c.startswith('# M2 KEEPALIVE'))
assert not any('M2 KEEPALIVE' in ''.join(c['source']) for c in parent['cells'])


def exercise_keepalive(start, healthy_at, hold_max=9000, stopped=False):
    module = types.ModuleType('d21_fixture_solver')
    sys.modules[module.__name__] = module
    class Session:
        def _play_inner(self):
            self.stock_calls += 1
        def should_stop(self):
            return stopped
        def _execute_auto_reset(self):
            self.resets += 1
    module._HarnessGameSession = Session
    solver_type = type('Solver', (), {'__module__': module.__name__})
    namespace = {'bm': types.SimpleNamespace(solver=solver_type()), 'SERVED_MODEL_PORT': 8001, 'NOTEBOOK_START_TIME': 0}
    exec(compile(ka, '<actual-keepalive-cell>', 'exec'), namespace)
    clock = [start]
    namespace['_ka_time'] = types.SimpleNamespace(time=lambda: clock[0], monotonic=lambda: clock[0], sleep=lambda s: clock.__setitem__(0, clock[0] + s))
    class Health:
        status = 200
        def __enter__(self):
            if clock[0] < healthy_at:
                raise OSError('loading')
            return self
        def __exit__(self, *args):
            pass
    namespace['_ka_url'] = types.SimpleNamespace(urlopen=lambda *a, **kw: Health())
    namespace['_KA_HOLD_MAX_S'] = hold_max
    session = Session()
    session.stock_calls = session.resets = 0
    session.game = types.SimpleNamespace(game_run=types.SimpleNamespace(game_id='fixture'))
    session._play_inner()
    return session, clock[0], namespace['_ka']


fast, time_fast, state = exercise_keepalive(720, 600)
assert fast.stock_calls == 1 and fast.resets == 0 and time_fast == 720 and state['ready']
slow, time_slow, state = exercise_keepalive(720, 1505)
assert slow.stock_calls == 1 and slow.resets == 3 and time_slow == 1505 and state['ready']
halted, time_halted, _ = exercise_keepalive(720, 9000, stopped=True)
assert halted.stock_calls == 1 and halted.resets == 0 and time_halted == 720
bounded, end, _ = exercise_keepalive(8995, 99999)
assert bounded.stock_calls == 1 and end == 9000

# Execute the parent customization branch with the exact unchanged source.
custom = next(c for c in cells if c.startswith('# Make one-off changes'))
for submission in (True, False):
    solver = types.SimpleNamespace()
    bm = types.SimpleNamespace(solver=solver)
    ns = {'TRUE_SUBMISSION': submission, 'USE_PRIORITY_SCHEDULING': True, 'bm': bm, 'target': types.SimpleNamespace(max_runtime_s=32400)}
    exec(compile(custom, '<actual-customization>', 'exec'), ns)
    assert solver.concurrency == (120 if submission else 25)
    assert solver.max_runtime_s_per_game == (31920 if submission else 7254)
    assert solver.save_request_logs is (not submission)

env = next(c for c in cells if 'setup_env = {' in c)
assert "'ARC3_MAX_ACTIVE_STREAMS': 14" in env
assert "'LOCAL_ANALYZER_CONTEXT_WINDOW': (116+16)*1024" in env
assert "'LOCAL_ANALYZER_MAX_OUTPUT': 16*1024" in env
assert 'ARC3_ANIMATION_IMAGE' not in env
run = next(c for c in cells if c.startswith("print('Starting benchmark...')"))
soft = run[run.index('soft_end = None'):run.index('# Play the benchmark;')]
for submission in (True, False):
    ns = {'TRUE_SUBMISSION': submission, 'target': types.SimpleNamespace(max_runtime_s=32400), 'datetime': dt.datetime, 'timedelta': dt.timedelta}
    exec(compile(soft, '<actual-soft-end>', 'exec'), ns)
    assert (ns['soft_end'] is None) is submission
assert 'seconds=60*60' in soft
serve = next(c for c in cells if c.startswith('# Paste this entire file'))
assert 'if not TRUE_SUBMISSION and time.time() - NOTEBOOK_START_TIME > LATE_READY_LIMIT:' in serve
assert 'LATE_READY_LIMIT = 1300' in serve
assert "target_receipt['target_experts'] == (448 if PRUNE_MODE == 'pruned' else 512)" in serve
assert "draft_receipt['draft_experts'] == 512" in serve

# Real observer calculation: an85minute queue must not consume the runtime bound.
accepted = '2026-10-30T08:00:00+00:00'
rows = [{'utc': accepted, 'status': 'KernelWorkerStatus.QUEUED'}, {'utc': '2026-10-30T09:25:00+00:00', 'status': 'KernelWorkerStatus.RUNNING'}]
assert observation_bounds(rows, accepted) == dt.datetime.fromisoformat('2026-10-30T11:35:00+00:00').timestamp()
assert observation_bounds(rows[:1], accepted) == dt.datetime.fromisoformat('2026-10-30T13:00:00+00:00').timestamp()
assert observation_bounds([{'utc': accepted, 'status': None}] + rows, accepted) == observation_bounds(rows, accepted)
assert observation_bounds(json.loads(json.dumps(rows)), accepted) == observation_bounds(rows, accepted)
print('PASS: exact one-cell insertion; actual keepalive fast/slow/stop/hold-cap; competition/public runtime branches; context/experts/14 streams;85min queue separation and restart clock')
