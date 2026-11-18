"""Copy of ../f1x-hidden-zero-diagnosis-20261021/faults/run_competition_cpu.py plus: --idle-close S (the gateway
closes the scorecard after S s with no action; later actions get HTTP 400, per Kaggle discussion 751598) and
execution of a "# M2 KEEPALIVE" cell after the customization cell.

Run a notebook's own competition-branch cells on CPU, against a local 110-game competition Arcade and a
controllable model endpoint. Fault-injection harness for the hidden-run silent zero.

usage: run_competition_cpu.py NOTEBOOK REPLICA_DIR ENV_FILES OUT_DIR --llm {dead,http500,http400,process} [K=V ...]
       [--port N] [--server-args 'fake_llm args'] [--startup-timeout S] [--kill-at S] [--per-game-s S]

Executed from the notebook, unchanged except where noted:
  env cell (TRUE_SUBMISSION forced by KAGGLE_IS_COMPETITION_RERUN=1; '!' shell lines and the git apply dropped,
  the replica is already patched), source-path cell, pickle cell, customization cell, and the run cell
  (competition branch). Not executed: precache, pip, SGLang launch, system monitor, diagnostics display.
The SGLang server is replaced by the --llm endpoint on the notebook's port 8001.
"""
import ast
import asyncio
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

nb_path, replica, env_files, out = map(Path, sys.argv[1:5])
args = sys.argv[5:]
llm = args[args.index('--llm') + 1]
overrides = [a for i, a in enumerate(args) if '=' in a and not a.startswith('--') and not args[i - 1].startswith('--')]
out.mkdir(parents=True, exist_ok=True)
T0 = time.time()


def mark(event, **kw):
    row = {'t': round(time.time() - T0, 2), 'event': event, **kw}
    with (out / 'lifecycle.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
    print('LIFECYCLE', json.dumps(row), flush=True)


code = [''.join(c['source']) for c in json.loads(nb_path.read_text())['cells'] if c['cell_type'] == 'code']
def cell(prefix_or_marker):
    return next(c for c in code if prefix_or_marker in c)

env_cell = cell("NOTEBOOK_START_TIME = time.time()")
env_cell = '\n'.join(l for l in env_cell.splitlines()
                     if not l.lstrip().startswith('!') and 'git", "apply"' not in l and "print('harness patch" not in l)
env_cell = env_cell.replace("BUNDLE_DIR = Path('/kaggle/taaf-kaggle-source-share')", f"BUNDLE_DIR = Path({str(replica)!r})")
env_cell = env_cell.replace('WORKING_DIR = Path("/kaggle/working")', f'WORKING_DIR = Path({str(out)!r})')
os.environ['KAGGLE_IS_COMPETITION_RERUN'] = '1'

ns = {'__name__': '__notebook__'}
exec(compile(env_cell, '<env-cell>', 'exec'), ns)
assert ns['TRUE_SUBMISSION'] is True
for kv in overrides:
    k, v = kv.split('=', 1); os.environ[k] = v
mark('env', true_submission=ns['TRUE_SUBMISSION'], overrides=overrides,
     max_failures=os.environ.get('ARC3_MAX_ANALYZER_FAILURES'), grace=os.environ.get('ARC3_HTTP_RETRY_INITIAL_SECONDS'))
exec(compile(cell('def _source_path_entries'), '<source-cell>', 'exec'), ns)

# ---- model endpoint stand-in ----
def opt(name, default=None):
    return args[args.index(name) + 1] if name in args else default
port = int(opt('--port', ns['SERVED_MODEL_PORT']))
if port != ns['SERVED_MODEL_PORT']:  # keep concurrent harness runs apart; the notebook reads these at import
    ns['SERVED_MODEL_PORT'] = port
    for k in ('LOCAL_ANALYZER_BASE_URL', 'OPENAI_BASE_URL'):
        os.environ[k] = f'http://127.0.0.1:{port}/v1'
calls = {'n': 0}
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _reply(self, code, body):
        b = json.dumps(body).encode(); self.send_response(code)
        self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def do_GET(self): self._reply(200, {'ok': True})
    def do_POST(self):
        calls['n'] += 1
        n = int(self.headers.get('Content-Length') or 0); self.rfile.read(n)
        self._reply(400 if llm == 'http400' else 500, {'error': 'injected'})
if llm in ('http500', 'http400'):
    srv = ThreadingHTTPServer(('127.0.0.1', port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
elif llm == 'process':
    # The SGLang launch is replaced by fake_llm.py with --server-args; then the notebook's own startup wait loop
    # (the tail of its server cell) and, if present, the guard supervisor cell run unchanged.
    import shlex, subprocess, urllib.request
    server_cell = cell('# Paste this entire file into the launcher cell.')
    wait_loop = server_cell[server_cell.index("def show_log_tail():"):]
    ns.update(subprocess=subprocess, urllib=urllib, shlex=shlex, Path=Path, json=json,
              LOG=str(out / 'serve.log'), env=os.environ.copy(),
              args=[sys.executable, str(Path(__file__).resolve().parents[1] / 'f1x-hidden-zero-diagnosis-20261021/faults/fake_llm.py'), '--port', str(port),
                    *shlex.split(opt('--server-args', ''))])
    ns['SERVER_STARTUP_TIMEOUT'] = float(opt('--startup-timeout', ns['SERVER_STARTUP_TIMEOUT']))
    with open(ns['LOG'], 'ab', buffering=0) as logf:
        ns['proc'] = subprocess.Popen(ns['args'], env=ns['env'], stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    ns['log_offset'] = 0
    exec(compile(wait_loop, '<server-wait-loop>', 'exec'), ns)
    mark('server_wait_done', alive=ns['proc'].poll() is None, code=ns['proc'].poll())
    guard_cells = [c for c in code if c.startswith('# M2 GUARD supervise')]
    if guard_cells:
        exec(compile(guard_cells[0], '<guard-supervise-cell>', 'exec'), ns)
        mark('guard_supervisor_started')
    if opt('--kill-at'):  # crash the (current) server at a fixed time from start, as an external fault
        def _kill(at=float(opt('--kill-at'))):
            time.sleep(at)
            p = ns.get('_guard', {}).get('proc', ns['proc'])
            mark('injected_kill', pid=p.pid); p.kill()
        threading.Thread(target=_kill, daemon=True).start()
elif llm != 'dead':
    raise SystemExit(f'--llm {llm} not implemented')
mark('llm_endpoint', mode=llm)


# ---- gateway idle close (harness-only stand-in for the real gateway's 15-min inactivity cleanup) ----
idle = {'idle_s': float(opt('--idle-close')) if opt('--idle-close') else None, 'opened': None, 'last': None,
        'closed_at': None, 'accepted': 0, 'rejected': 0, 'resets_accepted': 0}
if idle['idle_s']:
    from arc_agi import server as _arc_server
    _create_app = _arc_server.create_app
    _idle_lock = threading.Lock()
    def _create_app_idle(*a, **k):
        app, api = _create_app(*a, **k)
        inner = app.wsgi_app
        def mw(environ, start_response):
            path, now = environ.get('PATH_INFO', ''), time.time()
            with _idle_lock:
                if idle['opened'] and idle['closed_at'] is None and now - idle['last'] > idle['idle_s']:
                    idle['closed_at'] = now
                    mark('gateway_idle_close', idle_s=idle['idle_s'], last_action_t=round(idle['last'] - T0, 1))
                if path.startswith('/api/cmd/'):
                    if idle['closed_at'] is not None:
                        idle['rejected'] += 1
                        start_response('400 BAD REQUEST', [('Content-Type', 'application/json')])
                        return [b'{"error": "scorecard closed after inactivity"}']
                    idle['accepted'] += 1; idle['last'] = now
                    idle['resets_accepted'] += path.endswith('/RESET')
                elif path == '/api/scorecard/open':
                    idle['opened'] = idle['last'] = now
                    mark('scorecard_open')
            return inner(environ, start_response)
        app.wsgi_app = mw
        return app, api
    _arc_server.create_app = _create_app_idle

# ---- local competition gateway (taaf's own simulator, 110 unique clones) ----
# The bundle's competition_arcade imports an unshipped re_arc; load the audited simulator (hardcoded 25 IDs) by file.
import importlib.util
_sim = Path(__file__).resolve().parents[2] / 'analysis/m2-winners-20261001/audit/dfranzen-repo/tufa-arc-agi-framework/src/taaf/competition_arcade.py'
_spec = importlib.util.spec_from_file_location('probe_competition_arcade', _sim)
_mod = importlib.util.module_from_spec(_spec); sys.modules[_spec.name] = _mod; _spec.loader.exec_module(_mod)
gw = _mod.CompetitionArcadeServer.official_110()
gw.environments_dir = str(env_files)
gw.start()
os.environ['ARC_BASE_URL'] = gw.base_url + '/'
os.environ['ARC_API_KEY'] = gw.api_key
mark('gateway', url=gw.base_url)

exec(compile(cell('with open(BUNDLE_DIR / "deploy_target.pkl"'), '<pickle-cell>', 'exec'), ns)
exec(compile(cell('bm.solver.max_runtime_s_per_game'), '<custom-cell>', 'exec'), ns)
for c in code:
    if c.startswith('# M2 KEEPALIVE'):
        exec(compile(c, '<keepalive-cell>', 'exec'), ns); mark('keepalive_armed')
if opt('--per-game-s'):  # harness-only shortening of the production per-game cap, recorded in lifecycle
    ns['bm'].solver.max_runtime_s_per_game = float(opt('--per-game-s'))
mark('config', concurrency=ns['bm'].solver.concurrency, per_game_s=ns['bm'].solver.max_runtime_s_per_game,
     save_request_logs=ns['bm'].solver.save_request_logs)

run_cell = cell("print('Starting benchmark...')")
flags = ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
async def main():
    mark('run_start')
    try:
        r = eval(compile(run_cell, '<run-cell>', 'exec', flags=flags), ns)
        if asyncio.iscoroutine(r): await r
        mark('run_cell_returned_normally')
    except BaseException as e:
        mark('run_cell_raised', error=f'{type(e).__name__}: {e}')
        raise
    finally:
        bm = ns['bm']
        rows = [{'game_id': r.game_id, 'state': r.state, 'actions': len(r.history), 'levels': r.levels_completed,
                 'note': r.solver_note} for r in bm.game_runs]
        notes = {}
        for r in rows: notes[str(r['note'])[:60]] = notes.get(str(r['note'])[:60], 0) + 1
        final = {'games': len(rows), 'levels': sum(r['levels'] for r in rows),
                 'games_with_actions': sum(r['actions'] > 0 for r in rows),
                 'states': {s: sum(r['state'] == s for r in rows) for s in {r['state'] for r in rows}},
                 'notes': notes, 'llm_posts': calls['n'], 'wall_s': round(time.time() - T0, 1),
                 'idle': {k: (round(v - T0, 1) if k in ('opened', 'last', 'closed_at') and v else v) for k, v in idle.items()},
                 'keepalive_resets': ns.get('_ka', {}).get('resets'), 'submission_parquet_written_by_notebook': (out / 'submission.parquet').exists()}
        (out / 'final.json').write_text(json.dumps({'summary': final, 'games': rows}, indent=1))
        mark('final', **final)
        gw.stop()
asyncio.run(main())
