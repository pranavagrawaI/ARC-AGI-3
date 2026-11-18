"""Matched, bounded CodeAct model-repair replay. No environment actions are executed."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CAPTURE = ROOT / 'analysis/deployment-probes/20260905-q6-quick-host/raw/nooa-join-public3/solver-workspaces/ls20-170yp9tb'
BASELINE = ROOT / 'deployment/kaggle/q6-quick-probe-20260905'
CANDIDATE = ROOT / 'deployment/kaggle/q6-prediction-feedback-20260906'
MODEL = 'qwen/qwen3.8-27b'
KICKOFF = (
    'Resume from the recorded step 6 state and the saved worldmodel helper. '
    'Inspect the actual transition history, check the executable model against observations, '
    'and repair any contradictions you find. Keep encode(grid) and predict(z, action) usable. '
    'Preserve agent_top_row, agent_col and band_right as measured integer fields. '
    'Then submit one justified next action with a prediction through the normal action gates. '
    'Do not just describe a repair: save it in the helper. This is a paused replay; '
    'the probe stops on your first accepted submission. No successor to that submission is supplied.'
)

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def append(path, row):
    with path.open('a') as f:
        f.write(json.dumps(row) + '\n')

def seed(run_dir):
    """Only evidence through step six goes into the Actor's jailed workspace."""
    ipc = run_dir / 'ipc'
    helper = run_dir / 'team_nemo/shared/helpers'
    ipc.mkdir(parents=True)
    helper.mkdir(parents=True)
    for name in ('states', 'transitions'):
        selected = [r for r in rows(CAPTURE / 'ipc' / f'{name}.jsonl') if r['step'] <= 6]
        (ipc / f'{name}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in selected))
    shutil.copyfile(CAPTURE / 'team_nemo/shared/helpers/worldmodel.py', helper / 'worldmodel.py')
    # The current grid is available in states; do not copy final notes or later policy.
    (ipc / 'observations.json').write_text('{}\n')
    return ipc

class Budget:
    def __init__(self):
        self.reserved = 0.0
        self.calls = 0
        self.arm_calls = 0
    def reserve(self, size):
        # Byte count upper-bounds input token count; provider prices are ceilings.
        amount = size * .42 / 1_000_000 + 4096 * 3 / 1_000_000 + .002
        if size > 80000 or self.arm_calls >= 12 or self.reserved + amount > 1.0:
            raise RuntimeError('call/input/one-dollar reservation cap')
        self.reserved += amount
        self.calls += 1
        self.arm_calls += 1
        return amount

def credential():
    key = os.environ.get('OPENROUTER_API_KEY') or os.environ.get('ARC_LLM_API_KEY')
    if key and key != 'local-relay':
        return key
    for p in (ROOT / '.env', Path('/home/pranav/projects/arc-agi-3/.env')):
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                name, value = line.split('=', 1)
                if name.strip() in ('OPENROUTER_API_KEY', 'ARC_LLM_API_KEY'):
                    return value.strip().strip('\"\'')
    raise RuntimeError('No configured OpenRouter credential')

def worker_env(candidate, endpoint):
    env = {k:v for k,v in os.environ.items() if not ('KEY' in k or 'TOKEN' in k or 'SECRET' in k)}
    env.update(LITELLM_LOCAL_MODEL_COST_MAP='True', ARC_LLM_MODEL=MODEL,
        ARC_LLM_API_KEY='local-relay', ARC_LLM_BASE_URL=endpoint,
        ARC_LLM_EXTRA_BODY='{"reasoning":{"enabled":false}}', ARC_STALL_REVIEW='0',
        Q6_REPLAY_CANDIDATE=str(candidate), MPLCONFIGDIR=str(HERE / 'mpl'))
    return env

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='New output directory; never reused')
    parser.add_argument('--execute', action='store_true', help='Send bounded paid requests')
    args = parser.parse_args()
    out = Path(args.run).resolve()
    if out.exists():
        raise SystemExit('Run directory exists; use a new run identity')
    out.mkdir(parents=True)
    for candidate in (BASELINE, CANDIDATE):
        if not (candidate / 'chassis/solver_agent.py').exists():
            raise RuntimeError(f'Missing candidate: {candidate}')
    sources = [HERE / 'probe.py', HERE / 'worker.py', HERE / 'score.py', HERE / 'EXPERIMENT-CARD.md']
    sources += list(BASELINE.rglob('*.py')) + list(CANDIDATE.rglob('*.py'))
    sources += [CAPTURE / 'ipc' / n for n in ('states.jsonl','transitions.jsonl')]
    sources += [CAPTURE / 'team_nemo/shared/helpers/worldmodel.py']
    pins = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    (out / 'SOURCE-PIN.json').write_text(json.dumps(pins, indent=2)+'\n')
    shutil.copyfile(HERE / 'EXPERIMENT-CARD.md', out / 'EXPERIMENT-CARD.md')
    # ABBA order limits a simple provider/run-order confound, not model stochasticity.
    arms = [('baseline-a', BASELINE), ('candidate-a', CANDIDATE), ('candidate-b', CANDIDATE), ('baseline-b', BASELINE)]
    for arm, _ in arms:
        seed(out / arm)
    if not args.execute:
        print(json.dumps({'prepared': str(out), 'paid_calls': 0}))
        return
    key = credential()
    budget = Budget()
    state = {'arm': '', 'stop': None}
    def abort(reason):
        state['stop'] = reason
        (out / state['arm'] / 'ipc/ABORT').write_text(reason)
    class Relay(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            data = self.rfile.read(int(self.headers.get('Content-Length', '0')))
            try:
                if state['stop']:
                    raise RuntimeError(state['stop'])
                amount = budget.reserve(len(data))
            except RuntimeError as exc:
                abort(str(exc)); self.send_error(429, 'diagnostic cap'); return
            payload = json.loads(data)
            payload.update(model=MODEL, max_tokens=4096, stream=False, reasoning={'enabled':False},
                provider={'max_price':{'prompt':.42,'completion':3,'request':0}, 'allow_fallbacks':False})
            ident = {'arm':state['arm'], 'call':budget.arm_calls, 'time':time.time()}
            append(out/'requests.jsonl', {**ident, 'request':payload, 'reserved_dollars':amount})
            try:
                req = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',
                    data=json.dumps(payload).encode(), headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
                with urllib.request.urlopen(req, timeout=120) as response:
                    raw = response.read()
                parsed = json.loads(raw)
                append(out/'responses.jsonl', {**ident,'response':parsed})
                self.send_response(200); self.send_header('Content-Type','application/json'); self.end_headers(); self.wfile.write(raw)
            except Exception as exc:
                append(out/'responses.jsonl', {**ident,'error':type(exc).__name__})
                abort('upstream failure; no retry'); self.send_error(502,'upstream failure')
    server = HTTPServer(('127.0.0.1',0), Relay)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        for arm, candidate in arms:
            if any(sha(ROOT / p) != digest for p, digest in pins.items()):
                raise RuntimeError('Pinned source changed; do not continue this run')
            budget.arm_calls = 0
            state.update(arm=arm, stop=None)
            log = (out/arm/'worker.log').open('w')
            start = time.monotonic()
            proc = subprocess.Popen([sys.executable,str(HERE/'worker.py'),'--run-dir',str(out/arm)],
                env=worker_env(candidate,f'http://127.0.0.1:{server.server_port}/v1'),stdout=log,stderr=subprocess.STDOUT)
            try:
                proc.wait(timeout=360)
            except subprocess.TimeoutExpired:
                abort('360-second arm cap'); proc.terminate()
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            finally:
                log.close()
            append(out/'RESULTS.jsonl', {'arm':arm,'returncode':proc.returncode,'calls':budget.arm_calls,
                'stop':state['stop'],'wall_seconds':time.monotonic()-start,'total_reserved_dollars':budget.reserved})
            if state['stop'] or proc.returncode != 0:
                break
    finally:
        server.shutdown()
    subprocess.run([sys.executable,str(HERE/'score.py'),str(out)],check=True)

if __name__ == '__main__': main()
