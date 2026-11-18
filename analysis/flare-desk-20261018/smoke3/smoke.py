# FLARE stage-3 smoke on the Kaggle RTX PRO 6000 (card: ../SMOKE-CARD.md, addendum in SMOKE3-CARD.md).
# 2a/2b died rebuilding the package on local disk; probe 5 showed in-place reads survive. So the
# package is now one file in the private dataset flare-gb202-package-20261020, and Flare reads it in
# place. Start Flare through the shipped glibc, check tool calls on logged requests, then replay the
# 2026-10-01 control traffic (the MTP sweep's harness) at S=10/14/16 and report emitted tok/s with
# the same estimator used for SGLang M3A. Everything after the package location is 2b unchanged,
# plus the Flare process's read bytes in the watchdog line (load rate from /kaggle/input).
import glob, hashlib, json, os, re, shutil, signal, subprocess, sys, threading, time, urllib.request
from pathlib import Path

T0 = time.monotonic()
DEADLINE = T0 + 110 * 60          # self-imposed session cap (card)
OUT = Path('/kaggle/working/flare'); OUT.mkdir(parents=True, exist_ok=True)
RESULT = {'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
TOTAL = 107011338717
PUBLISHER = 'local-inference-lab'
PORT = 8002
MODEL = 'flashnext'

def log(msg):
    line = f'[{time.monotonic() - T0:7.1f}s] {msg}'
    print(line, flush=True)
    with open(OUT / 'smoke.log', 'a') as f:
        f.write(line + '\n')

def save():
    (OUT / 'result.json').write_text(json.dumps(RESULT, indent=1, default=str))

def mem_gib():
    try:
        return int(open('/sys/fs/cgroup/memory.current').read()) / 2**30
    except Exception:
        return -1.0

WATCH = {'peak_gib': 0.0, 'killed': False, 'proc': None}
def watchdog():
    last = 0.0
    while True:
        m = mem_gib(); WATCH['peak_gib'] = max(WATCH['peak_gib'], m)
        if m > 160 and WATCH['proc'] is not None and WATCH['proc'].poll() is None and not WATCH['killed']:
            WATCH['killed'] = True
            log(f'WATCHDOG: cgroup memory {m:.1f} GiB > 160; killing flare')
            try:
                os.killpg(WATCH['proc'].pid, signal.SIGKILL)
            except Exception as e:
                log(f'watchdog kill failed: {e!r}')
        if time.monotonic() - last > 30:
            last = time.monotonic()
            tail = ''
            try:
                tail = (OUT / 'flare-server.log').read_bytes()[-300:].decode(errors='replace').replace(chr(10), ' | ')
            except Exception:
                pass
            io = ''
            try:
                if WATCH['proc'] is not None and WATCH['proc'].poll() is None:
                    rd = dict(l.split(': ') for l in open(f'/proc/{WATCH["proc"].pid}/io').read().splitlines())
                    io = f'flare rchar {int(rd["rchar"]) / 1e9:.1f} GB read_bytes {int(rd["read_bytes"]) / 1e9:.1f} GB; '
            except Exception:
                pass
            log(f'mem {m:.1f} GiB (peak {WATCH["peak_gib"]:.1f}); {io}server tail: {tail}')
        time.sleep(2)
threading.Thread(target=watchdog, daemon=True).start()

def one(pattern):
    hits = glob.glob(f'/kaggle/input/**/{pattern}', recursive=True)
    assert len(hits) == 1, (pattern, hits)
    return hits[0]

# ---- 1. locate inputs ---------------------------------------------------------------------------
BUNDLE = os.path.dirname(one('ld-linux-x86-64.so.2'))
DATA = Path(os.path.dirname(one('replay.py')))
PKG = Path(one('qwen3.8-flash-next-tp1-c16-mtp5.safetensors'))
assert PKG.stat().st_size == TOTAL, PKG.stat().st_size
RESULT['inputs'] = {'bundle': BUNDLE, 'replay_data': str(DATA), 'package': str(PKG),
                    'package_mount': [l for l in open('/proc/mounts').read().splitlines() if '/kaggle/input' in l][:6],
                    'replay_manifest_sha256': hashlib.sha256((DATA / 'manifest.json').read_bytes()).hexdigest()}
log(f'inputs located: {RESULT["inputs"]}')

# ---- 2. stage the binary behind the trixie loader ---------------------------------------------
W = Path('/tmp/flare'); (W / 'libs').mkdir(parents=True, exist_ok=True)
sums = dict(reversed(l.split('  ', 1)) for l in open(f'{BUNDLE}/SHA256SUMS').read().splitlines())
for name, digest in sums.items():
    if name == 'blob.bin':
        continue
    dst = W / ('flare' if name == 'flare' else f'libs/{name}')
    shutil.copy(f'{BUNDLE}/{name}', dst)
    assert hashlib.sha256(dst.read_bytes()).hexdigest() == digest, name
os.chmod(W / 'flare', 0o755); os.chmod(W / 'libs/ld-linux-x86-64.so.2', 0o755)
SHIM = [str(W / 'libs/ld-linux-x86-64.so.2'), '--library-path',
        ':'.join([str(W / 'libs'), '/usr/local/nvidia/lib64', '/usr/lib/x86_64-linux-gnu']), str(W / 'flare')]

# ---- 3. no reassembly: the package is read in place from the dataset mount -----------------
RESULT['reassembly'] = {'skipped': 'single-file dataset, read in place', 'mem_gib_before_start': round(mem_gib(), 2)}
log(f'package in place: {PKG} ({TOTAL} bytes); mem {mem_gib():.1f} GiB'); save()

# ---- 4. start Flare ----------------------------------------------------------------------------
STATE = Path('/tmp/flare-state'); STATE.mkdir(exist_ok=True)
FINGERPRINT = '937c0143f6856d3ee8ad164f62e3d804bd825d7a8750d0accb9a5849f703a8c1'  # sha256(publisher.json key)

def http(method, path, body=None, timeout=10):
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}{path}', method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()

PREFIX_CACHE = '64G'  # stage 3: SGLang's reference ran with its radix cache (replay cache fraction ~0.92)

def start(fingerprint, prefix_cache=PREFIX_CACHE):
    cmd = SHIM + ['run', str(PKG), '--gpus', '0', '--max-concurrency', '16', '--mtp-draft-tokens', '5',
                  '--embedding-table-storage', 'ram', '--trust-store', str(STATE / 'trust.toml'),
                  '--host', '127.0.0.1', '--port', str(PORT), '--served-model-name', MODEL,
                  '--dangerously-approve-key', fingerprint, '--dangerously-accept-publisher', PUBLISHER]
    if prefix_cache:
        cmd += ['--prefix-cache-memory', prefix_cache]
    logf = open(OUT / 'flare-server.log', 'ab', buffering=0)
    logf.write(f'\n==== {" ".join(cmd)}\n'.encode())
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT,
                         start_new_session=True)
    WATCH['proc'] = p
    return p

def wait_healthy(proc, limit_s):
    t = time.monotonic()
    while time.monotonic() - t < limit_s and time.monotonic() < DEADLINE:
        if proc.poll() is not None:
            return False, f'exited rc={proc.returncode}'
        for path in ('/v1/models', '/health'):
            try:
                if http('GET', path, timeout=5)[0] == 200:
                    return True, round(time.monotonic() - t, 1)
            except Exception:
                pass
        time.sleep(5)
    return False, 'health timeout'

def server_tail(n=4000):
    return (OUT / 'flare-server.log').read_bytes()[-n:].decode(errors='replace')

attempts = []
fp = FINGERPRINT
proc = start(fp)
ok, info = wait_healthy(proc, 40 * 60)
attempts.append({'fingerprint': fp, 'prefix_cache': PREFIX_CACHE, 'ok': ok, 'info': info})
if not ok and proc.poll() is not None:
    other = [h for h in re.findall(r'\b[0-9a-f]{64}\b', server_tail()) if h != FINGERPRINT]
    if other:  # Flare named the key it authenticated; accept that one once
        fp = other[-1]
        proc = start(fp)
        ok, info = wait_healthy(proc, 40 * 60)
        attempts.append({'fingerprint': fp, 'prefix_cache': PREFIX_CACHE, 'ok': ok, 'info': info})
if not ok and proc.poll() is not None:  # one recorded fallback: no prefix cache (a lower bound on r)
    proc = start(fp, prefix_cache=None)
    ok, info = wait_healthy(proc, 40 * 60)
    attempts.append({'fingerprint': fp, 'prefix_cache': None, 'ok': ok, 'info': info})
RESULT['start'] = {'attempts': attempts, 'server_tail': server_tail(), 'watchdog': dict((k, v) for k, v in WATCH.items() if k != 'proc')}
smi = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader'],
                     capture_output=True, text=True).stdout.strip()
RESULT['start']['gpu_memory'] = smi
log(f'start: {attempts} gpu {smi}'); save()
if not ok:
    log('FLARE did not reach health; stopping (card: INVALID/KILL by cause)')
    raise SystemExit(0)
status, raw = http('GET', '/v1/models'); RESULT['models'] = raw.decode(errors='replace')[:2000]
status, raw = http('GET', '/metrics'); RESULT['metrics_endpoint'] = {'status': status, 'head': raw[:1500].decode(errors='replace')}

# ---- 5. functional: tool calls on logged requests ---------------------------------------------
sys.path.insert(0, str(DATA))
import payload, replay, sim
records = []
for path in sorted(DATA.glob('*_p0_requests.jsonl'))[:6]:
    with open(path) as f:
        reqs = [json.loads(l) for l in f if '"event": "request"' in l]
    records += [reqs[len(reqs) // 2]]          # mid-game: long context, retained functions
STRIP = []
def build(rec, max_tokens):
    body = payload.build(rec, MODEL, max_tokens)
    for k in STRIP:
        body.pop(k, None)
    return body
def probe_once(rec):
    body = build(rec, 16 * 1024)
    t = time.monotonic()
    status, raw = http('POST', '/v1/chat/completions', body, timeout=900)
    row = {'status': status, 'seconds': round(time.monotonic() - t, 1)}
    try:
        resp = json.loads(raw)
        msg = resp['choices'][0]['message']; calls = msg.get('tool_calls') or []
        args = [json.loads(c['function']['arguments']) for c in calls]
        row.update(finish=resp['choices'][0].get('finish_reason'), usage=resp.get('usage'),
                   tool_calls=len(calls), names=[c['function']['name'] for c in calls],
                   code_ok=bool(args) and all(isinstance(a.get('code'), str) and a['code'].strip() for a in args),
                   reasoning_chars=len(msg.get('reasoning_content') or msg.get('reasoning') or ''),
                   content_head=(msg.get('content') or '')[:300], code_head=(args[0].get('code', '')[:300] if args else ''))
    except Exception as e:
        row['parse_error'] = repr(e)[:300]; row['raw_head'] = raw[:600].decode(errors='replace')
    return row

first = probe_once(records[0])                    # one request alone: also the single-stream speed
if first['status'] == 400:                        # unknown-field rejection: strip vendor extras once
    STRIP[:] = ['top_k', 'chat_template_kwargs']
    first = probe_once(records[0])
functional = [first]
rows = [None] * (len(records) - 1)
def _run(i):
    rows[i] = probe_once(records[i + 1])
ths = [threading.Thread(target=_run, args=(i,)) for i in range(len(rows))]
[th.start() for th in ths]; [th.join() for th in ths]
functional += rows
for row in functional:
    log(f'functional: {row}')
RESULT['functional'] = {'rows': functional, 'stripped_fields': STRIP}; save()
if STRIP:
    _orig = payload.build
    def _stripped(rec, model, max_tokens):
        body = _orig(rec, model, max_tokens)
        for k in STRIP:
            body.pop(k, None)
        return body
    payload.build = _stripped
    replay.payload.build = _stripped

# ---- 6. closed-loop replay at S = 10, 14, 16 ---------------------------------------------------
def u_estimates(events_path):
    ev = [json.loads(l) for l in open(events_path)]
    ph = {e['name']: e['t'] for e in ev if e['kind'] == 'phase'}
    m0, m1 = ph.get('measure_start'), ph.get('measure_end')
    if m0 is None or m1 is None:
        return {'valid': False}
    resp = [e for e in ev if e['kind'] == 'response' and e['status'] == 200]
    inb = [e for e in resp if m0 <= e['t'] <= m1]
    tok = sum(e.get('completion_tokens') or 0 for e in inb)
    ov = sum((e.get('completion_tokens') or 0) * max(0, min(e['t'], m1) - max(e['t_send'], m0)) / (e['t'] - e['t_send'])
             for e in resp if e['t'] > e['t_send'])
    fin = {}
    for e in (x for x in ev if x['kind'] == 'response'):
        k = f"{e['status']}:{e.get('finish_reason')}"; fin[k] = fin.get(k, 0) + 1
    return {'valid': True, 'block_s': m1 - m0, 'responses_block': len(inb), 'U_resp': tok / (m1 - m0),
            'U_overlap': ov / (m1 - m0), 'mean_latency_s': sum(e['t'] - e['t_send'] for e in inb) / max(1, len(inb)),
            'finish': fin, 'errors': sum(1 for e in ev if e['kind'] == 'response' and e['status'] != 200),
            'early_stop': sum(1 for e in resp if e.get('finish_reason') == 'stop'
                              and (e.get('completion_tokens') or 0) < e['recorded_completion'])}

alpha = json.loads((DATA / 'alpha.json').read_text())['alpha']
RESULT['replay'] = {}
for S in (10, 14, 16):
    if DEADLINE - time.monotonic() < 16 * 60:
        log(f'S={S}: skipped, {int((DEADLINE - time.monotonic()) / 60)} min left'); break
    d = OUT / f'S{S}'; d.mkdir(exist_ok=True)
    r = replay.Replay(workload=DATA / 'workload.json', logs_dir=DATA, alpha=alpha,
                      base_url=f'http://127.0.0.1:{PORT}', model=MODEL, events_path=d / 'events.jsonl',
                      slots=S, header={'server': 'flare', 'slots': S, 'stripped_fields': STRIP})
    before = set(threading.enumerate())
    r.run(measure_s=480.0, warmup_max_s=600.0, settle_s=30.0)
    time.sleep(5)
    RESULT['replay'][S] = u_estimates(d / 'events.jsonl')
    try:
        RESULT['replay'][S]['summarize'] = replay.summarize(d / 'events.jsonl')
    except Exception as e:
        RESULT['replay'][S]['summarize'] = repr(e)[:300]
    log(f'S={S}: {RESULT["replay"][S]}'); save()
    # let the stopped replay's in-flight requests finish before the next level starts
    t = time.monotonic()
    while time.monotonic() - t < 900 and [th for th in threading.enumerate() if th not in before]:
        time.sleep(5)
    log(f'S={S}: drained in {time.monotonic() - t:.0f}s')

SGLANG_M3A = {'U_resp': 518.2, 'U_overlap': 477.6}   # 2026-10-08 retry1, same harness, S=10
r10 = RESULT['replay'].get(10, {})
if r10.get('valid'):
    RESULT['ratio_vs_sglang_m3a'] = {k: r10[k] / v for k, v in SGLANG_M3A.items()}
log(f'ratio vs SGLang M3A: {RESULT.get("ratio_vs_sglang_m3a")}')
RESULT['server_tail_end'] = server_tail(6000)
os.killpg(proc.pid, signal.SIGTERM)
save()
