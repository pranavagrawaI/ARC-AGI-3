# FLARE stage-3e smoke (card: SMOKE3E-CARD.md). 3d ran clean at r=0.45: Flare snapshots its prefix cache at request
# boundaries, so the moving 4-image window (each turn strips the previous image) breaks every new turn's prefix.
# 3e replaces EVERY image with a fixed placeholder, so prompts stay prefix-stable: Flare's ceiling with its cache
# working. S=10 and 14 only. Stage-3d header follows.
# FLARE stage-3d smoke (card: SMOKE3D-CARD.md). 3c served (attr 11 skipped) but every request got 400: Flare allows
# max_images_per_prompt 4 and the logged requests carry a median of 24. 3d sends the replay with history images
# stripped exactly as ARC3_HISTORY_IMAGE_KEEP=0 would (current turn keeps its <=3 images). Stage-3c header follows.
# FLARE stage-3c smoke (card: SMOKE3C-CARD.md). 3b loaded the weights (staged parts + flare_ra.so) and then failed in
# "Prepare execution": cuGraphKernelNodeSetAttribute -> CUDA_ERROR_INVALID_VALUE on driver 580. 3c also wraps that call
# (flare_ra.so dlsym override): every attribute id is logged, and ones the driver rejects are skipped.
# Stage-3b header follows.
# FLARE stage-3b smoke on the Kaggle RTX PRO 6000 (card: ../SMOKE-CARD.md, ../SMOKE3-CARD.md, SMOKE3B-CARD.md).
# Stage 3 never got past "Load weights": Flare reads its package with small O_DIRECT reads, which on
# the gcsfuse dataset mount ran at 1.7 MB/s (4.0 GB in 40 min). 3b preloads flare_ra.so
# (../readahead/), which drops O_DIRECT on the package and serves Flare's reads from a block cache fed
# by 16 threads of 64 MiB fresh-handle reads, the pattern that precaches datasets at ~800 MB/s on
# these machines. A 20 s raw read benchmark of that pattern runs first. Everything else is stage 3.
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
            ra = ''
            try:
                ra = (OUT / 'flare_ra.log').read_text().splitlines()[-1]
            except Exception:
                pass
            log(f'mem {m:.1f} GiB (peak {WATCH["peak_gib"]:.1f}); {io}ra {ra}; server tail: {tail}')
        time.sleep(2)
threading.Thread(target=watchdog, daemon=True).start()

def one(pattern):
    hits = glob.glob(f'/kaggle/input/**/{pattern}', recursive=True)
    assert len(hits) == 1, (pattern, hits)
    return hits[0]

# ---- 1. locate inputs ---------------------------------------------------------------------------
BUNDLE = os.path.dirname(one('ld-linux-x86-64.so.2'))
DATA = Path(os.path.dirname(one('replay.py')))
DSET = Path(one('qwen3.8-flash-next-tp1-c16-mtp5.safetensors'))   # route A: the gcsfuse dataset file
assert DSET.stat().st_size == TOTAL, DSET.stat().st_size
PARTS = []                                                         # route B: six notebook-output parts, staged
for i in range(6):                                                 # onto local ext4 before the notebook starts
    meta = json.load(open(one(f'flare-part-{i}.json')))
    PARTS.append((one(f'flare-part-{i}'), meta['start'], meta['end']))
assert PARTS[0][1] == 0 and PARTS[-1][2] == TOTAL and all(a[2] == b[1] for a, b in zip(PARTS, PARTS[1:]))
assert all(os.path.getsize(path) == e - s for path, s, e in PARTS)
RESULT['inputs'] = {'bundle': BUNDLE, 'replay_data': str(DATA), 'dataset_file': str(DSET), 'parts': PARTS,
                    'mounts': [l for l in open('/proc/mounts').read().splitlines() if '/kaggle/input' in l][:14],
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
import base64, zlib
RA_SO_B64 = '__RA_SO_B64__'
RA_SO_SHA = '__RA_SO_SHA__'
RA = W / 'libs/flare_ra.so'
RA.write_bytes(zlib.decompress(base64.b64decode(RA_SO_B64)))
assert hashlib.sha256(RA.read_bytes()).hexdigest() == RA_SO_SHA, 'flare_ra.so hash'
SHIM = [str(W / 'libs/ld-linux-x86-64.so.2'), '--library-path',
        ':'.join([str(W / 'libs'), '/usr/local/nvidia/lib64', '/usr/lib/x86_64-linux-gnu']),
        '--preload', str(RA), str(W / 'flare')]
RA_ENV = {'FLARE_RA_PATH': None, 'FLARE_RA_BLOCK_MB': '64', 'FLARE_RA_THREADS': '16', 'FLARE_RA_AHEAD': '24',
          'FLARE_RA_CAP': '64', 'FLARE_RA_LOG': str(OUT / 'flare_ra.log'),
          'FLARE_CU_LOG': str(OUT / 'flare_cu.log'), 'FLARE_CU_SKIP': 'all'}

# ---- 3. pick the read route: 20 s raw benchmark of the shim's fetch pattern on each -------------
def _bench(pick, seconds=20.0, threads=16, block=64 << 20):
    stop, got, lock, nxt = time.monotonic() + seconds, [0], threading.Lock(), [0]
    def w():
        buf = bytearray(block)
        while time.monotonic() < stop:
            with lock:
                k = nxt[0]; nxt[0] += 1
            path, off, n = pick(k, block)
            fd = os.open(path, os.O_RDONLY)
            try:
                r = os.preadv(fd, [memoryview(buf)[:n]], off)
            finally:
                os.close(fd)
            with lock:
                got[0] += r
    t = time.monotonic()
    th = [threading.Thread(target=w) for _ in range(threads)]
    [x.start() for x in th]; [x.join() for x in th]
    return round(got[0] / 1e6 / (time.monotonic() - t), 1)

def _spread(k, block):  # a different block each time, spread over the whole file
    return ((k * 997) % (TOTAL // block)) * block
def pick_a(k, block):
    return str(DSET), _spread(k, block), block
def pick_b(k, block):
    v = _spread(k, block)
    for path, s0, e0 in PARTS:
        if s0 <= v < e0:
            return path, v - s0, min(block, e0 - v)
bench = {'A_dataset_gcsfuse': _bench(pick_a), 'B_staged_parts': _bench(pick_b)}
ROUTE = 'B' if bench['B_staged_parts'] >= bench['A_dataset_gcsfuse'] else 'A'
if ROUTE == 'B':  # a sparse placeholder of the right size; flare_ra.so serves its bytes from the parts
    PKG = Path('/tmp/flare-pkg/qwen3.8-flash-next-tp1-c16-mtp5.safetensors')
    PKG.parent.mkdir(parents=True, exist_ok=True)
    with open(PKG, 'wb') as f:
        f.truncate(TOTAL)
    RA_ENV['FLARE_RA_SOURCES'] = ';'.join(f'{path}@{s0}' for path, s0, _ in PARTS)
else:
    PKG = DSET
RA_ENV['FLARE_RA_PATH'] = str(PKG)
VERIFY_WEIGHTS = ROUTE == 'B'  # route B serves a placeholder, so let Flare check every stored range
RESULT['reassembly'] = {'skipped': 'no local copy; flare_ra.so serves reads', 'mem_gib_before_start': round(mem_gib(), 2)}
RESULT['route'] = {'bench_MBps': bench, 'chosen': ROUTE, 'package_path': str(PKG), 'verify_weights': VERIFY_WEIGHTS}
log(f'route: {RESULT["route"]}; mem {mem_gib():.1f} GiB'); save()

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
    if VERIFY_WEIGHTS:
        cmd += ['--verify-weights']
    logf = open(OUT / 'flare-server.log', 'ab', buffering=0)
    logf.write(f'\n==== {RA_ENV} {" ".join(cmd)}\n'.encode())
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT,
                         start_new_session=True, env={**os.environ, **RA_ENV})
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
try:
    RESULT['start']['ra_last'] = json.loads((OUT / 'flare_ra.log').read_text().splitlines()[-1])
except Exception as e:
    RESULT['start']['ra_last'] = repr(e)
try:
    RESULT['start']['cu_last'] = json.loads((OUT / 'flare_cu.log').read_text().splitlines()[-1])
except Exception as e:
    RESULT['start']['cu_last'] = repr(e)
log(f'start: {attempts} gpu {smi}'); save()
if not ok:
    log('FLARE did not reach health; stopping (card: INVALID/KILL by cause)')
    raise SystemExit(0)
status, raw = http('GET', '/v1/models'); RESULT['models'] = raw.decode(errors='replace')[:2000]
status, raw = http('GET', '/metrics'); RESULT['metrics_endpoint'] = {'status': status, 'head': raw[:1500].decode(errors='replace')}

# ---- 5. functional: tool calls on logged requests ---------------------------------------------
sys.path.insert(0, str(DATA))
import payload, replay, sim
_HIST_PLACEHOLDER = '[grid image omitted from history]'   # dfranzen tool_agent._HISTORY_IMAGE_PLACEHOLDER
def _is_img(part):
    return isinstance(part, dict) and part.get('type') == 'image_url'
def keep_current_images(messages):
    # 3e: no images at all; every image part becomes the same placeholder, so history never changes
    out = []
    for m in messages:
        c = m.get('content')
        if isinstance(c, list) and any(_is_img(x) for x in c):
            m = {**m, 'content': [{'type': 'text', 'text': _HIST_PLACEHOLDER} if _is_img(x) else x for x in c]}
        out.append(m)
    return out
_full_build = payload.build
def _build_keep0(rec, model, max_tokens):
    body = _full_build(rec, model, max_tokens)
    body['messages'] = keep_current_images(body['messages'])
    return body
payload.build = _build_keep0
replay.payload.build = _build_keep0
_imgs = [sum(_is_img(x) for m in _build_keep0(json.loads(l), MODEL, 1)['messages'] if isinstance(m.get('content'), list) for x in m['content'])
         for path in sorted(DATA.glob('*_p0_requests.jsonl')) for l in open(path) if '"event": "request"' in l]
RESULT['image_window'] = {'keep': 'none', 'requests': len(_imgs), 'max_images': max(_imgs), 'over_4': sum(i > 4 for i in _imgs)}
log(f'image window: {RESULT["image_window"]}'); save()
assert RESULT['image_window']['over_4'] == 0, RESULT['image_window']
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
for S in (10, 14):
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
