"""Executed after the donor definitions. No donor watchdog, restart or benchmark."""
import concurrent.futures
import signal
import statistics
import threading
import traceback

RESULT = {'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'status': 'started',
          'contrast': 'historical screen only; runtime plus quantization package', 'full_images': True}
proc = None


def save():
    tmp = OUT / 'result.json.tmp'
    tmp.write_text(json.dumps(RESULT, indent=2, default=str) + '\n')
    tmp.replace(OUT / 'result.json')


def digest(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def http(path, body=None, timeout=10):
    import urllib.error
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(VLLM_SERVER_URL + path, data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r: return r.status, r.read()
    except urllib.error.HTTPError as e: return e.code, e.read()


def tool_ok(resp):
    try:
        msg = resp['choices'][0]['message']
        calls = msg.get('tool_calls') or []
        return any(c['function']['name'] == 'python' and isinstance(json.loads(c['function']['arguments']).get('code'), str)
                   and json.loads(c['function']['arguments'])['code'].strip() for c in calls)
    except (KeyError, IndexError, TypeError, ValueError):
        return False


def estimates(events):
    phases = {x['name']: x['t'] for x in events if x['kind'] == 'phase'}
    lo, hi = phases['measure_start'], phases['measure_end']
    resp = [x for x in events if x['kind'] == 'response']
    good = [x for x in resp if x['status'] == 200]
    block = [x for x in good if lo <= x['t'] <= hi]
    completed = sum(x.get('completion_tokens') or 0 for x in block) / (hi - lo)
    overlap = sum((x.get('completion_tokens') or 0) * max(0, min(x['t'], hi) - max(x['t_send'], lo)) / (x['t'] - x['t_send'])
                  for x in good if x['t'] > x['t_send']) / (hi - lo)
    warm = next(x['criterion_met'] for x in events if x.get('name') == 'warmup_end')
    errors = sum(x['status'] != 200 for x in resp)
    missing = sum(not isinstance(x.get('completion_tokens'), int) for x in good)
    valid = warm and len(block) >= 30 and missing == 0 and errors / max(1, len(resp)) <= .02
    prompt = sum(x.get('prompt_tokens') or 0 for x in block)
    return {'valid': bool(valid), 'measure_s': hi-lo, 'responses': len(resp), 'responses_block': len(block),
            'errors': errors, 'missing_usage': missing, 'warmup_criterion_met': warm,
            'U_resp': completed, 'U_overlap': overlap,
            'median_latency_s': statistics.median([x['t']-x['t_send'] for x in block]) if block else None,
            'cached_fraction': sum(x.get('cached_tokens') or 0 for x in block)/prompt if prompt else None,
            'historical_U_resp_ratio': completed/518.2, 'historical_U_overlap_ratio': overlap/477.6}


def main():
    global proc
    save()
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True)
    (OUT / 'hardware.json').write_text(json.dumps({'gpu': gpu, 'cgroup_memory_max': Path('/sys/fs/cgroup/memory.max').read_text() if Path('/sys/fs/cgroup/memory.max').exists() else None}, indent=2))
    assert_expected_cuda_gpu()  # before unpack or weight allocation
    assert shutil.disk_usage('/tmp').free >= 30 * 1024**3, 'less than 30 GiB for extracted runtime'
    memory_limit = Path('/sys/fs/cgroup/memory.max')
    if memory_limit.exists() and memory_limit.read_text().strip().isdigit():
        assert int(memory_limit.read_text()) >= 160 * 1024**3, 'host memory cap below 160 GiB'
    assert digest(RUNTIME_DATASET / 'runtime-manifest.json') == '__RUNTIME_MANIFEST_SHA__', 'runtime manifest changed'
    manifest = read_runtime_manifest()
    RESULT['runtime'] = {k: manifest[k] for k in ('image', 'vllm_version', 'torch_version', 'nvcc')}
    save()
    data = [p.parent for p in Path('/kaggle/input').rglob('manifest.json') if (p.parent / 'replay.py').is_file()]
    assert len(data) == 1, data
    DATA = data[0]
    assert digest(DATA / 'manifest.json') == 'c21852dc33715d0efe6b4d62b70d7869040b859b4f1c2ec36ad1ea22e64b9713', 'replay manifest differs'
    hashes = json.loads((DATA / 'manifest.json').read_text())
    actual = {name: digest(DATA / name) for name in hashes}
    (OUT / 'replay-readback.json').write_text(json.dumps(actual, indent=2))
    assert actual == hashes, 'replay file identity mismatch'
    # Verify all pinned image blobs, rather than relying only on the donor's size check.
    for layer in manifest['selected_layers']:
        assert digest(RUNTIME_DATASET / layer['file']) == layer['sha256'], layer['file']
    manifest = install_image_runtime()
    dist = IMAGE_RUNTIME_ROOT / manifest['dist_packages']
    installed = {x['target']: digest(dist / x['target']) for x in manifest['patch']['overlay_files']}
    assert all(installed[x['target']] == x['sha256'] for x in manifest['patch']['overlay_files']), 'installed overlay mismatch'
    (OUT / 'installed-source-hashes.json').write_text(json.dumps(installed, indent=2))
    server_env = build_image_runtime_env(manifest)
    cmd = build_vllm_server_command() + ['--enable-prompt-tokens-details']
    RESULT.update(command=cmd, image_max_pixels=VLLM_IMAGE_MAX_PIXELS, slots=10)
    save()
    recorded_env = {k: v for k, v in server_env.items() if k.startswith(('VLLM_', 'TORCH_', 'PYTORCH_', 'CUDA_')) or k in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE')}
    (OUT / 'launch.json').write_text(json.dumps({'command': cmd, 'environment': recorded_env}, indent=2))
    log = (OUT / 'server.log').open('wb', buffering=0)
    # Same process group as the supervised worker: one cutoff reaps all descendants.
    proc = subprocess.Popen(cmd, env=server_env, stdout=log, stderr=subprocess.STDOUT)
    start = time.monotonic()
    while time.monotonic() - start < 2400:
        if proc.poll() is not None: raise RuntimeError(f'server died at startup: {proc.returncode}')
        try:
            if http('/health', timeout=3)[0] == 200: break
        except Exception: pass
        time.sleep(3)
    else: raise TimeoutError('startup exceeded 2400 seconds')
    RESULT['ready_seconds'] = time.monotonic() - start
    save()
    sys.path.insert(0, str(DATA))
    import payload, replay, re
    wl = json.loads((DATA / 'workload.json').read_text())
    bodies = replay.Bodies(DATA, wl['games'])
    records = [(g, bodies.get(g, len(wl['games'][g]['requests']) // 2)) for g in sorted(wl['games'])[:6]]
    lock = threading.Lock()
    rawlog = (OUT / 'responses.jsonl').open('a', buffering=1)

    def record_response(body, status, raw, elapsed, label):
        row = {'phase': label, 'status': status, 'elapsed_s': elapsed, 'request_sha256': hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(), 'raw': raw.decode(errors='replace')}
        with lock: rawlog.write(json.dumps(row) + '\n')

    def functional(pair):
        g, rec = pair
        body = payload.build(rec, SERVED_MODEL_NAME, 16384)
        start = time.monotonic()
        try: status, raw = http('/v1/chat/completions', body, timeout=900)
        except Exception as e: status, raw = -1, repr(e).encode()
        record_response(body, status, raw, time.monotonic()-start, 'functional:'+g)
        try: resp = json.loads(raw)
        except ValueError: resp = {}
        return {'game': g, 'status': status, 'code_ok': bool(status == 200 and tool_ok(resp)), 'usage': resp.get('usage')}

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool: rows = list(pool.map(functional, records))
    RESULT['functional'] = rows
    save()
    assert sum(x['code_ok'] for x in rows) >= 3, 'functional tool-call gate failed'
    if proc.poll() is not None: raise RuntimeError('server died after functional gate')
    replay.METRIC = re.compile(r'^(vllm:[a-z_]+)(\{[^}]*\})? ([0-9.eE+-]+|NaN)$')
    replay.WANTED = {'vllm:num_requests_running': 'running', 'vllm:num_requests_waiting': 'queue', 'vllm:generation_tokens_total': 'gen_tokens'}

    class RecordedReplay(replay.Replay):
        def _post(self, body):
            start = time.monotonic()
            status, raw = super()._post(body)
            record_response(body, status, raw, time.monotonic()-start, 'replay')
            return status, raw

    d = OUT / 'replay'; d.mkdir(exist_ok=True)
    (OUT / 'metrics-before.txt').write_bytes(http('/metrics')[1])
    r = RecordedReplay(workload=DATA/'workload.json', logs_dir=DATA,
                       alpha=json.loads((DATA/'alpha.json').read_text())['alpha'],
                       base_url=VLLM_SERVER_URL, model=SERVED_MODEL_NAME,
                       events_path=d/'events.jsonl', slots=10, request_timeout=900,
                       header={'engine': 'vllm-e975732', 'slots': 10, 'full_images': True})
    r.run(measure_s=480, warmup_max_s=600, settle_s=30)
    r.stop.set()
    (OUT / 'metrics-after.txt').write_bytes(http('/metrics')[1])
    result = estimates([json.loads(x) for x in (d/'events.jsonl').read_text().splitlines()])
    RESULT['replay'] = result
    ratio = result['historical_U_resp_ratio']
    RESULT['status'] = 'invalid' if not result['valid'] else ('candidate_for_matched_test' if ratio >= 1.30 else 'inconclusive' if ratio >= 1.10 else 'closed_configuration')
    save()
    print(json.dumps(RESULT, indent=2), flush=True)


if __name__ == '__main__':
    try: main()
    except Exception as e:
        RESULT.update(status='failed', error=repr(e), traceback=traceback.format_exc())
        save()
        raise
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try: proc.wait(timeout=20)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait(timeout=10)
