# FLARE probe 4: reassembly ONLY (no Flare), then hold 120 s and exit. Survives -> Flare kills the
# container (2a/2b); dies -> the 107 GB local write does.
# (derived from stage-2b smoke) (2a died at 53 min with an empty log; disk ruled out by probe 3, so watch memory)
# FLARE stage-2a smoke on the Kaggle RTX PRO 6000 (card: ../SMOKE-CARD.md).
# Rebuild the gb202 package from six notebook-output parts, start Flare through the shipped glibc,
# check tool calls on logged requests, then replay the 2026-10-01 control traffic (the MTP sweep's
# harness) at S=10/14/16 and report emitted tok/s with the same estimator used for SGLang M3A.
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
            log(f'mem {m:.1f} GiB (peak {WATCH["peak_gib"]:.1f}); server tail: {tail}')
        time.sleep(2)
threading.Thread(target=watchdog, daemon=True).start()

def one(pattern):
    hits = glob.glob(f'/kaggle/input/**/{pattern}', recursive=True)
    assert len(hits) == 1, (pattern, hits)
    return hits[0]

# ---- 1. locate inputs ---------------------------------------------------------------------------
BUNDLE = os.path.dirname(one('ld-linux-x86-64.so.2'))
DATA = Path(os.path.dirname(one('replay.py')))
PARTS = [(one(f'flare-part-{i}'), json.load(open(one(f'flare-part-{i}.json')))) for i in range(6)]
assert [m['start'] for _, m in PARTS] == [sum(m['bytes'] for _, m in PARTS[:i]) for i in range(6)]
assert sum(m['bytes'] for _, m in PARTS) == TOTAL
RESULT['inputs'] = {'bundle': BUNDLE, 'replay_data': str(DATA),
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

# ---- 3. parallel reassembly with per-part hashes ----------------------------------------------
PKG = Path('/tmp/flare-model/qwen3.8-flash-next-tp1-c16-mtp5.safetensors')
PKG.parent.mkdir(parents=True, exist_ok=True)
with open(PKG, 'wb') as f:
    f.truncate(TOTAL)
part_ok = {}
def copy_part(path, meta):
    h = hashlib.sha256()
    fd = os.open(PKG, os.O_WRONLY)
    off = meta['start']
    try:
        with open(path, 'rb', buffering=0) as src:
            while True:
                b = src.read(64 << 20)
                if not b:
                    break
                h.update(b); os.pwrite(fd, b, off)
                os.posix_fadvise(src.fileno(), off - meta['start'], len(b), os.POSIX_FADV_DONTNEED)
                if (off - meta['start']) % (1 << 30) < len(b):   # every ~1 GiB: flush and drop our dirty pages
                    os.fdatasync(fd)
                    os.posix_fadvise(fd, meta['start'], off + len(b) - meta['start'], os.POSIX_FADV_DONTNEED)
                off += len(b)
    finally:
        os.close(fd)
    part_ok[meta['index']] = h.hexdigest() == meta['sha256'] and off == meta['end']
    log(f"part {meta['index']} copied ok={part_ok[meta['index']]} mem {mem_gib():.1f} GiB")
t = time.monotonic()
threads = [threading.Thread(target=copy_part, args=p) for p in PARTS]
[th.start() for th in threads]; [th.join() for th in threads]
RESULT['reassembly'] = {'peak_mem_gib': WATCH['peak_gib'], 'seconds': round(time.monotonic() - t, 1), 'parts_ok': part_ok,
                        'MBps': round(TOTAL / 1e6 / (time.monotonic() - t), 1)}
log(f'reassembly: {RESULT["reassembly"]}'); save()
assert all(part_ok.values()) and len(part_ok) == 6, 'part hash mismatch'

log(f'holding 120 s with the package on disk; mem {mem_gib():.1f} GiB')
time.sleep(120)
RESULT['survived'] = True; save()
log('probe 4 done: reassembly survived; removing package')
os.remove(PKG)
