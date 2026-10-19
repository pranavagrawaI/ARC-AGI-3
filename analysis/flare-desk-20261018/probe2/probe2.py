# FLARE probe 2 (after smoke 2a died with an empty log after 53 min while writing 107 GB to /tmp):
# can the package live in RAM instead? Records the memory cgroup limit, /dev/shm, memfd, the RAM
# accounting of one 16.6 GiB part held in memory, and whether flare can open a /proc/self/fd path.
# Exits cleanly in every branch so the log survives.
import ctypes, glob, json, os, shutil, subprocess, time
T0 = time.monotonic()
OUT = {}
def log(k, v):
    OUT[k] = v
    print(f'[{time.monotonic() - T0:6.1f}s] {k}: {json.dumps(v, default=str)[:1500]}', flush=True)
    json.dump(OUT, open('/kaggle/working/probe2.json', 'w'), indent=1, default=str)

def read(p):
    try:
        return open(p).read().strip()
    except Exception as e:
        return repr(e)

def meminfo():
    m = {l.split(':')[0]: int(l.split()[1]) // 1024 for l in open('/proc/meminfo') if l.split(':')[0] in
         ('MemTotal', 'MemAvailable', 'Cached', 'Shmem', 'MemFree')}
    m['cgroup_current_MiB'] = int(read('/sys/fs/cgroup/memory.current') or 0) // 2**20 if read('/sys/fs/cgroup/memory.current').isdigit() else read('/sys/fs/cgroup/memory.current')
    return m

log('cgroup', {k: read(f'/sys/fs/cgroup/{k}') for k in ('memory.max', 'memory.high', 'memory.swap.max', 'cgroup.controllers', 'io.max', 'pids.max')})
log('cgroup_v1', {k: read(f'/sys/fs/cgroup/memory/{k}') for k in ('memory.limit_in_bytes', 'memory.usage_in_bytes')})
log('meminfo_start', meminfo())
log('dev_shm', subprocess.run(['df', '-B1', '/dev/shm', '/tmp', '/'], capture_output=True, text=True).stdout)
log('mounts', [l for l in read('/proc/self/mountinfo').splitlines() if ' / ' in l.split(' - ')[0][:200] or 'shm' in l or 'overlay' in l][:12])
parts = sorted(glob.glob('/kaggle/input/**/flare-part-[0-9]', recursive=True))
log('parts_visible', parts)

# memfd: anonymous RAM file, sized like a package part
libc = ctypes.CDLL(None, use_errno=True)
fd = libc.memfd_create(b'flare-pkg', 0)
log('memfd_create', {'fd': fd, 'errno': ctypes.get_errno()})
if fd >= 0 and parts:
    t = time.time(); n = 0
    with open(parts[0], 'rb', buffering=0) as src:
        while True:
            b = src.read(64 << 20)
            if not b:
                break
            n += os.write(fd, b)
            if n % (4 << 30) < (64 << 20):
                log(f'copied_{n >> 30}GiB', {'s': round(time.time() - t, 1), **meminfo()})
    log('memfd_part0', {'bytes': n, 's': round(time.time() - t, 1), 'MBps': round(n / 1e6 / (time.time() - t), 1), **meminfo()})
    # can flare open a /proc/self/fd path (and a symlink to it)?
    W = '/tmp/flare'; os.makedirs(f'{W}/libs', exist_ok=True)
    B = os.path.dirname(glob.glob('/kaggle/input/**/ld-linux-x86-64.so.2', recursive=True)[0])
    for name in os.listdir(B):
        if name.startswith(('ld-linux', 'lib')):
            shutil.copy(f'{B}/{name}', f'{W}/libs/{name}')
    shutil.copy(f'{B}/flare', f'{W}/flare'); os.chmod(f'{W}/flare', 0o755); os.chmod(f'{W}/libs/ld-linux-x86-64.so.2', 0o755)
    shim = [f'{W}/libs/ld-linux-x86-64.so.2', '--library-path', f'{W}/libs:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu', f'{W}/flare']
    os.symlink(f'/proc/{os.getpid()}/fd/{fd}', '/tmp/pkg-link.safetensors')
    for label, path in (('procfd', f'/proc/self/fd/{fd}'), ('symlink', '/tmp/pkg-link.safetensors'), ('missing', '/tmp/does-not-exist.safetensors')):
        try:
            r = subprocess.run(shim + ['run', path, '--gpus', '0', '--trust-store', '/tmp/trust.toml'], pass_fds=(fd,),
                               capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
            log(f'flare_open_{label}', {'rc': r.returncode, 'out': r.stdout[-800:], 'err': r.stderr[-1500:]})
        except subprocess.TimeoutExpired as e:
            log(f'flare_open_{label}', {'timeout': True, 'out': (e.stdout or b'')[-800:], 'err': (e.stderr or b'')[-1500:]})
# /dev/shm write test (1 GiB) to see whether it is a usable tmpfs
try:
    t = time.time()
    with open('/dev/shm/probe.bin', 'wb') as f:
        for _ in range(16):
            f.write(b'\0' * (64 << 20))
    log('dev_shm_1GiB', {'ok': True, 's': round(time.time() - t, 2)}); os.remove('/dev/shm/probe.bin')
except Exception as e:
    log('dev_shm_1GiB', {'ok': False, 'err': repr(e)})
log('meminfo_end', meminfo())
