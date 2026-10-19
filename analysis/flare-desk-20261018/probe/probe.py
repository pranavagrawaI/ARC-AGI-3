# FLARE portability probe for the Kaggle RTX PRO 6000 session (no model package needed).
# Answers, before any 107 GB upload: does the binary start under a shipped glibc, can it load
# libcuda and detect gb202, are pinned host registrations (RAM-backed PLE) and io_uring (disk
# I/O) allowed in this container, and how fast does /kaggle/input read.
import ctypes, glob, hashlib, json, mmap, os, platform, resource, shutil, subprocess, time

OUT = {}
src = glob.glob('/kaggle/input/**/flare-probe-bundle-20261018', recursive=True) or \
      [os.path.dirname(p) for p in glob.glob('/kaggle/input/**/SHA256SUMS', recursive=True)]
SRC = src[0]
OUT['bundle'] = SRC

def sh(cmd, timeout=120):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {'rc': r.returncode, 'out': r.stdout[-6000:], 'err': r.stderr[-6000:]}
    except Exception as e:
        return {'exc': repr(e)}

# --- container facts ----------------------------------------------------------
OUT['uname'] = platform.uname()._asdict()
OUT['ldd'] = sh(['ldd', '--version'])['out'].splitlines()[:1]
OUT['id'] = sh(['id'])['out'].strip()
status = open('/proc/self/status').read().splitlines()
OUT['proc_status'] = [l for l in status if l.split(':')[0] in ('CapEff', 'CapBnd', 'Seccomp', 'NoNewPrivs')]
OUT['memlock_before'] = resource.getrlimit(resource.RLIMIT_MEMLOCK)
try:
    resource.setrlimit(resource.RLIMIT_MEMLOCK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    OUT['memlock_raise'] = 'ok'
except Exception as e:
    OUT['memlock_raise'] = repr(e)
OUT['memlock_after'] = resource.getrlimit(resource.RLIMIT_MEMLOCK)
try:
    OUT['io_uring_disabled_sysctl'] = open('/proc/sys/kernel/io_uring_disabled').read().strip()
except Exception as e:
    OUT['io_uring_disabled_sysctl'] = repr(e)
OUT['nvidia_smi'] = sh(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,compute_cap', '--format=csv,noheader'])
OUT['libcuda'] = sh(['bash', '-lc', 'ldconfig -p | grep -E "libcuda.so|libnvidia-ml.so"; ls /usr/local/nvidia/lib64 2>/dev/null | head'])
OUT['meminfo'] = [l for l in open('/proc/meminfo').read().splitlines() if l.startswith(('MemTotal', 'MemAvailable'))]
OUT['df'] = sh(['df', '-h', '/', '/tmp', '/kaggle/input', '/kaggle/working'])['out']

# --- io_uring_setup(8, params) ------------------------------------------------
libc = ctypes.CDLL(None, use_errno=True)
params = ctypes.create_string_buffer(120)
fd = libc.syscall(425, 8, params)
OUT['io_uring_setup'] = {'fd': fd, 'errno': ctypes.get_errno(), 'strerror': os.strerror(ctypes.get_errno()) if fd < 0 else ''}
if fd >= 0:
    os.close(fd)

# --- pinned host registration, the RAM-backed PLE path ------------------------
def cuda_register():
    res = {}
    cu = ctypes.CDLL('libcuda.so.1')
    res['cuInit'] = cu.cuInit(0)
    dev = ctypes.c_int(); res['cuDeviceGet'] = cu.cuDeviceGet(ctypes.byref(dev), 0)
    ver = ctypes.c_int(); cu.cuDriverGetVersion(ctypes.byref(ver)); res['driver_api_version'] = ver.value
    ctx = ctypes.c_void_p(); res['cuDevicePrimaryCtxRetain'] = cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev)
    res['cuCtxSetCurrent'] = cu.cuCtxSetCurrent(ctx)
    for gib in (1, 8, 32):
        size = gib << 30
        m = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(m))
        t = time.time()
        rc = cu.cuMemHostRegister_v2(ctypes.c_void_p(addr), ctypes.c_size_t(size), ctypes.c_uint(0))
        res[f'register_{gib}GiB'] = {'rc': rc, 's': round(time.time() - t, 2)}
        if rc == 0:
            cu.cuMemHostUnregister(ctypes.c_void_p(addr))
        del addr
        try:
            m.close()
        except BufferError:
            pass
        if rc != 0:
            break
    has = {}
    for name in ('cuMemcpyBatchAsync_v2', 'cuMemcpyBatchAsync', 'cuStreamWaitValue32_v2', 'cuGraphLaunch'):
        try:
            getattr(cu, name); has[name] = True
        except AttributeError:
            has[name] = False
    res['symbols'] = has
    return res

try:
    OUT['cuda'] = cuda_register()
except Exception as e:
    OUT['cuda'] = repr(e)

# --- the binary itself ----------------------------------------------------------
W = '/tmp/flare'
shutil.rmtree(W, ignore_errors=True)
shutil.copytree(SRC, W, ignore=shutil.ignore_patterns('blob.bin'))
sums = dict(reversed(l.split('  ', 1)) for l in open(f'{W}/SHA256SUMS').read().splitlines())
OUT['sha_ok'] = all(hashlib.sha256(open(f'{W}/{p}', 'rb').read()).hexdigest() == h
                    for p, h in sums.items() if p != 'blob.bin')
os.makedirs(f'{W}/libs', exist_ok=True)  # Kaggle uploads files only, so the libs ship flat
for name in os.listdir(W):
    if name != 'libs' and name.startswith(('ld-linux', 'lib')):
        shutil.move(f'{W}/{name}', f'{W}/libs/{name}')
for p in ['flare', 'libs/ld-linux-x86-64.so.2']:
    os.chmod(f'{W}/{p}', 0o755)
libpath = ':'.join([f'{W}/libs', '/usr/local/nvidia/lib64', '/usr/lib/x86_64-linux-gnu', '/usr/lib64'])
shim = [f'{W}/libs/ld-linux-x86-64.so.2', '--library-path', libpath, f'{W}/flare']
OUT['native_help'] = sh([f'{W}/flare', '--help'])
OUT['shim_version'] = sh(shim + ['--version'])
OUT['shim_help'] = sh(shim + ['run', '--help'])
os.makedirs('/tmp/flare-cache', exist_ok=True)
OUT['shim_detect'] = sh(shim + ['download', 'local-inference-lab/Qwen3.8-Flash-Next-FLARE', '--gpus', '0',
                                '--offline', '--cache-dir', '/tmp/flare-cache'], timeout=180)

# --- /kaggle/input read speed (the disk-PLE and load path) -------------------
blob = os.path.join(SRC, 'blob.bin')
t = time.time(); n = 0
with open(blob, 'rb', buffering=0) as f:
    while True:
        b = f.read(64 << 20)
        if not b:
            break
        n += len(b)
OUT['input_read'] = {'bytes': n, 's': round(time.time() - t, 2), 'MBps': round(n / 1e6 / max(time.time() - t, 1e-9), 1)}

print(json.dumps(OUT, indent=1, default=str))
json.dump(OUT, open('/kaggle/working/probe.json', 'w'), indent=1, default=str)
