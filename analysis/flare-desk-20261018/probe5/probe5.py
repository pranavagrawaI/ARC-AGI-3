# FLARE probe 5: can a GPU session READ the 107 GB package from /kaggle/input? Six parts in parallel,
# hashed and discarded (no local writes), page cache dropped as we go. Copying them to local disk
# killed three sessions (2a, 2b, probe 4); a fallocate of 108 GB did not. Survives -> reads are fine
# and the local write is the killer; dies -> any route that loads 107 GB from /kaggle/input is dead.
import glob, hashlib, json, os, threading, time
T0 = time.monotonic()
def log(m):
    print(f'[{time.monotonic() - T0:6.1f}s] {m}', flush=True)
    with open('/kaggle/working/probe5.log', 'a') as f:
        f.write(m + '\n')
def mem():
    return int(open('/sys/fs/cgroup/memory.current').read()) / 2**30
parts = [glob.glob(f'/kaggle/input/**/flare-part-{i}', recursive=True)[0] for i in range(6)]
metas = [json.load(open(p + '.json')) for p in parts]
ok = {}
def read(i):
    h = hashlib.sha256(); n = 0
    with open(parts[i], 'rb', buffering=0) as f:
        while True:
            b = f.read(64 << 20)
            if not b:
                break
            h.update(b)
            os.posix_fadvise(f.fileno(), n, len(b), os.POSIX_FADV_DONTNEED)
            n += len(b)
    ok[i] = h.hexdigest() == metas[i]['sha256']
    log(f'part {i} read ok={ok[i]} mem {mem():.1f} GiB')
def monitor():
    while len(ok) < 6:
        log(f'mem {mem():.1f} GiB'); time.sleep(60)
threading.Thread(target=monitor, daemon=True).start()
t = time.monotonic()
ths = [threading.Thread(target=read, args=(i,)) for i in range(6)]
[x.start() for x in ths]; [x.join() for x in ths]
dt = time.monotonic() - t
log(f'done: {ok} in {dt:.0f} s = {107011338717 / 1e6 / dt:.0f} MB/s aggregate')
