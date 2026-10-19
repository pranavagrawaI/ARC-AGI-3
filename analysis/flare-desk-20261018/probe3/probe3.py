# FLARE probe 3: is the session evicted for local disk use? fallocate 30 -> 60 -> 108 GB in /tmp,
# holding each step 90 s (kubelet measures periodically). COMPLETE = 108 GB of local disk is fine;
# ERROR with an empty log = evicted (smoke 2a's signature) somewhere above 30 GB.
import os, subprocess, time
T0 = time.monotonic()
def log(m):
    print(f'[{time.monotonic() - T0:6.1f}s] {m}', flush=True)
    with open('/kaggle/working/probe3.log', 'a') as f:
        f.write(m + '\n')
fd = os.open('/tmp/fill.bin', os.O_CREAT | os.O_WRONLY)
for gb in (30, 60, 108):
    os.posix_fallocate(fd, 0, gb * 10**9)
    log(f'allocated {gb} GB: ' + subprocess.run(['du', '-sh', '/tmp/fill.bin'], capture_output=True, text=True).stdout.strip())
    time.sleep(90)
    log(f'survived {gb} GB for 90 s')
os.close(fd); os.remove('/tmp/fill.bin')
log('done: 108 GB local disk survived')
