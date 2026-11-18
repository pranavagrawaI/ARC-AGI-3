"""Wait for usable upload bandwidth; at most ONE GPU push, never retry it."""
import datetime as dt
import fcntl
import json
import subprocess
import time
from prepare import HERE
from run import KPY, save, now, verify

DEADLINE = dt.datetime(2026, 10, 31, 12, 0, tzinfo=dt.timezone.utc)
MIN_BPS = 130000


def main():
    with (HERE / 'uplink-gate.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while dt.datetime.now(dt.timezone.utc) < DEADLINE:
            verify()
            if (HERE / 'launch-attempt.json').exists():
                raise SystemExit('attempt marker already exists; this gate never retries')
            probe = subprocess.run(['curl', '--fail', '-sS', '-o', '/dev/null', '-w', '%{http_code} %{speed_upload} %{time_total}', '-X', 'POST', '--data-binary', '@' + str(HERE / 'evidence/uplink-probe.bin'), 'https://httpbin.org/post', '--max-time', '30'], capture_output=True, text=True, timeout=40)
            parts = probe.stdout.split()
            bps = float(parts[1]) if probe.returncode == 0 and len(parts) == 3 and parts[0] == '200' else 0
            row = {'utc': now(), 'http': parts[0] if parts else None, 'upload_bps': bps, 'ready': bps >= MIN_BPS}
            with (HERE / 'uplink-history.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
            if bps >= MIN_BPS:
                launch = subprocess.run([KPY, '-B', str(HERE / 'run.py'), 'launch'], cwd=HERE, timeout=300)
                save(HERE / 'gate-launch-exit.json', {'utc': now(), 'returncode': launch.returncode})
                if launch.returncode:
                    return  # marker/ambiguous response preserved; never retry
                subprocess.run([KPY, '-B', str(HERE / 'run.py'), 'observe'], cwd=HERE, timeout=QUEUE_PLUS_RUNTIME)
                if (HERE / 'terminal.json').exists():
                    subprocess.run([KPY, '-B', str(HERE / 'run.py'), 'collect'], cwd=HERE, timeout=900, check=True)
                    subprocess.run([KPY, '-B', str(HERE / 'assess.py')], cwd=HERE, timeout=300, check=True)
                return
            time.sleep(300)
        save(HERE / 'gate-deadline.json', {'utc': now(), 'state': 'no attempt; upload admission deadline reached', 'gpu_pushes': 0})


QUEUE_PLUS_RUNTIME = 18000 + 7800 + 300
if __name__ == '__main__':
    main()
