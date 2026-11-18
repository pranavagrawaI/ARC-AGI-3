"""R6 leftover-descendant guard failure and R7 trusted-shutdown reap."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from test_runtime import PACKAGE, ROOT


def _guard(tmp_path, child, timeout=20):
    receipt = tmp_path / 'PROCESS-GUARD.json'
    proc = subprocess.run(
        [sys.executable, '-I', '-S', '-B', str(PACKAGE / 'process_guard.py'), str(os.getpid()), '8', str(receipt), *child],
        capture_output=True, timeout=timeout)
    assert receipt.exists(), proc.stderr.decode()
    return proc, json.loads(receipt.read_text())


def _wait_gone(pid):
    deadline = time.monotonic() + 3
    while Path(f'/proc/{pid}').exists() and time.monotonic() < deadline:
        time.sleep(.02)
    assert not Path(f'/proc/{pid}').exists()


def test_process_guard_leftover_check_unchanged_from_r6():
    prior = ROOT / 'deployment/kaggle/actor-critic-construction-kaggle-r6/process_guard.py'
    assert prior.read_bytes() == (PACKAGE / 'process_guard.py').read_bytes()
    source = prior.read_text()
    assert "record['status'] = 'failed'" in source
    assert "record['error'] = 'successful child left live descendants'" in source


def test_r6_successful_child_with_setsid_leftover_fails_guard(tmp_path):
    pidfile = tmp_path / 'leftover.json'
    script = tmp_path / 'r6_child.py'
    script.write_text(
        'import json, subprocess, sys\n'
        'from pathlib import Path\n'
        'proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)\n'
        'Path(sys.argv[1]).write_text(json.dumps({"pid": proc.pid, "comm": "synthetic-setsid-orphan"}))\n'
    )
    proc, record = _guard(tmp_path, [sys.executable, str(script), str(pidfile)])
    leftover = json.loads(pidfile.read_text())['pid']
    assert proc.returncode == 1
    assert record['returncode'] == 0
    assert record['status'] == 'failed'
    assert record['error'] == 'successful child left live descendants'
    assert leftover in record['killed_remaining_descendants']
    assert record['cleanup'] == 'retired'
    _wait_gone(leftover)


def test_r7_trusted_shutdown_reaps_setsid_orphan_and_guard_passes(tmp_path):
    pidfile = tmp_path / 'leftover.json'
    reapfile = tmp_path / 'SHUTDOWN-REAP.json'
    script = tmp_path / 'r7_child.py'
    script.write_text(
        'import json, subprocess, sys\n'
        'from pathlib import Path\n'
        'sys.path.insert(0, sys.argv[1])\n'
        'import donor_bootstrap\n'
        'donor_bootstrap.become_subreaper()\n'
        'proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)\n'
        'Path(sys.argv[2]).write_text(json.dumps({"pid": proc.pid, "comm": "synthetic-setsid-orphan"}))\n'
        'donor_bootstrap.reap_owned_descendants(Path(sys.argv[3]))\n'
    )
    proc, record = _guard(tmp_path, [sys.executable, str(script), str(PACKAGE), str(pidfile), str(reapfile)])
    leftover = json.loads(pidfile.read_text())['pid']
    reap = json.loads(reapfile.read_text())
    assert proc.returncode == 0, record
    assert record['returncode'] == 0
    assert record['status'] == 'passed'
    assert record['killed_remaining_descendants'] == []
    assert reap['status'] == 'retired'
    assert reap['remaining'] == []
    assert leftover in reap['reaped'] or any(item.get('pid') == leftover for item in reap['seen'])
    assert not Path(f'/proc/{leftover}').exists()


def test_guard_still_fails_when_descendant_cannot_be_reaped(tmp_path):
    pidfile = tmp_path / 'leftover.json'
    reapfile = tmp_path / 'SHUTDOWN-REAP.json'
    nested = tmp_path / 'orphan.py'
    nested.write_text(
        'import json, os, time\n'
        'from pathlib import Path\n'
        'import sys\n'
        'if os.fork() == 0:\n'
        '    os.setsid()\n'
        '    Path(sys.argv[1]).write_text(json.dumps({"pid": os.getpid(), "comm": "synthetic-unreapable-orphan"}))\n'
        '    time.sleep(30)\n'
        '    os._exit(0)\n'
        'os._exit(0)\n'
    )
    script = tmp_path / 'unreapable_child.py'
    script.write_text(
        'import subprocess, sys, time\n'
        'from pathlib import Path\n'
        'sys.path.insert(0, sys.argv[1])\n'
        'import donor_bootstrap\n'
        'pidfile, reapfile, nested = Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]\n'
        'subprocess.run([sys.executable, nested, str(pidfile)], check=True)\n'
        'deadline = time.monotonic() + 3\n'
        'while not pidfile.exists() and time.monotonic() < deadline:\n'
        '    time.sleep(0.02)\n'
        'donor_bootstrap.reap_owned_descendants(reapfile)\n'
    )
    proc, record = _guard(tmp_path, [sys.executable, str(script), str(PACKAGE), str(pidfile), str(reapfile), str(nested)])
    leftover = json.loads(pidfile.read_text())['pid']
    reap = json.loads(reapfile.read_text())
    assert reap['remaining'] == []
    assert leftover not in reap['reaped']
    assert proc.returncode == 1
    assert record['returncode'] == 0
    assert record['status'] == 'failed'
    assert record['error'] == 'successful child left live descendants'
    assert leftover in record['killed_remaining_descendants']
    _wait_gone(leftover)


def test_r7_reaps_orphan_reparented_after_intermediate_exits(tmp_path):
    """The R6 shape: the supervisor (intermediate) exits 0 leaving a setsid orphan.

    The orphan is not bootstrap's direct child, so it is visible only because
    bootstrap is the subreaper it reparents to."""
    pidfile = tmp_path / 'leftover.json'
    reapfile = tmp_path / 'SHUTDOWN-REAP.json'
    intermediate = tmp_path / 'supervisor.py'
    intermediate.write_text(
        'import json, subprocess, sys\n'
        'from pathlib import Path\n'
        'proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)\n'
        'Path(sys.argv[1]).write_text(json.dumps({"pid": proc.pid, "comm": "synthetic-setsid-orphan"}))\n'
    )
    script = tmp_path / 'r7_bootstrap.py'
    script.write_text(
        'import subprocess, sys\n'
        'from pathlib import Path\n'
        'sys.path.insert(0, sys.argv[1])\n'
        'import donor_bootstrap\n'
        'donor_bootstrap.become_subreaper()\n'
        'subprocess.run([sys.executable, sys.argv[4], sys.argv[2]], check=True)\n'
        'donor_bootstrap.reap_owned_descendants(Path(sys.argv[3]))\n'
    )
    proc, record = _guard(tmp_path, [sys.executable, str(script), str(PACKAGE), str(pidfile), str(reapfile), str(intermediate)])
    leftover = json.loads(pidfile.read_text())['pid']
    reap = json.loads(reapfile.read_text())
    assert proc.returncode == 0, record
    assert record['status'] == 'passed'
    assert record['killed_remaining_descendants'] == []
    assert reap['status'] == 'retired' and reap['remaining'] == []
    assert leftover in reap['reaped']
    seen = {item['pid']: item for item in reap['seen']}
    assert seen[leftover]['cmdline'] and 'time.sleep(30)' in seen[leftover]['cmdline']
    _wait_gone(leftover)
