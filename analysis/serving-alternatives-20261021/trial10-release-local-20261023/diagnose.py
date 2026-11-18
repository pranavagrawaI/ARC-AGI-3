"""CPU/source-only witnesses. Never invokes release.run, nvidia-smi or the SDK.

Run with a fresh --output directory. Source copies and trial10 are read-only.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
TRIAL = HERE.parent / 'trial-10-exact-wire-local'
PRIMARY = TRIAL / 'remote-readback/priority-output/serving-wire-capacity10/output'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / 'sources' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, state=fields[0], ppid=int(fields[1]), pgid=int(fields[2]),
                    sid=int(fields[3]), starttime=int(fields[19]))
    except (FileNotFoundError, ProcessLookupError):
        return None


def live(value):
    return value is not None and value['state'] != 'Z'


def expected_error(fn, message):
    try:
        fn()
    except ValueError as error:
        assert str(error) == message, repr(error)
        return str(error)
    raise AssertionError('expected unchanged eligibility rejection')


def eligibility(release):
    raw = json.loads((PRIMARY / 'control-release.json').read_text())['raw']
    first = expected_error(lambda: release.validate(**raw), 'GPU memory not released below2048MiB')
    zero = copy.deepcopy(raw)
    gpu = zero['gpu'].split(',')
    gpu[1] = ' 0'
    zero['gpu'] = ','.join(gpu)
    second = expected_error(lambda: release.validate(**zero), 'remaining reclaimable host headroom below140GiB')
    fields = {k: int(v) for k, v in (line.split() for line in raw['stat'].splitlines())}
    required = 140 * 1024**3
    computed = int(raw['maximum']) - int(raw['current']) + fields['inactive_file']
    boundary = copy.deepcopy(zero)
    boundary['current'] = str(int(raw['maximum']) + fields['inactive_file'] - required)
    assert release.validate(**boundary)['reclaimable_host_bytes'] == required
    boundary['current'] = str(int(boundary['current']) + 1)
    expected_error(lambda: release.validate(**boundary), second)
    return dict(first_observed_rejection=first, gpu_zero_only_counterfactual_rejection=second,
                computed_bytes=computed, required_bytes=required, shortfall_bytes=required-computed,
                captured_cgroup={k: fields[k] for k in ('anon', 'file', 'kernel', 'active_file', 'inactive_file', 'file_mapped', 'file_dirty')},
                original_boundary_preserved=True, counterfactuals_are_not_remote_observations=True)


def process_witness(supervisor, output, detached):
    """Intervene only on descendant session inheritance; both are owned CPU processes."""
    root = output / ('detached' if detached else 'inherited')
    root.mkdir()
    ready = root / 'child.json'
    child = "import os,json,time;from pathlib import Path;Path(%r).write_text(json.dumps({'pid':os.getpid()}));time.sleep(10)" % str(ready)
    parent = "import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',%r],start_new_session=%r);time.sleep(10)" % (child, detached)
    config = json.loads((TRIAL / 'packet-spec.json').read_text())['config']
    config.update(mode='CPU_FIXTURE_ONLY', cwd=str(root), total_seconds=8, cleanup_seconds=1,
                  run_id='LOCAL-ONLY-release-visibility')
    config['hardware']['argv'] = [sys.executable, '-c', 'raise SystemExit(0)']
    sup = supervisor.Supervisor(config, root / 'trace')
    p = None
    owned = None
    try:
        p = sup.start('cpu-owned-parent', {'argv': [sys.executable, '-c', parent]}, sup.deadline(3))
        end = time.monotonic() + 2
        while not ready.exists() and time.monotonic() < end:
            time.sleep(.01)
        assert ready.exists(), 'owned child did not initialize within local ceiling'
        owned = identity(json.loads(ready.read_text())['pid'])
        assert live(owned)
        assert (owned['pgid'] != p.pid) == detached
        before = supervisor.group_members(p.pid)
        assert p.pid in before
        assert (owned['pid'] in before) != detached
        sup.cleanup([p.pid])
        after = identity(owned['pid'])
        remaining = supervisor.group_members(p.pid)
        assert remaining == []
        assert live(after) == detached
        return dict(intervention='descendant start_new_session=' + str(detached),
                    child_before=owned, original_group_before=before, original_group_after=remaining,
                    child_after_original_group_cleanup=after,
                    detached_child_alive_with_empty_original_group=detached,
                    correspondence='CPU visibility witness only; no evidence native trial10 descendants detached')
    finally:
        if p is not None and p.pid in sup.groups:
            sup.cleanup([p.pid])
        if owned:
            now = identity(owned['pid'])
            if live(now):
                assert now['starttime'] == owned['starttime'], 'refuse signaling reused PID'
                os.kill(owned['pid'], signal.SIGKILL)
            end = time.monotonic() + 1
            while live(identity(owned['pid'])) and time.monotonic() < end:
                time.sleep(.01)
            assert not live(identity(owned['pid'])), 'owned CPU child cleanup incomplete'
        sup.trace.close()
        for handle in sup.handles:
            handle.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    receipt = json.loads((HERE / 'SOURCE-RECEIPT.json').read_text())
    for name, sha in receipt['source_sha256'].items():
        assert digest(HERE / 'sources' / name) == sha
        assert digest(TRIAL / name) == sha
    freeze = json.loads((TRIAL / 'freeze.json').read_text())
    assert digest(TRIAL / 'freeze.json') == receipt['trial10_freeze_sha256']
    for name, sha in freeze['files'].items():
        assert digest(TRIAL / name) == sha, 'trial10 input drift: ' + name
    wheel = HERE.parent.parent / 'serving-headroom-v3-20261008/evidence/packages/sglang-0.5.19+gd00d88efc8d6-cp312-cp312-linux_x86_64.whl'
    wheel_sha = digest(wheel)
    assert wheel_sha == 'd0620216732b82b6efd4f8a78cc9b5f55343a4d3837fdfa3bcbb841b7718a225'
    mapping = {'pinned-communication-mp.py': 'sglang/kernels/ops/communication/mp.py',
               'pinned-utils-common.py': 'sglang/srt/utils/common.py',
               'pinned-engine.py': 'sglang/srt/entrypoints/engine.py',
               'pinned-launch-server.py': 'sglang/launch_server.py',
               'pinned-http-server.py': 'sglang/srt/entrypoints/http_server.py'}
    for path in (HERE / 'sources').glob('pinned-sglang-*'):
        if path.name.endswith('entry_points.txt'):
            mapping[path.name] = 'sglang-0.5.19+gd00d88efc8d6.dist-info/entry_points.txt'
        else:
            mapping[path.name] = path.name.removeprefix('pinned-').replace('-', '/')
    with zipfile.ZipFile(wheel) as archive:
        for name, member in mapping.items():
            assert (HERE / 'sources' / name).read_bytes() == archive.read(member), name
    release = load('release_check')
    supervisor = load('supervisor')
    checks = eligibility(release)
    inherited = process_witness(supervisor, output, False)
    detached = process_witness(supervisor, output, True)
    events = list(map(json.loads, (PRIMARY / 'supervisor.jsonl').read_text().splitlines()))
    assert not any(e.get('phase', '').startswith('candidate.') for e in events)
    tail = [e for e in events if e['seq'] >= 1718]
    sources = {str(p.relative_to(HERE)): digest(p) for p in sorted((HERE / 'sources').iterdir()) if p.is_file()}
    proof = dict(scope=receipt['scope'], checks_passed=6,
                 checks=['actual_gpu_rejection', 'gpu_zero_only_exposes_host_rejection', 'unchanged_host_boundary_accepts',
                         'one_byte_below_boundary_rejects', 'inherited_descendant_removed', 'detached_visibility_counterexample'],
                 elapsed_seconds=time.monotonic()-started, source_sha256=sources,
                 pinned_wheel_sha256=wheel_sha, pinned_wheel_source_members=mapping,
                 diagnosis_script_sha256=digest(Path(__file__)),
                 primary_sha256={n: digest(PRIMARY / n) for n in ('control-release.json', 'supervisor.jsonl', 'result.json')},
                 eligibility=checks, process_witnesses=[inherited, detached], actual_trial10_tail=tail,
                 no_native_escape_or_driver_lag_cause_established=True, remote_calls=0, gpu_calls=0,
                 original_frozen_bindings_exact=len(freeze['files']))
    (output / 'PROOF.json').write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(dict(pass_=True, checks=6, elapsed_seconds=proof['elapsed_seconds'],
                          proof=str(output / 'PROOF.json'), proof_sha256=digest(output / 'PROOF.json'))))


if __name__ == '__main__':
    main()
