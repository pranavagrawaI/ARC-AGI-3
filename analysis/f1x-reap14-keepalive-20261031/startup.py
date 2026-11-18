"""Dated startup entry point. Restarts can observe an attempt but never repeat it."""
import argparse
import datetime as dt
import fcntl
import json
import subprocess
import time
from pathlib import Path
from prepare import HERE
from run import KPY, REF, TERMINAL, api_setup, observation_bounds, save, now, verify

LAUNCH_DATE = dt.date(2026, 10, 31)
LAUNCH_CUTOFF = dt.datetime(2026, 10, 31, 12, tzinfo=dt.timezone.utc)
OBSERVE_CUTOFF = dt.datetime(2026, 11, 1, 12, tzinfo=dt.timezone.utc)


def next_action(folder, current):
    """Pure decision seam: date guards, acceptance, and terminal artifacts choose one action."""
    folder = Path(folder)
    if (folder / 'QUALIFICATION.json').exists():
        return 'QUALIFIED_OR_FAILED_DONE'
    marker = folder / 'launch-attempt.json'
    if marker.exists():
        attempt = json.loads(marker.read_text())
        if attempt.get('state') != 'accepted':
            return 'HOLD_AMBIGUOUS_NEVER_PUSH'
        if current >= OBSERVE_CUTOFF:
            return 'OBSERVATION_CUTOFF_NEVER_PUSH'
        return 'COLLECT' if (folder / 'terminal.json').exists() else 'OBSERVE'
    if current.date() < LAUNCH_DATE:
        return 'WAIT_FOR_DATE'
    if current.date() != LAUNCH_DATE or current >= LAUNCH_CUTOFF:
        return 'LAUNCH_EXPIRED'
    return 'WAIT_UPLINK_THEN_PUSH_ONCE'


def command(*args, timeout):
    subprocess.run([KPY, '-B', *map(str, args)], cwd=HERE, timeout=timeout, check=True)


def resume_accepted():
    api, lw = api_setup()
    observation = api.kernels_status(REF)
    row = {'utc': now(), 'status': str(observation.status), 'failure': getattr(observation, 'failure_message', None), 'source': 'startup-resume'}
    with (HERE / 'status-history.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
    if row['status'].split('.')[-1] in TERMINAL:
        save(HERE / 'terminal.json', row)
        used, quota = lw.quota_used_s(api)
        save(HERE / 'evidence/quota-terminal.json', {'utc': now(), 'used_plus_reserved_s': used, 'quota': quota})
    else:
        rows = [json.loads(line) for line in (HERE / 'status-history.jsonl').read_text().splitlines()]
        attempt = json.loads((HERE / 'launch-attempt.json').read_text())
        if time.time() <= observation_bounds(rows, attempt['utc']):
            command(HERE / 'run.py', 'observe', timeout=26400)
        # Beyond the observation bound, this invocation has made one read-only probe.
        # A future dated timer/agent may probe again; no new push is permitted.


def finish_collection():
    if not (HERE / 'terminal.json').exists():
        return
    if not (HERE / 'output').exists():
        marker = HERE / 'collection-attempts.json'
        attempts = json.loads(marker.read_text()) if marker.exists() else []
        if len(attempts) >= 3:
            raise RuntimeError('three read-only collection attempts exhausted; preserve and hand off')
        partial = HERE / 'output.partial'
        if partial.exists():
            partial.rename(HERE / ('output.partial-preserved-' + str(len(attempts))))
        attempts.append({'utc': now(), 'operation': 'read-only output download; no GPU run'})
        save(marker, attempts)
        command(HERE / 'run.py', 'collect', timeout=900)
    if not (HERE / 'source-readback/identity.json').exists():
        api, _ = api_setup()
        from run import source_readback
        # Preserve a failed/partial earlier readback before trying this read-only operation.
        path = HERE / 'source-readback'
        if path.exists():
            path.rename(HERE / ('source-readback.partial-' + str(time.time_ns())))
        source_readback(api, 'source-readback')
    command(HERE / 'assess.py', timeout=300)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    with (HERE / 'startup.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify()
        current = dt.datetime.now(dt.timezone.utc)
        action = next_action(HERE, current)
        state = {'utc': now(), 'action': action, 'dry_run': args.dry_run, 'candidate': REF}
        print(json.dumps(state), flush=True)
        if args.dry_run:
            return
        save(HERE / 'startup-state.json', state)
        if action == 'WAIT_UPLINK_THEN_PUSH_ONCE':
            # The bandwidth gate has its own exclusive lock and run.py creates the
            # irreversible attempt marker before SaveKernel. It never retries.
            command(HERE / 'uplink_gate.py', timeout=24 * 3600)
        elif action == 'OBSERVE':
            resume_accepted()
        if action in {'OBSERVE', 'COLLECT'}:
            finish_collection()


if __name__ == '__main__':
    main()
