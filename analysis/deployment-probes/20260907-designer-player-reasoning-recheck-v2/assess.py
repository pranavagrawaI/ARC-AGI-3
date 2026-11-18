"""Assess retained recheck evidence; no external calls or model reasoning text."""
import hashlib
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT/'deployment/kaggle/designer-player-reasoning-20260907'))
import summarize_live
from freeze_candidate import verify


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []


def assess():
    summarize_live.OUT = OUT
    result = summarize_live.summarize()
    preflight = json.loads((OUT/'PREFLIGHT.json').read_text())
    result['source_identity_intact'] = verify()['contract_sha256'] == preflight['contract'] and all(
        hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
        for name,digest in preflight['files'].items())
    closed = set()
    admissions = 0
    violations = []
    for event in rows(OUT/'lifecycle.jsonl'):
        window = event['window']
        if event['event'] == 'closed':
            closed.add(window)
        elif event['event'] == 'admitted':
            admissions += 1
            if window in closed or event['monotonic'] >= event['deadline']:
                violations.append(event)
    result['admission_events'] = admissions
    result['post_close_or_deadline_admissions'] = violations
    result['relay_cleanup'] = rows(OUT/'cleanup.jsonl')
    result['prediction_checks'] = [r for p in OUT.rglob('step_checks.jsonl') for r in rows(p)]
    pids = set()
    for path in OUT.rglob('worker-metadata.json'):
        pids.add(int(json.loads(path.read_text())['agent_pid']))
    for path in OUT.rglob('worker-stdout.log'):
        for line in path.read_text().splitlines():
            try:
                record = json.loads(line[line.index('{'):])
            except (ValueError, json.JSONDecodeError):
                continue
            if record.get('type') == 'ready':
                pids.add(int(record['worker_id']))
                pids.add(int(record['metadata']['agent_pid']))
    states = {}
    for pid in pids:
        try:
            states[str(pid)] = Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]
        except FileNotFoundError:
            states[str(pid)] = 'absent'
    result['local_process_states'] = states
    result['observed_local_processes_retired'] = bool(states) and all(s in ('absent','Z') for s in states.values())
    if violations or not result['source_identity_intact'] or result['relay_cleanup'] or not result['observed_local_processes_retired']:
        result['status'] = 'INVALID'
    return result


if __name__ == '__main__':
    result = assess()
    text = json.dumps(result,indent=2)+'\n'
    (OUT/'QUALIFICATION.json').write_text(text)
    print(text)
