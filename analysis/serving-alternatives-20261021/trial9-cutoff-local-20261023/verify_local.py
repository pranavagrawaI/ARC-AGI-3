"""Offline review proof. No network, remote status, launch or credential calls."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PRIOR=ROOT.parent/'trial-9-readiness-local'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):return json.loads(path.read_text())
def http_rows(case):return [load(p) for p in sorted(case.glob('out/*/http.json'))]

def verify():
    freeze=load(PRIOR/'freeze.json')
    assert sha(PRIOR/'freeze.json')=='336f0c2d6a89951178e247a9a92825a5e6053099bb53694a06a14a2cd6d82d53'
    mismatches=[name for name,digest in freeze['files'].items() if sha(PRIOR/name)!=digest]
    assert not mismatches,mismatches
    baseline={p.name:sha(p) for p in (ROOT/'baseline').iterdir() if p.is_file()}
    variant={p.name:sha(p) for p in (ROOT/'variant').iterdir() if p.is_file()}
    assert len(baseline)==len(variant)==24
    assert all(sha(PRIOR/name)==digest for name,digest in baseline.items())
    changed=[name for name in baseline if baseline[name]!=variant[name]]
    assert sorted(changed)==['reduce_api.py','replay.py'],changed
    for name in ('test_replay.py','test_persistence.py'):
        assert sha(ROOT/'tests'/name)==sha(PRIOR/'tests'/name)
    before=ROOT/'evidence/baseline/deadline-race'
    after=ROOT/'evidence/variant-final/deadline-race'
    b=load(before/'witness.json');a=load(after/'witness.json')
    assert b['source_sha256']==baseline['replay.py'] and not b['run_ok'] and b['failed']
    assert any(row['error'].startswith('TimeoutError(')
               and b['cutoff']<=row['error_observed_at']<row['cutoff_shutdown_at'] for row in http_rows(before))
    assert a['source_sha256']==variant['replay.py'] and a['run_ok'] and not a['failed']
    assert all(row['cause_type']=='RemoteDisconnected' and row['termination']=='INTENTIONAL_CUTOFF'
               and row['read_timeout']['actual_seconds']==900.
               and row['read_timeout']['cutoff']==a['cutoff'] for row in http_rows(after))
    for arm in ('baseline','variant-final'):
        path=ROOT/'evidence'/arm/'delayed-timeout'
        witness=load(path/'witness.json')
        assert all(t<witness['cutoff'] for t in witness['detected_at'])
        assert all(row['error_observed_at']>row['cutoff_shutdown_at']>=witness['cutoff'] for row in http_rows(path))
        assert witness['run_ok']==(arm=='baseline')
        if arm=='variant-final':
            assert all(row['termination']=='ERROR' and row['error'].startswith('TimeoutError(')
                       and row['phase_at_error']=='READ_HEADERS'
                       and row['read_timeout']['actual_seconds']==.05 for row in http_rows(path))
    log=(ROOT/'evidence/local-suite-final.log').read_text()
    assert 'Ran 26 tests' in log and log.rstrip().endswith('OK')
    disabled=load(PRIOR/'remote-readback/scheduled-monitor-disabled.json')
    assert disabled['enabled'] is False and disabled['nextRunAt'] is None
    closure=load(ROOT.parent/'coordination-20261023/trial9-attempt-closed.json')
    # Bind primary collection without claiming a complete ZIP or API audit.
    primary={name:sha(PRIOR/'remote-readback'/name) for name in (
        'TERMINAL-HANDOFF.json','PRIMARY-INVENTORY.json','PENDING-CLOCK-RECEIPT.json',
        'PARTIAL-ARCHIVE-RECEIPT.json','scheduled-monitor-disabled.json')}
    local={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*')
           if p.is_file() and '__pycache__' not in p.parts and p.name!='LOCAL-PROOF.json'}
    return dict(scope='LOCAL_CPU_LOOPBACK_ONLY_NO_LAUNCH_AUTHORITY',trial9_stays_invalid=True,
                trial9_freeze_sha256=sha(PRIOR/'freeze.json'),trial9_input_bindings_exact=len(freeze['files']),
                payload_count=24,changed_payloads=sorted(changed),baseline_sha256=baseline,variant_sha256=variant,
                test_count=26,test_result='PASS',collection_receipts_sha256=primary,
                coordinator_closure_sha256=sha(ROOT.parent/'coordination-20261023/trial9-attempt-closed.json'),
                local_files_sha256=local,
                limitations=['Scheduling injection demonstrates mechanism, not incidence or exact trial9 timer provenance.',
                            '900s default verified as configured; actual expiry tested with explicit .05s compression, not a 900s wait.',
                            'No serving, paired-capacity, response-quality or adoption qualification.',
                            'Partial ZIP central directory absent; no retry or completed-response API gate audit.'])

if __name__=='__main__':print(json.dumps(verify(),indent=2)+'\n',end='')
