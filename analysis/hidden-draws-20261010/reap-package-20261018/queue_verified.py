"""Queue the exact qualified version once; never submits or launches a kernel."""
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile

HERE = Path(__file__).resolve().parent
DAILY = HERE.parent
PROJECT = HERE.parents[2]
DATE = '2026-10-20'
DRAW = 'D11'
sys.path.insert(0, str(PROJECT/'analysis/marlin-fixed-replay-r3-20261011'))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dispatch_proof(queue_bytes, slug, version):
    """Run the real dispatcher against a private copy and a recording subprocess."""
    script = DAILY/'daily_draw.py'
    spec = importlib.util.spec_from_file_location('reap_dispatch_check', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    queue = json.loads(queue_bytes)
    original_log = (DAILY/'draws.log').read_bytes()
    calls = []
    with tempfile.TemporaryDirectory(prefix='reap-dispatch-') as directory:
        sandbox = Path(directory)
        (sandbox/'queue.json').write_bytes(queue_bytes)
        (sandbox/'draws.log').write_bytes(original_log)
        module.HERE = sandbox
        module.subprocess = SimpleNamespace(run=lambda argv, **kw: (
            calls.append(dict(argv=argv, cwd=str(kw['cwd']))) or SimpleNamespace(returncode=0)))
        for date in ['2026-10-19', DATE]:
            instant = dt.datetime.fromisoformat(date+'T00:30:00+00:00')
            class FixedDateTime(dt.datetime):
                @classmethod
                def now(cls, tz=None):
                    return instant if tz else instant.replace(tzinfo=None)
            module.dt = SimpleNamespace(datetime=FixedDateTime, date=dt.date, timezone=dt.timezone)
            try:
                module.main()
            except SystemExit as exc:
                assert exc.code == 0
            expected = queue[date]
            argv = calls[-1]['argv']
            draw = f'D{4+(dt.date.fromisoformat(date)-dt.date(2026,10,13)).days}'
            assert argv[3:6] == [draw, expected['kernel'], str(expected['version'])], argv
            assert argv[-2:] == ['--not-before', date], argv
        assert calls[-1]['argv'][3:6] == [DRAW, slug, str(version)]
    assert (DAILY/'draws.log').read_bytes() == original_log
    return dict(passed=True, no_submissions=True, no_live_log_writes=True,
                script_sha256=sha(script.read_bytes()), queue_sha256=sha(queue_bytes),
                before_fallback_hour=True, calls=calls)


def main():
    from kaggle_backend import KaggleBackend
    assert not (HERE/'QUEUE-RECEIPT.json').exists(), 'Queue receipt exists; inspect it instead of requeueing'
    authority = json.loads((HERE/'AUTHORITY.json').read_text())
    assert authority['hidden_draws'] == 1 and authority['conditional_queue_authorized']
    # Recompute the full artifact audit, including native replay and frozen-file hashes.
    spec = importlib.util.spec_from_file_location('reap_queue_readout', HERE/'readout.py')
    readout = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(readout)
    readout.main()
    result = json.loads((HERE/'RESULT.json').read_text())
    assert result['status'] == 'PASS'
    response = json.loads((HERE/'launch-response.json').read_text())
    metadata = json.loads((HERE/'kernel/kernel-metadata.json').read_text())
    notebook = json.loads((HERE/'kernel'/metadata['code_file']).read_text())
    backend = KaggleBackend()
    remote = backend.verify_remote(metadata['id'], response, notebook, metadata)
    remote.pop('source')
    remote['status'] = backend.status(metadata['id'])
    assert remote['status']['status'] == 'COMPLETE', remote['status']
    (HERE/'prequeue-readback.json').write_text(json.dumps(remote, indent=2)+'\n')
    assert dt.datetime.now(dt.timezone.utc).date() < dt.date.fromisoformat(DATE), 'Date must still be future'
    path = DAILY/'queue.json'
    before = path.read_bytes()
    queue = json.loads(before)
    assert DATE not in queue, 'Do not displace an existing candidate'
    assert queue['2026-10-19']['kernel'] == 'pranavagrawai/m2-f1-screen2-f1x'
    entry = dict(kernel=metadata['id'], version=response['versionNumber'], message=(
        '(candidate, REAP448 + c18): retained target-bank pruning (448 experts, native top-10) + '
        '18 serving streams, full 512-expert drafter. Exact v1 passed bounded operational '
        'qualification and independent native action replay. Prior full public pair 106 vs 103 '
        'levels, inconclusive; one exploratory bundle draw, no causal efficacy or promotion claim. '
        'Authorized 2026-10-18; card reap-package-20261018/HIDDEN-DRAW-CARD.md.'))
    queue[DATE] = entry
    after = (json.dumps(queue, indent=1)+'\n').encode()
    proof = dispatch_proof(after, metadata['id'], response['versionNumber'])
    (HERE/'dispatch-proof.json').write_text(json.dumps(proof, indent=2)+'\n')
    (HERE/'queue-before.json').write_bytes(before)
    (HERE/'queue-proposed.json').write_bytes(after)
    # Preserve concurrent changes: fail rather than replace a queue that changed during checks.
    assert path.read_bytes() == before, 'Concurrent queue change; inspect before proceeding'
    temporary = path.with_name('.queue-reap-20261018.tmp')
    with temporary.open('xb') as stream:
        stream.write(after)
        stream.flush()
        os.fsync(stream.fileno())
    assert path.read_bytes() == before, 'Concurrent queue change; prepared queue retained'
    temporary.replace(path)
    assert path.read_bytes() == after
    receipt = dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), state='QUEUED_NOT_SUBMITTED',
                   date=DATE, draw=DRAW, entry=entry, kernel_id=response['kernelId'],
                   notebook_sha256=sha((HERE/'kernel'/metadata['code_file']).read_bytes()),
                   queue_before_sha256=sha(before), queue_after_sha256=sha(after),
                   preserved_other_entries=True, qualification='RESULT.json',
                   exact_remote_readback='prequeue-readback.json', dispatch_proof='dispatch-proof.json')
    (HERE/'QUEUE-RECEIPT.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
