"""Read-only-to-candidate adversarial controls; all fixture writes use a tempdir."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from tools.experiment_checks import preserve

with tempfile.TemporaryDirectory(prefix='experiment-check-review-') as folder:
    base = Path(folder)
    raw = base / 'raw'
    raw.mkdir()
    (raw / 'EXPERIMENT-CARD.md').write_text(
        'Invalid if: Wrong model/runtime/GPU identity, missing request/response '
        'accounting, timeout, OOM, failed cleanup, or missing terminal stream/usage.')
    (raw / 'vllm-openai-server.log').write_text('CUDA out of memory\n')
    (raw / 'vllm-setup-provenance.json').write_text('{"runtime":null}')
    result = subprocess.run([sys.executable, str(ROOT / 'tools/experiment_checks.py'),
                             'flash', str(raw)], capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert result.returncode == 1 and report['status'] == 'invalid'
    assert any(f['condition'] == 'OOM' for f in report['failures'])
    assert any('runtime provenance' in p for p in report['unresolved'])
    print('OOM plus malformed provenance: invalid JSON report retains OOM and pending provenance')
    (raw / 'vllm-openai-server.log').write_text('server ready\n')
    result = subprocess.run([sys.executable, str(ROOT / 'tools/experiment_checks.py'),
                             'flash', str(raw)], capture_output=True, text=True)
    assert result.returncode == 2 and json.loads(result.stdout)['status'] == 'unresolved'
    print('Malformed provenance without OOM: unresolved JSON report')
    (raw / 'resources.jsonl').write_text('{"epoch":1,"gpu":", , , "}\n')
    result = subprocess.run([sys.executable, str(ROOT / 'tools/experiment_checks.py'),
                             'flash', str(raw)], capture_output=True, text=True)
    print('Empty GPU CSV fields:', result.returncode, 'json_report=', bool(result.stdout.strip()),
          result.stderr.strip().splitlines()[-1:] or [])

    source = base / 'source'
    source.mkdir()
    os.mkfifo(source / 'server.log')
    destination = base / 'snapshot'
    try:
        preserve(source, destination)
    except ValueError as error:
        assert 'regular files' in str(error)
    else:
        raise AssertionError('FIFO silently accepted')
    assert not destination.exists()
    print('FIFO preservation: rejected before destination creation')

    ipc = base / 'ipc'
    ipc.mkdir()
    (ipc / 'model_feedback.jsonl').write_text('{"batch_allowed":false}\n')
    (ipc / 'actions.jsonl').write_text('{"actions":null}\n')
    for name in ['model_predictions.jsonl', 'batch_stops.jsonl']:
        (ipc / name).write_text('')
    result = subprocess.run([sys.executable, str(ROOT / 'tools/experiment_checks.py'),
                             'batching', str(ipc)], capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert result.returncode == 2 and report['status'] == 'unresolved'
    assert report['observations']['malformed_submission_rows'] == 1
    print('Malformed action row: unresolved JSON report and malformed row counted')
