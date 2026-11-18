"""Issue exact approval only after repeated offline reconciliation, then real gate check."""
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
with contextlib.redirect_stdout(io.StringIO()):
    review = runpy.run_path(str(HERE / 'verify_review.py'))
gate, run = review['gate'], review['RUN']
assert not os.path.lexists(run / 'CONSUMED.json')
assert not os.path.lexists(run / 'launch.json')
assert not os.path.lexists(run / 'launch-review.json')
assert gate.load(run / 'authorization.json')['used'] is False
assert gate.sha(run / 'manifest.json') == review['EXPECTED']
provider_calls = []
def no_provider(*args, **kwargs):
    provider_calls.append(repr(args))
    raise AssertionError('Provider invocation forbidden')
with patch.object(gate.subprocess, 'run', no_provider):
    gate.exclusive(run / 'launch-review.json', review['prospective'])
    # Immediately validate the REAL review file using the unmodified real gate.
    code = gate.main(['check', '--run-dir', str(run)])
    assert code == 0
    assert gate.load(run / 'launch-review.json') == review['prospective']
    assert not os.path.lexists(run / 'CONSUMED.json')
assert provider_calls == []
print(json.dumps({'status': 'APPROVED', 'manifest_sha256': review['EXPECTED'], 'exact_real_gate_exit': code, 'provider_calls': provider_calls, 'consumed': False, 'verified_sources': len(review['rows'])}, indent=2))
