"""Exercise real children with synthetic caller stdin, FD and env canaries."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PROBE = '''import os
result = {'stdin': os.read(0, 100).decode(), 'env': os.environ.get('RELAY_CANARY')}
try:
    result['fd'] = os.read(20, 100).decode()
except OSError:
    result['fd'] = 'closed'
'''


class InheritedInputTests(unittest.TestCase):
    def invoke(self, mode):
        outer = '''import json, sys, tempfile
from pathlib import Path
from sandbox_eval import CandidateRPC, RUNNER, execute_inspection
from real_trace import load_case
probe = %r
if %r == 'inspection':
    execution = execute_inspection(probe + "print(__import__('json').dumps(result))", load_case())
    assert execution['status'] == 'completed', execution
    print(execution['stdout'])
else:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = 'def initialize(observation):\\n' + '\\n'.join('    ' + line for line in probe.splitlines()) + '\\n    return result\\n'
        (root / 'candidate.py').write_text(source)
        (root / 'runner.py').write_text(RUNNER)
        print(json.dumps(CandidateRPC(root).initialize({})))
''' % (PROBE, mode)
        # Supply FD 20 explicitly to the outer harness; its generated child must
        # not inherit it. No real credentials enter this test.
        launcher = "import os,sys; os.dup2(int(sys.argv[1]),20); exec(sys.argv[2])"
        with tempfile.TemporaryFile() as canary:
            canary.write(b'FD_CANARY'); canary.seek(0)
            completed = subprocess.run(
                [sys.executable, '-c', launcher, str(canary.fileno()), outer],
                input='CONTAINMENT_REVIEW_STDIN_CANARY', text=True,
                capture_output=True, timeout=20, cwd=ROOT,
                env={**os.environ, 'RELAY_CANARY': 'SYNTHETIC_ENV_CANARY'},
                pass_fds=(canary.fileno(),),
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return json.loads(completed.stdout)

    def test_inspection_does_not_inherit_caller_input(self):
        self.assertEqual(self.invoke('inspection'), {'stdin': '', 'env': None, 'fd': 'closed'})

    def test_candidate_rpc_does_not_inherit_caller_input(self):
        self.assertEqual(self.invoke('candidate'), {'stdin': '', 'env': None, 'fd': 'closed'})
