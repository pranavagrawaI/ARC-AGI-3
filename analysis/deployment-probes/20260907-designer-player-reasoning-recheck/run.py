"""Run the frozen repaired probe in a fresh directory with lifecycle evidence."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
CANDIDATE = ROOT/'deployment/kaggle/designer-player-reasoning-20260907'
sys.path.insert(0, str(CANDIDATE))
import openrouter_probe as probe
from freeze_candidate import verify
from probe_lifecycle import AdmissionWindow


class RecordedWindow(AdmissionWindow):
    serial = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.identity = RecordedWindow.serial
        RecordedWindow.serial += 1
        self.event('created')

    def event(self, name):
        probe.append('lifecycle.jsonl', dict(event=name, window=self.identity,
            monotonic=time.monotonic(), time=time.time(), deadline=self.deadline,
            reason=self.reason, active=self.active))

    def close(self, reason):
        super().close(reason)
        self.event('closed')

    @contextmanager
    def admit(self):
        try:
            with super().admit():
                self.event('admitted')
                try:
                    yield
                finally:
                    self.event('transport_returned')
        except PermissionError:
            self.event('rejected')
            raise
        finally:
            self.event('released')


if __name__ == '__main__':
    manifest = verify()
    expected = '6b8a5c1d6955c09d2f324f9ce806539fceec37254e9b7832de8f604dec89733e'
    assert manifest['contract_sha256'] == expected
    assert not (OUT/'STARTED.json').exists(), 'Fresh run only'
    fingerprints = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), OUT/'EXPERIMENT-CARD.md', CANDIDATE/'probe_lifecycle.py', CANDIDATE/'openrouter_probe.py']}
    preflight = dict(contract=expected, files=fingerprints)
    if '--execute' in sys.argv:
        (OUT/'PREFLIGHT.json').write_text(json.dumps(preflight,indent=2)+'\n')
    probe.OUT = OUT
    probe.AdmissionWindow = RecordedWindow
    probe.main()
