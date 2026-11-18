"""CPU-only witnesses; writes only temporary fixtures, never the packet.

Run: python3 reproduce.py replay
     python3 reproduce.py reducer
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

PACKET = Path(__file__).resolve().parents[2] / 'trial-8-wire-capacity-local'
sys.dont_write_bytecode = True
sys.path[:0] = [str(PACKET), str(PACKET / 'tests')]


def replay_witness():
    from test_replay import ReplayTests

    original_open = Path.open

    class LateFailure:
        def __init__(self, file):
            self.file = file

        def __enter__(self):
            self.file.__enter__()
            return self

        def __exit__(self, *args):
            return self.file.__exit__(*args)

        def write(self, chunk):
            time.sleep(.3)
            raise OSError('synthetic persistence failure after final body read')

    def opened(path, *args, **kwargs):
        file = original_open(path, *args, **kwargs)
        if path.name == 'body.bin' and args and args[0] == 'xb':
            return LateFailure(file)
        return file

    case = ReplayTests()
    try:
        with patch.object(Path, 'open', opened):
            root, server, replay, events, ok, elapsed = case.run_fixture('ok', (.05, .05))
        print(json.dumps({
            'run_ok': ok,
            'failed': replay.failed,
            'response_events': sum(e['kind'] == 'response' for e in events),
            'pending': len(replay.pending),
            'origin': replay.origin,
            'cutoff': replay.end,
            'elapsed': elapsed,
            'http_receipts': [json.loads(p.read_text()) for p in (root / 'out').glob('*/http.json')],
            'events': events,
        }, indent=2))
    finally:
        case.doCleanups()


def reducer_witness():
    from test_reducer import fixture
    import catalog
    import reduce_api

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        cat, _ = fixture(root, 'control', 0, [100, 700, 1000, 1300])
        fixture(root, 'candidate', 2000, [100, 700, 800, 1000, 1100, 1300, 1400])
        path = root / 'control/control-g0-4/http.json'
        receipt = json.loads(path.read_text())
        receipt.update(
            received_bytes=228,
            headers=[['Content-Length', '228']],
            error="OSError('persistence failed after final body read')",
        )
        path.write_text(json.dumps(receipt))
        (path.parent / 'body.bin').write_bytes(b'')
        # Same provenance isolation as the existing analytical positive test.
        # No claim that synthetic fixtures qualify real package provenance.
        with patch.object(catalog, 'load', return_value=cat), patch.object(
            reduce_api.provenance, 'check', return_value={'CPU_FIXTURE_ONLY': True}
        ):
            value = reduce_api.compare(root)
        print(json.dumps({
            'verdict': value['verdict'],
            'ratio': value['ratio'],
            'issues': value['issues'],
            'control_valid': value['control']['valid'],
            'control_issues': value['control']['issues'],
            'modified_pending_receipt': receipt,
        }, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('witness', choices=['replay', 'reducer'])
    args = parser.parse_args()
    {'replay': replay_witness, 'reducer': reducer_witness}[args.witness]()
