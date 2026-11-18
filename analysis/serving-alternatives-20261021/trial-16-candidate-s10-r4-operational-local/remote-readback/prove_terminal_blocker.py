"""Offline saved-terminal/source proof; no API and no runtime outcome inference."""
import base64, datetime as dt, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT.parent / 'trial16-collection-decimal-overlay-local'
sys.path[:0] = [str(OVERLAY), str(ROOT), str(ROOT / 'collection_tools')]
import decimal_quota, reconciliation, transport

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prove():
    freeze = json.loads((ROOT / 'freeze.json').read_text())
    assert sha(ROOT / 'freeze.json') == 'b51120fedef231e6b5fdecd8882f03bb78653e4e221fa9b256cdd7d908c2ed5b'
    for name, expected in freeze['files'].items():
        assert sha(ROOT / name) == expected, name
    for name, expected in freeze['references'].items():
        assert sha(name) == expected, name
    assert len(freeze['files']) + len(freeze['references']) == freeze['bindings'] == 13118
    pins = json.loads((OVERLAY / 'SOURCE-PINS.json').read_text())
    assert sha(OVERLAY / 'SOURCE-PINS.json') == 'c609808e1a09042bb018a786eeda292c9eeb8e495962d65fb1746714613c2568'
    for name, expected in pins['overlay_files'].items():
        assert sha(OVERLAY / name) == expected, name
    for name, expected in pins['references'].items():
        assert sha(name) == expected, name
    poll = ROOT / 'remote-readback/poll-20261024T014018277296Z'
    rb = json.loads((poll / 'READBACK.json').read_text())
    metadata = json.loads((poll / 'metadata.json').read_text())
    spec = json.loads((ROOT / 'packet-spec.json').read_text())
    attempt = json.loads((ROOT / 'launch-attempt.json').read_text())
    assert not rb.get('error') and rb['source_exact'] is True
    assert (rb['kernel_ref'], rb['kernel_id'], rb['version']) == ('pranavagrawai/serving-candidate-s10-16-20261024', 140493086, 1)
    assert (metadata['ref'], metadata['id'], metadata['currentVersionNumber']) == (rb['kernel_ref'], rb['kernel_id'], rb['version'])
    remote = (poll / 'source.ipynb').read_bytes()
    local = (ROOT / 'kernel' / spec['metadata']['code_file']).read_bytes()
    assert transport.cell_source(json.loads(remote)) == transport.cell_source(json.loads(local))
    inner = transport.decode(remote)
    assert transport.sha(inner) == attempt['inner_notebook_sha'] == rb['inner_sha256']
    values = transport.literals(transport.cell_source(json.loads(inner)))
    for name, item in values['PAYLOAD'].items():
        raw = base64.b64decode(item['base64'], validate=True)
        assert transport.sha(raw) == item['sha256']
        assert raw == (ROOT / name).read_bytes(), name
    quota = decimal_quota.validate(rb['typed_quota'], reconciliation.load_policy(), now=dt.datetime.fromisoformat(rb['typed_quota']['utc']))
    assert rb['status']['status'] == 'COMPLETE' and quota['reserved_s'] == 0
    primary = json.loads((ROOT / 'remote-readback/primary-v1/PRIMARY-RECEIPT.json').read_text())
    assert primary['complete'] is False and primary['error']['type'] == 'HTTPError'
    disabled = json.loads((ROOT / 'remote-readback/schedule-disabled-primary404.json').read_text())
    assert disabled['enabled'] is False and disabled['nextRunAt'] is None
    return dict(local_terminal_source_proof=True, original_bindings=freeze['bindings'], overlay_pins_exact=True,
                payloads=len(values['PAYLOAD']), terminal_status=rb['status']['status'],
                wire_seconds={k:str(v) for k,v in quota.items()}, quota_validated_at_original_receipt=True,
                primary_complete=False, archive_attempted=False, rate=None, runtime_verdict='UNQUALIFIED_COLLECTION_GAP',
                cleanup=None, collection_complete=False, followup_disabled=True, network_calls=0,
                gaps=['Primary list request HTTP404; no primary download', 'No archive or raw reducer',
                      'Actual result, supervisor, health, cleanup, raw resource, host/OOM, window/accounting, ownership, final180 and publication joins unavailable'])

if __name__ == '__main__':
    print(json.dumps(prove(), indent=2))
