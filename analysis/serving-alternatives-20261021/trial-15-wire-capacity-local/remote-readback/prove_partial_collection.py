"""Read-only source/terminal/partial-member proof; never qualifies a paired rate."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import transport


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def references(value):
    if isinstance(value, dict):
        if {'path', 'sha256', 'bytes', 'complete'} <= value.keys():
            yield value
        else:
            for item in value.values():
                yield from references(item)
    elif isinstance(value, list):
        for item in value:
            yield from references(item)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(exist_ok=False)
    remote = HERE / 'remote-readback'
    freeze = read(HERE / 'freeze.json')
    for name, digest in freeze['files'].items():
        assert sha(HERE / name) == digest, name
    for name, digest in freeze['references'].items():
        assert sha(name) == digest, name

    partial = remote / 'archive-v1-kernel140466291/output.zip'
    recovery = read(remote / 'partial-final-members/RECOVERY.json')
    assert partial.stat().st_size == recovery['archive_bytes'] == 1689764
    assert sha(partial) == recovery['archive_sha256'] == sha(remote / 'archive-final-partial.zip')
    assert not recovery['complete_archive'] and not recovery['central_directory_verified']
    assert not zipfile.is_zipfile(partial), 'unexpected complete archive; review separately'
    archive_receipt = read(remote / 'archive-v1-kernel140466291/ARCHIVE.json')
    assert archive_receipt['complete'] is False
    assert archive_receipt['error']['error_type'] == 'ChunkedEncodingError'
    inventory = recovery['entries']
    for entry in inventory:
        path = remote / 'partial-final-members/members' / entry['name']
        assert entry['retained'] and path.stat().st_size == entry['bytes']
        assert sha(path) == entry['sha256'], entry['name']

    spec = read(HERE / 'packet-spec.json')
    terminal = remote / 'poll-20261023T225348969480Z'
    readback = read(terminal / 'READBACK.json')
    metadata = read(terminal / 'metadata.json')
    assert metadata['id'] == readback['kernel_id'] == 140466291
    assert metadata['currentVersionNumber'] == readback['version'] == 1
    assert metadata['ref'] == spec['metadata']['id'] == readback['kernel_ref']
    assert readback['status']['status'] == 'ERROR' and readback['source_exact'] is True
    raw_source = (terminal / 'source.ipynb').read_bytes()
    armed = (HERE / 'kernel' / spec['metadata']['code_file']).read_bytes()
    assert transport.cell_source(json.loads(raw_source)) == transport.cell_source(json.loads(armed))
    inner = transport.decode(raw_source)
    attempt = read(HERE / 'launch-attempt.json')
    assert transport.sha(inner) == attempt['inner_notebook_sha'] == readback['inner_sha256']
    values = transport.literals(transport.cell_source(json.loads(inner)))
    assert values['CONFIG'] == spec['config']
    manifest = read(HERE / 'review-packet.json')
    assert {k: v['sha256'] for k, v in values['PAYLOAD'].items()} == manifest['files']
    for name, record in values['PAYLOAD'].items():
        data = base64.b64decode(record['base64'], validate=True)
        assert hashlib.sha256(data).hexdigest() == record['sha256'] == sha(HERE / name)
    slot = read(HERE / 'slot-confirmation.json')
    expected_assignment = {key: slot[key] for key in ('launch_authorized', 'serialized_assignment', 'sources_sha256', 'config_sha256', 'kernel_ref', 'confirmed_utc', 'local_only', 'approved_freeze_sha256')}
    expected_assignment.update(total_seconds=10200, platform_hard_timeout_s=10500)
    assert values['ASSIGNMENT'] == expected_assignment

    primary = read(remote / 'PRIMARY-RESULT.json')
    native_log = remote / 'primary-native-log/native.log'
    assert sha(native_log) == primary['native_log_sha256']
    results = []
    for event in read(native_log):
        if event['stream_name'] == 'stdout':
            for line in event['data'].splitlines():
                try:
                    value = json.loads(line)
                except ValueError:
                    continue
                if value.get('run_id') == spec['config']['run_id']:
                    results.append(value)
    assert results == [primary['result']]
    assert results[0]['pass_'] is False and results[0]['qualification'] == 'FEASIBILITY_ONLY'
    quota = readback['typed_quota']
    assert quota['used_s'] == 77433.978 and quota['reserved_s'] == 0
    reduced = read(remote / 'derived-original/REDUCTION.json')
    assert reduced['status'] == 'FEASIBILITY_ONLY' and reduced['ratio'] is None
    assert reduced['issues'] == ["ValueError('archive incomplete/hash')"]
    disabled = read(remote / 'schedule-disabled-terminal.json')
    assert disabled['enabled'] is False and disabled['nextRunAt'] is None

    root = remote / 'partial-final-members/members/serving-wire-capacity15/output'
    missing = [name for name in ('result.json', 'supervisor.jsonl', 'release.json',
        'control.qualify.stdout', 'control/summary.json', 'candidate-release.json',
        'candidate-server-binding.json', 'paired-result.json', 'host-boundaries.jsonl',
        'host-series.jsonl') if not (root / name).exists()]
    raw_refs = list(references(read(root / 'candidate-admission.json')))
    absent_refs = [ref for ref in raw_refs if not (root / 'query-artifacts' / ref['path']).exists()]
    proof = dict(status='TERMINAL_ERROR_PARTIAL_COLLECTION_ONLY',
        freeze_sha256=sha(HERE / 'freeze.json'), frozen_files=len(freeze['files']),
        frozen_references=len(freeze['references']), kernel_id=140466291, version=1,
        source_exact=True, decoded_source_payloads_exact=len(values['PAYLOAD']),
        executed_archive_payloads_complete=False, config_assignment_exact=True,
        inner_sha256=transport.sha(inner), terminal_status='ERROR', typed_quota=quota,
        delta_used_s=round(quota['used_s'] - 71670.895, 3),
        partial_archive_sha256=sha(partial), partial_archive_bytes=partial.stat().st_size,
        local_CRC_verified_members=len(inventory), local_decoded_bytes=recovery['decoded_bytes'],
        full_archive_verified=False, central_directory_verified=False,
        primary_result=results[0], native_log_archive_join=False,
        missing_primary_or_raw=missing, admission_raw_refs=len(raw_refs),
        missing_admission_raw_refs=len(absent_refs),
        candidate_health_probes=len(list(root.glob('candidate.health-*.stdout'))),
        control_health_probes_retained=len(list(root.glob('control.health-*.stdout'))),
        candidate_stderr_sha256=sha(root / 'candidate.server.stderr'),
        independent_resource_rederivation='UNAVAILABLE_MISSING_RAW_REFERENCES',
        configured_final180_and_candidate_replay_cleanup_join='NOT_REACHED_NOT_QUALIFIED',
        original_reducer_sha256=sha(HERE / 'collection_tools/reduce.py'),
        original_reduction_sha256=sha(remote / 'derived-original/REDUCTION.json'),
        ratio=None, rate_claim=False, adoption=False, followup_disabled=True,
        no_API_calls=True, no_retry=True, slot_release_claim=False,
        gaps=['Incomplete archive prevents raw paired qualification and independent resource audit.',
              'Primary native-log result lacks archive equality join; supervisor trace unavailable.',
              'Candidate incomplete loading is observed; exact failed phase and cause remain unresolved.',
              'No control rate claim from unrecovered raw arms; original140 and prior failures unchanged.'])
    (out / 'INVENTORY.json').write_text(json.dumps(inventory, indent=2) + '\n')
    (out / 'MISSING-RAW-REFERENCES.json').write_text(json.dumps(absent_refs, indent=2) + '\n')
    (out / 'COLLECTION-PROOF.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
