"""Fail-closed receipt audit. Local fixtures never establish remote provenance."""
import hashlib
import json
from pathlib import Path
import re
import sys
import accelerator

NAMES = ('benign','filesystem','inherited','syscalls','readonly','wall','output','forged')
DENIED = 'socket connect ptrace process_vm_readv pidfd_getfd mount chroot setns unshare setuid setgid capset prctl clone fork vfork clone3 execve execveat io_uring_setup open_by_handle_at prlimit64 fcntl sched_getaffinity x32_getpid'.split()
QUALIFIED = 'QUALIFIED_SYNTHETIC_PROBE_ONLY'
CLASSIFICATIONS = {QUALIFIED, 'UNSUPPORTED_OR_QUALIFICATION_FAILED', 'CLEANUP_FAILED', 'INCOMPLETE_QUALIFICATION'}


def require(ok, reason):
    if not ok:
        raise ValueError('AUDIT:' + reason)


def provenance(status, server, *, kernel, version, image, expected_notebook_sha256, notebook):
    require(type(version) is int and version > 0, 'EXACT_VERSION_REQUIRED')
    terminal = status.get('status')
    require(status.get('returncode') == 0 and terminal in ('COMPLETE','ERROR','CANCELLED','CANCELED'), 'STATUS_QUERY_FAILED_OR_NONTERMINAL')
    require(status.get('stdout', '').strip() == f'{kernel}/{version} has status "KernelWorkerStatus.{terminal}"', 'STATUS_TARGET_VERSION_MISMATCH')
    require(server.get('id') == kernel and server.get('current_version_number') == version, 'SERVER_TARGET_VERSION_MISMATCH')
    require(server.get('is_private') is True and server.get('enable_internet') is False, 'SERVER_PRIVACY_NETWORK')
    require(server.get('machine_shape') == 'NvidiaRtxPro6000' and server.get('docker_image') == image, 'SERVER_RESOURCE_IMAGE')
    require(server.get('competition_sources') == ['arc-prize-2026-arc-agi-3'], 'SERVER_COMPETITION_SOURCE')
    require(all(server.get(k) == [] for k in ('dataset_sources','kernel_sources','model_sources')), 'SERVER_UNINTENDED_SOURCES')
    require(hashlib.sha256(notebook).hexdigest() == expected_notebook_sha256, 'SERVER_NOTEBOOK_DRIFT')
    return {'kernel': kernel, 'version': version, 'terminal_status': terminal, 'provenance': 'EXACT_TARGET_RECONCILED'}


def audit_directory(directory, *, identity, sources, local_fixture=False):
    directory = Path(directory)
    result = json.loads((directory/'result.json').read_text())
    require(result.get('identity') == identity and result.get('source_sha256') == sources, 'RESULT_SOURCE_IDENTITY')
    require(result.get('classification') in CLASSIFICATIONS, 'UNKNOWN_CLASSIFICATION')
    require(result.get('model_loaded') is False, 'MODEL_LOADED')
    rows = result['cases']
    require(isinstance(rows, list), 'CASE_ROWS')
    required = {(m,n) for m in ('candidate','inspection') for n in NAMES}
    observed = [(r['mode'],r['name']) for r in rows]
    require(len(set(observed)) == len(observed) and set(observed) <= required, 'CASE_IDENTITIES')
    require(all(type(r.get('check')) is bool for r in rows), 'CASE_CHECK_TYPE')
    require(result.get('completed_cases') == len(rows) and result.get('passed_cases') == sum(r['check'] for r in rows), 'CASE_COUNTS')
    require(result.get('required_cases') == len(required) and result.get('unexercised_cases') == len(required)-len(rows), 'REQUIRED_COUNTS')
    require(result.get('failed_cases') == sum(not r['check'] for r in rows), 'FAILED_COUNTS')
    qualified = result['classification'] == QUALIFIED
    if qualified:
        require(set(observed) == required and result['passed_cases'] == len(required), 'INCOMPLETE_QUALIFICATION')
        require(result.get('cleanup_complete') is True, 'CLEANUP_INCOMPLETE')
    else:
        require(bool(result.get('error')) or result['classification'] in ('CLEANUP_FAILED','INCOMPLETE_QUALIFICATION'), 'FAILURE_WITHOUT_REASON')
    platform = json.loads((directory/'platform.json').read_text()) if (directory/'platform.json').exists() else None
    if qualified:
        require(platform is not None and platform['uid'] == [0]*3 and platform['uname'][0] == 'Linux' and platform['uname'][4] == 'x86_64', 'PLATFORM')
        if not result.get('local_user_namespace'):
            device = accelerator.parse_device(platform['accelerator'])
            require(result.get('observed_accelerator') == device, 'OBSERVATION_MISMATCH')
        elif not local_fixture:
            raise ValueError('AUDIT:LOCAL_RESULT_IS_NOT_REMOTE_QUALIFICATION')
    checked = 0
    for row in rows:
        mode, name = row['mode'], row['name']
        # Failed case retains its diagnostic; it is never counted as a pass.
        if not row['check']:
            require(not qualified, 'FAILED_QUALIFIED_CASE')
            continue
        require(row.get('reaped') is True and row.get('parent_canary_unchanged') is True and row.get('parent_no_loopback_connection') is True and row.get('rootfs_unchanged') is True, 'CASE_PARENT_GATES')
        seal = json.loads(row['stderr'])['sealed']
        require(seal.get('uid') == [65534]*3 and seal.get('gid') == [65534]*3 and seal.get('groups') == [], 'SEAL_IDENTITY')
        require(seal.get('capabilities_zero') is True and seal.get('ambient_cleared') is True and seal.get('bounding_dropped',0) >= 38, 'SEAL_CAPABILITIES')
        require(seal.get('cwd') == '/' and seal.get('no_new_privs') is True and seal.get('seccomp') == 'native-allowlist' and seal.get('parent_death_signal') == 9, 'SEAL_PRIMITIVES')
        if name in ('wall','output'):
            require(row['status'] == ('timeout' if name=='wall' else 'output_limit') and row['returncode'] == -9 and 0 <= row['elapsed_s'] < 5, 'BOUNDED_CASE')
        else:
            require(row['status'] == 'completed' and row['returncode'] == 0, 'CASE_EXIT')
            if name == 'forged':
                require(row['stdout'] == 'PASS\n{"qualified": true}\n', 'FORGED_OUTPUT')
            else:
                value = json.loads(row['stdout'])
                if name == 'benign': expected = {'answer':5,'roundtrip':{'a':2,'b':3}}
                elif name == 'readonly': expected = 'denied'
                elif name == 'inherited': expected = {'stdin':'','env':{'LANG':'C','PYTHONDONTWRITEBYTECODE':'1'},'extra_fds':[]}
                elif name == 'syscalls': expected = {**{n:[-1,1] for n in DENIED},'subprocess':'denied','loopback':'denied'}
                else:
                    expected = {k:'denied' for k in ('absolute','traversal','symlink','proc','proc_environ','corpus','future','verifier','device','openat_fd','fchdir')}
                    expected.update(cwd='/', prefix={'visible':['SYNTHETIC_VISIBLE_PREFIX']} if mode=='inspection' else None)
                require(value == expected, 'CASE_OUTPUT')
        manifest = json.loads((directory/f'{mode}-{name}-rootfs.json').read_text())
        require({'work/candidate.py','work/args.json'} <= set(manifest), 'MANIFEST_MISSING_INPUTS')
        require(('work/prefix.json' in manifest) == (mode=='inspection'), 'MANIFEST_PREFIX')
        for path, digest in manifest.items():
            require(str(Path(path)) == path and not Path(path).is_absolute() and '..' not in Path(path).parts, 'MANIFEST_PATH')
            require(path.startswith('runtime/') or path in ('work/candidate.py','work/args.json','work/prefix.json','work/escape-link'), 'MANIFEST_ALLOWLIST')
            require(bool(re.fullmatch('[0-9a-f]{64}', digest)) or (path=='work/escape-link' and name=='filesystem' and digest.startswith('synthetic-test-symlink:')), 'MANIFEST_HASH')
        checked += 1
    require(checked == result['passed_cases'], 'PASSED_COUNT_NOT_INDEPENDENTLY_RECONCILED')
    return {'classification': result['classification'], 'qualified': qualified,
            'scope': 'LOCAL_FIXTURE_ONLY' if local_fixture else 'ARTIFACTS_ONLY_REMOTE_PROVENANCE_NOT_ESTABLISHED',
            'required': len(required), 'executed': len(rows), 'passed': checked, 'failed': len(rows)-checked, 'unexercised': len(required)-len(rows)}


def audit_remote(directory, build, status, server, notebook):
    origin = provenance(status, server, kernel=build['kernel'], version=build['version'], image=build['image'],
                        expected_notebook_sha256=build['notebook_sha256'], notebook=notebook)
    report = audit_directory(directory, identity=build['identity'], sources=build['assets'])
    require(not report['qualified'] or origin['terminal_status']=='COMPLETE', 'QUALIFIED_NONCOMPLETE_JOB')
    return {**report, **origin, 'scope':'REMOTE_EXACT_VERSION'}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--status', type=Path, required=True)
    parser.add_argument('--server', type=Path, required=True)
    parser.add_argument('--notebook', type=Path, required=True)
    args = parser.parse_args()
    report = audit_remote(args.directory, json.loads(args.contract.read_text()), json.loads(args.status.read_text()),
                          json.loads(args.server.read_text()), args.notebook.read_bytes())
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['qualified'] else 2)
