"""Verify the exact packaged wheel and installed source without importing SGLang."""
import hashlib
from pathlib import Path
import zipfile


def package_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_wheel(wheel, contract):
    wheel = Path(wheel)
    expected = contract['wheel']
    if wheel.name != expected['filename'] or package_sha256(wheel) != expected['sha256']:
        raise ValueError('packaged wheel identity mismatch')
    with zipfile.ZipFile(wheel) as archive:
        members = archive.namelist()
        for relative, digest in contract['files'].items():
            member = 'sglang/' + relative
            if members.count(member) != 1 or hashlib.sha256(archive.read(member)).hexdigest() != digest:
                raise ValueError('wheel source mismatch: ' + relative)
    return expected['sha256']


def verify_installed(root, contract):
    for relative, digest in contract['files'].items():
        if package_sha256(Path(root) / relative) != digest:
            raise ValueError('installed source mismatch: ' + relative)
    return contract['files']


def verify_bundle(wheel_directory, lock_path, contract):
    wheels = sorted(Path(wheel_directory).glob('sglang-*.whl'))
    if len(wheels) != 1:
        raise ValueError('expected one packaged SGLang wheel')
    verify_wheel(wheels[0], contract)
    if package_sha256(lock_path) != contract['wheel']['requirements_lock_sha256']:
        raise ValueError('package requirements lock mismatch')
    return contract['wheel']
