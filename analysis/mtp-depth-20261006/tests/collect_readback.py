"""Read every frozen private dataset file at an explicit version and compare hashes.

Run with the Kaggle tool's Python after dataset version creation has completed.
Raw downloads stay outside the checkout; only the matching receipt is committed.
"""
import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REF = 'pranavagrawai/mtp-depth-replay-data-20261006'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', type=int, required=True)
    args = ap.parse_args()
    ref = f'{REF}/{args.version}'
    root = Path(tempfile.mkdtemp(prefix=f'mtp-readback-v{args.version}-'))
    local = HERE / 'kaggle/dataset'
    manifest = json.loads((local / 'manifest.json').read_text())

    def fetch(name):
        out = root / name
        out.mkdir()
        # Each invocation owns a different directory; no concurrent file writes are shared.
        cmd = ['kaggle', 'datasets', 'download', ref, '-f', name, '-p', str(out), '-o', '-q']
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        (out / 'fetch.log').write_text(r.stdout + r.stderr)
        if r.returncode:
            raise RuntimeError(f'{name}: download failed; see {out / "fetch.log"}')
        target = out / name
        if not target.is_file():
            for archive in out.glob('*.zip'):
                with zipfile.ZipFile(archive) as z:
                    matches = [member for member in z.namelist() if Path(member).name == name]
                    if len(matches) == 1:
                        target.write_bytes(z.read(matches[0]))
        if not target.is_file():
            raise RuntimeError(f'{name}: no downloaded file at {out}')
        return name, sha(target)

    _, remote_manifest_sha = fetch('manifest.json')
    if remote_manifest_sha != sha(local / 'manifest.json'):
        raise RuntimeError('uploaded manifest differs from the frozen local payload')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        readback = dict(pool.map(fetch, manifest))
    bad = sorted(name for name in manifest if readback[name] != manifest[name])
    receipt = {'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
               'dataset': REF, 'version': args.version,
               'method': 'version-specific per-file kaggle datasets download -f',
               'manifest_sha256': remote_manifest_sha, 'files': readback,
               'mismatches': bad, 'raw_download_dir': str(root)}
    (HERE / 'evidence/prelaunch-20261007/upload-readback.json').write_text(json.dumps(receipt, indent=2)+'\n')
    if bad:
        raise RuntimeError(f'dataset readback mismatches: {bad}')
    (HERE / 'evidence/dataset-readback.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(f'Version {args.version}: {len(readback)}/{len(manifest)} files match, manifest {remote_manifest_sha}', flush=True)


if __name__ == '__main__':
    main()
