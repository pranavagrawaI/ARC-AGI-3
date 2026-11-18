"""Build the successor locally; never imports or calls the Kaggle SDK."""
import json
from pathlib import Path
import sys
import tempfile
import transport

HERE=Path(__file__).resolve().parent
RUNNER=HERE
sys.path.insert(0,str(RUNNER))
import build as runner_build


def prepare(destination, assignment=None):
    spec=json.loads((HERE/'packet-spec.json').read_text())
    destination=Path(destination)
    # The builder validates reviewed source/config identities against NEW authority.
    with tempfile.TemporaryDirectory(prefix='release12-inner-') as directory:
        inner_dir=Path(directory)/'inner'
        manifest=runner_build.build(spec,inner_dir,assignment=assignment)
        raw=(inner_dir/spec['metadata']['code_file']).read_bytes()
        outer=transport.wrap(raw)
        if transport.decode(outer)!=raw:raise ValueError('lossless transport mismatch')
        if len(outer)>=1_000_000:raise ValueError('outer notebook exceeds local transport target')
        destination.mkdir(exist_ok=False)
        (destination/spec['metadata']['code_file']).write_bytes(outer)
        (destination/'kernel-metadata.json').write_bytes((inner_dir/'kernel-metadata.json').read_bytes())
    manifest.update(inner_notebook_sha256=transport.sha(raw),inner_source_sha256=transport.sha(transport.cell_source(json.loads(raw)).encode()),
                    outer_notebook_sha256=transport.sha(outer),outer_notebook_bytes=len(outer),
                    kernel_ref=spec['metadata']['id'],platform_hard_timeout_s=3900)
    # Previous manifest packet_files described temporary INNER notebook; replace explicitly.
    manifest['packet_files']={p.name:transport.sha(p.read_bytes()) for p in destination.iterdir()}
    return manifest


if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('packet.py REVIEW_DESTINATION (unassigned only)')
    report=prepare(Path(sys.argv[1]))
    (HERE/'review-packet.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
