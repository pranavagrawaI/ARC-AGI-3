"""Freeze/reconcile this offline audit without network or paid requests."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
PACKAGE=REPO/'deployment/kaggle'/HERE.name
MANIFEST=HERE/'MANIFEST.json'


def inventory():
    return {str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest()
            for root in (HERE,PACKAGE) for p in sorted(root.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p!=MANIFEST}


def verify():
    manifest=json.loads(MANIFEST.read_text())
    actual=inventory()
    assert actual==manifest['files'], {'missing':sorted(set(manifest['files'])-set(actual)),
        'added':sorted(set(actual)-set(manifest['files'])),
        'changed':sorted(p for p in set(actual)&set(manifest['files']) if actual[p]!=manifest['files'][p])}
    import verify_package
    verify_package.verify()
    return {'verified':True,'identity':HERE.name,'files_verified':len(actual),
            'manifest_sha256':hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),'launch_authorized':False}

if __name__=='__main__':
    if '--freeze' in sys.argv:
        with MANIFEST.open('x') as stream:
            json.dump({'identity':HERE.name,'scope':'OFFLINE_AUDIT_NO_LAUNCH','launch_authorized':False,'files':inventory()},stream,indent=2)
            stream.write('\n')
    print(json.dumps(verify(),indent=2))
