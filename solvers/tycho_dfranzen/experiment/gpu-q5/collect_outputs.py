"""Fetch critical evidence directly before optional checkpoint/blob downloads."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import time


def safe_path(name):
    path=PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('unsafe remote output path')
    return path.as_posix()


class KaggleOutput:
    def __init__(self, kernel, version):
        from kaggle.api.kaggle_api_extended import KaggleApi
        self.owner,self.slug=kernel.split('/')
        self.version=version
        self.api=KaggleApi(); self.api.authenticate()

    def fetch(self,name):
        import requests
        from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest
        request=ApiDownloadKernelOutputRequest()
        request.owner_slug=self.owner; request.kernel_slug=self.slug
        request.version_number=self.version; request.file_path=safe_path(name)
        with self.api.build_kaggle_client() as client:
            redirect=client.kernels.kernels_api_client.download_kernel_output(request)
        response=requests.get(redirect.url,timeout=(10,30))
        response.raise_for_status()
        return response.content


def collect(remote,directory,*,legacy_index=None,bulk=False):
    directory=Path(directory); output=directory/'output'; output.mkdir(parents=True,exist_ok=True)
    receipt={'files':[], 'errors':[], 'critical_complete':False,
             'interpretation':'Retrieval only; gameplay and validity require an audit.'}
    receipt_path=directory/'collection-receipt.json'
    def save(): receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    def fetch(name,*,required=True,expected=None,phase='critical'):
        started=time.monotonic()
        try:
            name=safe_path(name); data=remote.fetch(name)
            digest=hashlib.sha256(data).hexdigest()
            if expected and digest!=expected: raise ValueError('output hash mismatch')
            dest=output/name; dest.parent.mkdir(parents=True,exist_ok=True)
            temporary=dest.with_name(dest.name+'.partial'); temporary.write_bytes(data); temporary.replace(dest)
            receipt['files'].append({'file':name,'sha256':digest,'bytes':len(data),
                                     'phase':phase,'seconds':time.monotonic()-started})
            save(); return data
        except Exception as error:
            # URLs can contain credentials. Persist exception type, not its URL.
            receipt['errors'].append({'file':name,'required':required,'phase':phase,
                                      'error_type':type(error).__name__,
                                      'seconds':time.monotonic()-started})
            save(); return None
    for name in ('tycho-dfranzen/benchmark.json','tycho-dfranzen/candidate-config.json',
                 'qualification-start.json'):
        fetch(name)
    for name in ('qualification-complete.json','watchdog-receipt.json','serve.log'):
        fetch(name,required=False)
    raw=fetch('tycho-dfranzen/evidence-index.json',required=legacy_index is None)
    if raw is not None:
        index=json.loads(raw)
        files=index['files']
    elif legacy_index is not None:
        # Historical q4 has no index emitted by its runner. Use its preserved
        # remote inventory without listing thousands of dependency files again.
        names=json.loads(Path(legacy_index).read_text())
        files=[{'path':safe_path(name),'critical':bool(re.fullmatch(
            r'tycho-dfranzen/hybrid-games/[^/]+/[^/]+\.(json|jsonl|log)',name))}
            for name in names if name.startswith('tycho-dfranzen/hybrid-games/')]
    else:
        save(); return receipt
    done={item['file'] for item in receipt['files']}
    for item in files:
        if item['critical'] and item['path'] not in done:
            fetch(item['path'],expected=item.get('sha256'))
    receipt['critical_complete']=not any(e['required'] and e['phase']=='critical' for e in receipt['errors'])
    save()
    # Never bury absent core evidence under a bulk transfer.
    if bulk and receipt['critical_complete']:
        for item in files:
            if not item['critical']:
                fetch(item['path'],expected=item.get('sha256'),required=False,phase='bulk')
    save(); return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kernel'); parser.add_argument('directory',type=Path)
    parser.add_argument('--version',type=int,required=True)
    parser.add_argument('--legacy-index',type=Path)
    parser.add_argument('--bulk',action='store_true')
    args=parser.parse_args()
    receipt=collect(KaggleOutput(args.kernel,args.version),args.directory,
                    legacy_index=args.legacy_index,bulk=args.bulk)
    print(json.dumps({'critical_complete':receipt['critical_complete'],
                      'files':len(receipt['files']),'errors':receipt['errors']}))
    raise SystemExit(0 if receipt['critical_complete'] else 1)


if __name__=='__main__': main()
