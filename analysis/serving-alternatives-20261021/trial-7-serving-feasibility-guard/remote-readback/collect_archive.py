"""Version-bound, read-only full output download. Never overwrite collected evidence."""
import datetime as dt,hashlib,json,sys,zipfile
from pathlib import Path
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parent))
from http_error_capture import exception_record
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest
out={'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'kernel_ref':'pranavagrawai/serving-candidate-feas7-20261023','version':1,'kernel_id':140347143,'policy':'read-only; version-bound; no overwrite'}
api=KaggleApi();api.authenticate()
archive=P/'output-v1-kernel140347143.zip'
try:
    request=ApiDownloadKernelOutputRequest();request.owner_slug='pranavagrawai';request.kernel_slug='serving-candidate-feas7-20261023';request.version_number=1
    with api.build_kaggle_client() as c:response=c.kernels.kernels_api_client.download_kernel_output(request)
    out['download_status']=response.status_code
    out['download_headers']={k:v for k,v in response.headers.items() if k.lower() in ('content-type','content-length','content-disposition','etag','last-modified')}
    count=0;digest=hashlib.sha256()
    with response,archive.open('xb') as stream:
        for chunk in response.iter_content(chunk_size=1024*1024):
            count+=len(chunk)
            if count>1024**3:raise ValueError('output archive exceeds1GiB collection bound')
            digest.update(chunk);stream.write(chunk)
    out['archive_bytes']=count;out['archive_sha256']=digest.hexdigest()
    extraction=P/'archive-extracted';extraction.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as zipped:
        out['members']=[{'name':info.filename,'bytes':info.file_size,'compressed_bytes':info.compress_size,'CRC':info.CRC} for info in zipped.infolist()]
        for info in zipped.infolist():
            name=Path(info.filename)
            if name.is_absolute() or '..' in name.parts:raise ValueError('unsafe archive path')
            if info.is_dir():continue
            path=extraction/name;path.parent.mkdir(parents=True,exist_ok=True)
            data=zipped.read(info)
            with path.open('xb') as stream:stream.write(data)
    # Compare overlap without replacing anything collected earlier.
    mismatches=[];overlap=0
    for file in extraction.rglob('*'):
        if not file.is_file():continue
        prior=P/'output'/file.relative_to(extraction)
        if prior.is_file():
            overlap+=1
            if hashlib.sha256(prior.read_bytes()).digest()!=hashlib.sha256(file.read_bytes()).digest():mismatches.append(str(file.relative_to(extraction)))
    out.update(extracted_files=sum(x.is_file() for x in extraction.rglob('*')),prior_overlap_files=overlap,prior_overlap_differences=mismatches,completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
except Exception as e:
    out['error']=exception_record(e)
    (P/'archive-collection.json').write_text(json.dumps(out,indent=2)+'\n')
    raise
(P/'archive-collection.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='members'},indent=2))
