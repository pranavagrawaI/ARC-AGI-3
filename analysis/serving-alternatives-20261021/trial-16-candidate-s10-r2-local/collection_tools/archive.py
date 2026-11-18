"""Single future version-bound ZIP collection; exclusive, bounded, never retries."""
import argparse,datetime as dt,hashlib,json,sys,zipfile
from pathlib import Path
from identity import HERE,accepted
import space
sys.path.insert(0,str(HERE))
from http_error_capture import exception_record

def main():
    p=argparse.ArgumentParser();p.add_argument('destination',type=Path);a=p.parse_args();a.destination.mkdir(exist_ok=False)
    local_space=space.check(a.destination)
    (a.destination/'space-preflight.json').write_text(json.dumps(local_space,indent=2)+'\n')
    if not local_space['eligible']:raise ValueError('simultaneous2GiB archive+8GiB extraction+2GiB derived+1GiB reserve unavailable; no API')
    ref,kid,version=accepted();out=dict(started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),kernel_ref=ref,kernel_id=kid,version=version)
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest
    api=KaggleApi();api.authenticate();archive=a.destination/'output.zip'
    try:
        request=ApiDownloadKernelOutputRequest();request.owner_slug,request.kernel_slug=ref.split('/');request.version_number=version
        with api.build_kaggle_client() as c:response=c.kernels.kernels_api_client.download_kernel_output(request)
        response.raise_for_status();count=0
        with response,archive.open('xb') as stream:
            for block in response.iter_content(chunk_size=1024*1024):
                count+=len(block)
                if count>space.ARCHIVE:raise ValueError('2GiB compressed archive cap')
                stream.write(block)
        out.update(archive_bytes=count,archive_sha256=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest())
        extraction=a.destination/'members';extraction.mkdir();inventory=[];total=0
        with zipfile.ZipFile(archive) as zipped:
            if len(zipped.infolist())>30000:raise ValueError('member bound')
            for info in zipped.infolist():
                name=Path(info.filename);total+=info.file_size
                if name.is_absolute() or '..' in name.parts or total>space.EXTRACTED or (info.external_attr>>16)&0o170000==0o120000:raise ValueError('unsafe/oversized archive')
                if info.is_dir():continue
                data=zipped.read(info);target=extraction/name;target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as stream:stream.write(data)
                inventory.append(dict(name=info.filename,bytes=len(data),CRC=info.CRC,sha256=hashlib.sha256(data).hexdigest()))
        out.update(complete=True,CRC_verified=True,members=inventory,finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    except BaseException as error:out['error']=exception_record(error);out['complete']=False;raise
    finally:(a.destination/'ARCHIVE.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
