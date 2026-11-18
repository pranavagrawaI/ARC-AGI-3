"""Read-only exact-version SDK status/output retrieval; never CLI version suffixes.
Run with the installed Kaggle SDK Python. T3 does not invoke this tool remotely.
"""
import argparse,hashlib,json
from pathlib import Path
TARGET='pranavagrawai/arc3-ac-competition-20260917-v1'

def request_for(action,version,file_path=None):
    if type(version) is not int or version<=0:raise ValueError('Positive exact version required')
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest,ApiDownloadKernelOutputRequest
    owner,slug=TARGET.split('/')
    if action=='status':
        r=ApiGetKernelSessionStatusRequest();r.user_name=owner;r.kernel_slug=slug;r.version_label=str(version)
    elif action=='output':
        if not file_path or Path(file_path).is_absolute() or '..' in Path(file_path).parts:raise ValueError('Safe output file required')
        r=ApiDownloadKernelOutputRequest();r.owner_slug=owner;r.kernel_slug=slug;r.version_number=version;r.file_path=file_path
    else:raise ValueError('Unknown action')
    return r

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['status','output']);p.add_argument('--version',type=int,required=True);p.add_argument('--file');p.add_argument('--destination',type=Path);a=p.parse_args()
    r=request_for(a.action,a.version,a.file)
    if a.action=='output' and (a.destination is None or a.destination.exists()):raise ValueError('New destination required')
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    with api.build_kaggle_client() as client:
        if a.action=='status':
            response=client.kernels.kernels_api_client.get_kernel_session_status(r)
            print(json.dumps(dict(target=TARGET,version=a.version,status=response.status.name,failure_message=response.failure_message)))
        else:
            response=client.kernels.kernels_api_client.download_kernel_output(r)
            try:
                response.raise_for_status();data=bytearray()
                for chunk in response.iter_content(65536):
                    data.extend(chunk)
                    if len(data)>64*1024*1024:raise RuntimeError('Output exceeds 64 MiB read bound')
                with a.destination.open('xb') as f:f.write(data)
                print(json.dumps(dict(target=TARGET,version=a.version,file=a.file,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())))
            finally:response.close()
if __name__=='__main__':main()
