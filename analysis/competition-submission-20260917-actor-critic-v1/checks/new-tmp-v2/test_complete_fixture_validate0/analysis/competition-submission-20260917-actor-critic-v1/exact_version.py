"""Read-only exact-version SDK status/output retrieval. Run with Kaggle SDK Python.
Never use CLI status/output owner/slug/version: installed CLI discards version.
"""
import argparse,json

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['status','output']);p.add_argument('--version',type=int,required=True);a=p.parse_args()
    if a.version<=0:raise ValueError('Positive exact version required')
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest,ApiDownloadKernelOutputRequest
    api=KaggleApi();api.authenticate()
    request=ApiGetKernelSessionStatusRequest() if a.action=='status' else ApiDownloadKernelOutputRequest()
    request.user_name='pranavagrawai';request.kernel_slug='arc3-ac-competition-20260917-v1'
    if a.action=='status':request.version_label=str(a.version)
    else:request.version_number=a.version
    with api.build_kaggle_client() as client:
        method=client.kernels.kernels_api_client.get_kernel_session_status if a.action=='status' else client.kernels.kernels_api_client.download_kernel_output
        response=method(request)
    # Output response can contain signed URLs. Caller must retain privately, not paste.
    print(response)
if __name__=='__main__':main()
