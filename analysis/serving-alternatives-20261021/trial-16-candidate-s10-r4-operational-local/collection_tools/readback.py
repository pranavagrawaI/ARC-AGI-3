"""One future read-only source/status/typed-quota readback. No auth repair."""
import argparse,datetime as dt,json,sys
from pathlib import Path
from identity import HERE,accepted
sys.path.insert(0,str(HERE))
import transport,reconciliation,quota_wire
from http_error_capture import exception_record

def main():
    parser=argparse.ArgumentParser();parser.add_argument('destination',type=Path);a=parser.parse_args();a.destination.mkdir(exist_ok=False)
    ref,kid,version=accepted();out=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),kernel_ref=ref,kernel_id=kid,version=version)
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    api=KaggleApi();api.authenticate()
    try:
        with api.build_kaggle_client() as c:
            request=ApiGetKernelRequest();request.user_name,request.kernel_slug=ref.split('/');response=c.kernels.kernels_api_client.get_kernel(request)
        meta=type(response.metadata).to_dict(response.metadata)
        if meta['id']!=kid or meta['currentVersionNumber']!=version or meta['ref']!=ref:raise ValueError('remote current version identity drift')
        remote=response.blob.source.encode();spec=json.loads((HERE/'packet-spec.json').read_text());local=(HERE/'kernel'/spec['metadata']['code_file']).read_bytes()
        attempt=json.loads((HERE/'launch-attempt.json').read_text())
        if transport.cell_source(json.loads(remote))!=transport.cell_source(json.loads(local)) or transport.sha(transport.decode(remote))!=attempt['inner_notebook_sha']:raise ValueError('exact armed source mismatch')
        (a.destination/'source.ipynb').write_bytes(remote);(a.destination/'metadata.json').write_text(json.dumps(meta,indent=2))
        status=api.kernels_status(ref)
        out.update(source_exact=True,inner_sha256=transport.sha(transport.decode(remote)),status=type(status).to_dict(status))
        out['typed_quota']=quota_wire.read(api)
        policy=reconciliation.load_policy()
        out['typed_quota'].update(baseline_used_s=policy['current_quota']['used_s'],historical_predecessor_used_s=policy['historical_predecessor_used_s'])
        reconciliation.validate_quota(out['typed_quota'],policy,prelaunch=False)
    except BaseException as error:out['error']=exception_record(error);raise
    finally:(a.destination/'READBACK.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
