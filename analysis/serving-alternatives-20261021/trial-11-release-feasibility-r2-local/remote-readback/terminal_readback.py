import datetime as dt,json,sys
from pathlib import Path
STAGE=sys.argv[1] if len(sys.argv)>1 else 'terminal'
assert STAGE in ('initial','terminal')
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent));import transport
from http_error_capture import exception_record
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
api=KaggleApi();api.authenticate();ref='pranavagrawai/serving-release11-20261023'
out={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'kernel_ref':ref,'accepted_version':1,'accepted_kernel_id':140425916}
try:
    with api.build_kaggle_client() as c:
        req=ApiGetKernelRequest();req.user_name='pranavagrawai';req.kernel_slug='serving-release11-20261023'
        response=c.kernels.kernels_api_client.get_kernel(req)
    meta=type(response.metadata).to_dict(response.metadata)
    (P/f'get-kernel-{STAGE}.json').write_text(json.dumps(type(response).to_dict(response),indent=2)+'\n')
    (P/f'source-{STAGE}.ipynb').write_text(response.blob.source)
    (P/f'metadata-{STAGE}.json').write_text(json.dumps(meta,indent=2)+'\n')
    assert meta['id']==140425916 and meta['currentVersionNumber']==1 and meta['ref']==ref
    remote=response.blob.source.encode();local=(P.parent/'kernel/serving-release11-20261023.ipynb').read_bytes()
    inner=transport.decode(remote);marker=json.loads((P.parent/'launch-attempt.json').read_text())
    assert transport.cell_source(json.loads(remote))==transport.cell_source(json.loads(local))
    assert transport.sha(inner)==marker['inner_notebook_sha']
    out.update(metadata_version_identity_verified=True,outer_cell_exact=True,decoded_inner_sha256=transport.sha(inner),decoded_inner_matches_armed=True,
               remote_source_sha256=transport.sha(remote),remote_metadata_sha256=transport.sha((P/f'metadata-{STAGE}.json').read_bytes()))
except Exception as e:out['source_error']=exception_record(e)
try:
    s=api.kernels_status(ref);out['status']=type(s).to_dict(s)
except Exception as e:out['status_error']=exception_record(e)
with api.build_kaggle_client() as c:q=c.kernels.kernels_api_client.get_accelerator_quota_statistics()
out['quota']={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'used_s':q.gpu_quota.time_used.total_seconds(),'reserved_s':q.gpu_quota.time_reserved.total_seconds()}
(P/f'{STAGE}-source-status-quota.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
