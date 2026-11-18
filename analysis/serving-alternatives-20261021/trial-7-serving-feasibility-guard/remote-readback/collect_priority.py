import datetime as dt,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parent))
from http_error_capture import exception_record
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi();api.authenticate();ref='pranavagrawai/serving-candidate-feas7-20261023'
pattern=r'(/result\.json$|/supervisor\.jsonl$|/candidate-functional/(verdict|raw-readback-verdict|progress)\.json$|/actual-gpu-query\.json$|/actual-nvidia-smi\.json$|/host-headroom\.json$|/candidate\.server\.(stdout|stderr)$|/candidate\.(prepare|identity|functional|qualify)\.(stdout|stderr)$|/runtime-input-hashes\.json$|/model-source-identity\.json$|/installed-package-identity\.json$|/installed-source-identity\.json$|/candidate-launch\.json$|/prepared\.json$)'
report={'utc':dt.datetime.now(dt.timezone.utc).isoformat()}
try:
    paths,token=api.kernels_output(ref,path=str(P/'output'),file_pattern=pattern,force=False,quiet=True,page_size=100)
    report['files']=paths;report['next_page_token']=token
except Exception as e:report['error']=exception_record(e)
s=api.kernels_status(ref);report['status']=type(s).to_dict(s)
with api.build_kaggle_client() as c:q=c.kernels.kernels_api_client.get_accelerator_quota_statistics()
report['quota']={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'used_s':q.gpu_quota.time_used.total_seconds(),'reserved_s':q.gpu_quota.time_reserved.total_seconds(),'delta_used_s':q.gpu_quota.time_used.total_seconds()-59275.88}
(P/'priority-collection.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))
