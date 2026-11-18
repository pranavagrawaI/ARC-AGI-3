"""Exact reviewed capacity-purpose boundary; no SDK or launch at import."""
import argparse,hashlib,json,signal,time
from pathlib import Path
from urllib.parse import urlsplit
from phase_tools import require,positive
HERE=Path(__file__).resolve().parent

def validate(config,check_cwd=True):
 require(config['mode'] in ('GPU_SCREEN','CPU_FIXTURE_ONLY'),'mode')
 require(config['purpose']=='PAIRED_CAPACITY' and config['estimand']=='WIRE_EQUIVALENT_TOTAL_PACKAGE_API_CAPACITY_EXPERIMENTAL_NONHOST','capacity estimand')
 require(config['arm_order']==['control','candidate'],'fixed order')
 require(Path(config['cwd']).is_absolute() and (not check_cwd or Path(config['cwd']).is_dir()),'absolute existing cwd')
 require(positive(config['total_seconds']) and positive(config['cleanup_seconds']) and config['total_seconds']>config['cleanup_seconds'],'absolute cap')
 if config['mode']=='GPU_SCREEN':
  contract=json.loads((HERE/'capacity-contract.json').read_text())
  require(hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()==contract['config_sha256'],'exact reviewed capacity config')
  require((config['total_seconds'],config['cleanup_seconds'],config['platform_seconds'])==(10200,300,10500),'caps')
 else:require(config['total_seconds']<=600,'CPU cap')
 for arm in ('control','candidate'):
  a=config[arm]
  require(set(a)=={'prepare','identity','server','ready_url','ready_seconds','request_seconds','ready_poll_seconds','ready_retry_http_statuses','replay','qualify','drain_seconds'},'arm fields')
  for phase in ('prepare','identity','server','replay','qualify'):require(a[phase]['argv'] and positive(a[phase]['seconds']),'phase')
  endpoint=urlsplit(a['ready_url']);require(endpoint.hostname=='127.0.0.1' and endpoint.port is not None,'loopback')
 return config

from capacity_runner import Runner

def main():
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('output');p.add_argument('--assignment',required=True);a=p.parse_args()
 config=validate(json.loads(Path(a.config).read_text()));assignment=json.loads(Path(a.assignment).read_text())
 require(assignment['launch_authorized'] is True and assignment['local_only'] is False and assignment['serialized_assignment'],'real serialized authority required')
 require(assignment['config_sha256']==hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest() and assignment['total_seconds']==10200 and assignment['platform_hard_timeout_s']==10500,'authority config/caps')
 def stop(signum,frame):raise RuntimeError('signal '+str(signum))
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 value=Runner(config,a.output).run();print(json.dumps(value),flush=True);return 0 if value['pass_'] else 1
if __name__=='__main__':raise SystemExit(main())
