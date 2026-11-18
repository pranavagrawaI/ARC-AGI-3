"""Exact candidate-first paired purpose and authority boundary; no SDK at import."""
import argparse,hashlib,json,signal
from pathlib import Path
from phase_tools import require,positive
HERE=Path(__file__).resolve().parent

def validate(config,check_cwd=True):
    require(config['mode'] in ('GPU_SCREEN','CPU_FIXTURE_ONLY'),'mode')
    require(config['purpose']=='CANDIDATE_FIRST_PAIRED_CAPACITY' and config['arm_order']==['candidate','control'],'candidate alone purpose/order')
    require(config['estimand']=='WIRE_EQUIVALENT_FIXED_CANDIDATE_FIRST_TOTAL_PACKAGE_API_CAPACITY','estimand')
    require(Path(config['cwd']).is_absolute() and (not check_cwd or Path(config['cwd']).is_dir()),'absolute existing cwd')
    require(positive(config['total_seconds']) and positive(config['cleanup_seconds']) and config['total_seconds']>config['cleanup_seconds'],'absolute cap')
    if config['mode']=='GPU_SCREEN':
        contract=json.loads((HERE/'candidate-contract.json').read_text())
        require(hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()==contract['config_sha256'],'exact candidate config')
        require((config['total_seconds'],config['cleanup_seconds'],config['platform_seconds'])==(10200,300,10500),'caps')
        require(config['candidate']['server']['seconds']==config['candidate']['ready_seconds']==1800,'startup unchanged')
    else:require(config['total_seconds']<=600,'CPU cap')
    for arm in ('candidate','control'):
        for name in ('prepare','identity','server','replay','qualify'):
            spec=config[arm][name];require(spec['argv'] and positive(spec['seconds']),'phase')
    require(config['candidate']['ready_url']=='http://127.0.0.1:8001/health','exact loopback endpoint')
    require(config['final_resource_seconds']==180 and config['final_host_policy']=='MEASURE_REQUIRED_ORIGINAL140_REPORT_ONLY','final resource policy')
    return config


from pair_runner import Runner

def main():
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('output');p.add_argument('--assignment',required=True);a=p.parse_args()
    config=validate(json.loads(Path(a.config).read_text()));assignment=json.loads(Path(a.assignment).read_text())
    require(assignment['launch_authorized'] is True and assignment['local_only'] is False and assignment['serialized_assignment'],'real serialized authority required')
    require(assignment['config_sha256']==hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest() and assignment['total_seconds']==10200 and assignment['platform_hard_timeout_s']==10500,'authority config/caps')
    def stop(signum,frame):raise RuntimeError('signal '+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    value=Runner(config,a.output).run();print(json.dumps(value),flush=True)
    return 0 if value['pass_'] else 1
if __name__=='__main__':raise SystemExit(main())
