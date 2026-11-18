"""Chunk-configuration ABBA, exact successor cache controls; no tools executed."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import baseline as base

OUT = Path('/kaggle/working/prefix-cache-20260907')
CHUNKS = (2048, 4096, 4096, 2048)
RUNNER_CAP_S = 9000
REQUEST_CAP = 148
REQUEST_COUNT = 0
REQUEST_LOCK = threading.Lock()


def admit_request():
    global REQUEST_COUNT
    with REQUEST_LOCK:
        if REQUEST_COUNT >= REQUEST_CAP:
            raise RuntimeError('148-request cap reached; no replacement request')
        REQUEST_COUNT += 1
        base.save(OUT/'request-count.json', {'admitted':REQUEST_COUNT,'cap':REQUEST_CAP})


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def engine_command(runtime, model, chunk):
    if chunk not in (2048, 4096):
        raise ValueError('Unsupported experimental chunk')
    command = runtime.sglang_command(model, '127.0.0.1', 30000,
                                    attention_backend='triton', sampling_backend='pytorch')
    def option(flag, value):
        if flag in command:
            at = command.index(flag)
            command[at+1] = str(value)
        else:
            command.extend([flag, str(value)])
    option('--chunked-prefill-size', chunk)
    option('--mem-fraction-static', 0.85)
    option('--stream-interval', 1)
    option('--sampling-defaults', 'model')
    if '--enable-metrics' not in command:
        command.append('--enable-metrics')
    return command


def validate_server_info(info, chunk):
    if info.get('chunked_prefill_size') != chunk:
        raise RuntimeError('Resolved chunk differs from requested arm')
    if info.get('mem_fraction_static') != 0.85:
        raise RuntimeError('Resolved memory fraction differs from fixed 0.85')
    if info.get('speculative_algorithm') is not None or info.get('sampling_defaults') != 'model':
        raise RuntimeError('Unexpected speculative/sampling configuration')
    keys = [k for k in info if any(word in k for word in
            ('chunk', 'mamba', 'graph', 'page', 'overlap', 'cache', 'mem_fraction'))]
    resolved = {k: info[k] for k in keys}
    for key in ['mamba_cache_chunk_size', 'mamba_track_interval', 'mamba_radix_cache_strategy', 'cuda_graph_config']:
        resolved.setdefault(key, None)
    resolved['missing_meaning'] = 'null means unavailable, not disabled or zero; retain full server-info and startup log'
    return resolved


def fixed_payload(ids, tokens=256):
    return base.payload(ids, tokens)


def checked_generate(name, body):
    admit_request()
    row = base.generate(name, body)
    m = row['meta_info']
    total, cached = m.get('prompt_tokens'), m.get('cached_tokens')
    if not isinstance(cached, int) or not isinstance(total, int) or not 0 <= cached <= total:
        raise RuntimeError('Missing or invalid cached-token telemetry: '+name)
    prefill = row.get('server_forward_to_prefill_finished_s')
    if not isinstance(prefill, (float, int)) or not 0 <= prefill < float('inf'):
        raise RuntimeError('Missing or invalid prefill timing: '+name)
    last = None
    stream = base.OUT/(name+'-stream.jsonl')
    if stream.exists():
        with stream.open() as handle:
            for line in handle:
                last = json.loads(line)['data']
    if last is not None:
        row['output_fingerprint'] = {
            'token_ids_sha256':digest(last['output_ids']) if isinstance(last.get('output_ids'),list) else None,
            'text_sha256':digest(last['text']) if isinstance(last.get('text'),str) else None}
    base.save(base.OUT/(name+'-result.json'),row)
    return row


def render_records(corpus, model, reference, expected=None):
    records = {record['name']: record for pair in corpus['pairs']
               for record in (pair['seed'], pair['successor'])}
    records.update({record['name']: record for record in corpus['batch_records']})
    if len(records) != 5:
        raise RuntimeError('Expected five frozen distinct natural requests')
    rendered = {}
    for name, record in records.items():
        base.http('/flush_cache')
        admit_request()
        rendered[name] = base.chat('natural-'+name, record, model, reference)
        if expected is not None and rendered[name] != expected[name]:
            raise RuntimeError('Cross-arm rendered token drift: '+name)
    base.save(base.OUT/'rendered-inputs.json', rendered)
    for pair in corpus['pairs']:
        base.http('/flush_cache')
        start=time.perf_counter()
        for role in ['seed','successor']:
            record=pair[role]
            admit_request()
            ids=base.chat('natural-warm-'+pair['name']+'-'+role,record,model,reference)
            if ids != rendered[record['name']]:
                raise RuntimeError('Natural warm rendered token drift')
        base.save(base.OUT/(pair['name']+'-natural-check.json'),{
            'pair':pair['name'],'warm_sequence_wall_s':time.perf_counter()-start,
            'cold_successor':'natural-'+pair['successor']['name']+'-result.json',
            'warm_successor':'natural-warm-'+pair['name']+'-successor-result.json',
            'interpretation':'Natural outputs/schema checks; different lengths forbid attributing e2e difference to cache alone'})
    return rendered


def experiments(corpus, model, reference, rendered=None, reverse=False):
    rendered = render_records(corpus, model, reference, expected=rendered)
    pairs = [{'name': p['name'], 'kind': p['kind'],
              'seed': rendered[p['seed']['name']],
              'successor': rendered[p['successor']['name']]}
             for p in corpus['pairs']]
    long_pair = pairs[-1]
    def stretch(ids):
        return (ids*((95000+len(ids)-1)//len(ids)))[:95000]
    pairs.append({'name': 'mechanical-95000', 'kind': 'mechanical repetition, not natural history',
                  'seed': stretch(long_pair['seed']), 'successor': stretch(long_pair['successor'])})
    base.save(base.OUT/'inputs.json', [{'name':p['name'], 'kind':p['kind'],
        'seed_tokens':len(p['seed']), 'successor_tokens':len(p['successor']),
        'seed_sha256':digest(p['seed']), 'successor_sha256':digest(p['successor']),
        'lcp_tokens':next((i for i,(a,b) in enumerate(zip(p['seed'],p['successor'])) if a!=b),
                         min(len(p['seed']),len(p['successor'])))} for p in pairs])
    for p in pairs:
        base.http('/flush_cache')
        checked_generate('warm-'+p['name'], fixed_payload(p['successor'],16))
    pair_order = list(reversed(pairs)) if reverse else pairs
    for p in pair_order:
        for condition in (['predecessor','cold'] if reverse else ['cold','predecessor']):
            name = p['name']+'-'+condition
            base.http('/flush_cache')
            start = time.perf_counter()
            prime = None
            if condition == 'predecessor':
                prime = checked_generate(name+'-prime',fixed_payload(p['seed'],16))
                if prime['cached_input_tokens'] != 0:
                    raise RuntimeError('Cold predecessor prime had cached tokens')
            successor = checked_generate(name+'-successor',fixed_payload(p['successor']))
            if condition == 'cold' and successor['cached_input_tokens'] != 0:
                raise RuntimeError('Cold successor had cached tokens')
            base.save(base.OUT/(name+'-condition.json'), {
                'condition':condition,'pair':p['name'],'kind':p['kind'],
                'condition_wall_s':time.perf_counter()-start,'prime':prime,'successor':successor,
                'interpretation':'Priming cost included in condition wall; predecessor prompt with fresh16 tokens does not recreate recorded live continuation'})
    cases = {r['name']:rendered[r['name']] for r in corpus['batch_records']}
    workloads = {'captured':cases,'long95000':{n:stretch(ids) for n,ids in cases.items()}}
    for workload in (list(reversed(workloads)) if reverse else workloads):
        for c in ([4,1] if reverse else [1,4]):
            base.http('/flush_cache')
            name = f'batch-{workload}-{c}'
            items = [(name+'-'+n,fixed_payload(ids)) for n,ids in workloads[workload].items()]
            result = base.finite_batch(items,c,checked_generate,base.OUT/(name+'-events.jsonl'))
            result.update(workload=workload,input_sha256={n:digest(ids) for n,ids in workloads[workload].items()})
            base.save(base.OUT/(name+'.json'),result)
    return rendered


def run_arm(runtime, model, reference, corpus, index, chunk, rendered):
    base.OUT = OUT/f'arm-{index}-chunk-{chunk}'
    base.OUT.mkdir(parents=True)
    base.ROWS = []
    base.PROCESS = None
    command = engine_command(runtime, model, chunk)
    base.save(base.OUT/'launch.json',{'command':command,'chunk':chunk,'arm_index':index,
        'interpretation':'Chunk configuration includes resolved graph and Mamba cache policies'})
    env = dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',R8_L3B_OWNER=runtime.R8_L3B_OWNER)
    stop=threading.Event(); target=None; monitor=None; ready=False
    with (base.OUT/'server.log').open('w') as log:
        try:
            base.PROCESS=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,text=True)
            target=runtime.process_identity(base.PROCESS.pid)
            runtime.wait_for_server(base.BASE,base.PROCESS,1000)
            ready=True
            runtime.wait_for_sglang_warmup(base.OUT/'server.log',base.PROCESS,120)
            info=json.loads(base.http('/server_info'));base.save(base.OUT/'server-info.json',info)
            base.save(base.OUT/'resolved-policy.json',validate_server_info(info,chunk))
            monitor=threading.Thread(target=base.telemetry,args=(stop,),daemon=True);monitor.start()
            return experiments(corpus,model,reference,rendered,reverse=bool(index%2))
        finally:
            original=sys.exc_info()[1]
            stop.set()
            if monitor:monitor.join(timeout=7)
            monitor_pending = monitor is not None and monitor.is_alive()
            base.save(base.OUT/'natural-results.json',base.ROWS)
            if base.PROCESS is not None:
                try:
                    if not ready or base.PROCESS.poll() is not None:
                        raise RuntimeError('Server not live for orderly shutdown')
                    result=runtime.qualify_lifecycle(base.PROCESS,log,base.OUT,port=30000)
                    base.save(base.OUT/'lifecycle.json',result)
                    if result['status'] != 'passed':raise RuntimeError('Lifecycle failed')
                except BaseException as cleanup:
                    base.save(base.OUT/'cleanup-error.json',{'error':repr(cleanup)})
                    base.save(base.OUT/'containment.json',runtime.emergency_contain_process_group(base.PROCESS,target))
                    if original is None:raise
            if monitor_pending and original is None:
                raise RuntimeError('Telemetry thread not retired; do not start another engine')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    def expire(*_):raise TimeoutError('9000-second whole-runner cap')
    signal.signal(signal.SIGALRM,expire);signal.alarm(RUNNER_CAP_S)
    import torch
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability()!=(12,0) or 'RTX PRO 6000' not in torch.cuda.get_device_name():
        raise RuntimeError('Unexpected target GPU')
    base.save(OUT/'hardware.json',{'device':torch.cuda.get_device_name(),
                                 'capability':torch.cuda.get_device_capability()})
    runtime=base.module('frozen_runtime','/tmp/official_launcher_runtime.py')
    reference=base.module('actor_reference','/tmp/actor-reference.py')
    corpus=json.loads(Path('/tmp/prefix-corpus.json').read_text())
    model=runtime.discover_model(Path('/kaggle/input'),Path('/tmp/PROTOTYPE-qwen3_8_fp8-manifest.json'))
    base.save(OUT/'model-verification.json',runtime.verify_model(model,Path('/tmp/PROTOTYPE-qwen3_8_fp8-manifest.json')))
    for name in ['config.json','generation_config.json']:
        base.save(OUT/name,json.loads((model/name).read_text()))
    original_corpus=json.loads(Path('/tmp/actor-corpus.json').read_text())
    if json.loads((model/'generation_config.json').read_text()) != original_corpus['generation_config']:
        raise RuntimeError('Generation config drift')
    base.save(OUT/'experiment-inputs.json',corpus)
    rendered=None
    for index,chunk in enumerate(CHUNKS):
        rendered=run_arm(runtime,model,reference,corpus,index,chunk,rendered)
    base.save(OUT/'VERDICT.json',{'status':'MEASURED_REVIEW_REQUIRED','promotion':False,
        'limits':'Chunk configuration effect, not isolated kernel speed; compare cache counts and graph/Mamba policy. No summary-boundary fixture, natural quality comparison, host/game deadlines or deployment qualification.'})


if __name__=='__main__':
    try:main()
    except BaseException as error:
        OUT.mkdir(parents=True,exist_ok=True)
        base.save(OUT/'FAILURE.json',{'type':type(error).__name__,'error':str(error)})
        raise
