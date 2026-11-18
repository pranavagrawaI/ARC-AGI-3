"""Baseline cost structure and finite-workload concurrency; never execute tools."""
import concurrent.futures
import hashlib
import gzip
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.request

OUT = Path('/kaggle/working/baseline-cost-20260906')
BASE = 'http://127.0.0.1:30000'
PROCESS = None
ROWS = []


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def http(route, body=None):
    req = urllib.request.Request(BASE + route, data=None if body is None else json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=180) as response:
        return response.read().decode()


def guarded(fn):
    # Daemon transport may outlive an HTTP error; never admit replacement work.
    from watchdog import guarded_call
    return guarded_call(fn, PROCESS)


def payload(ids, tokens=256):
    return {'input_ids': ids, 'stream': True, 'sampling_params': {
        'temperature': 1.0, 'top_k': 20, 'top_p': .95,
        'max_new_tokens': tokens, 'ignore_eos': True}}


def decode_stats(meta):
    n = meta['completion_tokens']
    rate = meta.get('decode_throughput')
    span = (n - 1) / rate if rate and n > 1 else None
    entry=meta.get('forward_entry_time');prefill_end=meta.get('prefill_finished_time')
    return {'server_tokenizer_decode_span_s': span,
            'server_tokenizer_ms_per_token': 1000 / rate if span else None,
            'server_forward_to_prefill_finished_s':prefill_end-entry if entry is not None and prefill_end is not None else None,
            'server_queue_s':meta.get('queue_time'),
            'timing_definition': '(N-1)/(tokenizer finished_time-first_token_time); includes tokenizer output handling, not pure GPU time'}


def cache_stats(meta):
    total=meta.get('prompt_tokens'); cached=meta.get('cached_tokens')
    return {'total_input_tokens':total,'cached_input_tokens':cached,
            'new_input_tokens':total-cached if isinstance(total,int) and isinstance(cached,int) else None}


def generate(name, body):
    save(OUT / (name + '-request.json'), body)
    def run():
        start = time.perf_counter()
        req = urllib.request.Request(BASE + '/generate', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        last = None
        with (OUT / (name + '-stream.jsonl')).open('w') as log:
            with urllib.request.urlopen(req, timeout=600) as response:
                for line in response:
                    if not line.startswith(b'data: '):
                        continue
                    data = line[6:].strip()
                    if data == b'[DONE]':
                        break
                    last = json.loads(data)
                    log.write(json.dumps({'client_elapsed_s': time.perf_counter()-start, 'data': last})+'\n')
                    log.flush()
        if not last or not last.get('meta_info', {}).get('finish_reason'):
            raise RuntimeError('Missing terminal generation: ' + name)
        meta = last['meta_info']
        if meta['completion_tokens'] != body['sampling_params']['max_new_tokens']:
            raise RuntimeError('Fixed generation count not met: ' + name)
        if meta['prompt_tokens'] != len(body['input_ids']):
            raise RuntimeError('Prompt truncated or changed: ' + name)
        return dict(name=name, client_start_monotonic_s=start,
                    client_end_monotonic_s=time.perf_counter(),
                    client_e2e_s=time.perf_counter()-start, meta_info=meta,
                    effective_context_start=meta['prompt_tokens'],
                    effective_context_end=meta['prompt_tokens']+meta['completion_tokens'],
                    **decode_stats(meta), **cache_stats(meta))
    result = guarded(run)
    save(OUT / (name+'-result.json'), result)
    return result


def chat(name, record, model, reference):
    request = dict(record['request'], model=str(model), return_meta_info=True,
                   return_token_ids=True, return_prompt_token_ids=True)
    save(OUT / (name+'-request.json'), request)
    response, elapsed = guarded(lambda: reference.post(request))
    save(OUT / (name+'-response.json'), response)
    ids = response['choices'][0].get('prompt_token_ids')
    if not ids or len(ids) != record['captured_usage']['prompt_tokens']:
        raise RuntimeError('Rendered Actor prompt differs from capture')
    row = dict(name=name, client_e2e_s=elapsed, usage=response['usage'],
               meta_info=response['choices'][0].get('meta_info'),
               quality=reference.inspect_response(response, request),
               provenance=record['provenance'],
               interpretation='Natural completion; schema validation only, not a usable gameplay decision')
    row.update(cache_stats(row['meta_info'] or {}))
    save(OUT / (name+'-result.json'), row)
    ROWS.append(row)
    return ids


def finite_batch(items, concurrency, call):
    start = time.perf_counter()
    executor=concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)
    remaining=iter(items);pending=set();results=[]
    def admit():
        item=next(remaining,None)
        if item is not None: pending.add(executor.submit(call,*item))
    try:
        for _ in range(concurrency):admit()
        while pending:
            done,pending=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            # Observe every completed outcome before opening another admission slot.
            results.extend(future.result() for future in done)
            for _ in done:admit()
    except BaseException:
        executor.shutdown(wait=False,cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    elapsed = time.perf_counter()-start
    return {'concurrency_limit': concurrency, 'wall_s': elapsed, 'requests': results,
            'aggregate_generated_tok_s': sum(r['meta_info']['completion_tokens'] for r in results)/elapsed,
            'gpu_allocation_seconds_per_completed_request': elapsed/len(results),
            'server_overlap': 'Inspect metrics time series and scheduler log; client limit alone is not proof'}


def telemetry(stop):
    with (OUT / 'metrics.jsonl').open('w') as log:
        while not stop.is_set():
            try:
                row = {'monotonic_s': time.perf_counter(), 'metrics': http('/metrics')}
                gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,clocks.sm,clocks.mem,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
                row['gpu_csv']=gpu.stdout.strip();row['gpu_error']=gpu.stderr.strip()
            except Exception as e:
                row = {'monotonic_s': time.perf_counter(), 'error': repr(e)}
            log.write(json.dumps(row)+'\n'); log.flush()
            stop.wait(.5)


def profile(n, ids):
    http('/flush_cache')
    trace_dir=OUT/f'trace-{n}';trace_dir.mkdir()
    try:
        http('/start_profile', {'output_dir':str(trace_dir),'activities':['CPU','GPU'],
                                'with_stack':False,'record_shapes':False})
        try:
            generate(f'profiled-{n}',payload(ids,32))
        finally:
            http('/stop_profile', {})
        summaries=[]
        for path in trace_dir.rglob('*.json*'):
            opener=gzip.open if path.suffix=='.gz' else open
            with opener(path,'rt') as handle: trace=json.load(handle)
            events=trace.get('traceEvents',[]) if isinstance(trace,dict) else []
            kernels=[event for event in events if event.get('cat')=='kernel']
            summaries.append({'file':str(path),'kernel_events':len(kernels),
                'cuda_graph_launch_events':sum('cudaGraphLaunch' in event.get('name','') for event in events),
                'kernel_names':sorted({event.get('name','') for event in kernels})})
        save(trace_dir/'trace-check.json',summaries)
        if not any(row['kernel_events'] for row in summaries):
            raise RuntimeError('Profiler produced no GPU kernel evidence')
    except Exception as error:
        save(trace_dir/'UNAVAILABLE.json', {'error':repr(error),'consequence':'Resource diagnosis unqualified; no backend selected'})
        raise


def experiments(corpus, model, reference):
    # Obtain exact server-rendered IDs, preserving captured requests and natural EOS.
    indices = [0, 5, 12, 13]
    ids = {}
    for i in indices:
        http('/flush_cache')
        ids[i] = chat('natural-'+str(i), corpus['records'][i], model, reference)
    # 12->13 is a captured chronological continuation, not a claim that this
    # newly sampled response authored the successor's frozen history.
    http('/flush_cache')
    chat('continuation-seed-12', corpus['records'][12], model, reference)
    chat('continuation-successor-13', corpus['records'][13], model, reference)
    lcp = next((i for i,(a,b) in enumerate(zip(ids[12], ids[13])) if a != b), min(len(ids[12]),len(ids[13])))
    save(OUT/'continuation.json', {'prompt_lcp_tokens': lcp, 'prefix_reuse': 'Use response cached_tokens, not LCP as cache-hit claim'})
    source = ids[13]
    contexts = {n: (source*((n+len(source)-1)//len(source)))[:n] for n in [1024,10240,21279,95000]}
    save(OUT/'context-provenance.json', {str(n): {'tokens':len(v), 'sha256':hashlib.sha256(json.dumps(v).encode()).hexdigest(),
        'construction':'prefix truncated or repeated from rendered Actor case 13; mechanical stress input, NOT natural Actor traffic'} for n,v in contexts.items()})
    # Warm all measured shapes independently of prefix cache, then alternate length order.
    for n,v in contexts.items():
        http('/flush_cache'); generate('warm-'+str(n),payload(v,16))
    # Establish serving-worker profiling before spending the measured sweep budget.
    # Warmup is excluded; graphs stay enabled, and profile timings stay out of ROWS.
    for n in [1024,21279]:
        profile(n,contexts[n])
    for repeat, order in enumerate([list(contexts),list(reversed(contexts)),list(contexts)]):
        for n in order:
            http('/flush_cache')
            ROWS.append(generate(f'cold-{n}-{repeat}',payload(contexts[n])))
            save(OUT/'results.json',ROWS)
    # Identical finite set across concurrency; distinct recorded Actor prompts.
    for repeat, order in enumerate([[1,2,4],[4,2,1]]):
        for concurrency in order:
            http('/flush_cache')
            name=f'batch-{concurrency}-{repeat}'
            items=[(f'{name}-case-{i}',payload(ids[i])) for i in indices]
            save(OUT/(name+'.json'),finite_batch(items,concurrency,generate))
    save(OUT/'results.json',ROWS)


def main():
    global PROCESS
    OUT.mkdir(parents=True,exist_ok=True)
    def expire(*_): raise TimeoutError('5400 s runner cap')
    signal.signal(signal.SIGALRM,expire);signal.alarm(5400)
    import torch
    assert torch.cuda.device_count()==1 and torch.cuda.get_device_capability()==(12,0)
    assert 'RTX PRO 6000' in torch.cuda.get_device_name()
    save(OUT/'hardware.json', {'device':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability()})
    runtime=module('frozen_runtime','/tmp/official_launcher_runtime.py')
    reference=module('actor_reference','/tmp/actor-reference.py')
    corpus=json.loads(Path('/tmp/actor-corpus.json').read_text())
    module('contract','/tmp/actor-corpus-contract.py').validate(corpus)
    assert len(corpus['records'])==14
    model=runtime.discover_model(Path('/kaggle/input'),Path('/tmp/PROTOTYPE-qwen3_8_fp8-manifest.json'))
    save(OUT/'model-verification.json',runtime.verify_model(model,Path('/tmp/PROTOTYPE-qwen3_8_fp8-manifest.json')))
    for name in ['config.json','generation_config.json']:
        save(OUT/name,json.loads((model/name).read_text()))
    assert json.loads((model/'generation_config.json').read_text())==corpus['generation_config']
    command=runtime.sglang_command(model,'127.0.0.1',30000,attention_backend='triton',sampling_backend='pytorch')
    command+=['--enable-metrics','--stream-interval','1','--sampling-defaults','model']
    save(OUT/'launch.json',{'command':command,'sampling_diagnostic':payload([])['sampling_params']})
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',R8_L3B_OWNER=runtime.R8_L3B_OWNER)
    stop=threading.Event(); target=None; ready=False; monitor=None
    with (OUT/'server.log').open('w') as log:
        try:
            PROCESS=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,text=True)
            target=runtime.process_identity(PROCESS.pid)
            runtime.wait_for_server(BASE,PROCESS,1000)
            ready=True
            runtime.wait_for_sglang_warmup(OUT/'server.log',PROCESS,120)
            info=json.loads(http('/server_info'));save(OUT/'server-info.json',info)
            assert info['speculative_algorithm'] is None and info['sampling_defaults']=='model'
            monitor=threading.Thread(target=telemetry,args=(stop,),daemon=True);monitor.start()
            experiments(corpus,model,reference)
        finally:
            original=sys.exc_info()[1]
            stop.set()
            if monitor: monitor.join(timeout=2)
            save(OUT/'results.json',ROWS)
            if PROCESS is not None:
                try:
                    if not ready or PROCESS.poll() is not None: raise RuntimeError('Server not live for orderly shutdown')
                    result=runtime.qualify_lifecycle(PROCESS,log,OUT,port=30000)
                    save(OUT/'lifecycle.json',result)
                    assert result['status']=='passed'
                except BaseException as cleanup:
                    save(OUT/'cleanup-error.json',{'error':repr(cleanup)})
                    save(OUT/'containment.json',runtime.emergency_contain_process_group(PROCESS,target))
                    if original is None: raise
    save(OUT/'VERDICT.json',{'status':'MEASURED_REVIEW_REQUIRED','promotion':False,
        'limits':'No actions, no game deadlines tested; synthetic context stress; batching bypasses host semaphore; traces do not establish bandwidth or arithmetic saturation without counters'})


if __name__=='__main__':
    try: main()
    except BaseException as error:
        OUT.mkdir(parents=True,exist_ok=True)
        save(OUT/'FAILURE.json',{'type':type(error).__name__,'error':str(error)})
        raise
