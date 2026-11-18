"""Recompute qualification and summarize captured CUDA kernels without relaunching."""
import collections
import datetime
import gzip
import hashlib
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
RUN=HERE.parent/'serving-headroom-v3-20261008'
ROOT=RUN/'readout/raw/headroom'
sys.path.insert(0,str(RUN))
import audit


def trace_summary(path):
    with gzip.open(path,'rt') as stream:
        trace=json.load(stream)
    kernels=[e for e in trace['traceEvents'] if e.get('ph')=='X' and e.get('cat')=='kernel' and e.get('dur',0)>0]
    totals=collections.defaultdict(float);counts=collections.Counter();spans=[]
    for event in kernels:
        totals[event['name']]+=event['dur'];counts[event['name']]+=1
        spans.append((event['ts'],event['ts']+event['dur']))
    merged=[]
    for start,end in sorted(spans):
        if merged and start<=merged[-1][1]:merged[-1][1]=max(end,merged[-1][1])
        else:merged.append([start,end])
    summed=sum(totals.values());union=sum(b-a for a,b in merged)
    elapsed=max(b for a,b in spans)-min(a for a,b in spans)
    experts=sum(value for name,value in totals.items() if 'marlin_moe::Marlin' in name)
    return dict(file=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        kernel_events=len(kernels),kernel_sum_ms=summed/1000,kernel_interval_union_ms=union/1000,
        first_to_last_kernel_ms=elapsed/1000,kernel_union_fraction_within_capture=union/elapsed,
        marlin_moe_sum_ms=experts/1000,marlin_moe_fraction_of_kernel_sum=experts/summed,
        top_kernels=[dict(name=name,count=counts[name],sum_ms=duration/1000,fraction_of_kernel_sum=duration/summed)
            for name,duration in sorted(totals.items(),key=lambda item:-item[1])[:10]])


def main():
    verified=audit.audit(RUN/'readout/raw')
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run_id=verified['run_id'],
        qualified=verified['qualified'],matches_collector_audit=verified==json.loads((RUN/'readout/audit.json').read_text()),
        verified_raw_hashes=verified.get('verified_hashes'),capacity=verified.get('capacity'),blocks=[],
        limitations=['S10 instrumentation qualification only; no S16 treatment contrast or gameplay score.',
                     'Profiled blocks excluded from baseline throughput; capture spans are short.',
                     'Kernel duration sums overlap; interval union applies only within captured window.',
                     'Expert kernel duration share does not establish memory bandwidth saturation or causal speedup.'])
    for block in verified['blocks']:
        summary=json.loads((ROOT/block['label']/'summary.json').read_text())
        result['blocks'].append(dict(label=block['label'],n=block['n'],tokens=block['generated_tokens'],
            plateau_tokens_per_second=block['plateau']['U'],plateau_seconds=block['plateau']['seconds'],
            five_second_window_CV=block['plateau']['CV'],wall_seconds=summary['wall_seconds'],
            minimum_cache_fraction=min(r['cache_fraction'] for r in summary['responses']),accounting=block['accounting']))
    result['traces']=[trace_summary(p) for p in sorted((ROOT/'traces').glob('*.trace.json.gz'))]
    (HERE/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('qualified','matches_collector_audit','verified_raw_hashes','capacity','blocks')},indent=2))
    for row in result['traces']:
        print(json.dumps({key:value for key,value in row.items() if key!='top_kernels'}))
    if not verified['qualified']:raise SystemExit(1)


if __name__=='__main__':main()
