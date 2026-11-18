"""CPU-only full-corpus event sensitivity, never measured engine performance."""
import hashlib
import json
import signal
import time
import capacity
import corpus
import fixtures


def main():
    signal.alarm(1800);began=time.monotonic()
    sources=('capacity.py','fixtures.py','corpus.py','sensitivity.py')
    hashes={name:corpus.digest((corpus.HERE/name).read_bytes()) for name in sources}
    catalog=corpus.load();reader=fixtures.Reader(catalog);rows=[]
    try:
        for prefill in (.25,1.,4.):
            for cache in (0.,.5,1.):
                control=fixtures.build(catalog,reader,label=f'control-{prefill}-{cache}',prefill=prefill,cache=cache)
                for decode in (1.,1.2,1.5):
                    candidate=fixtures.build(catalog,reader,label=f'candidate-{prefill}-{cache}-{decode}',
                                             prefill=prefill,cache=cache,decode=decode,origin=2000.)
                    result=capacity.compare(control,candidate,catalog,reader)
                    rows.append(dict(prefill_rate_multiplier=prefill,recorded_cache_hit_multiplier=cache,
                                     hypothetical_decode_multiplier=decode,result=result))
                    print(prefill,cache,decode,result['verdict'],flush=True)
        lag=fixtures.build(catalog,reader,label='generator-fault',lag=.3)
        fault=capacity.reduce_arm(lag,catalog,reader);fault.pop('measured_rows',None)
        if hashes!={name:corpus.digest((corpus.HERE/name).read_bytes()) for name in sources}:
            raise RuntimeError('sources changed during sensitivity')
        report=dict(scope='CPU_FIXTURE_ONLY; no engine, GPU, runtime adapter or model executed',
                    source_hashes=hashes,
                    catalog_sha=corpus.CATALOG_SHA,rows=rows,generator_fault=fault,elapsed_s=time.monotonic()-began,
                    model_limits=['Decode curve already folds historical prefill interleave; additional prefill model is not calibrated engine prediction',
                                  'Cache reuse scales recorded hits, not an implemented cache',
                                  'Fixture generator refill lag quantized by50ms simulation step; no network or CPU serialization timing prediction',
                                  'Capacity load checks are separate from warmup/sample/stability/mix qualification',
                                  'Local renderer matches audit; unresolved real-runtime11-token offset remains a strict fidelity blocker'])
        (corpus.HERE/'sensitivity.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:signal.alarm(0)


if __name__=='__main__':main()
