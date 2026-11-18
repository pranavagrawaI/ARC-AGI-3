#!/usr/bin/env python3
"""Fixed-evidence, unpaid before/after prompt and sandbox measurements."""
import hashlib, json, statistics, subprocess, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASELINE=HERE.parent/'model-construction-real-prefix-20260913'
if len(sys.argv)>1:
    root=Path(sys.argv[1]); sys.path.insert(0,str(root))
    from real_trace import load_case
    from constructor import construction_messages, TOOLS
    from sandbox_eval import evaluate_candidate, execute_inspection
    case=load_case(); source=case.legacy_incumbent_source()
    inspection="import json; d=json.load(open('/evidence/case.json')); print(d['frontier']); print(d['visible_transition_summaries'])"
    times={'inspection':[], 'visible_evaluation':[]}
    for _ in range(5):
        t=time.perf_counter(); r=execute_inspection(inspection,case); times['inspection'].append(time.perf_counter()-t); assert r['status']=='completed'
        t=time.perf_counter(); r=evaluate_candidate(source,case,include_suffix=False); times['visible_evaluation'].append(time.perf_counter()-t); assert r['visible']['status']=='passed'
    packet={'messages':construction_messages(case,None),'tools':TOOLS}
    print(json.dumps({'prompt_utf8_bytes':len(json.dumps(packet,separators=(',',':')).encode()),'latency_seconds':times,'median_seconds':{k:statistics.median(v) for k,v in times.items()},'prefix_sha256':hashlib.sha256(json.dumps(case.visible,sort_keys=True).encode()).hexdigest(),'candidate_sha256':hashlib.sha256(source.encode()).hexdigest(),'inspection_code':inspection}))
else:
    out={}
    for label,root in [('baseline',BASELINE),('v2',HERE)]:
        p=subprocess.run([sys.executable,'-B',str(__file__),str(root)],capture_output=True,text=True,check=True,env={'PATH':'/usr/bin:/bin'})
        out[label]=json.loads(p.stdout)
    assert out['baseline']['prefix_sha256']==out['v2']['prefix_sha256']
    out['prompt_byte_reduction_fraction']=1-out['v2']['prompt_utf8_bytes']/out['baseline']['prompt_utf8_bytes']
    out['limits']='Five sequential fixed-input samples per implementation, not randomized; local scheduling noise. Prompt bytes, not tokenizer estimates. V2 also makes all intermediate visible grids accessible, not preloaded. No live causal speedup comparison.'
    (HERE/'PERFORMANCE.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out))
