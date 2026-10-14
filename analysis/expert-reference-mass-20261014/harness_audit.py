"""Post-launch audit of the frozen harness's process-wide gate and request eviction exposure."""
import hashlib
import json
from pathlib import Path
import threading

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]/'analysis/model-serving-20261011/wave-1/e2-c14/kernel/wave1-e2-c14-20261011.ipynb'


def clock_replay():
    notebook=json.loads(BASE.read_text());patch=''.join(notebook['cells'][2]['source'])
    block=patch[patch.index('+class _PriorityGate:'):patch.index('+def _max_active_streams()',patch.index('+class _PriorityGate:'))]
    source='\n'.join(line[1:] for line in block.splitlines() if line.startswith('+'))
    ns={'threading':threading,'_get_env_bool':lambda key,default:True}
    exec(compile('from __future__ import annotations\n'+source,'frozen-c14-gate','exec'),ns)
    gate=ns['_PriorityGate'](14)
    gate.configure_clock(100,1000);gate.configure_clock(1500,2400)
    retained={'start':gate._clock_start,'deadline':gate._clock_deadline}
    fresh=ns['_PriorityGate'](14);fresh.configure_clock(1500,2400)
    return {'base_notebook_sha256':hashlib.sha256(BASE.read_bytes()).hexdigest(),
            'extracted_gate_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'first_phase':[100,1000],'second_phase':[1500,2400],
            'reused_gate_clock':retained,'fresh_gate_clock':{'start':fresh._clock_start,'deadline':fresh._clock_deadline},
            'reused_clock_expired_at_second_start':gate._clock_deadline<1500,
            'interpretation':'configure_clock retains earliest deadline; a new benchmark in the same Python process does not reset the gate. Actual ranking impact requires handover/refresh exposure.'}


def evictions(out):
    events=[];requests=0;responses=0;lengths=0;completion_tokens=0;prompt_max=0
    paths=sorted(Path(out).glob('*_p0_requests.jsonl'))
    for path in paths:
        previous=[]
        for number,line in enumerate(path.open(),1):
            record=json.loads(line)
            if record.get('event')=='response':
                responses+=1;lengths+=record.get('finish_reason')=='length'
                usage=record.get('usage',{});completion_tokens+=usage.get('completion_tokens',0) or 0
                prompt_max=max(prompt_max,usage.get('prompt_tokens',0) or 0)
            if record.get('event')!='request':continue
            requests+=1;current=record.get('messages',[])
            hashes={hashlib.sha256(json.dumps(m,sort_keys=True,ensure_ascii=False).encode()).hexdigest() for m in current}
            lost=[m for m in previous if m.get('role')!='system' and hashlib.sha256(json.dumps(m,sort_keys=True,ensure_ascii=False).encode()).hexdigest() not in hashes]
            if lost:events.append({'file':path.name,'line':number,'analysis_step':record.get('analysis_step'),'lost_message_signatures':len(lost),
                                   'old_messages':len(previous),'new_messages':len(current)})
            previous=current
    return {'game_request_files':len(paths),'requests':requests,'responses':responses,'length_cutoffs':lengths,
            'completed_response_tokens':completion_tokens,'maximum_served_prompt_tokens':prompt_max,
            'changed_prior_message_events':events,
            'interpretation':'Prior message loss/change is exposure to cached-prefix invalidation; source inspection is required to distinguish mutations from actual handovers.'}


def main():
    result={'clock_replay':clock_replay()}
    for arm in ['control','candidate']:
        out=HERE/'output/reference-mass'/arm
        if out.is_dir():result[arm]=evictions(out)
    path=HERE/'evidence/harness-state-audit.json';path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:({a:b for a,b in v.items() if a!='changed_prior_message_events'} if k!='clock_replay' else v) for k,v in result.items()},indent=2))


if __name__=='__main__':main()
