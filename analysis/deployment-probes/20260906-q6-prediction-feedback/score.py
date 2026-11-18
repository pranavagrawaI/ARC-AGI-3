"""Score executable repair on held-out recorded transitions, not model prose."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from probe import CAPTURE, BASELINE, rows

FIELDS = ('agent_top_row','agent_col','band_right')

def measurement(grid):
    g=np.asarray(grid)
    r,c=np.where(g==12)
    _,b=np.where(g==11)
    return {'agent_top_row':int(r.min()),'agent_col':int(c.min()),'band_right':int(b.max())}

def evaluate(module, transitions):
    outcomes=[]
    for row in transitions:
        before=np.asarray(row['before']['grid'])
        after=np.asarray(row['after']['grid'])
        predicted=module.predict(module.encode(before),row['action'])
        encoded=module.encode(after)
        actual=measurement(after)
        fields=list(FIELDS)
        if 'band_left' in predicted or 'band_left' in encoded:
            actual['band_left']=int(np.where(after==11)[1].min())
            fields.append('band_left')
        checks={k:bool(predicted.get(k)==actual[k]) for k in fields}
        outcomes.append({'step':row['step'],'action':row['action'],'predicted':predicted,
            'measured':actual,'observed_band_left':int(np.where(after==11)[1].min()),'field_correct':checks,'encoder_correct':{k:bool(encoded.get(k)==actual[k]) for k in fields}})
    return {'transitions':outcomes,'band_correct':sum(r['field_correct']['band_right'] for r in outcomes),
        'all_fields_correct':sum(all(r['field_correct'].values()) for r in outcomes),'n':len(outcomes)}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run')
    parser.add_argument('--helper',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.helper:
        import resource
        resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
        resource.setrlimit(resource.RLIMIT_CPU,(10,10))
        # Actor helper AST is checked using the same import/load restriction as the Actor.
        sys.path.insert(0,str(BASELINE/'chassis'))
        from solver_agent import _check_helper_ast
        source=Path(args.helper).read_text()
        error=_check_helper_ast(source)
        if error: raise ValueError(error)
        spec=importlib.util.spec_from_file_location('evaluated_model',args.helper)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        selected=[r for r in rows(CAPTURE/'ipc/transitions.jsonl') if r['step']>6]
        print(json.dumps(evaluate(module,selected),default=lambda x:x.item() if hasattr(x,'item') else str(x)))
        return
    out=Path(args.run)
    results=[]
    for arm in ('baseline-a','candidate-a','candidate-b','baseline-b'):
        if not (out/arm).exists(): continue
        accepted=(out/arm/'REPLAY-COMPLETE.json').exists()
        helper_dir=out/arm/'team_nemo/shared/helpers'
        helper=helper_dir/('world_model.py' if (helper_dir/'world_model.py').exists() else 'worldmodel.py')
        selection='saved_unsubmitted_helper'
        commits=out/arm/'ipc/model_predictions.jsonl'
        if accepted and commits.exists():
            record=rows(commits)[-1]
            if record.get('step')!=6 or record.get('status')!='committed':
                raise RuntimeError('Accepted candidate did not retain a committed step6 model')
            source=record['source']
            if hashlib.sha256(source.encode()).hexdigest()!=record['model_sha256']:
                raise RuntimeError('Committed source hash mismatch')
            helper=out/arm/'EVALUATED-COMMITTED-MODEL.py'
            helper.write_text(source)
            selection='exact_action_committed_source'
        try:
            proc=subprocess.run([sys.executable,__file__,str(out),'--helper',str(helper)],
                capture_output=True,text=True,timeout=20)
            score=json.loads(proc.stdout.splitlines()[-1]) if proc.returncode==0 else {'error':proc.stderr[-1500:]}
        except Exception as exc: score={'error':str(exc)}
        results.append({'arm':arm,'accepted_submission':accepted,'source_selection':selection,'evaluated_source_sha256':hashlib.sha256(helper.read_bytes()).hexdigest(),**score})
    (out/'SCORE.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in row.items() if k!='transitions'} for row in results],indent=2))

if __name__=='__main__': main()
