"""Sequential isolated workers. No retry; stop matrix on integrity/transport failure."""
from pathlib import Path
import json,os,signal,subprocess,sys,time

HERE=Path(__file__).resolve().parent

def main():
    plan=json.loads((HERE/'matrix.json').read_text())
    env_dir=sys.argv[1];url=sys.argv[2]
    results=[];began=time.monotonic()
    for episode in plan:
        out=HERE/'episodes'/episode['id'];out.parent.mkdir(exist_ok=True)
        if out.exists():raise RuntimeError('episode already attempted; no retry')
        code=HERE/'sources'/episode['checkpoint']
        command=[sys.executable,'-B',str(HERE/'worker.py'),'--checkpoint',episode['checkpoint'],
                 '--arm',episode['arm'],'--seed',str(episode['seed']),'--mode','live','--output',str(out),
                 '--url',url,'--env-dir',env_dir,'--code-dir',str(code),'--taaf-dir',str(HERE/'sources/taaf')]
        with (out.parent/(episode['id']+'.log')).open('xb') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:rc=process.wait(timeout=960)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait();rc=-9
        path=out/'result.json'
        result=json.loads(path.read_text()) if path.exists() else {'status':'missing_result','new_model_calls':None}
        results.append({**episode,'exitcode':rc,**result})
        (HERE/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        print(episode['id'],result['status'],'tokens',result.get('generated_tokens'),flush=True)
        valid={'solved','token_budget','request_budget','episode_deadline','gave_up'}
        if rc or result['status'] not in valid or result.get('unresolved'):
            print('STOP: failure preserved; no later episode will be launched',flush=True);break
        if sum(r.get('generated_tokens') or 0 for r in results)>393216:raise RuntimeError('total token ceiling')
        if time.monotonic()-began>10800:break

if __name__=='__main__':main()
