"""Two isolated fresh-game phases under one absolute GPU-session deadline."""
import ast
import asyncio
import inspect
import json
import math
import os
from pathlib import Path
import signal
import threading
import time

PLAY_S = 900
ROOT = Path('/kaggle/working/reference-mass')
OWNED = {'proc':None}
START = time.monotonic()
DONE = threading.Event()


def save(name,value):
    ROOT.mkdir(parents=True,exist_ok=True)
    path=ROOT/name;tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def stop_server():
    proc=OWNED['proc']
    if proc is None:return
    # Group belongs exclusively to our start_new_session launcher, including worker children.
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:proc.wait(timeout=20)
    except Exception:
        try:os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        proc.wait(timeout=10)
    OWNED['proc']=None


def start_containment():
    def supervise():
        if DONE.wait(3450):return
        save('deadline-abort.json',{'elapsed_s':time.monotonic()-START,'reason':'absolute self cap'})
        stop_server()
        os._exit(124)
    threading.Thread(target=supervise,name='reference-mass-deadline',daemon=True).start()


async def execute(source,ns):
    value=eval(compile(source,'frozen-phase','exec',flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT),ns)
    if inspect.isawaitable(value):await value


def make_view(source,view):
    source,view=Path(source),Path(view)
    cfg=json.loads((source/'config.json').read_text())
    if cfg['text_config']['num_experts_per_tok']!=10 or cfg['text_config']['num_experts']!=512:
        raise ValueError('unexpected target routing config')
    view.mkdir(exist_ok=False)
    for entry in source.iterdir():
        if entry.name!='config.json':(view/entry.name).symlink_to(entry.resolve())
    cfg['text_config']['num_experts_per_tok']=5
    (view/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    index=json.loads((view/'model.safetensors.index.json').read_text())
    if any(not (view/p).is_file() for p in set(index['weight_map'].values())):
        raise ValueError('missing target shard')
    return str(view)


async def run_screen(ns):
    from urllib.request import Request, urlopen
    sources=json.loads((ns['KIT']/'phase-sources.json').read_text())
    model=ns['MODEL_DIR'];phases=[]
    try:
        for arm in ['control','candidate']:
            if 3300-(time.monotonic()-START)<PLAY_S+180:
                raise RuntimeError('insufficient remaining exposure for another full phase')
            out=ROOT/arm;out.mkdir(parents=True,exist_ok=False)
            ns.update(ARM=arm,WORKING_DIR=out,PLAY_S=PLAY_S,SERVER_STARTUP_TIMEOUT=660,NOTEBOOK_START_TIME=time.time())
            ns['MODEL_DIR']=model if arm=='control' else make_view(model,'/kaggle/reference-top5-view')
            os.environ['RECORDINGS_DIR']=str(out/'server_recording')
            save('progress.json',{'arm':arm,'state':'starting','elapsed_s':time.monotonic()-START,'phases':phases})
            await execute(sources['12'],ns)
            OWNED['proc']=ns['proc']
            with urlopen(f"http://127.0.0.1:{ns['SERVED_MODEL_PORT']}/health",timeout=5) as r:
                if r.status!=200:raise RuntimeError('server did not reach READY within phase startup cap')
            with urlopen(f"http://127.0.0.1:{ns['SERVED_MODEL_PORT']}/get_server_info",timeout=15) as r:
                (out/'server-info.json').write_bytes(r.read())
            payload={'model':ns['SERVED_MODEL_NAME'],'prompt':'Compute 2+2. Answer with the numeral only: ',
                     'max_tokens':32,'temperature':0,'logprobs':1}
            request=Request(ns['SERVER_BASE_URL']+'/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
            with urlopen(request,timeout=45) as r:answer=json.loads(r.read())
            (out/'finite-answer.json').write_text(json.dumps(answer,indent=2)+'\n')
            probabilities=answer['choices'][0]['logprobs']['token_logprobs']
            if not probabilities or any(not math.isfinite(float(p)) for p in probabilities if p is not None):
                raise RuntimeError('nonfinite or empty full-model answer')
            if 3300-(time.monotonic()-START)<PLAY_S+120:
                raise RuntimeError('startup left insufficient full play exposure')
            phase_start=time.monotonic()
            save('progress.json',{'arm':arm,'state':'playing','elapsed_s':time.monotonic()-START,'phases':phases})
            for i in ['14','16','20']:await execute(sources[i],ns)
            phases.append({'arm':arm,'play_elapsed_s':time.monotonic()-phase_start,'elapsed_s':time.monotonic()-START,
                           'benchmark_exists':(out/'benchmark.json').is_file()})
            save('progress.json',{'arm':arm,'state':'complete','phases':phases})
            stop_server()
            await asyncio.sleep(3)
        save('complete.json',{'phases':phases,'elapsed_s':time.monotonic()-START,'hard_cap_s':3600,'self_cap_s':3450})
    except BaseException as exc:
        save('failure.json',{'exception':repr(exc),'elapsed_s':time.monotonic()-START,'phases':phases})
        raise
    finally:
        # A startup exception may occur before the process is registered above.
        if OWNED['proc'] is None:OWNED['proc']=ns.get('proc')
        stop_server();DONE.set()
