"""One bounded continuation in its own process. Default mode is CPU qualification."""
from pathlib import Path
import argparse, copy, dataclasses, json, os, signal, time
import requests
import restore

REAL_POST=requests.post

class Stop(BaseException):
    def __init__(self,status):self.status=status;super().__init__(status)

def write(path,value):
    path.write_text(json.dumps(value,indent=2,default=str)+'\n')

def append(path,value):
    with path.open('a') as f:f.write(json.dumps(value,default=str)+'\n');f.flush();os.fsync(f.fileno())

class Transport:
    def __init__(self,out,url,seed,token_cap=32768,request_cap=32,episode_seconds=900,request_seconds=120):
        self.out,self.url,self.seed=out,url,seed
        self.token_cap,self.request_cap=token_cap,request_cap
        self.request_seconds=request_seconds
        self.deadline=time.monotonic()+episode_seconds
        self.calls=self.tokens=0
        self.unresolved=None
        self.status='running'
    def alarm(self,*_):raise Stop('request_deadline' if self.unresolved else 'episode_deadline')
    def set_deadline(self,when):
        signal.signal(signal.SIGALRM,self.alarm)
        signal.setitimer(signal.ITIMER_REAL,max(.001,when-time.monotonic()))
    def post(self,url,**kwargs):
        if self.calls>=self.request_cap:raise Stop('request_budget')
        if self.tokens>=self.token_cap:raise Stop('token_budget')
        if time.monotonic()>=self.deadline:raise Stop('episode_deadline')
        if url!=self.url.rstrip('/')+'/chat/completions':raise Stop('unexpected_endpoint')
        body=copy.deepcopy(kwargs['json'])
        if body.get('stream'):raise Stop('unexpected_stream_contract')
        body['max_tokens']=min(int(body.get('max_tokens') or 12288),self.token_cap-self.tokens)
        body['seed']=self.seed
        kwargs['json']=body
        self.calls+=1;started=time.monotonic()
        self.unresolved={'call':self.calls,'reserved_tokens':body['max_tokens']}
        append(self.out/'wire.jsonl',{'event':'request','call':self.calls,'at':time.time(),'body':body})
        self.set_deadline(min(self.deadline,started+self.request_seconds))
        response=None
        try:
            response=REAL_POST(url,**kwargs)
            raw=response.json()
            append(self.out/'wire.jsonl',{'event':'response','call':self.calls,'status':response.status_code,
                    'latency_seconds':time.monotonic()-started,'body':raw})
            if response.status_code>=400:raise Stop('http_error')
            usage=raw.get('usage',{});tokens=usage.get('completion_tokens')
            if not isinstance(tokens,int) or tokens<0 or tokens>body['max_tokens']:raise Stop('invalid_usage')
            self.tokens+=tokens;self.unresolved=None
            return response
        except BaseException as exc:
            append(self.out/'wire.jsonl',{'event':'failure','call':self.calls,'latency_seconds':time.monotonic()-started,
                     'error':repr(exc),'unresolved':self.unresolved})
            raise Stop(exc.status if isinstance(exc,Stop) else 'transport_error') from None
        finally:self.set_deadline(self.deadline)

def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    state={'status':'restoring','new_model_calls':0,'mode':args.mode,'checkpoint':args.checkpoint,'arm':args.arm,'seed':args.seed}
    transport=None;session_ref=[];first_wire=None
    def branch(a,session,messages,kwargs,first,original_chat,cfg):
        nonlocal transport,first_wire
        from inference.agent import tool_agent as ta
        import arcengine
        if first:
            session_ref.append(session)
            feedback=json.loads((restore.HERE/'inputs/feedback.json').read_text())[args.checkpoint][args.arm]
            if feedback:messages.append({'role':'user','content':feedback})
            write(out/'first-request.json',{'messages':messages,'tools':kwargs['tools'],'chat_template_kwargs':cfg['branch']['chat_template_kwargs']})
            first_wire=copy.deepcopy(messages)
            session.started_at=time.monotonic();session.solver.max_runtime_s_per_game=900
            original_step=session.game._execute_action
            def step(action):
                frame=original_step(action)
                if frame.levels_completed>=2:frame.raw.state=arcengine.GameState.WIN
                return frame
            session.game._execute_action=step
            state['status']='continuing';write(out/'progress.json',state)
            if args.mode=='live':
                transport=Transport(out,args.url,args.seed)
                requests.post=transport.post
                a._model=dataclasses.replace(a._model,base_url=args.url)
                transport.set_deadline(transport.deadline)
        if args.mode=='restore':raise Stop('restored')
        if args.mode=='fixture':
            if state.get('fixture_sent'):raise Stop('fixture_did_not_solve')
            state['fixture_sent']=True
            # Only the offline qualification receives this known route; never packaged as live feedback.
            route=json.loads(Path(args.fixture).read_text())
            call={'id':'qualification','type':'function','function':{'name':'python','arguments':json.dumps({'code':route})}}
            return ta._ChatCompletionResult(message={'role':'assistant','content':None,'tool_calls':[call]},finish_reason='tool_calls',usage={'completion_tokens':1,'prompt_tokens':1,'total_tokens':2})
        if transport.tokens>=transport.token_cap:raise Stop('token_budget')
        if transport.calls>=transport.request_cap:raise Stop('request_budget')
        result=original_chat(messages,tools=kwargs['tools'],request_timeout_seconds=120,
                             max_tokens=min(a._max_output_tokens,transport.token_cap-transport.tokens))
        state.update(new_model_calls=transport.calls,generated_tokens=transport.tokens)
        write(out/'progress.json',state)
        return result
    try:
        a,s,cfg,restored=restore.restore(args.checkpoint,out,on_branch=branch,env_dir=args.env_dir,code_dir=args.code_dir,taaf_dir=args.taaf_dir)
        state['restoration_status']=restored['status']
        if 'error' in restored:state['error']=restored['error']
        if s.game.current_state.levels_completed>=2:state['status']='solved'
        elif transport:state['status']=restored.get('halt_status','gave_up')
        else:state['status']=restored.get('halt_status',restored['status'])
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if transport:state.update(new_model_calls=transport.calls,known_receipted_tokens=transport.tokens,
             generated_tokens=transport.tokens if transport.unresolved is None else None,unresolved=transport.unresolved)
        if session_ref:
            s=session_ref[0];r=s.game.game_run
            state.update(levels_completed=s.game.current_state.levels_completed,score=r.final_score,
                         actions_per_level=r.actions_per_level,total_actions=len(r.history))
            write(out/'actions.json',[{'id':h.action.id.name,'data':dict(h.action.data or {})} for h in r.history])
        write(out/'result.json',state)
        print(json.dumps(state,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',choices=['b1','control'],required=True)
    p.add_argument('--arm',choices=['control','reminder','counterexample'],default='control')
    p.add_argument('--mode',choices=['restore','fixture','live'],default='restore');p.add_argument('--seed',type=int,default=1701)
    p.add_argument('--output',required=True);p.add_argument('--url',default='http://127.0.0.1:8001/v1')
    p.add_argument('--fixture');p.add_argument('--env-dir');p.add_argument('--code-dir');p.add_argument('--taaf-dir')
    run(p.parse_args())
