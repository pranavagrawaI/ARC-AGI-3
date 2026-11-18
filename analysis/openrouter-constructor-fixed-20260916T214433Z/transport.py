"""One-attempt transport, durable exposure reservations, no SDK implicit retries."""
import copy, hashlib, json, os, time, urllib.request, urllib.error
from decimal import Decimal
from pathlib import Path
from protocol import project_messages
HERE=Path(__file__).resolve().parent
MODEL='qwen/qwen3.8-27b'
ENDPOINT='reka/fp8'
URL='https://openrouter.ai/api/v1/chat/completions'
RATES={'prompt':Decimal('0.000000214'),'completion':Decimal('0.00000255')}
MAX_PROVIDER_COMPLETION=131072
PROVIDER={'only':[ENDPOINT],'order':[ENDPOINT],'allow_fallbacks':False,'require_parameters':True,'quantizations':['fp8'],'max_price':{'prompt':0.214,'completion':2.55}}

def digest(raw): return hashlib.sha256(raw).hexdigest()
def encode(value): return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f: f.write(encode(value));f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def exclusive(path,value):
    with Path(path).open('xb') as f: f.write(encode(value));f.flush();os.fsync(f.fileno())

def wire(body,arm):
    if arm not in ('off','on'): raise ValueError('arm')
    from request_contract import validate_request
    validate_request(body)
    w=copy.deepcopy(body)
    w.update(provider=copy.deepcopy(PROVIDER),stream=False,reasoning={'enabled':arm=='on','exclude':False})
    if arm=='on':w['reasoning']['effort']='medium'
    if w['messages'] != project_messages(w['messages']): raise ValueError('message projection')
    return w

class Stop(RuntimeError): pass
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs): return None

def initial_ledger(path):
    exclusive(path,{'known_cost_usd':'0','unknown_exposure_usd':'0','stopped':False,'attempts':[], 'arms':{a:{'requests':0,'prompt_tokens':0,'completion_tokens':0,'reasoning_tokens':0} for a in ('off','on')}})

class Transport:
    label='OpenRouter pinned hosted FP8, no checkpoint equivalence'
    def __init__(self,arm,output,ledger,deadline,key=None,fixture_url=None):
        self.arm,self.output,self.ledger,self.deadline=arm,Path(output),Path(ledger),deadline
        self.key=key;self.url=fixture_url or URL;self.fixture=fixture_url is not None
        if fixture_url and not fixture_url.startswith('http://127.0.0.1:'):raise ValueError('fixture must be loopback')
        if not fixture_url and not key:raise Stop('credentials unavailable')
        self.output.mkdir(parents=True,exist_ok=True)
    def __call__(self,body):
        w=wire(body,self.arm);raw=json.dumps(w,ensure_ascii=False,allow_nan=False).encode()
        reservation_prompt=len(raw)+4096
        state=json.loads(self.ledger.read_text());arm=state['arms'][self.arm]
        # Admission uses full advertised endpoint output allowance, not requested 4096.
        exposure=reservation_prompt*RATES['prompt']+MAX_PROVIDER_COMPLETION*RATES['completion']
        if state['stopped'] or Decimal(state['unknown_exposure_usd'])!=0:raise Stop('pair already stopped or unresolved')
        if time.monotonic()>=self.deadline-20:raise Stop('soft admission cutoff')
        if arm['requests']>=6 or arm['completion_tokens']+w['max_tokens']>24576 or arm['prompt_tokens']+reservation_prompt>600000:raise Stop('arm budget admission')
        if reservation_prompt+w['max_tokens']>262144:raise Stop('endpoint context admission')
        if Decimal(state['known_cost_usd'])+exposure>Decimal('2'):raise Stop('pair exposure admission')
        if arm['requests']==0:
            shared=copy.deepcopy(w);shared.pop('reasoning')
            first_hash=digest(encode(shared))
            if self.arm=='on' and not self.fixture and state.get('off_initial_shared_sha256')!=first_hash:raise Stop('initial prompt/schema/transport parity mismatch')
            if self.arm=='off':state['off_initial_shared_sha256']=first_hash
            arm['initial_shared_sha256']=first_hash
        n=arm['requests']+1;stem=self.output/f'{n:03}'
        # Validate and construct request BEFORE recording a possibly dispatched attempt.
        headers={'Content-Type':'application/json','X-OpenRouter-Metadata':'enabled'}
        if self.key:headers['Authorization']='Bearer '+self.key
        req=urllib.request.Request(self.url,data=raw,headers=headers,method='POST')
        with stem.with_suffix('.request.json').open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        rec={'arm':self.arm,'ordinal':n,'request_sha256':digest(raw),'url':self.url,'reserved_prompt_tokens':reservation_prompt,'reserved_completion_tokens':MAX_PROVIDER_COMPLETION,'exposure_usd':str(exposure),'status':'possibly_dispatched','actual_cost_usd':None}
        arm['requests']+=1;state['unknown_exposure_usd']=str(exposure);state['attempts'].append(rec);write(self.ledger,state)
        started=time.monotonic();response_raw=None;status=None;error=None
        try:
            opener=urllib.request.build_opener(NoRedirect)
            with opener.open(req,timeout=max(.001,self.deadline-time.monotonic())) as response:
                status=response.status;response_raw=response.read(8_000_001)
        except urllib.error.HTTPError as e:
            status=e.code;response_raw=e.read(8_000_001);error='HTTPError'
        except Exception as e:
            error=type(e).__name__  # never exception repr with request/auth state
        rec['latency_seconds']=time.monotonic()-started;rec['http_status']=status;rec['error_type']=error
        if response_raw is not None:
            with stem.with_suffix('.response.raw').open('xb') as f:f.write(response_raw);f.flush();os.fsync(f.fileno())
            rec['response_sha256']=digest(response_raw);rec['response_bytes']=len(response_raw)
        parsed=None;fatal=None
        try:
            if response_raw is None or len(response_raw)>8_000_000:raise Stop('missing/oversize response')
            parsed=json.loads(response_raw)
            u=parsed.get('usage')
            if not isinstance(u,dict):raise Stop('missing usage: unknown exposure')
            for k in ('prompt_tokens','completion_tokens','total_tokens'):
                if type(u.get(k)) is not int or u[k]<0:raise Stop('invalid usage tokens')
            if u['total_tokens']!=u['prompt_tokens']+u['completion_tokens']:raise Stop('inconsistent usage')
            cost=Decimal(str(u.get('cost')))
            if not cost.is_finite() or cost<0:raise Stop('invalid cost')
            # Preserve known actuals even if subsequent routing/treatment validation fails.
            rec['usage']=u;rec['actual_cost_usd']=str(cost);rec['status']='receipted'
            state['known_cost_usd']=str(Decimal(state['known_cost_usd'])+cost);state['unknown_exposure_usd']='0'
            arm['prompt_tokens']+=u['prompt_tokens'];arm['completion_tokens']+=u['completion_tokens']
            if cost>exposure or Decimal(state['known_cost_usd'])>2:raise Stop('actual cost exceeds exposure/pair cap')
            if u['prompt_tokens']>reservation_prompt or u['completion_tokens']>w['max_tokens']:raise Stop('reported token overrun')
            if status!=200 or error or parsed.get('error'):raise Stop('HTTP/API failure')
            if parsed.get('model') not in (MODEL,'qwen/qwen3.8-27b-20260814') or parsed.get('provider')!='Reka':raise Stop('model/provider mismatch')
            meta=parsed.get('openrouter_metadata')
            if not isinstance(meta,dict) or meta.get('attempt')!=1 or meta.get('requested')!=MODEL or meta.get('strategy')!='direct':raise Stop('missing/invalid routing evidence')
            if meta.get('pipeline'):raise Stop('provider pipeline changed prompt/response')
            selected=[e for e in meta.get('endpoints',{}).get('available',[]) if e.get('selected')]
            if len(selected)!=1 or selected[0].get('provider')!='Reka' or selected[0].get('model') not in (MODEL,'qwen/qwen3.8-27b-20260814'):raise Stop('selected routing evidence mismatch')
            if meta.get('attempts') and (len(meta['attempts'])!=1 or meta['attempts'][0].get('provider')!='Reka' or meta['attempts'][0].get('status')!=200):raise Stop('fallback evidence')
            choices=parsed.get('choices')
            if not isinstance(choices,list) or len(choices)!=1:raise Stop('choice shape')
            message=choices[0].get('message')
            if not isinstance(message,dict) or message.get('role')!='assistant':raise Stop('assistant shape')
            if message.get('content') is not None and not isinstance(message['content'],str):raise Stop('nontext content')
            if message.get('reasoning') is not None and not isinstance(message['reasoning'],str):raise Stop('reasoning string shape')
            if message.get('reasoning_details') is not None and (not isinstance(message['reasoning_details'],list) or any(not isinstance(x,dict) for x in message['reasoning_details'])):raise Stop('reasoning details shape')
            rt=(u.get('completion_tokens_details') or {}).get('reasoning_tokens')
            if type(rt) is not int or not 0<=rt<=u['completion_tokens']:raise Stop('missing/invalid reasoning count')
            arm['reasoning_tokens']+=rt
            payload=bool(message.get('reasoning_details') or message.get('reasoning') or message.get('reasoning_content'))
            if self.arm=='off' and (rt or payload):raise Stop('off treatment contradicted')
            if self.arm=='on' and ((rt>0)!=payload):raise Stop('reasoning count/payload inconsistency')
            rec['reasoning_tokens']=rt;rec['reasoning_payload_present']=payload;rec['finish_reason']=choices[0].get('finish_reason')
            if rec['finish_reason']=='length':raise Stop('completion truncation (consumed request)')
            if rec['finish_reason'] not in ('tool_calls','stop'):raise Stop('unsupported finish reason')
            if time.monotonic()>self.deadline:raise Stop('late response: no tools/publication')
        except Exception as e:
            fatal=str(e) if isinstance(e,Stop) else type(e).__name__
            state['stopped']=True;rec['stop']=fatal
        write(self.ledger,state);write(stem.with_suffix('.receipt.json'),rec)
        if fatal:raise Stop(fatal)
        return parsed
