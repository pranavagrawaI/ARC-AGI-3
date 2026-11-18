"""Local raw-evidence reducer for the distinct saturated S10 contrast.

No launcher or adapter. Corpus and actual artifact bytes are trusted inputs to
the public seam; producer-supplied verdicts, occupancy/mix and warmup flags are
never consumed. Errors in JSON-shaped evidence return a feasibility-only result.
"""
import collections
import json
import math
import statistics
import corpus

SLOTS=10
WARMUP=600.
MEASURE=900.
IDLE_FRACTION_MAX=.005
VACANCY_EPISODE_MAX=.25


class Invalid(ValueError):
    pass


def require(condition,message):
    if not condition:raise Invalid(message)


def number(value):
    return type(value) in (int,float) and math.isfinite(value)


def integer(value,minimum=0):
    return type(value) is int and value>=minimum


def unique(events,kind):
    rows=[row for row in events if row['kind']==kind]
    require(len(rows)==1,'phase count: '+kind)
    return rows[0]['t']


def derive_window(events):
    origin=unique(events,'replay_start')
    start=unique(events,'measure_start');end=unique(events,'measure_end')
    require(all(number(t) for t in (origin,start,end)),'nonfinite phase clock')
    require(abs(start-origin-WARMUP)<1e-6 and abs(end-start-MEASURE)<1e-6,'phase clock mismatch')
    return start,end


def validate_log(rows,first):
    require(isinstance(rows,list) and rows,'missing raw event log')
    require(rows[0]['kind']==first and rows[-1]['kind']=='capture_end','capture boundaries')
    pid=rows[0]['pid'];boot=rows[0]['boot_id'];last=rows[0]['t']
    require(integer(pid,1) and isinstance(boot,str) and bool(boot),'process/boot identity')
    for index,row in enumerate(rows):
        require(type(row['seq']) is int and row['seq']==index,'event sequence')
        require(row['pid']==pid and row['boot_id']==boot,'process/boot changed')
        require(number(row['t']) and row['t']>=last,'event clock')
        require(row['t']-last<=2.,'raw heartbeat gap')
        require(index in (0,len(rows)-1) or row['kind'] not in (first,'capture_end'),'interior capture boundary')
        last=row['t']
    return boot


def read_json(reader,reference):
    raw=reader(reference)
    require(isinstance(raw,bytes) and len(raw)<=64*1024**2,'artifact byte type/limit')
    value=json.loads(raw)
    require(isinstance(value,dict),'artifact must be JSON object')
    return value


class Client:
    def __init__(self,catalog,events,reader):
        self.catalog=catalog;self.events=events;self.reader=reader
        self.start,self.end=derive_window(events)
        self.origin=unique(events,'replay_start')
        self.queue=list(catalog['initial_holders'])+list(catalog['initial_queue'])
        self.holders=set();self.cursor={g:0 for g in catalog['streams']}
        self.pending={};self.sent={};self.done={};self.cutoffs=set();self.warm=set();self.needs_handover=set();self.response_ids=set()
        self.pending_seconds=collections.defaultdict(float);self.starved=0.;self.vacancy=0.;self.max_vacancy=0.

    def remaining(self):
        return sum(self.cursor[g]<len(stream) for g,stream in self.catalog['streams'].items())

    def integrate(self,previous,now):
        dt=max(0.,min(now,self.end)-max(previous,self.start))
        if not dt:return
        n=len(self.pending);self.pending_seconds[n]+=dt
        if self.remaining()<SLOTS:self.starved+=dt
        if n<SLOTS and self.remaining()>=SLOTS:
            self.vacancy+=dt;self.max_vacancy=max(self.max_vacancy,self.vacancy)
        else:self.vacancy=0.

    def admit(self,row):
        game=row['game']
        require(self.queue and game==self.queue[0] and len(self.holders)<SLOTS,'FIFO admission')
        require(game not in self.holders,'duplicate holder')
        self.queue.pop(0);self.holders.add(game);self.needs_handover.discard(game)

    def release(self,row):
        game=row['game'];stream=self.catalog['streams'][game];index=self.cursor[game]
        require(game in self.holders and all(x['record']['game']!=game for x in self.pending.values()),'release pending/nonholder')
        finished=index==len(stream)
        require(index>0 and (finished or not self.catalog['records'][stream[index]]['extends_previous']),'unjustified handover')
        require(finished or game in self.needs_handover,'handover without fresh completion')
        require(row['reason']==('finished' if finished else 'handover'),'release reason')
        self.holders.remove(game)
        if not finished:self.queue.append(game)

    def send(self,row):
        rid=row['rid'];key=row['key'];record=self.catalog['records'][key];game=record['game']
        require(self.origin<=row['t']<self.end,'send outside replay window')
        require(isinstance(rid,str) and rid and rid not in self.sent,'duplicate/invalid send')
        require(game in self.holders,'send without admission')
        require(game not in self.needs_handover,'trim handover skipped')
        require(all(x['record']['game']!=game for x in self.pending.values()),'parallel per-game sends')
        require(key==self.catalog['streams'][game][self.cursor[game]],'corpus order')
        wire=read_json(self.reader,row['wire_ref'])
        require(wire['headers'].get('X-Request-Id')==rid,'wire identity header')
        body=wire['body']
        require(body.get('model')=='flashnext','served model alias')
        require(corpus.wire_hash(body)==record['wire_sha'],'wire/corpus fidelity')
        require(corpus.image_observations(body)==record['images'],'source-image fidelity')
        attempt=dict(rid=rid,key=key,record=record,t=row['t'])
        self.sent[rid]=attempt;self.pending[rid]=attempt
        require(len(self.pending)<=SLOTS,'offered requests exceed S10')

    def response(self,row):
        rid=row['rid'];require(rid in self.pending and rid not in self.done,'unmatched/duplicate response')
        require(row['status']==200,'HTTP error')
        value=read_json(self.reader,row['response_ref']);usage=value['usage'];choices=value['choices']
        require(isinstance(usage,dict) and isinstance(choices,list) and len(choices)==1,'response shape')
        tokens=usage['completion_tokens'];prompt=usage['prompt_tokens']
        attempt=self.pending[rid];record=attempt['record']
        require(integer(tokens,1) and tokens<=record['cap'] and integer(prompt,1),'response usage/cap')
        require(choices[0]['finish_reason'] in ('length','stop','tool_calls'),'finish reason')
        response_id=value['id']
        require(isinstance(response_id,str) and response_id and response_id not in self.response_ids,'duplicate/invalid response identity')
        self.response_ids.add(response_id)
        self.done[rid]=dict(attempt,t_done=row['t'],tokens=tokens,prompt=prompt,response_id=response_id)
        require(row['t']>=attempt['t'],'response before send')
        self.pending.pop(rid);self.cursor[record['game']]+=1
        stream=self.catalog['streams'][record['game']];index=self.cursor[record['game']]
        if index<len(stream) and not self.catalog['records'][stream[index]]['extends_previous']:
            self.needs_handover.add(record['game'])
        if row['t']<=self.start and record['game'] in self.catalog['initial_holders']:self.warm.add(record['game'])

    def cutoff(self,row):
        rid=row['rid'];require(rid in self.pending and rid not in self.cutoffs,'bad cutoff identity')
        require(row['t']>=self.end,'early cutoff')
        self.cutoffs.add(rid)

    def run(self):
        dispatch={'admit':self.admit,'release':self.release,'send':self.send,'response':self.response,'cutoff_pending':self.cutoff}
        previous=self.events[0]['t']
        for row in self.events:
            self.integrate(previous,row['t']);previous=row['t'];kind=row['kind']
            if kind in dispatch:dispatch[kind](row)
            else:require(kind in ('replay_start','measure_start','measure_end','capture_end','heartbeat'),'client error/unknown event')
        require(self.events[-1]['t']>=self.end,'client window coverage')
        require(self.cutoffs==set(self.pending),'unaccounted pending attempts')
        require(self.warm==set(self.catalog['initial_holders']),'initial-holder warmup incomplete')
        require(abs(sum(self.pending_seconds.values())-MEASURE)<1e-6,'offered-load coverage')
        idle=sum((SLOTS-n)*dt for n,dt in self.pending_seconds.items())/(SLOTS*MEASURE)
        self.load=dict(idle_fraction=idle,max_vacancy_s=self.max_vacancy,starved_s=self.starved,
                       pending_mean=sum(n*dt for n,dt in self.pending_seconds.items())/MEASURE,
                       finished_games=sum(self.cursor[g]==len(s) for g,s in self.catalog['streams'].items()))
        return self


class Native:
    def __init__(self,client,events,reader):
        self.client=client;self.events=events;self.reader=reader
        self.accepted={};self.prefilled={};self.decoding=set();self.finished={};self.nids=set();self.response_ids=set()
        self.decode_seconds=collections.defaultdict(float)

    def accept(self,row):
        rid=row['rid'];nid=row['nid'];sent=self.client.sent[rid]
        require(rid not in self.accepted and isinstance(nid,str) and nid and nid not in self.nids,'native identity reused')
        response_id=row['response_id']
        require(isinstance(response_id,str) and response_id and response_id not in self.response_ids,'native response identity reused')
        self.response_ids.add(response_id)
        require(row['t']>=sent['t'],'native accept before send')
        self.accepted[rid]=row;self.nids.add(nid)

    def prefill(self,row):
        rid=row['rid'];record=self.client.sent[rid]['record']
        require(rid in self.accepted and rid not in self.prefilled,'prefill lifecycle')
        require(row['image_hashes']==record['images'],'native source images')
        require(row['grids']==[[1,40,40] for _ in record['images']],'native image geometry')
        require(corpus.digest(self.reader(row['render_ref']))==record['render_sha'],'native rendered prefix')
        require(row['prompt_tokens']==record['local_prompt'],'native prompt count unresolved')
        require(row['prompt_tokens']+record['cap']<=139264,'native context bound')
        self.prefilled[rid]=row

    def decode(self,row):
        rid=row['rid'];require(rid in self.prefilled and rid not in self.decoding and rid not in self.finished,'decode lifecycle')
        self.decoding.add(rid)

    def finish(self,row):
        rid=row['rid'];require(rid in self.decoding and rid not in self.finished,'finish lifecycle')
        require(integer(row['output_tokens'],1),'native output count')
        self.decoding.remove(rid);self.finished[rid]=row

    def run(self):
        previous=self.events[0]['t'];start,end=self.client.start,self.client.end
        dispatch={'accept':self.accept,'prefill':self.prefill,'decode_start':self.decode,'finish':self.finish}
        for row in self.events:
            dt=max(0.,min(row['t'],end)-max(previous,start));previous=row['t']
            self.decode_seconds[len(self.decoding)]+=dt;kind=row['kind']
            if kind in dispatch:
                if kind!='accept':require(row['nid']==self.accepted[row['rid']]['nid'],'native request join')
                dispatch[kind](row)
            else:require(kind in ('capture_start','capture_end','heartbeat'),'native abort/preempt/error')
            require(len(self.accepted)-len(self.finished)<=SLOTS,'native admission exceeds S10')
        require(self.events[0]['t']<=self.client.origin and self.events[-1]['t']>=end,'native window coverage')
        require(set(self.accepted)==set(self.client.sent),'send/native accept reconciliation')
        for rid,response in self.client.done.items():
            done=self.finished[rid];accepted=self.accepted[rid]
            require(done['t']<=response['t_done'] and done['output_tokens']==response['tokens'],'native/response finish mismatch')
            require(accepted['response_id']==response['response_id'],'raw response identity')
            require(self.prefilled[rid]['prompt_tokens']==response['prompt'],'native/HTTP prompt disagreement')
        for rid in self.client.pending:
            require(rid not in self.finished or self.finished[rid]['t']>=end,'lost finished response before cutoff')
        return self


def arm_summary(client,native):
    rows=[r for r in client.done.values() if client.start<=r['t_done']<client.end]
    totals=[sum(r['tokens'] for r in rows if client.start+i*300<=r['t_done']<client.start+(i+1)*300) for i in range(3)]
    total=sum(totals);mean=statistics.mean(totals)
    cv=statistics.pstdev(totals)/mean if mean else None
    seconds=sum(n*dt for n,dt in native.decode_seconds.items())
    issues=[]
    if client.load['starved_s']>0:issues.append('genuine corpus starvation')
    if client.load['idle_fraction']>IDLE_FRACTION_MAX or client.load['max_vacancy_s']>VACANCY_EPISODE_MAX+1e-9:issues.append('client generator not saturated')
    load_valid=not issues
    if len(rows)<30:issues.append('fewer than 30 completions')
    if cv is None or cv>.15:issues.append('block instability')
    return dict(valid=not issues,load_valid=load_valid,issues=issues,window=[client.start,client.end],load=client.load,
                completed=len(rows),completion_tokens=total,U_resp=total/MEASURE,
                native_mean_R=seconds/MEASURE,native_p9=sum(dt for n,dt in native.decode_seconds.items() if n>=9)/MEASURE,
                native_stream_seconds=seconds,per_stream_rate=total/seconds if seconds else None,
                block_cv=cv,measured_rows=rows,unfinished=len(client.pending))


def reduce_arm(bundle,catalog,reader):
    try:
        require(corpus.digest(corpus.canonical(catalog))==corpus.CATALOG_SHA,'immutable catalog identity')
        client_boot=validate_log(bundle['client_events'],'replay_start')
        native_boot=validate_log(bundle['native_events'],'capture_start')
        require(client_boot==native_boot,'cross-source boot mismatch')
        client_header=bundle['client_events'][0];native_header=bundle['native_events'][0]
        for field in ('session_id','gpu_uuid'):
            require(isinstance(client_header[field],str) and bool(client_header[field])
                    and client_header[field]==native_header[field],'cross-source '+field)
        client=Client(catalog,bundle['client_events'],reader).run()
        native=Native(client,bundle['native_events'],reader).run()
        result=arm_summary(client,native)
        result['capture']=dict(boot_id=client_boot,session_id=client_header['session_id'],gpu_uuid=client_header['gpu_uuid'],
                               native_pid=native_header['pid'],origin=client.origin,
                               start=min(client.origin,native_header['t']),
                               end=max(bundle['client_events'][-1]['t'],bundle['native_events'][-1]['t']))
        return result
    except (Invalid,KeyError,TypeError,ValueError,IndexError,AttributeError,OSError) as error:
        return dict(valid=False,issues=[str(error) or type(error).__name__])


def histogram(rows,field,bounds):
    counts=[0]*(len(bounds)-1)
    for row in rows:
        record=row['record'];value=len(record['images']) if field=='images' else record[field]
        index=next((i for i,(lo,hi) in enumerate(zip(bounds,bounds[1:])) if lo<value<=hi),None)
        require(index is not None,'histogram value outside fixed bins');counts[index]+=1
    return [n/len(rows) for n in counts]


def mix(a,b):
    out={}
    for field,bounds in [('prompt',[0,32768,65536,98304,139264]),('images',[0,1,8,16,32,58]),('cap',[0,1024,2048,4096,8192,12288])]:
        x=histogram(a,field,bounds);y=histogram(b,field,bounds)
        out[field+'_tvd']=sum(abs(u-v) for u,v in zip(x,y))/2
    early=lambda rows:sum(r['tokens']<r['record']['cap'] for r in rows)/len(rows)
    fraction=lambda rows:statistics.mean(r['tokens']/r['record']['cap'] for r in rows)
    out['early_stop_difference']=abs(early(a)-early(b))
    out['completion_fraction_ratio']=fraction(b)/fraction(a)
    return out


def compare(control,candidate,catalog,reader):
    a=reduce_arm(control,catalog,reader);b=reduce_arm(candidate,catalog,reader)
    issues=[];metrics=None;ratio=None
    if not a['valid'] or not b['valid']:issues.append('invalid arm')
    else:
        for field in ('boot_id','session_id','gpu_uuid'):
            if a['capture'][field]!=b['capture'][field]:issues.append('pair capture identity')
        if a['capture']['end']>b['capture']['start'] or a['capture']['native_pid']==b['capture']['native_pid']:
            issues.append('pair is not fresh sequential loads')
        metrics=mix(a['measured_rows'],b['measured_rows'])
        if any(metrics[k]>.10 for k in ('prompt_tvd','images_tvd','cap_tvd')):issues.append('workload mix')
        if metrics['early_stop_difference']>.05 or not .9<=metrics['completion_fraction_ratio']<=1.1:issues.append('output-length mix')
        ratio=b['U_resp']/a['U_resp']
    verdict='FEASIBILITY_ONLY' if issues else ('CLOSE_TESTED_CONFIG' if ratio<1.20 else 'QUALITY_INVESTIGATION_ONLY')
    for arm in (a,b):arm.pop('measured_rows',None)
    scope=lambda value:value.get('scope') if isinstance(value,dict) else None
    return dict(verdict=verdict,issues=issues,ratio=ratio if not issues else None,mix=metrics,control=a,candidate=b,
                evidence_scope=[scope(control),scope(candidate)])
