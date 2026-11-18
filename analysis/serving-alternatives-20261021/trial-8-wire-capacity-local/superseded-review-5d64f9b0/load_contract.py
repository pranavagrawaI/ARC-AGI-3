"""Trial4 API load/order gates reused for the separately approved API estimand.
Native fidelity and occupancy are deliberately not implemented here.
"""
import collections
import json
import math
import statistics
import catalog as corpus

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
    def boundary(kind):
        rows=[r for r in events if r['kind']==kind]
        require(len(rows)==1,'phase count: '+kind)
        r=rows[0];value=r['boundary']
        require(0<=r['t']-value<=.25,'boundary receipt jitter')
        return value
    start=boundary('measure_start');end=boundary('measure_end')
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
