"""CPU event fixtures over the verified REAL corpus; not model measurements."""
import heapq
import json
import math
import corpus


class Reader:
    def __init__(self,catalog):
        self.catalog=catalog;self.blobs={};self.requests={};self.bodies={};self.renders={}
        prior=corpus.HERE.parent/'trial-3-matched-preparation'
        self.fit=corpus.module('capacity_fixture_render',prior/'processor_fit.py')
        self.template=self.fit.template(prior/'processor-assets/control')

    def __call__(self,reference):
        if reference in self.blobs:return self.blobs[reference]
        kind,key=reference.split('/',1)
        if kind=='wire':
            rid=key;key=self.requests[rid];record=self.catalog['records'][key]
            if key not in self.bodies:self.bodies[key]=corpus.payload_for(record)
            return corpus.canonical(dict(headers={'X-Request-Id':rid},body=self.bodies[key]))
        if kind=='render':
            if key not in self.renders:
                record=self.catalog['records'][key]
                if key not in self.bodies:self.bodies[key]=corpus.payload_for(record)
                body=self.bodies[key]
                text=self.template.render(messages=self.fit.messages_for_template(body['messages']),tools=body.get('tools'),
                                          add_generation_prompt=True,**body.get('chat_template_kwargs',{}))
                self.renders[key]=text.encode()
            return self.renders[key]
        raise ValueError('missing artifact: '+reference)


class Simulation:
    def __init__(self,catalog,reader,label='fixture',decode=1.,prefill=1.,cache=1.,lag=0.,dt=.05,skip_handover=False,extra=0.,origin=0.):
        self.catalog=catalog;self.reader=reader;self.label=label
        self.decode_speed=decode;self.prefill_speed=11942.*prefill;self.cache=cache;self.lag=lag;self.dt=dt
        self.skip_handover=skip_handover;self.stop=1500.+extra;self.origin=origin
        self.queue=list(catalog['initial_holders'])+list(catalog['initial_queue'])
        self.holders=set();self.cursor={g:0 for g in catalog['streams']};self.pending={}
        self.prefill_queue=[];self.callbacks=[];self.serial=0;self.t=0.
        self.client=[];self.native=[]

    def emit(self,source,kind,**fields):
        target=self.client if source=='client' else self.native
        target.append(dict(t=self.t,kind=kind,**fields))

    def admit(self):
        while self.queue and len(self.holders)<10:
            game=self.queue.pop(0);self.holders.add(game)
            self.emit('client','admit',game=game)
            heapq.heappush(self.callbacks,(self.t+self.lag,game))

    def send(self,game):
        key=self.catalog['streams'][game][self.cursor[game]];record=self.catalog['records'][key]
        rid=self.label+'-'+str(self.serial);self.serial+=1
        self.reader.requests[rid]=key
        self.emit('client','send',rid=rid,key=key,wire_ref='wire/'+rid)
        self.emit('native','accept',rid=rid,nid=rid,response_id=rid)
        self.pending[rid]=dict(key=key,game=game,record=record,remaining=record['cap'],
                               pre=max(0.,record['prompt']-min(record['prompt'],record['cached']*self.cache)),decoding=False)
        self.prefill_queue.append(rid)

    def complete(self,rid):
        job=self.pending.pop(rid);record=job['record'];game=job['game'];tokens=record['cap']
        self.emit('native','finish',rid=rid,nid=rid,output_tokens=tokens)
        ref='response/'+rid
        self.reader.blobs[ref]=corpus.canonical(dict(id=rid,usage=dict(prompt_tokens=record['local_prompt'],completion_tokens=tokens),
                                                    choices=[dict(finish_reason='length')]))
        self.emit('client','response',rid=rid,status=200,response_ref=ref)
        self.cursor[game]+=1;stream=self.catalog['streams'][game];index=self.cursor[game]
        if index==len(stream):
            self.holders.remove(game);self.emit('client','release',game=game,reason='finished');self.admit()
        elif not self.skip_handover and not self.catalog['records'][stream[index]]['extends_previous']:
            self.holders.remove(game);self.queue.append(game)
            self.emit('client','release',game=game,reason='handover');self.admit()
        else:heapq.heappush(self.callbacks,(self.t+self.lag,game))

    def advance(self):
        if self.prefill_queue:
            rid=self.prefill_queue[0];job=self.pending[rid];job['pre']-=self.prefill_speed*self.dt
            if job['pre']<=0:
                self.prefill_queue.pop(0);job['decoding']=True;record=job['record']
                self.emit('native','prefill',rid=rid,nid=rid,prompt_tokens=record['local_prompt'],
                          image_hashes=record['images'],grids=[[1,40,40] for _ in record['images']],render_ref='render/'+job['key'])
                self.emit('native','decode_start',rid=rid,nid=rid)
        decoding=[rid for rid,job in self.pending.items() if job['decoding']]
        curve={1:175,2:285,3:316,4:532,5:495,6:558,7:660,8:669,9:687,10:704}
        rate=curve.get(len(decoding),0)/max(1,len(decoding))*self.decode_speed
        for rid in decoding:
            self.pending[rid]['remaining']-=rate*self.dt
            if self.pending[rid]['remaining']<=0:self.complete(rid)

    def run(self):
        self.emit('client','replay_start');self.emit('native','capture_start');self.admit()
        steps=round(self.stop/self.dt)
        for step in range(steps+1):
            self.t=round(step*self.dt,8)
            if self.t==600.:self.emit('client','measure_start')
            if self.t==1500.:self.emit('client','measure_end')
            if self.t<self.stop:
                while self.callbacks and self.callbacks[0][0]<=self.t+1e-9:
                    _,game=heapq.heappop(self.callbacks);self.send(game)
                self.advance()
                # Zero artificial gap means completions refill at the SAME
                # timestamp. Positive lag is an explicit generator fault probe.
                while self.callbacks and self.callbacks[0][0]<=self.t+1e-9:
                    _,game=heapq.heappop(self.callbacks);self.send(game)
            if step%round(1/self.dt)==0:
                self.emit('client','heartbeat');self.emit('native','heartbeat')
        for rid in self.pending:self.emit('client','cutoff_pending',rid=rid)
        # A callback waiting on generator lag is not an HTTP attempt, so the
        # offered-load integral (not a fabricated cutoff) detects that vacancy.
        self.emit('client','capture_end');self.emit('native','capture_end')
        for rows,pid in ((self.client,101),(self.native,202)):
            for seq,row in enumerate(rows):row.update(t=row['t']+self.origin,seq=seq,pid=pid+int(self.origin),boot_id='CPU_FIXTURE_ONLY')
        for rows in (self.client,self.native):rows[0].update(session_id='CPU_FIXTURE_ONLY',gpu_uuid='CPU_FIXTURE_ONLY')
        return dict(client_events=self.client,native_events=self.native,
                    scope='CPU_FIXTURE_ONLY; no engine or GPU executed')


def build(catalog,reader,**options):
    return Simulation(catalog,reader,**options).run()
