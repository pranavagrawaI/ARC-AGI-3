"""Fail-closed raw API-capacity reducer. No native/per-running-stream metric."""
import argparse
import json
from pathlib import Path
import statistics
import accounting
import catalog
import load_contract as contract
import provenance

def transport(raw,rid):
    head,body=raw.split(b'\r\n\r\n',1)
    lines=head.decode('ascii').split('\r\n')
    contract.require(lines[0]=='POST /v1/chat/completions HTTP/1.1','actual HTTP target')
    headers={}
    for line in lines[1:]:
        k,v=line.split(':',1);k=k.lower()
        contract.require(k not in headers,'duplicate transport header');headers[k]=v.strip()
    contract.require(headers['x-request-id']==rid,'transport request ID')
    contract.require(int(headers['content-length'])==len(body),'transport content length')
    return json.loads(body)

class Client(contract.Client):
    def __init__(self,cat,events,reader,arm):
        super().__init__(cat,events,reader);self.arm=arm;self.accounting={}
    def send(self,row):
        super().send(row)
        body=transport(self.reader(row['transport_ref']),row['rid'])
        planned=contract.read_json(self.reader,row['wire_ref'])['body']
        contract.require(body==planned,'actual/planned wire mismatch')
        contract.require(catalog.wire_hash(body)==self.catalog['records'][row['key']]['wire_sha'],'transport/corpus mismatch')
        contract.require('truncate_prompt_tokens' not in body,'request truncation forbidden')
        self.sent[row['rid']]['transport_ref']=row['transport_ref']
    def response(self,row):
        super().response(row)
        body=contract.read_json(self.reader,row['response_ref']);http=contract.read_json(self.reader,row['http_ref'])
        record=self.done[row['rid']]['record']
        contract.require(http['complete'] is True and http['error'] is None and http['status']==200,'incomplete/error HTTP')
        contract.require(http['rid']==row['rid'] and http['key']==self.done[row['rid']]['key'],'HTTP receipt join')
        raw=self.reader(row['response_ref'])
        lengths=[v for k,v in http['headers'] if k.lower()=='content-length']
        contract.require(len(lengths)==1 and int(lengths[0])==len(raw)==http['received_bytes'],'raw complete-body length')
        contract.require(self.done[row['rid']]['t']<=http['started']<=http['ended']<=row['t']<self.end,'raw completion clock')
        contract.require(row['t']-http['ended']<=.25,'response receipt lag')
        self.accounting[row['rid']]=accounting.observe(self.arm,record,body)
        self.done[row['rid']]['t_done']=http['ended']
        if http['ended']<=self.start and record['game'] in self.catalog['initial_holders']:self.warm.add(record['game'])
        writes=http['writes']
        contract.require(writes and all(self.origin<=x['t']<=x['ended']<self.end for x in writes),'actual write outside fixed window')
        contract.require(sum(x['bytes'] for x in writes)==len(self.reader(self.sent[row['rid']]['transport_ref'])),'wire writes')

def actual_load(client,reader):
    """HTTP intervals from actual socket-write/body clocks, not queued intent.
    Corpus starvation uses immutable stream completion progress on those clocks.
    Nothing here estimates native running streams or conditions on cache state.
    """
    intervals=[]
    for rid,attempt in client.sent.items():
        http=contract.read_json(reader,f'{rid}/http.json');writes=http['writes']
        contract.require(writes and all(client.origin<=w['t']<=w['ended']<client.end for w in writes),'missing/late actual socket write')
        start=writes[0]['t'];end=client.done[rid]['t_done'] if rid in client.done else client.end
        contract.require(attempt['t']<=start<=end,'dispatch/body clock')
        contract.require(sum(w['bytes'] for w in writes)==len(reader(attempt['transport_ref'])),'raw write accounting')
        intervals.append((start,end))
    boundaries=sorted({client.start,client.end}|{max(client.start,min(client.end,t)) for pair in intervals for t in pair})
    integral=0.;lost=0.;vacancy=0.;maximum=0.;starved=0.
    for left,right in zip(boundaries,boundaries[1:]):
        mid=(left+right)/2;dt=right-left;n=sum(a<=mid<b for a,b in intervals)
        contract.require(n<=10,'actual HTTP concurrency exceeds S10')
        integral+=n*dt;lost+=(10-n)*dt
        completed={g:0 for g in client.catalog['streams']}
        for row in client.done.values():
            if row['t_done']<=mid:completed[row['record']['game']]+=1
        remaining=sum(completed[g]<len(s) for g,s in client.catalog['streams'].items())
        if remaining<10:starved+=dt
        if n<10 and remaining>=10:vacancy+=dt;maximum=max(maximum,vacancy)
        else:vacancy=0.
    return dict(idle_fraction=lost/(10*900),max_vacancy_s=maximum,starved_s=starved,pending_mean=integral/900,
                clock='ACTUAL_SOCKET_WRITE_TO_COMPLETE_HTTP_BODY',scope='HTTP_OFFERED_LOAD_ONLY')

def arm(directory,cat,label):
    directory=Path(directory)
    def reader(ref):
        p=Path(ref)
        contract.require(not p.is_absolute() and '..' not in p.parts,'unsafe artifact path')
        return (directory/p).read_bytes()
    try:
        events=list(map(json.loads,(directory/'events.jsonl').read_bytes().splitlines()))
        boot=contract.validate_log(events,'replay_start');header=events[0]
        contract.require(events[-1]['threads_alive']==0 and not events[-1]['failed'],'worker cleanup/failure')
        client=Client(cat,events,reader,label).run()
        client.load=actual_load(client,reader)
        rows=[r for r in client.done.values() if client.start<=r['t_done']<client.end]
        totals=[sum(r['tokens'] for r in rows if client.start+i*300<=r['t_done']<client.start+(i+1)*300) for i in range(3)]
        mean=statistics.mean(totals);cv=statistics.pstdev(totals)/mean if mean else None
        issues=[]
        if client.load['starved_s']>0:issues.append('genuine corpus starvation')
        if client.load['idle_fraction']>.005 or client.load['max_vacancy_s']>.25+1e-9:issues.append('client generator not saturated')
        if len(rows)<30:issues.append('fewer than 30 completions')
        if cv is None or cv>.15:issues.append('block instability')
        # Pending requests have raw evidence but no successful completion/usage.
        # Their exact complete transport must still match immutable input.
        for rid,attempt in client.pending.items():
            http=contract.read_json(reader,f'{rid}/http.json')
            contract.require(http['rid']==rid,'cutoff receipt identity')
            contract.require(http['status'] in (None,200),'cutoff HTTP error')
            contract.require(all(x['t']<client.end for x in http['writes']),'post-cutoff write')
            contract.require(not (http['complete'] and http['status']==200 and http['ended']<client.end),'lost completed response before cutoff')
        capture={k:header[k] for k in ('session_id','gpu_uuid','server_pgid','server_start')}
        contract.require(all(isinstance(v,str) and v for v in capture.values()),'capture identity missing')
        capture.update(boot_id=boot,origin=client.origin,start=client.origin,end=events[-1]['t'])
        return dict(valid=not issues,issues=issues,completed=len(rows),completion_tokens=sum(totals),U_resp=sum(totals)/900,
                    block_tokens=totals,block_cv=cv,load=client.load,unfinished=len(client.pending),capture=capture,
                    accounting=client.accounting,measured_rows=rows,native_residence='UNAVAILABLE',
                    available_rate_metrics=['U_resp'],http_pending_scope='OFFERED_LOAD_ONLY')
    except (ValueError,KeyError,TypeError,IndexError,AttributeError,OSError) as exc:
        return dict(valid=False,issues=[str(exc) or type(exc).__name__])

def compare(root):
    root=Path(root);cat=catalog.load();a=arm(root/'control',cat,'control');b=arm(root/'candidate',cat,'candidate')
    issues=[];mix=None;ratio=None;binding=None
    if not a['valid'] or not b['valid']:issues.append('invalid arm')
    else:
        try:binding=provenance.check(root,a,b)
        except (ValueError,KeyError,TypeError,OSError,StopIteration) as error:issues.append(str(error) or type(error).__name__)
        for k in ('session_id','gpu_uuid','boot_id'):
            if a['capture'][k]!=b['capture'][k]:issues.append('pair capture identity '+k)
        if a['capture']['end']>b['capture']['start'] or (a['capture']['server_pgid'],a['capture']['server_start'])==(b['capture']['server_pgid'],b['capture']['server_start']):issues.append('fresh sequential loads')
        mix=contract.mix(a['measured_rows'],b['measured_rows'])
        if any(mix[k]>.10 for k in ('prompt_tvd','images_tvd','cap_tvd')):issues.append('reached workload mix')
        if mix['early_stop_difference']>.05 or not .9<=mix['completion_fraction_ratio']<=1.1:issues.append('output-length mix')
        if not issues:ratio=b['U_resp']/a['U_resp']
    for value in (a,b):value.pop('measured_rows',None)
    verdict='FEASIBILITY_ONLY' if issues else 'CLOSE_TESTED_CONFIG' if ratio<1.20 else 'QUALITY_INVESTIGATION_ONLY'
    return dict(estimand='WIRE_EQUIVALENT_TOTAL_PACKAGE_API_CAPACITY',verdict=verdict,issues=issues,ratio=ratio,mix=mix,control=a,candidate=b,package_binding=binding,
                limitations=accounting.LIMITATIONS+['One fixed-order pair assumes temporal stability; does not identify an engine-only effect.','Full-reply quality and production/adoption remain unqualified.'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--arm',choices=['control','candidate']);a=p.parse_args()
    value=arm(Path(a.root)/a.arm,catalog.load(),a.arm) if a.arm else compare(a.root)
    value.pop('measured_rows',None);print(json.dumps(value,indent=2))
    Path(a.root,('result-'+a.arm if a.arm else 'paired-result')+'.json').write_text(json.dumps(value,indent=2)+'\n')
    raise SystemExit(0 if value.get('valid',True) else 1)
