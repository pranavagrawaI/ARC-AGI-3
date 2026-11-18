"""CPU-only audit of the preserved corpus and invalid historical replay. No inference."""
import base64
import collections
import hashlib
import json
import math
import statistics
import struct
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=ROOT/'analysis/mtp-depth-20261006'
DATA=OLD/'kaggle/dataset'


def integrate(points,lo,hi,key):
    points=sorted((x['t'],x[key]) for x in points if key in x)
    total=covered=0.0
    for (t,v),(end,_) in zip(points,points[1:]):
        width=max(0,min(end,hi)-max(t,lo))
        total+=width*v;covered+=width
    return {'integral':total,'coverage_s':covered,'mean':total/covered if covered else None}


def http_intervals(events,lo,hi):
    sent={(x['game'],x['idx']):x['t'] for x in events if x['kind']=='send'}
    ended={(x['game'],x['idx']):x['t'] for x in events if x['kind']=='response'}
    changes=collections.defaultdict(float)
    for k,start in sent.items():
        stop=ended.get(k,hi)
        a,b=max(lo,start),min(hi,stop)
        if b>a:changes[a]+=1;changes[b]-=1
    r=0;last=lo;hist=collections.defaultdict(float)
    for t,delta in sorted(changes.items()):
        hist[r]+=t-last;r+=delta;last=t
    hist[r]+=hi-last
    return {'mean':sum(k*v for k,v in hist.items())/(hi-lo),'seconds_by_count':dict(hist)}


def main():
    sys.path.insert(0,str(OLD));import sim,payload
    wl=json.loads((DATA/'workload.json').read_text());games=sim.build(json.loads(json.dumps(wl)))
    valid={g:len(v['requests']) for g,v in wl['games'].items()}
    summary={'source_requests':0,'tail_requests':sum(len(x['reqs']) for x in games.values()),
             'games':len(games),'max_joint_recorded_tokens':0,'max_prompt_tokens':0,
             'max_recorded_completion_tokens':0,'max_images_per_request':0,'max_image_pixels':0,
             'external_image_urls':0,'unreadable_image_headers':0,'over_139264':0,
             'max_tools':0,'message_roles':collections.Counter(),'image_dimensions':collections.Counter(),
             'image_mime_types':collections.Counter(),'wire_fields':collections.Counter()}
    images={};pergame={};caps=[]
    for path in sorted(DATA.glob('*_p0_requests.jsonl')):
        g=path.name[:4];index=0
        for line in path.open():
            row=json.loads(line)
            if row.get('event')!='request':continue
            rec=wl['games'][g]['requests'][index] if index<valid[g] else None
            index+=1
            if rec is None:continue
            body=payload.build(row,'same-model-placeholder',rec['completion_tokens'])
            summary['source_requests']+=1;summary['wire_fields'].update(body.keys())
            summary['max_tools']=max(summary['max_tools'],len(body.get('tools',[])))
            pair=rec['prompt_tokens']+rec['completion_tokens'];caps.append(pair)
            summary['max_joint_recorded_tokens']=max(summary['max_joint_recorded_tokens'],pair)
            summary['max_prompt_tokens']=max(summary['max_prompt_tokens'],rec['prompt_tokens'])
            summary['max_recorded_completion_tokens']=max(summary['max_recorded_completion_tokens'],rec['completion_tokens'])
            summary['over_139264']+=pair>139264
            nimg=0
            for message in body['messages']:
                summary['message_roles'][message['role']]+=1
                content=message.get('content')
                if not isinstance(content,list):continue
                for part in content:
                    if part.get('type')!='image_url':continue
                    nimg+=1;value=part['image_url'];url=value['url'] if isinstance(value,dict) else value
                    if not url.startswith('data:'):
                        summary['external_image_urls']+=1;continue
                    head,encoded=url.split(',',1);summary['image_mime_types'][head]+=1
                    key=hashlib.sha256(url.encode()).hexdigest()
                    if key not in images:
                        blob=base64.b64decode(encoded[:128])
                        images[key]=struct.unpack('>II',blob[16:24]) if blob.startswith(b'\x89PNG\r\n\x1a\n') else None
                    dims=images[key]
                    if dims:
                        summary['image_dimensions'][f'{dims[0]}x{dims[1]}']+=1
                        summary['max_image_pixels']=max(summary['max_image_pixels'],dims[0]*dims[1])
                    else:summary['unreadable_image_headers']+=1
            summary['max_images_per_request']=max(summary['max_images_per_request'],nimg)
        assert valid[g]<=index<=valid[g]+1,(g,index,valid[g])
        pergame[g]=index
    summary['unique_image_payloads']=len(images)
    summary['token_bound_note']='Recorded SGLang token accounting only; candidate processor/tokenizer must be checked separately. No candidate fit claim.'
    summary['files_sha256']={n:hashlib.sha256((DATA/n).read_bytes()).hexdigest() for n in ('manifest.json','workload.json','payload.py','sim.py','alpha.json')}
    (HERE/'corpus-audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    events=[json.loads(x) for x in (ROOT/'analysis/mtp-depth-retry1-readout-20261008/raw/mtp/M3A/events.jsonl').read_text().splitlines()]
    phases={x['name']:x['t'] for x in events if x['kind']=='phase'};lo,hi=phases['measure_start'],phases['measure_end']
    mets=[x for x in events if x['kind']=='metrics']
    native=integrate(mets,lo,hi,'running');http=http_intervals(events,lo,hi)
    holding=integrate(mets,lo,hi,'holding');queue=integrate(mets,lo,hi,'queue')
    end={}
    for x in events:
        if x['kind']=='response':end[(x['game'],x['idx'])]=x['t']
    gaps=[]
    for x in events:
        if x['kind']=='send' and x['idx']>0 and (x['game'],x['idx']-1) in end:
            prev=end[(x['game'],x['idx']-1)]
            if lo<=prev<=hi:gaps.append(x['t']-prev)
    old={'measure_start':lo,'measure_end':hi,'native_running':native,'http_pending':http,
         'client_holding':holding,'native_waiting':queue,'idle_client_slot_mean':10-http['mean'],
         'HTTP_pending_minus_native_running':http['mean']-native['mean'],
         'between_request_gap_median_s':statistics.median(gaps),'between_request_gap_max_s':max(gaps),
         'simulator_measure_start_s':sim.MEASURE_FROM_S,'simulator_horizon_s':sim.HORIZON_S,
         'replay_measure_start_s':lo,'replay_window_s':hi-lo,
         'facts':['All client gate holders=10 during measurement; not all holders had pending HTTP.',
                  'Pending HTTP occupancy exceeds the native running gauge; queue/prefill/frontend and sampling semantics remain inside that gap.',
                  'Warmup stopped after each initial holder completed once plus 30s, earlier than the simulator 180s cutoff.',
                  'No game exhausted during this measured trace; corpus depletion is not the cause.',
                  'Simulator fits a 1020s late horizon; actual replay sampled an earlier 240s block.',
                  'Modeled server rates/gaps are estimates, not captured engine phase timings.'],
         'cause_boundary':'Underoccupancy is decomposed, but its exact prefill/frontend/gauge contribution is unresolved; no alpha refit or GPU runs.',
         'new_policy':'Retain original corpus/gaps and production bands. Use a fixed 600s warmup ceiling/floor policy defined in card; failure remains feasibility-only, no online adjustment.'}
    (HERE/'historical-occupancy-audit.json').write_text(json.dumps(old,indent=2)+'\n')
    print(json.dumps({'corpus':{k:summary[k] for k in ('source_requests','tail_requests','games','max_joint_recorded_tokens','max_images_per_request','max_image_pixels','over_139264','unique_image_payloads')},'historical':{k:old[k] for k in ('native_running','idle_client_slot_mean','HTTP_pending_minus_native_running','replay_measure_start_s')}}))

if __name__=='__main__':main()
