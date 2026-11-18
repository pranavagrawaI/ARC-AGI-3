"""Read-only raw artifact audit. Run with python3 -B; outputs stay beside this file."""
import collections as C
import datetime as D
import hashlib
import json
import re
import statistics as S
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = {
    'c14': 'analysis/model-serving-20261011/wave-1/e2-c14/output',
    'b1_c14': 'analysis/model-serving-20261011/wave-2/b1-c14/output',
    'base_c1': 'analysis/m2-winners-20261001/runs/out-control',
    'base_pc': 'solvers/m2_evidence/experiment/pair-1/control/output',
}
def load(p): return json.loads(p.read_text())
def parsed(v): return json.loads(v) if isinstance(v, str) else v
def stamp(v): return D.datetime.fromisoformat(v).timestamp()
def quant(v):
    v = sorted(v)
    return {f'p{q}': v[int((len(v)-1)*q/100)] if v else None for q in [0,50,90,99,100]}
def content(m):
    c = m.get('content', '')
    if isinstance(c, list): return '\n'.join(x.get('text','') for x in c if isinstance(x,dict))
    return c or ''
def level(m):
    hits = re.findall(r'Current state: step \d+, level (\d+)\.', content(m))
    return int(hits[-1]) if hits else None
def fingerprint(m):
    return hashlib.sha256(json.dumps(m, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
def sections(t):
    pat = re.compile(r'^\[(SYSTEM PROMPT|USER PROMPT|MODEL RESPONSE META|THINKING|MODEL RESPONSE|TOOL CALL: [^\]\n]+|TOOL RESULT: [^\]\n]+|ANALYZER STATUS|ROLLING SUMMARY(?: [A-Z]+)?)\]\s*$|^--- analysis_step=.*?---\s*$',re.M)
    ms=list(pat.finditer(t))
    for i,m in enumerate(ms):
        yield m.group(1) or 'TURN',t[m.end():ms[i+1].start() if i+1<len(ms) else len(t)]
def audit_run(label, rel):
    out=ROOT/rel; b=load(out/'benchmark.json'); games=[]; curves=C.Counter(); levels=[]
    for g in b['game_runs']:
        h=parsed(g['history']); apl=parsed(g['actions_per_level']); done=int(g['levels_completed']); n=g['number_of_levels']
        k=0; enter=0.; segments=[]
        for L in range(min(done+1,n)):
            end=k+apl[L]; seg=h[k:end]; solved=L<done
            endtime=(h[end-1]['wallclock_seconds'] if end else enter) if solved else g['final_wallclock_seconds']
            row=dict(game=g['game_id'][:4],level=L+1,solved=solved,entry_s=enter,end_s=endtime,
                     duration_s=endtime-enter,actions=len(seg),tokens=sum(x.get('generated_tokens',0) for x in seg))
            segments.append(row);levels.append(row);k=end;enter=endtime
            if solved:
                for minute in [5,10,15,20,30,40,50,60,75,90,100,110,120,122]:
                    if endtime<=minute*60:curves[minute]+=1
        games.append(dict(game=g['game_id'][:4],levels=done,total_levels=n,state=g['state'],score=g['final_score'],
                          started_at=g['started_at'],duration_s=g['final_wallclock_seconds'],
                          tokens=sum(x.get('generated_tokens',0) for x in h)+g.get('final_generated_tokens',0),
                          actions=len(h),resets=sum(x['action']['id']=='RESET' for x in h),segments=segments))
    counters=C.Counter(); touched=C.defaultdict(set); touched_levels=C.defaultdict(set); examples=C.defaultdict(list)
    error_types=C.Counter(); turn_counts=C.Counter(); turn_levels=C.defaultdict(set)
    for f in sorted((out/'transcripts').glob('*.txt')):
        game=f.name[:4]; L=1
        for kind,body in sections(f.read_text(errors='replace')):
            if kind=='USER PROMPT':
                match=re.search(r'Current state: step \d+, level (\d+)\.',body)
                if match:L=int(match.group(1))
            flags=[]
            if kind=='TOOL RESULT: python':
                counters['tool_results']+=1
                if 'Traceback (most recent call last)' in body or body.lstrip().startswith('error'):
                    flags.append('tool_errors')
                    es=re.findall(r'^(\w*Error|\w*Exception):',body,re.M);error_types[es[-1] if es else 'other']+=1
                if re.search(r'Tool execution timed out|Python tool timed out|timed out after|execution timeout',body,re.I):flags.append('tool_timeouts')
                if re.search(r'\[truncated \d+ chars',body):flags.append('tool_output_truncated')
                if re.search(r'stop_reason: (?:known_noop|stale_state|known_death)|refused|blocked',body,re.I):flags.append('guard_or_refusal_text')
            if kind=='ANALYZER STATUS':
                if 'step_executed:' in body:
                    counters['analyzer_invocations']+=1
                    if 'step_executed: False' in body:flags.append('no_action_invocations')
                if 'message: Yielded control' in body:
                    reason=re.search(r'message: Yielded control to solver: ([^.\n]+)',body)
                    flags.append('yield_'+(reason.group(1) if reason else 'other'))
                for key in ['context_overflow_recovered','request_timeout','rolling_summary:']:
                    if key in body:flags.append(key.rstrip(':'))
            for key in flags:
                counters[key]+=1;touched[key].add(game);touched_levels[key].add((game,L))
                if len(examples[key])<4:examples[key].append(dict(path=str(f.relative_to(ROOT)),level=L,text=body[:1400]))
    requests=C.Counter(); usages=[]; completions=[]; trim_events=[]; request_levels=set(); tool_seen=set(); image_counts=[]; pixels=[]
    no_action_tokens=0; length_levels=set(); max_prompt=0; total_prompt=0; total_cached=0
    response_turns={}; loss_by_level=C.Counter(); lost_roles=C.Counter(); tool_result_keys=C.Counter(); tool_error_calls=0; tool_timeout_calls=0; tool_truncated_calls=0; unique_tool_results=0
    length_tokens=0; length_rows=[]; tool_rows=[]; opening_losses=[]
    level_transitions=[]
    for f in sorted(out.glob('*_requests.jsonl')):
        game=f.name[:4]; prev=[]; currentL=1; prevL=1; reqno=0
        seen_tools=set(); openings={}
        for lineno,line in enumerate(f.open(),1):
            j=json.loads(line);messages=j.get('messages',[])
            newL=next((level(m) for m in reversed(messages) if level(m) is not None),currentL)
            if j.get('event')=='request':
                reqno+=1;requests['request']+=1;request_levels.add((game,newL))
                for m in messages:
                    ml=level(m)
                    if ml is not None:openings.setdefault(ml,content(m))
                fps={fingerprint(m) for m in messages[1:]}
                lost=[m for m in prev[1:] if fingerprint(m) not in fps]
                # Normal multimodal policy strips old images. Count only missing TEXT+role+tool identity.
                textfps={(m.get('role'),content(m),m.get('tool_call_id'),json.dumps(m.get('tool_calls'),sort_keys=True),m.get('reasoning_content')) for m in messages[1:]}
                lost=[m for m in lost if (m.get('role'),content(m),m.get('tool_call_id'),json.dumps(m.get('tool_calls'),sort_keys=True),m.get('reasoning_content')) not in textfps]
                if lost:
                    chars=sum(len(content(m))+len(str(m.get('reasoning_content',''))) for m in lost)
                    event=dict(game=game,level=newL,line=lineno,analysis_step=j.get('analysis_step'),
                               old_messages=len(prev),new_messages=len(messages),lost_messages=len(lost),lost_chars=chars,
                               lost_roles=dict(C.Counter(m.get('role') for m in lost)),
                               oldest_lost_level=next((level(m) for m in lost if level(m)),None),
                               same_level_user_message_lost=any(level(m)==newL for m in lost),
                               evidence=str(f.relative_to(ROOT)))
                    if len(trim_events)<500:trim_events.append(event)
                    requests['text_eviction_events']+=1;loss_by_level[(game,newL)]+=1
                    lost_roles.update(m.get('role') for m in lost)
                    if any(level(m)==newL and content(m)==openings.get(newL) for m in lost):opening_losses.append(event)
                    for m in lost:
                        if len(examples['evicted_text'])<6 and m.get('role') in ['assistant','tool']:
                            examples['evicted_text'].append(dict(path=str(f.relative_to(ROOT)),line=lineno,level=newL,role=m.get('role'),text=(content(m)+'\n'+str(m.get('reasoning_content','')))[:1200]))
                if newL!=currentL:
                    level_transitions.append(dict(game=game,level=newL,previous_level=currentL,request_line=lineno,
                                                  old_messages=len(prev),new_messages=len(messages),lost_text_messages=len(lost),
                                                  earlier_level_user_messages=sum(level(m) is not None and level(m)<newL for m in messages),evidence=str(f.relative_to(ROOT))))
                currentL=newL;prev=messages
                imgs=[x for m in messages for x in (m.get('content') or []) if isinstance(x,dict) and x.get('type')=='image_url']
                image_counts.append(len(imgs))
                # PNG IHDR yields actual uploaded dimensions without image decoding.
                for x in imgs:
                    u=x.get('image_url',{}).get('url','')
                    if u.startswith('data:image/png;base64,'):
                        import base64,struct
                        raw=base64.b64decode(u.split(',',1)[1][:64]);pixels.append(struct.unpack('>II',raw[16:24]))
                for m in messages:
                    if m.get('role')!='tool':continue
                    ident=m.get('tool_call_id')
                    if not ident or ident in seen_tools:continue
                    seen_tools.add(ident)
                    try:v=json.loads(content(m))
                    except (ValueError,TypeError):continue
                    unique_tool_results+=1
                    tool_result_keys.update(v.keys())
                    tool_error_calls+=bool(v.get('error') or v.get('returncode',0))
                    is_timeout=bool(re.search(r'Tool timed out after',str(v.get('error',''))))
                    tool_timeout_calls+=is_timeout
                    tool_truncated_calls+=bool(v.get('truncated',False))
                    if v.get('error') or v.get('truncated'):
                        tool_rows.append(dict(game=game,level=newL,line=lineno,path=str(f.relative_to(ROOT)),
                            timed_out=is_timeout,truncated=bool(v.get('truncated')),error=str(v.get('error',''))[:1200],
                            function_retention=v.get('function_retention','')))
            elif j.get('event')=='response':
                requests['response']+=1;u=j.get('usage') or {}; c=u.get('completion_tokens',0);completions.append(c)
                total_prompt+=u.get('prompt_tokens',0);max_prompt=max(max_prompt,u.get('prompt_tokens',0));total_cached+=(u.get('prompt_tokens_details') or {}).get('cached_tokens',0)
                requests['finish_'+str(j.get('finish_reason'))]+=1
                if j.get('finish_reason')=='length':
                    length_levels.add((game,currentL));length_tokens+=c
                    length_rows.append(dict(game=game,level=currentL,line=lineno,tokens=c,reasoning_tokens=u.get('reasoning_tokens'),evidence=str(f.relative_to(ROOT))))
                key=(game,j.get('analysis_step'));response_turns.setdefault(key,dict(level=currentL,tokens=0))['tokens']+=c
    events=C.Counter();ev_games=C.defaultdict(set);ev_levels=C.defaultdict(set);batch_sizes=[];logical={}
    for f in sorted((out/'artifacts').glob('*events.jsonl')):
        game=f.name[:4]
        for line in f.open():
            v=json.loads(line);kind=v.get('type');L=v.get('level',1)
            if kind=='action':
                events['actions']+=1
                if v.get('batch_index')==1:batch_sizes.append(v.get('batch_size',1))
                for key in ['game_over','level_completed','run_complete']:
                    if v.get(key):events[key]+=1;ev_games[key].add(game);ev_levels[key].add((game,L))
            if kind=='analysis':
                t=v.get('transcript','');events['analysis_events']+=1
                key=(game,v.get('analysis_step'))
                a=logical.setdefault(key,dict(acted=False,level=L))
                a['acted']=a['acted'] or bool(re.search(r'step_executed: True',t))
    events['logical_turns']=len(logical)
    for key,v in logical.items():
        if not v['acted']:
            events['never_acted_logical_turns']+=1;ev_games['never_acted_logical_turns'].add(key[0]);ev_levels['never_acted_logical_turns'].add((key[0],v['level']))
            no_action_tokens+=response_turns.get(key,{}).get('tokens',0)
    assert no_action_tokens<=sum(completions)
    serve=(out/'serve.log').read_text(errors='replace');decode=[]
    pat=re.compile(r'^\[([^]]+)\] Decode batch, #running-req: (\d+),.*?full token usage: ([\d.]+).*?gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)',re.M)
    for m in pat.finditer(serve):decode.append((stamp(m[1]),int(m[2]),float(m[3]),float(m[4]),int(m[5])))
    tw=C.Counter();tw_time=C.Counter();dt_bins=C.Counter()
    t0=stamp(b['start_time']);end=stamp(b['end_time']); game_ends=[stamp(g['started_at'])+g['duration_s'] for g in games]
    for a,nxt in zip(decode,decode[1:]):
        t,r,k,tps,q=a;w=max(0,min(10,nxt[0]-t));minute=(t-t0)/60
        binname='0-30' if minute<30 else '30-90' if minute<90 else '90-end'
        tw[binname]+=w*tps;tw_time[binname]+=w
        dt_bins[r]+=w
    capacity_loss=sum((max(game_ends)-x) for x in game_ends)
    startup={}
    for f in out.glob('*.log'):
        if f.name=='serve.log':continue
        raw=f.read_text(errors='replace')
        try:raw=''.join(x.get('data','') for x in json.loads(raw))
        except (ValueError,TypeError):pass
        startup[str(f.relative_to(ROOT))]={pat:sorted(set(re.findall(pat,raw)))[:5] for pat in [r'READY after \d+s',r'DEADLINE at \d+s',r'priority gate active: \d+ concurrent streams']}
        try:
            logs=json.loads(f.read_text());startup[str(f.relative_to(ROOT))]['ready_log_elapsed_s']=next(x['time'] for x in logs if 'READY after' in x.get('data',''))
        except (ValueError,TypeError,StopIteration):pass
    return dict(label=label,evidence=rel,n_games=len(games),total_levels=sum(g['total_levels'] for g in games),
                completed_levels=sum(g['levels'] for g in games),reached_levels=len(levels),uncompleted_levels=sum(g['total_levels']-g['levels'] for g in games),
                states=dict(C.Counter(g['state'] for g in games)),curve=dict(curves),games=games,segments=levels,
                counters=dict(counters),touched_games={k:sorted(v) for k,v in touched.items()},touched_levels={k:sorted(v) for k,v in touched_levels.items()},
                errors=dict(error_types),examples=dict(examples),requests=dict(requests),generated_tokens=sum(completions),
                completion_tokens=quant(completions),max_prompt_tokens=max_prompt,prompt_tokens=total_prompt,cached_tokens=total_cached,
                text_evictions=trim_events,eviction_levels=[list(k) for k in loss_by_level],evicted_roles=dict(lost_roles),opening_losses=opening_losses,
                length_tokens=length_tokens,length_rows=length_rows,tool_rows=tool_rows,
                length_levels=sorted(length_levels),request_levels=sorted(request_levels),image_count=quant(image_counts),png_dimensions=dict(C.Counter(str(x) for x in pixels)),
                unique_tool_results=unique_tool_results,tool_result_keys=dict(tool_result_keys),
                tool_errors_visible_in_requests=tool_error_calls,tool_timeouts_visible_in_requests=tool_timeout_calls,tool_truncations_visible_in_requests=tool_truncated_calls,
                level_transitions=level_transitions,event_counts=dict(events),event_games={k:sorted(v) for k,v in ev_games.items()},
                event_levels={k:sorted(v) for k,v in ev_levels.items()},batch_sizes=quant(batch_sizes),batched_calls=sum(x>1 for x in batch_sizes),
                no_action_logical_turn_tokens=no_action_tokens,serve=dict(decode_lines=len(decode),
                    first_decode=decode[0][0] if decode else None,benchmark_start=t0,benchmark_end=end,
                    throughput={k:tw[k]/tw_time[k] for k in tw if tw_time[k]},running_request_seconds=dict(dt_bins),
                    retraction_lines=sum('retract' in x.lower() and 'server_args' not in x for x in serve.splitlines()),
                    fully_idle_at_end_s=max(0,end-decode[-1][0]) if decode else None,
                    per_game_finished_early_seconds=capacity_loss,decode_span_s=decode[-1][0]-decode[0][0] if decode else None),startup=startup)

if __name__=='__main__':
    for label,rel in RUNS.items():
        print('Auditing',label,flush=True)
        data=audit_run(label,rel)
        (HERE/(label+'_metrics.json')).write_text(json.dumps(data,indent=2)+'\n')
        print(label,data['completed_levels'],data['requests'],data['counters'],flush=True)
