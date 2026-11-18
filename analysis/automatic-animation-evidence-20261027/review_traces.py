"""Card measure: for the first automatic-image exposure in each of the first ten exposed games, keep the next two
assistant turns (text + tool code) and the tool outputs between them, for grounding against the attached real frame.

  review_traces.py RUN_DIR      (reads RUN_DIR/peak/request-receipts.json and RUN_DIR/peak/output; writes TRACE-REVIEW.*)
"""
import json,sys
from pathlib import Path

def text_of(content):
 if isinstance(content,str):return content
 return '\n'.join(x.get('text','') for x in content if isinstance(x,dict) and x.get('type')=='text')

def assistant_view(m):
 calls=[{'name':c.get('function',{}).get('name'),'arguments':str(c.get('function',{}).get('arguments'))[:3000]} for c in m.get('tool_calls') or []]
 return {'role':'assistant','content':text_of(m.get('content') or '')[:3000],'reasoning':str(m.get('reasoning_content') or m.get('reasoning') or '')[:4000],'tool_calls':calls}

def followups(events,line_no,n=2):
 """Messages after the image-bearing user message, from the longest same-turn request that still contains it."""
 base=events[line_no-1];users=[i for i,m in enumerate(base['messages']) if m['role']=='user'];u=users[-1];anchor=base['messages'][u]
 best=base
 for e in events[line_no:]:
  if e.get('event')!='request':continue
  if len(e['messages'])>u and e['messages'][u]==anchor:best=e if len(e['messages'])>len(best['messages']) else best
  elif e.get('analysis_step')!=base.get('analysis_step'):break
 out=[];seen=0
 for m in best['messages'][u+1:]:
  if m['role']=='assistant':
   seen+=1
   if seen>n:break
   out.append(assistant_view(m))
  elif m['role']=='tool':out.append({'role':'tool','content':text_of(m.get('content') or '')[:1500]})
 return text_of(anchor['content']),out

def main(run):
 folder=Path(run)/'peak';receipts=json.loads((folder/'request-receipts.json').read_text())
 firsts={}
 for img in receipts['images']:firsts.setdefault(img['game'],img)
 chosen=list(firsts.values())[:10];review=[]
 for img in chosen:
  path=folder/'output'/img['request_file'];events=[json.loads(l) for l in path.open()]
  user_text,after=followups(events,img['line'])
  k=user_text.find('left the board area unchanged')
  review.append({**img,'user_cue':user_text[max(0,k-300):k+900] if k>=0 else user_text[-1200:],'followups':after,
                 'judgment':{'accurate_transient_description':None,'contradictory_description':None,'inspection_call':None,'subsequent_action':None,'note':''}})
 summary={'exposed_games':len(firsts),'distinct_automatic_messages':receipts['totals'].get('distinct_automatic_messages',0),
          'request_exposures':receipts['totals'].get('automatic_image_request_exposures',0),
          'qualifying_request_exposures':receipts['totals'].get('qualifying_request_exposures',0),
          'per_game_distinct':{g.split('_p0_')[0]:v.get('distinct_automatic_messages',0) for g,v in receipts['per_game'].items() if v.get('distinct_automatic_messages')}}
 target=Path(run)/'TRACE-REVIEW.json'
 if target.exists():raise SystemExit('refusing to overwrite '+str(target))
 target.write_text(json.dumps({'summary':summary,'cases':review},indent=2)+'\n')
 with (Path(run)/'TRACE-REVIEW-excerpts.md').open('x') as f:
  for c in review:
   f.write(f"## {c['game']} step {c['analysis_step']} action {c['action']} — {c['label']}\n\n![frame]({Path('peak')/c['png']})\n\n```\n{c['user_cue']}\n```\n\n")
   for m in c['followups']:
    if m['role']=='assistant':f.write(f"**assistant** reasoning:\n```\n{m['reasoning'][:2500]}\n```\ncontent: {m['content'][:1500]}\n\ntool calls: {json.dumps(m['tool_calls'])[:2500]}\n\n")
    else:f.write(f"**tool**:\n```\n{m['content'][:800]}\n```\n\n")
 print(json.dumps(summary,indent=1))

if __name__=='__main__':main(sys.argv[1])
