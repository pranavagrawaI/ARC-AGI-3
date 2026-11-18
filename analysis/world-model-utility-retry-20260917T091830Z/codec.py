"""Lossless representation only: integer RLE rows and request-local frame references."""
import copy,json

def grid_encode(g):
 rows=[];indices=[]
 for row in g:
  runs=[]
  for v in row:
   if runs and runs[-1][1]==v:runs[-1][0]+=1
   else:runs.append([1,v])
  flat=[x for pair in runs for x in pair]
  if flat not in rows:rows.append(flat)
  indices.append(rows.index(flat))
 return {'rows':rows,'row_indices':indices}

def grid_decode(f):
 return [[v for count,v in zip(f['rows'][i][::2],f['rows'][i][1::2]) for _ in range(count)] for i in f['row_indices']]

def pack(values):
 frames=[];originals=[];texts=[];grid_rows=[]
 def visit(v):
  if isinstance(v,str) and len(v)>=48:
   if v not in texts:texts.append(v)
   return {'text_ref':texts.index(v)}
  if isinstance(v,list):
   keys=None;rows=None
   if len(v)>=2 and all(isinstance(x,dict) and set(x)==set(v[0]) for x in v):
    keys=sorted(v[0]);rows=[[x[k] for k in keys] for x in v]
   elif len(v)>=2 and all(isinstance(x,list) and len(x)==len(v[0]) for x in v) and v[0]:rows=v
   if rows is not None:
    columns=[]
    for col in zip(*rows):
     runs=[]
     for x in col:
      if runs and runs[-1][1]==x:runs[-1][0]+=1
      else:runs.append([1,x])
     encoded_runs=[[n,visit(x)] for n,x in runs]
     if len(col)>2 and all(type(x) is int for x in col) and len(set(b-a for a,b in zip(col,col[1:])))==1:
      arithmetic={'arithmetic':[col[0],col[1]-col[0],len(col)]}
      if len(json.dumps(arithmetic))<len(json.dumps(encoded_runs)):encoded_runs=arithmetic
     columns.append(encoded_runs)
    return {'table_columns':keys,'table_runs':columns}
   return [visit(x) for x in v]
  if not isinstance(v,dict):return v
  if v.get('encoding') in ('hex_rows','integer_matrix'):
   kind=v['encoding'];g=[[int(c,16) for c in r] for r in v['rows']] if kind=='hex_rows' else v['grid']
   if g not in originals:
    local=grid_encode(g);indices=[]
    for j in local['row_indices']:
     row=local['rows'][j]
     if row not in grid_rows:grid_rows.append(row)
     indices.append(grid_rows.index(row))
    runs=[]
    for i in indices:
     if runs and runs[-1][1]==i:runs[-1][0]+=1
     else:runs.append([1,i])
    frame={'row_runs':[x for pair in runs for x in pair]}
    originals.append(g);frames.append(frame)
   return {**{k:visit(x) for k,x in v.items() if k not in ('encoding','rows','grid')},'encoding':'frame_ref','frame':originals.index(g),'original_encoding':kind}
  return {k:visit(x) for k,x in v.items()}
 result=visit(values)
 return {'codec':'rle-frame-v1','frames':frames,'grid_rows':grid_rows,'texts':texts,'value':result}

def unpack(packed):
 def frame_grid(i):
  runs=packed['frames'][i]['row_runs']
  indices=[j for n,j in zip(runs[::2],runs[1::2]) for _ in range(n)]
  return grid_decode({'rows':packed['grid_rows'],'row_indices':indices})
 def visit(v):
  if isinstance(v,list):return [visit(x) for x in v]
  if not isinstance(v,dict):return v
  if set(v)=={'text_ref'}:return packed['texts'][v['text_ref']]
  if set(v)=={'table_columns','table_runs'}:
   def column(runs):
    if isinstance(runs,dict):
     start,step,count=runs['arithmetic'];return [start+i*step for i in range(count)]
    return [visit(x) for n,x in runs for _ in range(n)]
   columns=[column(runs) for runs in v['table_runs']]
   return [dict(zip(v['table_columns'],row)) if v['table_columns'] is not None else list(row) for row in zip(*columns)]
  if v.get('encoding')=='frame_ref':
   g=frame_grid(v['frame']);kind=v['original_encoding']
   result={k:visit(x) for k,x in v.items() if k not in ('encoding','frame','original_encoding')};result['encoding']=kind
   result.update({'rows':[''.join(format(c,'x') for c in row) for row in g]} if kind=='hex_rows' else {'grid':g})
   return result
  return {k:visit(x) for k,x in v.items()}
 return visit(packed['value'])

def messages(packet,exchange):
 # Shared frame table across current packet AND immediately previous tool result.
 values={'packet':packet,'previous_tool_result':json.loads(exchange[-1]['content']) if exchange else None}
 p=pack(values)
 tail=copy.deepcopy(exchange)
 for message in tail:
  for call in message.get('tool_calls',[]):
   call['function']['arguments']=json.dumps(json.loads(call['function']['arguments']),ensure_ascii=False,separators=(',',':'))
 if tail:tail[-1]['content']=json.dumps({'result_location':'current user message value.previous_tool_result; shared frames apply'})
 return [{'role':'user','content':json.dumps(p,separators=(',',':'),sort_keys=True,ensure_ascii=False)}]+tail

EXPLANATION='''Lossless rle-frame-v1 decoder:
value.packet holds current evidence; value.previous_tool_result holds the preceding tool result referenced by the tool message. {text_ref:i} means texts[i]. For {table_columns:keys,table_runs:columns}, expand each column's [count,value] runs, then transpose into rows. A column {arithmetic:[start,step,count]} is start+i*step for i=0..count-1. With keys=null, rows are lists; otherwise zip rows with keys into objects. Apply substitutions recursively.
encoding=frame_ref selects frames[frame]. Its flat row_runs contains count,row_index pairs. Expand these, then expand grid_rows[row_index]'s flat count,color pairs into integer pixels. Rows are top-to-bottom, columns left-to-right; x=column,y=row. original_encoding is archival metadata. References preserve complete frames. executed_summary rows are [step,action,state,levels_completed].'''
