"""Offline trace preparation. Operators split known recording; model gets prefix only."""
import hashlib,json,os
from pathlib import Path

def raw_json(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def digest(raw):return hashlib.sha256(raw).hexdigest()
def read(path):return json.loads(Path(path).read_bytes(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
def put(path,value):
 path=Path(path);raw=raw_json(value)
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
 with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return digest(raw)
def journal(path,value):
 with Path(path).open('ab') as f:f.write(raw_json(value).replace(b'\n',b' ')+b'\n');f.flush();os.fsync(f.fileno())

def select(rows,session):
 from prediction_contract import grid
 if len(rows)<2:raise ValueError('no immediate step-2 holdout')
 a,b=rows[:2]
 if [a.get('step'),b.get('step')]!=[1,2] or [a.get('turn'),b.get('turn')]!=[0,1]:raise ValueError('chronology/turn boundary')
 if any(r.get('action')=='RESET' or not isinstance(r.get('action'),str) for r in (a,b)):raise ValueError('reset/action boundary')
 if any(r.get('session_id',session)!=session for r in (a,b)):raise ValueError('episode boundary')
 states=[a['before'],a['after'],b['before'],b['after']]
 if any(s.get('state')!='NOT_FINISHED' for s in states):raise ValueError('terminal episode boundary')
 levels=[s.get('levels_completed') for s in states]
 if any(type(x) is not int for x in levels) or len(set(levels))!=1:raise ValueError('level boundary/unknown')
 shapes=[grid(s['grid']) for s in states]
 if len(set(shapes))!=1:raise ValueError('grid shape boundary')
 if a['after']['grid']!=b['before']['grid']:raise ValueError('observation discontinuity')
 return a,b

def prepare(source,output,session,card):
 source=Path(source);output=Path(output);output.mkdir(exist_ok=False)
 raw=source.read_bytes();rows=[json.loads(x) for x in raw.splitlines() if x.strip()];a,b=select(rows,session)
 (output/'prefix.jsonl').write_bytes(raw.splitlines(keepends=True)[0])
 outcome={'session':session,'step':2,'action':b['action'],'grid':b['after']['grid']}
 (output/'heldout').mkdir(mode=0o700);outcome_sha=put(output/'heldout/outcome.json',outcome)
 query={'session':session,'prefix_step':1,'step':2,'action':b['action'],'outcome_sha256':outcome_sha,'prefix_sha256':digest((output/'prefix.jsonl').read_bytes())}
 put(output/'query.json',query)
 put(output/'SELECTION.json',{'selection_rule':'fixed first step and immediate second step; no outcome-based choice','kind':'retrospective chronological holdout; operators/history have seen data','source':str(source.resolve()),'source_sha256':digest(raw),'session':session,'steps':[1,2],'turns':[0,1],'same_level':a['before']['levels_completed'],'boundary_checks_passed':True,'query_sha256':digest((output/'query.json').read_bytes()),'card_sha256':digest(Path(card).read_bytes()),'prefix_sha256':query['prefix_sha256'],'outcome_sha256':outcome_sha,'operator_split_reads_outcome':True,'scorer_must_not_read_outcome_until_prediction_bound':True})
 return query
