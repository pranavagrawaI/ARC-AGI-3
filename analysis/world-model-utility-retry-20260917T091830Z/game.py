"""Trusted official environment adapter. No game internals enter actor packets."""
import contextlib,json,sys
from common import HERE,PREFIX,put,journal,digest,encoded
# arc_agi performs dotenv loading at import; explicitly disable before import.
import dotenv
dotenv.load_dotenv=lambda *a,**k:False
from arc_agi import Arcade,OperationMode
from arcengine import GameAction
from prediction_contract import grid
ACTIONS={'RESET':GameAction.RESET,'UP':GameAction.ACTION1,'DOWN':GameAction.ACTION2,'LEFT':GameAction.ACTION3,'RIGHT':GameAction.ACTION4,'SPACEBAR':GameAction.ACTION5,'CLICK':GameAction.ACTION6}

def observed(f):
 if f is None:raise ValueError('official frame missing')
 a=f.frame[-1];g=a.tolist() if hasattr(a,'tolist') else a
 grid(g)
 return {'grid':g,'state':f.state.name,'levels_completed':f.levels_completed,'available_actions':[k for k,v in ACTIONS.items() if v.value in f.available_actions]}

def parse_action(value,obs):
 if not isinstance(value,str):raise ValueError('action string required')
 parts=value.split();name=parts[0] if parts else ''
 if name not in obs['available_actions']:raise ValueError('unavailable action')
 if name=='CLICK':
  if len(parts)!=3 or any(not p.isdecimal() for p in parts[1:]):raise ValueError('CLICK x y integer coordinates')
  x,y=map(int,parts[1:]);h,w=grid(obs['grid'])
  if not 0<=x<w or not 0<=y<h:raise ValueError('CLICK outside grid')
  return ACTIONS[name],{'x':x,'y':y}
 if len(parts)!=1:raise ValueError('unexpected action arguments')
 return ACTIONS[name],{}

class Game:
 def __init__(self,out):
  self.out=out;self.rows=[]
  with contextlib.redirect_stdout(sys.stderr):
   self.arc=Arcade(operation_mode=OperationMode.OFFLINE,environments_dir=str(HERE/'assets/cd82'),recordings_dir=str(out/'recordings'))
   if [g.game_id for g in self.arc.available_environments]!=['cd82-fb555c5d']:raise ValueError('game identity')
   self.card=self.arc.open_scorecard(tags=['world-model-utility'])
   self.env=self.arc.make('cd82-fb555c5d',seed=0,scorecard_id=self.card,save_recording=True)
   if self.env is None:raise ValueError('no official environment')
   # make performs exactly one initialization reset; do not introduce a scored second reset.
   self.current=observed(self.env.observation_space)
  put(out/'INITIAL.json',self.current)
 def step(self,action):
  before=self.current;enum,data=parse_action(action,before)
  with contextlib.redirect_stdout(sys.stderr):self.current=observed(self.env.step(enum,data=data))
  row={'step':len(self.rows)+1,'turn':len(self.rows),'action':action,'before':before,'after':self.current}
  self.rows.append(row);journal(self.out/'transitions.jsonl',row);return row
 def prefix(self):
  initial=self.current
  for action in PREFIX:
   row=self.step(action)
   if row['after']['state']!='NOT_FINISHED' or row['after']['levels_completed']!=initial['levels_completed']:raise ValueError('fixed prefix terminal/level change; no reselection')
  raw=b''.join(encoded(r) for r in self.rows)
  (self.out/'prefix.jsonl').write_bytes(raw)
  record={'actions':PREFIX,'sha256':digest(raw),'post_prefix_sha256':digest(encoded(self.current)),'initial_sha256':digest(encoded(initial))}
  put(self.out/'PREFIX.json',record);return record
 def close(self):
  with contextlib.redirect_stdout(sys.stderr):card=self.arc.close_scorecard(self.card)
  # No API/key exists in offline worker; exclude SDK credential field even if future version adds one.
  value=card.model_dump(mode='json');value.pop('api_key',None);put(self.out/'SCORECARD.json',value)
  return value

def compact(obs):
 # Lossless hexadecimal grid, with decimal matrix fallback outside the16-color palette.
 g=obs['grid'];h,w=grid(g)
 if all(0<=v<16 for r in g for v in r):encoded_grid={'encoding':'hex_rows','rows':[''.join(format(v,'x') for v in r) for r in g],'height':h,'width':w}
 else:encoded_grid={'encoding':'integer_matrix','grid':g,'height':h,'width':w}
 return {**{k:v for k,v in obs.items() if k!='grid'},**encoded_grid}
