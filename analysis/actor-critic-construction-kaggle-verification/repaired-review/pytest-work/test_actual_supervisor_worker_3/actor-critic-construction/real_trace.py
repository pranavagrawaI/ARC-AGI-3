import hashlib,json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
HERE=Path(__file__).resolve().parent

def observe(state):
 g=state['grid']; cells=[(r,c) for r,row in enumerate(g) for c,v in enumerate(row) if v==12]
 return {'block':[min(r for r,c in cells),min(c for r,c in cells)],'bar_left':next(c for c,v in enumerate(g[61]) if v==11)}
def rows(state): return [''.join(format(int(v),'x') for v in row) for row in state['grid']]
@dataclass(frozen=True)
class RealTraceCase:
 visible:tuple[dict[str,Any],...]; sealed_suffix:tuple[dict[str,Any],...]; planned_actions:tuple[str,...]; incumbent_source:str; incumbent_sha256:str; frontier:int=5
 @property
 def suffix(self): return self.sealed_suffix
 @property
 def sealed_suffix_sha256(self): return hashlib.sha256(json.dumps(self.sealed_suffix,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 def constructor_view(self):
  return {'schema':'ls20-construction-case-v1','objective':'Construct the smallest model that supports the next useful decision, while preserving what is known and exposing what remains unresolved.','frontier':self.frontier,'visible_transition_summaries':[{'step':x['step'],'action':x['action'],'before':observe(x['before']),'after':observe(x['after'])} for x in self.visible],'initial_observation':{'grid_rows':rows(self.visible[0]['before'])},'current_observation':{'grid_rows':rows(self.visible[-1]['after'])},'planned_fixed_actions':list(self.planned_actions),'incumbent':{'source':self.incumbent_source,'sha256':self.incumbent_sha256,'visible_status':'five informative predictions matched'},'host_observer':{'owned_targets':['block','bar_left'],'authority':'factual measurement only; no transition rule or semantic diagnosis'}}
 def compact_view(self):
  p=self.constructor_view(); p.pop('initial_observation');p.pop('current_observation');p['full_evidence']={'path':'/evidence/case.json','scope':'chronological visible transitions only; no future observations'};return p
 def inspection_view(self):
  p=self.constructor_view();p['visible_transitions']=[{'step':x['step'],'action':x['action'],'before':{'grid_rows':rows(x['before'])},'after':{'grid_rows':rows(x['after'])}} for x in self.visible];return p
 def evaluator_payload(self,include_suffix=False): return {'frontier':self.frontier,'visible':list(self.visible),'suffix':list(self.sealed_suffix) if include_suffix else [],'branch_cap':32,'required_targets':['block','bar_left']}
def load_case():
 d=json.loads((HERE/'case.json').read_text()); c=RealTraceCase(tuple(d['visible']),tuple(d['suffix']),tuple(d['planned_actions']),d['incumbent_source'],d['incumbent_sha256'])
 if [x['step'] for x in c.visible]!=[1,2,3,4,5] or [x['step'] for x in c.suffix]!=[6,7,8]: raise ValueError('chronology drift')
 if hashlib.sha256(c.incumbent_source.encode()).hexdigest()!=c.incumbent_sha256: raise ValueError('incumbent drift')
 return c
