import sys,json
from pathlib import Path
p=Path(sys.argv[1]);sys.path[:0]=[str(p),str(p/'engine')]
from model import Model,audit
from common import put,encoded,digest
out=Path(sys.argv[2]);out.mkdir()
source='def initialize(observation):\n return {"grid":observation["grid"]}\ndef predict(state,action):\n return [{"state":{"grid":state["grid"],"actual_action":action},"commitments":{"type":"grid","predicted_grid":state["grid"]}}]\ndef assimilate(predicted_state,observation,verdict):\n return {"grid":observation["grid"],"last_action":predicted_state["actual_action"]}\n'
obs={'grid':[[0]],'state':'NOT_FINISHED','levels_completed':0,'available_actions':['RIGHT','RESET']}
row={'step':1,'turn':0,'action':'RIGHT','before':obs,'after':obs}
prefix=out/'prefix.jsonl';prefix.write_bytes(encoded(row))
f=out/'FROZEN.json';put(f,{'candidate_source':source,'source_digest':digest(source.encode()),'prefix_sha256':digest(prefix.read_bytes())})
with audit(out):
 m=Model(f,prefix,out);assert m.active
 m.query('RIGHT',obs,1);m.before_actual('RIGHT',obs,1)
 m.after_actual({'step':2,'action':'RIGHT','before':obs,'after':dict(obs,levels_completed=1)})
 assert not m.active and m.reason=='level/terminal boundary'
 m=Model(f,prefix,out);m.before_actual('RESET',obs,1)
 assert not m.active and m.reason=='reset committed'
