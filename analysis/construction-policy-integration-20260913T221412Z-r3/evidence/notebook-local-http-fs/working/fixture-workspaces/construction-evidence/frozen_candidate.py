
def _obs(o):
 g=o["grid"]; cells=[(r,c) for r,row in enumerate(g) for c,v in enumerate(row) if v==12]
 return [min(r for r,c in cells),min(c for r,c in cells)],next(c for c,v in enumerate(g[61]) if v==11)
def initialize(observation):
 b,m=_obs(observation); return {"block":b,"bar_left":m}
def predict(state,action):
 dr,dc={"UP":(-5,0),"DOWN":(5,0),"LEFT":(0,-5),"RIGHT":(0,5)}.get(action,(0,0))
 n={"block":[state["block"][0]+dr,state["block"][1]+dc],"bar_left":state["bar_left"]+1}
 return [{"state":n,"commitments":dict(n),"declared_dependencies":["block","bar_left"]}]
def assimilate(predicted_state,observation,verdict):
 b,m=_obs(observation); return {"block":b,"bar_left":m}
