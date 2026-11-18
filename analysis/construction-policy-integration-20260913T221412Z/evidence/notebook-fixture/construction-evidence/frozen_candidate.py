
def _o(o):
 g=o["grid"];q=[(r,c) for r,x in enumerate(g) for c,v in enumerate(x) if v==12];return [min(r for r,c in q),min(c for r,c in q)],next(c for c,v in enumerate(g[61]) if v==11)
def initialize(o): b,m=_o(o);return {"block":b,"bar_left":m}
def predict(s,a):
 d={"UP":(-5,0),"DOWN":(5,0),"LEFT":(0,-5),"RIGHT":(0,5)}.get(a,(0,0));n={"block":[s["block"][0]+d[0],s["block"][1]+d[1]],"bar_left":s["bar_left"]+1};return [{"state":n,"commitments":dict(n),"declared_dependencies":["block","bar_left"]}]
def assimilate(s,o,v): b,m=_o(o);return {"block":b,"bar_left":m}
