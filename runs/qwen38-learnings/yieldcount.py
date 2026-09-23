import json,glob,re,sys,collections
tot=collections.Counter(); zero=[]
for f in sorted(glob.glob(sys.argv[1]+"/*events.jsonl")):
    g=f.split("/")[-1][:4]; ne=n=acts=0
    for l in open(f):
        e=json.loads(l)
        if e["type"]=="action": acts+=1
        if e["type"]!="analysis": continue
        n+=1; t=e["transcript"]; tail=t[t.rfind("[ANALYZER STATUS]"):]
        if "step_executed: True" in tail: tot["exec"]+=1
        elif "turn_time_budget" in tail: tot["yield"]+=1; ne+=1
        else: tot["error/other"]+=1; ne+=1
    if acts==0: zero.append(g)
s=sum(tot.values()); print({k:f"{v} ({100*v/s:.0f}%)" for k,v in tot.items()}, "games with 0 actions:",zero)
