from collections import deque
cur={'Y':{'axis':'H','r':27,'c':13,'L':20},'O':{'axis':'H','r':39,'c':10,'L':2},
     'N':{'axis':'V','dir':1,'c':33,'r':28,'L':2},'S':{'axis':'V','dir':-1,'c':12,'r':30,'L':11}}
st=static_set()
acts=[('g',k) for k in 'YONS']+[('s',k) for k in 'YONS']
seen={keyS(cur):None}; q=deque([(cur,())]); best=(99,None,None); found=None; exp=0
while q:
    S,p=q.popleft()
    if len(p)>=24: continue
    exp+=1
    dt=dotS(S); dd=abs(dt[0]-31)+abs(dt[1]-52)
    if dd<best[0]: best=(dd,dt,p)
    for op,k in acts:
        n2=sim_step(S,op,k,st)
        if n2 is None: continue
        kk=keyS(n2)
        if kk in seen: continue
        seen[kk]=1; p2=p+((op,k),)
        if dotS(n2)==(31,52): found=p2; break
        q.append((n2,p2))
    if found: break
print('states',len(seen),'exp',exp)
print('found',found)
print('best',best)
