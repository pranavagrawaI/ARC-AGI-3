def gdir_of(b): return (0,3) if b['axis']=='H' else ((3,0) if b['dir']==1 else (-3,0))
def cellsS(b):
    out=set()
    if b['axis']=='H':
        for r in range(b['r'],b['r']+3):
            out.add((r,b['c']-1))
            for c in range(b['c'],b['c']+b['L']): out.add((r,c))
    else:
        for c in range(b['c'],b['c']+3):
            out.add((b['r']-1,c) if b['dir']==1 else (b['r']+b['L'],c))
            for r in range(b['r'],b['r']+b['L']): out.add((r,c))
    return out
def growth_region(b):
    if b['axis']=='H': return {(r,c) for r in range(b['r'],b['r']+3) for c in range(b['c']+b['L'],b['c']+b['L']+3)}
    if b['dir']==1: return {(r,c) for r in range(b['r']+b['L'],b['r']+b['L']+3) for c in range(b['c'],b['c']+3)}
    return {(r,c) for r in range(b['r']-3,b['r']) for c in range(b['c'],b['c']+3)}
def static_set():
    purple=set()
    for r in range(9,12):
        for c in range(33,57): purple.add((r,c))
    for r in range(12,27):
        for c in range(33,36): purple.add((r,c))
    for r in range(15,45):
        for c in range(42,45): purple.add((r,c))
    boxes=set()
    for c0 in (3,18,33,48):
        for r in range(54,61):
            for c in range(c0,c0+13): boxes.add((r,c))
    return purple|boxes
def keyS(S): return tuple((v['r'],v['c'],v['L']) for v in S.values())
def dotS(S):
    b=S['N']; return (b['r']+b['L']-2, b['c']+1) if b['dir']==1 else (b['r']+1,b['c']+1)
def adj(cs):
    out=set()
    for r,c in cs:
        out|={(r+1,c),(r-1,c),(r,c+1),(r,c-1)}
    return out
def removed_cells(b):
    cs=set()
    if b['axis']=='H':
        for r in range(b['r'],b['r']+3):
            for c in range(b['c']+b['L']-3,b['c']+b['L']): cs.add((r,c))
    else:
        if b['dir']==1:
            for c in range(b['c'],b['c']+3):
                for r in range(b['r']+b['L']-3,b['r']+b['L']): cs.add((r,c))
        else:
            for c in range(b['c'],b['c']+3):
                for r in range(b['r'],b['r']+3): cs.add((r,c))
    return cs
import copy
def sim_step(S,op,k,static):
    nb=copy.deepcopy(S); b=nb[k]; g=gdir_of(b)
    d=g if op=='g' else (0,-3) if b['axis']=='H' else ((-3,0) if b['dir']==1 else (3,0))
    occ={x:cellsS(nb[x]) for x in nb}
    if op=='g':
        R=growth_region(b)
        if R&static: return None
        if R&occ[k]: return None
        tgt={k:occ[k]|R}; M=set()
        seeds=[x for x in nb if x!=k and occ[x]&R]
    else:
        if b['L']-3<2: return None
        rm=removed_cells(b)
        tgt={k:occ[k]-rm}; M=set()
        seeds=[x for x in nb if x!=k and adj(occ[x])&rm]
    frontier=list(seeds)
    while frontier:
        x=frontier.pop()
        if x in M or x==k: continue
        M.add(x); tgt[x]={(r+d[0],c+d[1]) for r,c in occ[x]}
        for y in nb:
            if y in M or y==k: continue
            if adj(occ[y])&occ[x]: frontier.append(y)
    allT=set().union(*tgt.values())
    if allT&static: return None
    for x in nb:
        if x not in M and x!=k and allT&occ[x]: return None
    for r,c in allT:
        if not(0<=r<=62 and 0<=c<=62): return None
    if op=='g':
        if b['axis']=='H': nb[k]['L']+=3
        elif b['dir']==1: nb[k]['L']+=3
        else: nb[k]['r']-=3; nb[k]['L']+=3
    else:
        nb[k]['L']-=3
        if nb[k]['axis']=='V' and nb[k]['dir']==-1: nb[k]['r']+=3
    for x in M: nb[x]['r']+=d[0]; nb[x]['c']+=d[1]
    return nb

def successors(s):
    for op in ('g','s'):
        for k in 'YONS':
            yield (op,k),sim_step(s,op,k,static_set())
def goal(s): return dotS(s)==(31,52)
def heuristic(s):
    import math
    r,c=dotS(s)
    return math.ceil((abs(r-31)+abs(c-52))/3)
initial={'O': {'axis': 'H', 'r': 39, 'c': 10, 'L': 2}, 'S': {'axis': 'V', 'r': 30, 'c': 12, 'L': 11, 'dir': -1}, 'Y': {'axis': 'H', 'r': 27, 'c': 13, 'L': 20}, 'N': {'axis': 'V', 'r': 28, 'c': 33, 'L': 2, 'dir': 1}}

plan=bounded_search(initial,successors,state_key=keyS,is_goal=goal,heuristic=heuristic,
                    max_depth=60,max_nodes=100000,max_expanded=100000,max_seconds=20)
assert plan['status']=='FOUND'
print('SANDBOX_SEARCH',plan['depth'],plan['expanded'])
buttons={'Og':12,'Os':6,'Sg':27,'Ss':21,'Yg':42,'Ys':36,'Ng':57,'Ns':51}
for op,k in plan['path']:
    action({'action':'MOUSE','row':57,'col':buttons[k+op]})
print('NATIVE_ACTIONS_COMPLETE')
