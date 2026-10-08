import sys, copy, time, hashlib, collections, numpy as np
sys.path.insert(0,str(__import__('pathlib').Path(__file__).parent))
from replay import game, GameAction, ActionInput
from arcengine import GameState
def H(fr): return hashlib.md5(np.asarray(fr.frame[-1]).tobytes()).hexdigest()
def bfs(g, fr, acts, cap=30000, tmax=60):
    L0=fr.levels_completed; start=H(fr); seen={start}; q=collections.deque([(g,0)]); t=time.time(); n=0
    while q and n<cap and time.time()-t<tmax:
        s,d=q.popleft()
        for a in acts:
            c=copy.deepcopy(s); f=c.perform_action(a); n+=1
            if f.levels_completed>L0: return d+1, n, len(seen)
            if f.state==GameState.GAME_OVER: continue
            k=H(f)
            if k in seen: continue
            seen.add(k); q.append((c,d+1))
    return None, n, len(seen)
for gid in sys.argv[1:]:
    g=game(gid); fr=g.perform_action(ActionInput(id=GameAction.RESET))
    av=fr.available_actions; print(gid, 'available', av)
    acts=[ActionInput(id=GameAction[f'ACTION{a}']) for a in av if a != 6]
    if 6 in av:
        # candidate clicks: one per connected colour component (4-conn) of the current frame
        F=np.asarray(fr.frame[-1]); seenc=np.zeros_like(F,bool)
        for y in range(64):
            for x in range(64):
                if seenc[y,x]: continue
                col=F[y,x]; st=[(y,x)]; seenc[y,x]=True; cells=[]
                while st:
                    yy,xx=st.pop(); cells.append((yy,xx))
                    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny,nx=yy+dy,xx+dx
                        if 0<=ny<64 and 0<=nx<64 and not seenc[ny,nx] and F[ny,nx]==col: seenc[ny,nx]=True; st.append((ny,nx))
                if len(cells)<1500: yy,xx=cells[len(cells)//2]; acts.append(ActionInput(id=GameAction.ACTION6, data={'x':xx,'y':yy}))
    print('  branching', len(acts), '-> BFS', bfs(g, fr, acts), 'baseline', g.__class__.__name__)
