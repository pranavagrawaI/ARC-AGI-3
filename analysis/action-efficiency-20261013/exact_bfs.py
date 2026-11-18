"""Certify Tu93 L0-L2 shortest routes with a source-derived semantic state key.

Includes exact timer, direction queues, internal phase, and every sprite field.
Unlike the archived oracle, does not merge by rendered pixels. All four valid actions.
Stops on first goal at BFS depth; this proves shortest distance without exhausting deeper states.
"""
import collections
import copy
import json
import pickle
import time
from replay_audit import engine, act
from analyze import HERE, ROOT, parse


def key(g):
    sprites = g.current_level.get_sprites()
    states = [tuple((k, (v.shape,v.dtype.str,v.tobytes()) if hasattr(v,'tobytes') else v)
                    for k,v in sorted(vars(s).items())) for s in sprites]
    # Empty queues still matter: their dictionary membership causes later moves to enqueue directions.
    # Removed sprites cannot interact again; their queues are irrelevant to future game transitions.
    queue = tuple((sprites.index(s), tuple(q)) for s,q in g.ylmdnwbdyy.items() if s in sprites)
    return pickle.dumps((g._current_level_index, g._state, g.kdkehgjrzq,
                         tuple(sorted(vars(g.ksulgrfyqx).items())), states, queue), protocol=5)


def search(g0, lv, cap=90):
    t = time.monotonic(); q = collections.deque([()]); seen = {key(g0)}; expansions=0
    while q and time.monotonic()-t < cap:
        path = q.popleft(); g=copy.deepcopy(g0)
        for a in path:
            act(g, {'id':a})
        for name in ('ACTION1','ACTION2','ACTION3','ACTION4'):
            c=copy.deepcopy(g); f=act(c,{'id':name}); expansions+=1
            p=path+(name,)
            if f.levels_completed > lv:
                check=copy.deepcopy(g0)
                for a in p:
                    last=act(check,{'id':a})
                assert last.levels_completed==lv+1
                return dict(level=lv, exact_shortest=True, distance=len(p), route=[{'id':a} for a in p],
                            states=len(seen), expanded_actions=expansions, seconds=time.monotonic()-t,
                            certificate='Breadth-first depth with all valid actions; source-derived key includes hidden state')
            if f.state.name=='GAME_OVER':
                continue
            k=key(c)
            if k not in seen:
                seen.add(k); q.append(p)
    return dict(level=lv,exact_shortest=False,states=len(seen),expanded_actions=expansions,
                seconds=time.monotonic()-t,reason='time cap' if q else 'no goal')


def main():
    b=json.loads((ROOT/'analysis/m2-winners-20261001/runs/out-control/benchmark.json').read_text())
    r=next(r for r in b['game_runs'] if r['game_id'].startswith('tu93'))
    h=parse(r['history']); apl=parse(r['actions_per_level']); k=0; results=[]
    for lv in range(3):
        g=engine(r['game_id']);act(g,{'id':'RESET'})
        for x in h[:k]:
            act(g,x['action'])
        s=search(g,lv);s['game_id']=r['game_id'];results.append(s)
        (HERE/'exact-bfs.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps({a:v for a,v in s.items() if a!='route'}),flush=True)
        k+=apl[lv]


if __name__=='__main__':
    main()
