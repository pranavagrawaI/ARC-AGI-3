def blk(f):
    for nd in f.segmentation['nodes']:
        if nd['color']=='O': return (nd['boundary'][0][0], nd['boundary'][0][1])
base=history[21].frame.ascii.split('\n')
def g(r,c): return base[r][c] if 0<=r<64 and 0<=c<len(base[r]) else 'c'
box=(39,45,13,19)
def mk(wcols, boxopen):
    def walk(r,c):
        ch=g(r,c)
        if box[0]<=r<=box[1] and box[2]<=c<=box[3]:
            return boxopen and ch!='b'
        return ch in wcols
    return walk
def legal(w,p,d):
    dr,dc={'UP':(-5,0),'DOWN':(5,0),'LEFT':(0,-5),'RIGHT':(0,5)}[d]
    n=(p[0]+dr,p[1]+dc)
    return all(w(n[0]+i,n[1]+j) for i in range(5) for j in range(5))
from collections import deque
targets={'ring0':(16,15,18,17),'ring1':(51,40,53,42),'plus':(46,50,48,52),'boxint':(40,44,14,18),'boxany':box}
for name,(wc,bo) in {'A':('GW',False),'B':('GWY',False),'C':('GWY',True),'D':('GWc',False)}.items():
    w=mk(wc,bo); bad=0
    for i in range(21,39):
        tr=transitions[i]; p=blk(tr.before_frame); exp=blk(tr.after_frame)!=p
        if legal(w,p,tr.action)!=exp: bad+=1
    start=(35,14); dist={start:0}; q=deque([start])
    while q:
        p=q.popleft()
        for d in ['UP','DOWN','LEFT','RIGHT']:
            if legal(w,p,d):
                dr,dc={'UP':(-5,0),'DOWN':(5,0),'LEFT':(0,-5),'RIGHT':(0,5)}[d]
                n=(p[0]+dr,p[1]+dc)
                if n not in dist: dist[n]=dist[p]+1; q.append(n)
    out=[]
    for t,(r0,c0,r1,c1) in targets.items():
        hits=[(p,dist[p]) for p in dist if p[0]<=r0 and p[0]+4>=r1 and p[1]<=c0 and p[1]+4>=c1]
        out.append((t,min([h[1] for h in hits],default=None),len(hits)))
    print(name,'bad',bad,'states',len(dist),out)

