import json,glob,re,sys
for f in sorted(glob.glob(sys.argv[1]+"/*/wire.jsonl")):
    ep=f.split("/")[-2]
    reqs=[json.loads(l) for l in open(f) if '"request"' in l[:30]]
    first=set(json.dumps(m,sort_keys=True) for m in reqs[0]["body"]["messages"])
    seen=[];res=[];codes=[]
    for r in reqs:
        for m in r["body"]["messages"]:
            k=json.dumps(m,sort_keys=True)
            if k in first or k in seen: continue
            seen.append(k)
            if m["role"]=="tool": res.append(m["content"] if isinstance(m["content"],str) else json.dumps(m["content"]))
            if m["role"]=="user": res.append(m["content"] if isinstance(m["content"],str) else json.dumps(m["content"]))
            if m["role"]=="assistant":
                for t in m.get("tool_calls") or []: codes.append(t["function"]["arguments"])
    allr="\n".join(res); allc="\n".join(codes)
    search=bool(re.search(r"heapq|deque|astar|bfs",allc,re.I))
    route=len(re.findall(r"WIN at|\(\d+, \[\'[A-Z][+-]\'|len \d+|path|plan len|FOUND",allr))
    native_mismatch=len(re.findall(r"MISMATCH|DIFF |\bmismatch @|pred .{0,80} got",allr))
    exh=len(re.findall(r"EXH|CAP|no solution|not found",allr))
    hist_ok=len(re.findall(r"mismatches:? 0",allr))
    errs=len(re.findall(r'"error"',allr))
    print(f"{ep:24} search_code={search!s:5} route_hits={route:2} native_mismatch={native_mismatch:2} exhausted={exh:2} hist_zero_mismatch={hist_ok} tool_errors={errs:2} calls={len(reqs)}")
