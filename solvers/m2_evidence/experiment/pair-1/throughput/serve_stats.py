"""Per-arm SGLang serving stats from serve.log, binned by minutes since the first decode line.
python3 serve_stats.py control.log treatment.log [bin_min]"""
import re, sys, datetime as dt, statistics as st
D = re.compile(r"^\[(\S+ \S+)\] Decode batch, #running-req: (\d+).*?accept len: ([\d.]+).*?gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)")
P = re.compile(r"^\[(\S+ \S+)\] Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+).*?#running-req: (\d+)")
def parse(path):
    dec, pre = [], []
    for line in open(path, errors="replace"):
        m = D.match(line)
        if m: dec.append((dt.datetime.fromisoformat(m[1]), int(m[2]), float(m[3]), float(m[4]), int(m[5]))); continue
        m = P.match(line)
        if m: pre.append((dt.datetime.fromisoformat(m[1]), int(m[2]), int(m[3]), int(m[4])))
    return dec, pre
def summarize(name, path, bin_min):
    dec, pre = parse(path); t0 = dec[0][0]; rows = {}
    for t, run, acc, thr, q in dec:
        b = int((t - t0).total_seconds() // (bin_min * 60)); r = rows.setdefault(b, {"run": [], "acc": [], "thr": [], "q": [], "new": 0, "cached": 0})
        r["run"].append(run); r["acc"].append(acc); r["thr"].append(thr); r["q"].append(q)
    for t, seqs, new, cached in pre:
        b = int((t - t0).total_seconds() // (bin_min * 60))
        if b in rows: rows[b]["new"] += new; rows[b]["cached"] += cached
    allthr = [d[3] for d in dec if d[1] > 0]; new = sum(p[2] for p in pre); cached = sum(p[3] for p in pre)
    print(f"== {name}: decode lines {len(dec)}, mean gen tok/s {st.mean(allthr):.0f}, mean running {st.mean(d[1] for d in dec):.2f}, "
          f"mean accept len {st.mean(d[2] for d in dec):.2f}, prefill new {new/1e6:.1f}M cached {cached/1e6:.1f}M ({cached/max(1,new+cached):.0%}), span {(dec[-1][0]-t0).total_seconds()/60:.0f} min")
    print(" bin  run   acc   tok/s  queue  prefillNewK cachedK")
    for b in sorted(rows):
        r = rows[b]
        print(f" {b*bin_min:4d} {st.mean(r['run']):5.2f} {st.mean(r['acc']):5.2f} {st.mean(r['thr']):6.0f} {st.mean(r['q']):5.2f} {r['new']/1e3:9.0f} {r['cached']/1e3:9.0f}")
    return dec, pre
if __name__ == "__main__":
    bm = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    summarize("control", sys.argv[1], bm); summarize("treatment", sys.argv[2], bm)
