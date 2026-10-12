"""SGLang decode-batch series per run: minute since first decode line, running requests, gen tok/s, per-stream tok/s."""
import datetime, json, re, sys
from pathlib import Path

LINE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] Decode batch, #running-req: (\d+).*?gen throughput \(token/s\): ([\d.]+)", re.M)


def series(serve):
    rows = LINE.findall(Path(serve).read_text(errors="replace"))
    t0 = None
    out = []
    for ts, running, gen in rows:
        t = datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").timestamp()
        t0 = t0 or t
        out.append(((t - t0) / 60, int(running), float(gen)))
    # 5-minute bins: median running, mean gen tok/s, gen per running stream
    bins = {}
    for m, r, g in out:
        bins.setdefault(int(m // 5) * 5, []).append((r, g))
    res = []
    for b in sorted(bins):
        rs = sorted(r for r, _ in bins[b]); gs = [g for _, g in bins[b]]
        run_med = rs[len(rs) // 2]; gen = sum(gs) / len(gs)
        res.append({"min": b, "running": run_med, "gen": round(gen, 1), "per_stream": round(gen / max(run_med, 1), 1)})
    return res


if __name__ == "__main__":
    spec = json.loads(sys.argv[1])
    print(json.dumps({k: series(v) for k, v in spec.items()}))
