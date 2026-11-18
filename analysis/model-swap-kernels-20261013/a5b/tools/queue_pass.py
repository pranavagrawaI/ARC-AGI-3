"""Queue only a terminal, exact-version, all-gates-passing A5B smoke. Never submits.

Date rule (Pranav brief): 2026-10-13 if before 20:00Z, else the next free date; never overwrite an entry."""
import datetime as dt
import fcntl
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = ROOT / 'a5b/RESULT.json'
    r = json.loads(p.read_text())
    assert r['smoke_pass'] is True and all(r['gates'].values())
    assert r['collection']['status'] == 'COMPLETE' and r['collection']['version'] == r['version']
    now = dt.datetime.now(dt.timezone.utc)
    day = now.date() if now.hour < 20 else now.date() + dt.timedelta(days=1)
    m, s = r['metrics'], r['a5b']['serve_5_30_benchmark_clock']
    message = (f'A5B top-5 routing (Intel W4A16 unchanged weights, num_experts_per_tok 10->5, unhealed) c14 smoke COMPLETE v{r["version"]}: '
               f'READY, finite logprobs, routing live (server model_path view, SGLang parse target 5/draft 10), 0/25 crashed; '
               f'{s["agg_tok_s"]:.1f} tok/s = {m["tok_ratio_vs_c14_wave1"]:.2f}x c14 (bench-clock min 5-30), accept {s["accept_len_median"]:.2f}; '
               f'levels@30 bench clock {m["levels30_benchmark_clock"][0]} vs c14 {m["levels30_benchmark_clock"][1]}. '
               f'Evidence analysis/model-swap-kernels-20261013/a5b/RESULT.json + collection.json. Breakage/throughput gate only; no efficacy claim.')
    path = ROOT.parents[1] / 'analysis/hidden-draws-20261010/queue.json'
    with path.open('r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        queue = json.load(f)
        while day.isoformat() in queue:
            e = queue[day.isoformat()]
            if e.get('kernel') == r['kernel'] and e.get('version') == r['version']:
                break
            day += dt.timedelta(days=1)
        assert day < dt.date(2026, 11, 2)
        entry = {'kernel': r['kernel'], 'version': r['version'], 'message': message}
        queue[day.isoformat()] = entry
        f.seek(0); json.dump(queue, f, indent=2); f.write('\n'); f.truncate(); f.flush(); os.fsync(f.fileno())
    assert json.loads(path.read_text())[day.isoformat()] == entry
    r.update(queued=True, queued_utc_day=day.isoformat(), queue_entry=entry)
    p.write_text(json.dumps(r, indent=2) + '\n')
    print(json.dumps({'date': day.isoformat(), **entry}, indent=2))


if __name__ == '__main__':
    main()
