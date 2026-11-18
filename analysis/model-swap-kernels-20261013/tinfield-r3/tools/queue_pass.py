"""Queue only a terminal, exact-version, all-gates-passing smoke. Never submits."""
import datetime as dt
import fcntl
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
def main():
    p=ROOT/'tinfield-r3/RESULT.json';r=json.loads(p.read_text())
    assert r['smoke_pass'] is True and all(r['gates'].values())
    assert r['collection']['status']=='COMPLETE' and r['collection']['version']==r['version']
    now=dt.datetime.now(dt.timezone.utc);day=now.date()
    if now.hour>=20:day+=dt.timedelta(days=1)
    stats=r['tinfield']['serve_5_30'];lv=r['tinfield']['levels_over_time']['30']
    message=f'Tinfield NVFP4 c14 smoke COMPLETE v{r["version"]}: READY, finite token logprobs, 0/25 crashed games, {lv} levels at game minute 30; MTP acceptance {stats["accept_len_median"]:.2f}, matched-window {stats["tw_gen_tok_s"]:.1f} tok/s. thinking/action {r["tinfield"]["thinking_0_40"]["thinking_tokens_per_action"]:.0f}. Evidence analysis/model-swap-kernels-20261013/tinfield-r3/RESULT.json and exact-version collection.json. Breakage gate only; no selection/efficacy claim.'
    path=ROOT.parents[1]/'analysis/hidden-draws-20261010/queue.json'
    with path.open('r+') as f:
        fcntl.flock(f,fcntl.LOCK_EX);queue=json.load(f)
        while day.isoformat() in queue:
            if queue[day.isoformat()].get('kernel')==r['kernel'] and queue[day.isoformat()].get('version')==r['version']:break
            day+=dt.timedelta(days=1)
        assert day<dt.date(2026,11,2)
        entry={'kernel':r['kernel'],'version':r['version'],'message':message};queue[day.isoformat()]=entry
        f.seek(0);json.dump(queue,f,indent=2);f.write('\n');f.truncate();f.flush();os.fsync(f.fileno())
    verify=json.loads(path.read_text());assert verify[day.isoformat()]==entry
    r.update(queued=True,queued_utc_day=day.isoformat(),queue_entry=entry)
    p.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'date':day.isoformat(),**entry},indent=2))

if __name__=='__main__':main()
