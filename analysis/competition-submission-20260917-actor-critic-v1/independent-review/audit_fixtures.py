"""Independent programmatic audit of T3's three cp312 gateway fixture runs."""
import json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'checks/local-deps-cp312'))
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
R=ROOT/'analysis/competition-submission-20260917-actor-critic-v1'
P=ROOT/'deployment/kaggle/submission-20260917-actor-critic-competition-v1'
assets=json.loads((P/'ASSETS.json').read_text())
report={}
for mode,(games,actor,critic) in {'110':(110,550,110),'reserve':(2,2,2),'critic-cap':(2,24,8)}.items():
    base=R/'checks'/f'gateway-{mode}-cp312';q=base/'notebook/working/qualification';o={}
    res=json.loads((base/'RESULT.json').read_text());o['RESULT']= {k:res[k] for k in ('status','games','actor','critic')}
    # embedded asset bytes materialized by the notebook == ASSETS.json
    o['materialized_assets_match']=all(hashlib.sha256((base/'notebook'/n).read_bytes()).hexdigest()==d for n,d in assets.items())
    # model requests as seen by the scripted model server
    reqs=[json.loads(l) for l in (base/'model-requests.jsonl').read_text().splitlines()]
    o['server_requests']=dict(actor=sum(r['purpose']=='actor' for r in reqs),critic=sum(r['purpose']=='critic' for r in reqs),total=len(reqs))
    o['critic_requests_have_no_tools']=all(not r['request'].get('tools') for r in reqs if r['purpose']=='critic')
    o['critic_max_tokens']=sorted({r['request'].get('max_tokens') for r in reqs if r['purpose']=='critic'})
    # game progress
    prog=[json.loads(l) for l in (q/'game-progress.jsonl').read_text().splitlines()]
    o['progress_games']=len(prog);o['unique_game_ids']=len({g['game_id'] for g in prog})
    o['stop_reasons']=sorted({str(g['stop_reason']) for g in prog});o['states']=sorted({g['state'] for g in prog});o['levels']=sorted({g['levels_completed'] for g in prog})
    o['max_game_elapsed_s']=round(max(g['game_elapsed_seconds'] for g in prog),2);o['total_elapsed_s']=round(prog[-1]['elapsed_seconds'],2)
    # launcher result
    lr=json.loads((q/'launcher-result.json').read_text());o['launcher_status']=lr['status'];o['launcher_games']=len(lr['games']);o['operation_mode']=lr['operation_mode']
    # per-game shared ledgers
    ledgers=[[json.loads(l) for l in (q/f'request-budget-{i}.jsonl').read_text().splitlines()] for i in range(len(prog))]
    la=[sum(r['request_purpose']=='actor' for r in L) for L in ledgers];lc=[sum(r['request_purpose']=='critic' for r in L) for L in ledgers]
    o['ledger_actor']=dict(total=sum(la),max=max(la),min=min(la));o['ledger_critic']=dict(total=sum(lc),max=max(lc),min=min(lc))
    o['ledger_only_actor_or_critic']=all({r['request_purpose'] for r in L}<= {'actor','critic'} for L in ledgers)
    # worker ipc model_calls / lifecycle / budget stop evidence
    calls=[json.loads(l) for f in q.rglob('model_calls.jsonl') for l in f.read_text().splitlines()]
    o['ipc_model_calls']=dict(actor=sum((c.get('request_purpose') or 'actor')=='actor' for c in calls),critic=sum(c.get('request_purpose')=='critic' for c in calls),summarizer=sum(c.get('request_purpose')=='summarizer' for c in calls))
    stops=[json.loads(f.read_text()) for f in q.rglob('budget_exhausted.json')]
    o['budget_stops']=len(stops);o['budget_stop_cap_exhausted']=sum(s.get('request_cap_exhausted') is True for s in stops)
    o['budget_stop_actor_calls']=sorted({s.get('actor_calls') for s in stops});o['budget_stop_critic_calls']=sorted({s.get('critic_calls') for s in stops})
    life=[json.loads(l) for f in q.rglob('request_lifecycle.jsonl') for l in f.read_text().splitlines()]
    adm=[e for e in life if e['event']=='admitted'];done=[e for e in life if e['event']=='completed'];fail=[e for e in life if e['event']=='failed']
    o['lifecycle']=dict(admitted=len(adm),completed=len(done),failed=len(fail),admitted_ids_equal_completed=({e['request_id'] for e in adm}=={e['request_id'] for e in done}))
    o['critic_triggers']=sorted({e.get('review_trigger') for e in adm if e.get('request_purpose')=='critic'})
    # worker metadata / sessions
    metas=list(q.rglob('worker-metadata.json'));o['worker_sessions']=len(metas)
    # parquet
    frame=pd.read_parquet(base/'notebook/working/submission.parquet')
    o['parquet']=dict(rows=len(frame),columns=list(frame.columns),dtypes={c:str(t) for c,t in frame.dtypes.items()},unique_games=frame.game_id.nunique(),nulls=int(frame.isna().sum().sum()),dup_row_ids=int(frame.row_id.duplicated().sum()),end_of_game_all_true=bool(frame.end_of_game.all()),score_values=sorted(frame.score.unique().tolist()))
    o['parquet_games_equal_progress']=set(frame.game_id.astype(str))=={g['game_id'] for g in prog}
    # scorecard
    sc=json.loads((base/'gateway-scorecard.json').read_text());o['scorecard_envs']=len(sc['environments']);o['scorecard_keys']=sorted(sc.keys())[:12]
    # import origins inside the materialized framework; dependency origins in site-packages
    io=json.loads((q/'import-origins.json').read_text());o['import_origins_in_materialized_framework']=all(Path(p).is_relative_to(base/'notebook/working/ARC-AGI-3-Agents') for p in io.values())
    do=json.loads((base/'dependency-origins.json').read_text());o['dependency_origins']=do['origins'];o['verified_solver_sources']=do['verified_solver_sources'];o['python']=do['python'].split()[0]
    o['supervisor_facts']=res['supervisor']
    # notebook stderr/stdout: any Traceback?
    o['stderr_has_traceback']='Traceback' in (base/'stderr.log').read_text()
    report[mode]=o
print(json.dumps(report,indent=1,default=str))
