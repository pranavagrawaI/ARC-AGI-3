"""Check that individually shortened level routes also compose from each game's initial state."""
import json
from analyze import HERE,ROOT,RUNS,parse,game_score
from replay_audit import engine,act


def main():
    deletes={(r['run'],r['game_id'],r['level']):r for r in map(json.loads,(HERE/'deletion-checks.jsonl').open())}
    modes=('remove_reset_prefix','remove_interior_unchanged','remove_repeated_noops')
    records=[];summaries=[]
    for name,(rel,role) in RUNS.items():
        run=[]
        for r in json.loads((ROOT/rel/'benchmark.json').read_text())['game_runs']:
            gid=r['game_id'];done=int(r['levels_completed'] or 0)
            hist,apl,bases=map(parse,(r['history'],r['actions_per_level'],r['base_actions_per_level']))
            for mode in modes:
                g=engine(gid);act(g,{'id':'RESET'});k=0;newapl=list(apl);failure=None
                for lv in range(done):
                    original=[h['action'] for h in hist[k:k+apl[lv]]]
                    check=deletes[name,gid,lv]['trials'][mode]
                    seq=(check['kept_actions'][:check['completion_action']]
                         if check['solves'] and check['kept_actions'] else original)
                    k+=apl[lv]
                    for j,a in enumerate(seq):
                        fr=act(g,a)
                        if fr.levels_completed>lv:
                            if fr.levels_completed!=lv+1 or j+1!=len(seq):
                                failure=dict(level=lv,reason='unexpected completion boundary',action=j+1)
                            break
                    if fr.levels_completed!=lv+1:
                        failure=dict(level=lv,reason='route did not complete from composed entry',state=fr.state.name)
                    if failure:
                        break
                    newapl[lv]=len(seq)
                score=game_score(bases,newapl,done) if not failure else r['final_score']
                rec=dict(run=name,game_id=gid,mode=mode,composed_routes_solve=failure is None,
                         failure=failure,counterfactual_score_or_original_on_failure=score,
                         removed_actions=sum(apl[:done])-sum(newapl[:done]) if not failure else 0,
                         original_score=r['final_score'])
                run.append(rec);records.append(rec)
        summary=dict(run=name,modes={mode:dict(
            games_validated=sum(r['composed_routes_solve'] for r in run if r['mode']==mode),
            removed_actions=sum(r['removed_actions'] for r in run if r['mode']==mode),
            mean_score=sum(r['counterfactual_score_or_original_on_failure'] for r in run if r['mode']==mode)/25,
            direct_mean_gain=sum(r['counterfactual_score_or_original_on_failure']-r['original_score'] for r in run if r['mode']==mode)/25)
            for mode in modes})
        summaries.append(summary);print(json.dumps(summary),flush=True)
    (HERE/'chain-checks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (HERE/'chain-summary.json').write_text(json.dumps(summaries,indent=2)+'\n')


if __name__=='__main__':
    main()
