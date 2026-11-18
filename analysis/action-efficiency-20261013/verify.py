"""Independent installed scorer and event-frame validation, plus certified oracle attachment."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np
from analyze import HERE, ROOT, ENV, RUNS, parse, game_score, sha
# arc_agi imports plotting dependencies; keep their cache inside the authorized directory.
os.environ['MPLCONFIGDIR']=str(HERE/'.runtime-cache'/'matplotlib')
os.environ['XDG_CACHE_HOME']=str(HERE/'.runtime-cache')
from arc_agi.scorecard import EnvironmentScoreCalculator
from replay_audit import engine, act, trial
from exact_bfs import key


def independent_score(bases, actions, done):
    c=EnvironmentScoreCalculator()
    for i,b in enumerate(bases):
        c.add_level(i+1,i<done,actions[i],b)
    return c.to_score().score


def main():
    proofs=json.loads((HERE/'exact-bfs.json').read_text())
    assert all(p['exact_shortest'] for p in proofs)
    cert={p['level']:p for p in proofs}
    rows=list(map(json.loads,(HERE/'levels.jsonl').open()))
    index={(r['run'],r['game_id'],r['level']):r for r in rows}
    games=list(map(json.loads,(HERE/'games.jsonl').open()))
    gindex={(r['run'],r['game_id']):r for r in games}
    scorer_checks, frame_checks, oracle_checks, phase_rows=[],[],[],[]
    orig=json.loads((ROOT/RUNS['control'][0]/'benchmark.json').read_text())
    rr=next(r for r in orig['game_runs'] if r['game_id'].startswith('tu93'))
    original_roots={};gh=engine(rr['game_id']);act(gh,{'id':'RESET'});k=0
    for lv in range(3):
        original_roots[lv]=key(gh)
        for h in parse(rr['history'])[k:k+parse(rr['actions_per_level'])[lv]]:
            act(gh,h['action'])
        k+=parse(rr['actions_per_level'])[lv]
    plans={(r['run'],r['game_id'],r['level']):r for r in json.loads((HERE/'plan-audit.json').read_text())}
    for name,(rel,role) in RUNS.items():
        benchmark=json.loads((ROOT/rel/'benchmark.json').read_text())
        for r in benchmark['game_runs']:
            gid=r['game_id'];done=int(r['levels_completed'] or 0)
            bases,apl,hist=map(parse,(r['base_actions_per_level'],r['actions_per_level'],r['history']))
            ep=ROOT/rel/'artifacts'/f'{gid}_p0_events.jsonl'
            events=[json.loads(l) for l in ep.open()]
            ae=[e for e in events if e['type']=='action']
            g=engine(gid);fr=act(g,{'id':'RESET'});frames_ok=0;lv=0
            certified_actions=list(apl)
            # Actual, baseline, one action, ceil half: independently evaluate every game's relevant bounds.
            for scenario,aa in [('actual',apl),('baseline',bases),('ceiling',[1]*len(bases)),
                                ('exploration_half_upper',[max(1,math.ceil(a/2)) for a in apl])]:
                value=independent_score(bases,aa,done)
                assert abs(value-game_score(bases,aa,done))<1e-9
                scorer_checks.append(dict(run=name,game_id=gid,scenario=scenario,score=value,ok=True))
            for j,h in enumerate(hist):
                if gid.startswith('tu93') and lv in cert and j==sum(apl[:lv]) and lv<done:
                    same=key(g)==original_roots[lv]
                    check=trial(g,cert[lv]['route'],lv)
                    assert same and check['solves'] and check['exact_end']
                    row=index[name,gid,lv]
                    row['global_oracle_actions']=cert[lv]['distance']
                    row['global_oracle_level_score']=min(115,100*(bases[lv]/cert[lv]['distance'])**2)
                    row['global_oracle_evidence']='analysis/action-efficiency-20261013/exact-bfs.json'
                    certified_actions[lv]=cert[lv]['distance']
                    oracle_checks.append(dict(run=name,game_id=gid,level=lv,semantic_root_equal=same,
                                              exact_shortest=cert[lv]['distance'],route_replay=check))
                fr=act(g,h['action'])
                assert np.array_equal(np.asarray(fr.frame[-1]),np.asarray(ae[j]['board'])), (name,gid,j+1)
                assert fr.levels_completed==ae[j]['score']
                frames_ok+=1
                lv=fr.levels_completed
            frame_checks.append(dict(run=name,game_id=gid,frames_checked=frames_ok,ok=True))
            gindex[name,gid]['global_certified_oracles_only_score']=independent_score(bases,certified_actions,done)
            gindex[name,gid]['exploration_halved_most_generous_score_upper']=gindex[name,gid]['all_actions_halved_score']
            for lv in range(done):
                row=index[name,gid,lv]
                row.setdefault('global_oracle_actions',None)
                row.setdefault('global_oracle_level_score',None)
                plan=plans.get((name,gid,lv))
                start=sum(apl[:lv]); seg=ae[start:start+apl[lv]]
                c={'fatal_or_reset_actions':0,'other_interior_unchanged_actions':0,
                   'before_explicit_full_plan_other_actions':0,'after_explicit_full_plan_other_actions':0,
                   'phase_unknown_other_actions':0}
                for e in seg:
                    if e.get('game_over') or e['action_name']=='RESET':
                        field='fatal_or_reset_actions'
                    else:
                        before=events[0]['board'] if e['action_num']==1 else ae[e['action_num']-2]['board']
                        unchanged=np.array_equal(np.asarray(before)[1:-1,1:-1],np.asarray(e['board'])[1:-1,1:-1])
                        if unchanged and not e.get('level_completed') and not e.get('run_complete'):
                            field='other_interior_unchanged_actions'
                        elif plan:
                            field='before_explicit_full_plan_other_actions' if e['action_num']<=plan['known_after_action'] else 'after_explicit_full_plan_other_actions'
                        else:
                            field='phase_unknown_other_actions'
                    c[field]+=1
                assert sum(c.values())==apl[lv]
                phase_rows.append(dict(run=name,game_id=gid,level=lv,actions=apl[lv],disjoint_observation_partition=c,
                                       insight_boundary_known=bool(plan),
                                       known_post_plan_removable_actions=plan['verified_post_plan_saving'] if plan else None,
                                       cognitive_waste_unidentified=True))
        print(name,'frames verified',sum(r['frames_checked'] for r in frame_checks if r['run']==name),flush=True)
    (HERE/'levels.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (HERE/'games.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in games))
    (HERE/'phase-accounting.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in phase_rows))
    verification=dict(independent_scorer='arc_agi.scorecard.EnvironmentScoreCalculator',
                      scorer_game_scenarios=len(scorer_checks),max_error=0,scorer_checks=scorer_checks,
                      games_replayed=len(frame_checks),frames_compared=sum(r['frames_checked'] for r in frame_checks),
                      frame_checks=frame_checks,global_certified_oracle_checks=oracle_checks,
                      solved_level_partition_rows=len(phase_rows),manual_insight_boundaries=len(plans))
    (HERE/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    accounting=json.loads((HERE/'accounting.json').read_text())
    for r in accounting['runs']:
        rr=[g for g in games if g['run']==r['run']]
        r['means']['exploration_halved_most_generous_score_upper']=r['means']['all_actions_halved_score']
        r['means']['global_certified_oracles_only_score']=sum(g['global_certified_oracles_only_score'] for g in rr)/len(rr)
        r['global_certified_oracle_levels']=sum(x['run']==r['run'] and x['global_oracle_actions'] is not None for x in rows)
    (HERE/'accounting.json').write_text(json.dumps(accounting,indent=2)+'\n')
    paths=[Path(__file__),HERE/'analyze.py',HERE/'replay_audit.py',HERE/'plan_audit.py',HERE/'exact_bfs.py',
           Path('/home/pranav/projects/arc-agi-3-reference/dfranzen-m2/tufa-arc-agi-framework/src/taaf/game.py')]
    for gid in {r['game_id'] for r in rows}:
        paths += [ENV/gid[:4]/gid[5:]/'metadata.json',ENV/gid[:4]/gid[5:]/(gid[:4]+'.py')]
    (HERE/'source-manifest.json').write_text(json.dumps([dict(path=str(p),sha256=sha(p)) for p in paths],indent=2)+'\n')


if __name__=='__main__':
    main()
