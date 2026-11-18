"""Verify offline donor-model plans on both stalled native-engine states."""
from pathlib import Path
import json
import replay

OUT=Path(__file__).resolve().parent
R=json.loads((OUT/'records.json').read_text())
M=json.loads((OUT/'model-check.json').read_text())
COLS={'Og':12,'Os':6,'Sg':27,'Ss':21,'Yg':42,'Ys':36,'Ng':57,'Ns':51}
checks=[]
for label,plan in M['offline_plans'].items():
    p=replay.ROOT/Path(R[label]['source']).parent.parent
    b=next(r for r in json.loads((p/'benchmark.json').read_text())['game_runs'] if r['game_id']=='s5i5-18d95033')
    prefix_count=b['actions_per_level'][0]+plan['after_local_action']
    game=replay.make()
    for h in b['history'][:prefix_count]:fr=replay.perform(game,h['action'])
    available=game.gwiuiwqizb.current_steps
    result=[]
    for token in plan['tokens']:
        fr=replay.perform(game,{'id':'ACTION6','data':{'x':COLS[token],'y':57}})
        board=game.inspection_wins[-1] if fr.levels_completed>1 else fr.frame[-1]
        result.append({'token':token,'levels_completed':fr.levels_completed,'state':fr.state.name,
                       'tip_row_col':replay.tip(board)})
        if fr.levels_completed>1:break
    assert fr.levels_completed==2,(label,result[-1])
    assert len(result)==len(plan['tokens'])
    checks.append({'run':label,'after_local_action':plan['after_local_action'],
                   'available_engine_steps':available,'plan_clicks':len(plan['tokens']),
                   'solves':True,'result':result,
                   'scope':'Offline donor-model route; does not demonstrate the failed model would repair itself.'})
    print(label,'available engine steps',available,'plan',len(result),'solves',True)
(OUT/'plan-verification.json').write_text(json.dumps(checks,indent=2)+'\n')
