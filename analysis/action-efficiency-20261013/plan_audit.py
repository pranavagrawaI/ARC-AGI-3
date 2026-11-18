"""Manually anchored, replay-checked model-plan examples; not an all-level insight detector."""
import copy
import json
from replay_audit import engine, act, trial
from analyze import HERE, ROOT, RUNS, parse, game_score

CASES = [
    dict(run='control', game='vc33', level=0, known_after_action=13, analysis_step=6,
         reason='Model identifies one B click aligning yellow bar and wall gap; next action completes.',
         quote='If I press B once more', replacement=None, optimal_suffix_certified=True),
    dict(run='control', game='ka59', level=0, known_after_action=69, analysis_step=20,
         reason='Model corrects goal to both rings occupied and batches select + LEFT five + DOWN.',
         quote='fill the left ring with A as well', replacement=None, optimal_suffix_certified=False),
    dict(run='control', game='sb26', level=7, known_after_action=142, analysis_step=26,
         reason='Model corrects permutation cycles; six swaps plus SPACE directly execute the chosen arrangement.',
         quote='Total 6 swaps = 12 clicks', replacement=None, optimal_suffix_certified=False),
    dict(run='control-r2', game='tr87', level=4, known_after_action=222, analysis_step=17,
         reason='Model corrects tile-versus-glyph indexing; suffix cycles UP 5,4,6 on seven-cycle tiles. Inverse DOWN uses 2,3,1.',
         quote='Now the mechanics are clear',
         replacement=['ACTION2']*2+['ACTION4']+['ACTION2']*3+['ACTION3']*2+['ACTION2'],
         optimal_suffix_certified=False),
    dict(run='c14', game='cd82', level=4, known_after_action=134, analysis_step=64,
         reason='Corrected inclusive SW region makes own backward painting search return a six-stamp plan.',
         quote='corrected base solvable', replacement=None, optimal_suffix_certified=False),
]


def main():
    result=[]
    for case in CASES:
        rel=RUNS[case['run']][0]
        r=next(r for r in json.loads((ROOT/rel/'benchmark.json').read_text())['game_runs'] if r['game_id'].startswith(case['game']))
        hist,apl,bases=map(parse,(r['history'],r['actions_per_level'],r['base_actions_per_level']))
        lv=case['level']; start=sum(apl[:lv]); end=start+apl[lv]; anchor=case['known_after_action']
        g=engine(r['game_id']);act(g,{'id':'RESET'})
        for h in hist[:anchor]:
            fr=act(g,h['action'])
        assert fr.levels_completed==lv
        original=[h['action'] for h in hist[anchor:end]]
        replacement=[{'id':a} for a in case['replacement']] if case['replacement'] else original
        check=trial(g,replacement,lv)
        assert check['solves'] and check['exact_end'], case
        actual=trial(g,original,lv)
        assert actual['solves'] and actual['exact_end']
        transformed=list(apl);transformed[lv]=anchor-start+len(replacement)
        exploration_half=list(apl);exploration_half[lv]=(anchor-start+1)//2+len(original)
        rec=dict(case, game_id=r['game_id'], benchmark=rel+'/benchmark.json',
                 transcript=rel+'/transcripts/'+r['game_id']+'_p0.txt',
                 pre_full_plan_actions=anchor-start, original_execution_actions=len(original),
                 verified_execution_actions=len(replacement), verified_post_plan_saving=len(original)-len(replacement),
                 original_route=original, verified_route=replacement, replay=check,
                 direct_public_mean_gain_this_case=(game_score(bases,transformed,r['levels_completed'])-r['final_score'])/25,
                 exploration_half_this_level_public_mean_gain=(game_score(bases,exploration_half,r['levels_completed'])-r['final_score'])/25,
                 interpretation='Prefix is before this explicit full-plan witness, not proven earliest rule insight; other exploration/knowledge boundaries unknown')
        result.append(rec)
        print(json.dumps({k:v for k,v in rec.items() if k not in ['original_route','verified_route','replacement']}))
    (HERE/'plan-audit.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
