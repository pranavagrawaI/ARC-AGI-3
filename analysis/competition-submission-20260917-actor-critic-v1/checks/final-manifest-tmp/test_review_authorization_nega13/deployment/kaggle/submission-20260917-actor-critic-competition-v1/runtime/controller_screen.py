"""Closed finite-world diagnostic. Supplied hypotheses, no neural generalization claim."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent/'chassis'))
from contingent_controller import compile_plan, select_branch
from test_contingent_controller import relevant_proposal, candidate, outcome


class TinyWorld:
    """Independent primitive executor: a wrong route costs four recovery steps."""
    def __init__(self, hidden, case):
        self.hidden, self.case = hidden, case
        self.phase='root'; self.actions=[]; self.done=False; self.wait=0

    def step(self, action):
        assert not self.done
        self.actions.append(action)
        if self.phase=='root':
            self.root=action; self.phase='route'
            if self.case=='all_wrong':return 99
            if self.case=='wrong_goal':return 0
            return self.hidden+1 if action in ('UP','DOWN') else 3
        if self.phase=='route':
            correct=action==('LEFT' if self.hidden==0 or self.case=='same_route' else 'RIGHT')
            self.wait=0 if correct else 4
            if correct and self.case in ('progress','same_route') and self.root=='UP':self.wait+=1
            self.phase='recovery';self.done=self.wait==0
        else:
            assert action=='SPACEBAR'
            self.wait-=1;self.done=self.wait==0
        return -1


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    records=[]
    for case in ('relevant','same_route','progress','all_wrong','wrong_goal'):
        p=relevant_proposal()
        p['candidates'][0]['predictions']={'a':3,'b':3}
        if case=='same_route':
            for c in p['candidates']:
                c['continuations']=[dict(action='LEFT',outcomes={h:outcome(2 if c['id']=='probe' else 1) for h in p['hypotheses']})]
        if case=='progress':
            p['candidates'][1]['continuations'][0]['outcomes']['a']=outcome(2)
            p['candidates'][1]['continuations'][1]['outcomes']['b']=outcome(2)
            p['candidates'].append(candidate('progress','DOWN',{'a':1,'b':2},
                {'LEFT':{'a':1,'b':5},'RIGHT':{'a':5,'b':1}}))
        if case=='wrong_goal':
            p['candidates']=p['candidates'][1:]
            p['candidates'][0]['metric']='levels_completed_delta'
            p['candidates'][0]['predictions']={'a':1,'b':1}
        plan=compile_plan(p)
        (output/f'{case}-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
        for hidden in (0,1):
            for policy in ('fixed_commit','controller'):
                world=TinyWorld(hidden,case)
                observed=world.step(plan['selected']['action'] if policy=='controller' else 'RIGHT')
                branch=select_branch(plan,observed) if policy=='controller' else {'action':'LEFT'}
                if branch:
                    world.step(branch['action'])
                    while not world.done and len(world.actions)<8:world.step('SPACEBAR')
                records.append(dict(case=case,hidden=hidden,policy=policy,observation=observed,
                    actions=world.actions,completed=world.done,stopped_unknown=branch is None))
    (output/'trajectories.json').write_text(json.dumps(records,indent=2)+'\n')
    summary={}
    for case in ('relevant','same_route','progress','all_wrong','wrong_goal'):
        summary[case]={}
        for policy in ('fixed_commit','controller'):
            rs=[r for r in records if r['case']==case and r['policy']==policy]
            summary[case][policy]={'completed':sum(r['completed'] for r in rs),
                'mean_actions':sum(len(r['actions']) for r in rs)/len(rs),
                'unknown_stops':sum(r['stopped_unknown'] for r in rs)}
    assert summary['relevant']['controller']==dict(completed=2,mean_actions=2,unknown_stops=0)
    assert summary['relevant']['fixed_commit']['mean_actions']==4
    assert summary['same_route']['controller']['mean_actions']==2
    assert summary['progress']['controller']['mean_actions']==2
    assert summary['all_wrong']['controller']['unknown_stops']==2
    assert summary['wrong_goal']['controller']['completed']==0
    summary['scope']='Five constructed finite cases with supplied hypotheses; action counts are not RHAE. Fixed commit is not the full Qwen baseline.'
    (output/'RESULTS.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':run(sys.argv[1])
