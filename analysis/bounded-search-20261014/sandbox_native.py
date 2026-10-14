"""Real sandbox planning and native action execution; analyst fixture, no LLM."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=ROOT/'solvers/m2_search/src'
sys.path.insert(0,str(SOURCE))
pkg=types.ModuleType('inference.agent');pkg.__path__=[str(SOURCE/'inference/agent')]
sys.modules['inference.agent']=pkg
from inference.agent import python_tool_sandbox as sandbox


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def main():
    screen=module('sandbox_fixture_adapters',HERE/'run_screen.py')
    prior=module('sandbox_native_helpers',ROOT/'analysis/planner-qualification-v2-20261013/run_probe.py')
    helpers,_=prior.native_context()
    case=next(c for c in json.loads((HERE/'cases.json').read_text()) if c['episode']=='b1-2903-reminder')
    game,frame=screen.replay_prefix(case,helpers)
    _,predict,_,_,_=screen.model(case)
    receipts=[]
    tokens={12:('g','O'),6:('s','O'),27:('g','S'),21:('s','S'),
            42:('g','Y'),36:('s','Y'),57:('g','N'),51:('s','N')}
    def state_payload(fr):
        return {'current_frame':{'grid':fr.frame[-1],'step':len(case['actions'])+len(receipts),
                                'level':fr.levels_completed+1},'history':[],'valid_actions':['MOUSE']}
    def handler(actions,**kwargs):
        nonlocal frame
        assert len(actions)==1 and actions[0]['action']=='MOUSE'
        action=actions[0]
        assert action['row']==57
        token=tokens[action['col']]
        before=helpers['state_from_board'](frame.frame[-1])
        expected=predict(before,token)
        frame=helpers['perform'](game,{'id':'ACTION6','data':{'x':action['col'],'y':action['row']}})
        board=game.inspection_wins[-1] if frame.levels_completed>1 else frame.frame[-1]
        actual=helpers['state_from_board'](board)
        assert expected==actual,(token,expected,actual)
        receipt={'executed':True,'executed_count':1,'executed_actions':['MOUSE'],
                 'action_num':len(case['actions'])+len(receipts)+1,'gameplay_changed':True,
                 'level_completed':frame.levels_completed>1,'before':before,'predicted':expected,'actual':actual}
        receipts.append(receipt)
        return {'action_result':receipt,'state':state_payload(frame)}
    retained=case['functions']
    names=['gdir_of','cellsS','growth_region','static_set','keyS','dotS','adj','removed_cells','sim_step']
    code='\n'.join(retained[n] for n in names)+'\n'
    code+='''
def successors(s):
    for op in ('g','s'):
        for k in 'YONS':
            yield (op,k),sim_step(s,op,k,static_set())
def goal(s): return dotS(s)==(31,52)
def heuristic(s):
    import math
    r,c=dotS(s)
    return math.ceil((abs(r-31)+abs(c-52))/3)
'''
    code+='initial='+repr(case['start'])+'\n'
    code+='''
plan=bounded_search(initial,successors,state_key=keyS,is_goal=goal,heuristic=heuristic,
                    max_depth=60,max_nodes=100000,max_expanded=100000,max_seconds=20)
assert plan['status']=='FOUND'
print('SANDBOX_SEARCH',plan['depth'],plan['expanded'])
buttons={'Og':12,'Os':6,'Sg':27,'Ss':21,'Yg':42,'Ys':36,'Ng':57,'Ns':51}
for op,k in plan['path']:
    action({'action':'MOUSE','row':57,'col':buttons[k+op]})
print('NATIVE_ACTIONS_COMPLETE')
'''
    (HERE/'sandbox-fixture.py').write_text(code)
    result=sandbox.run_sandboxed_python(code=code,timeout_seconds=30,initial_state=state_payload(frame),action_handler=handler)
    assert not result.get('error'),result
    assert frame.levels_completed==2 and len(receipts)==29
    assert len(result['action_results'])==29
    report={'scope':'Analyst-authored callback adapter through actual sandbox helper/action IPC and native engine; no model uptake',
            'native_completed':True,'actual_actions':len(receipts),'result':result,'receipts':receipts}
    (HERE/'sandbox-native.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Native sandbox integration PASS:29 real action receipts, all successors match, L2 clears')


if __name__=='__main__':main()
