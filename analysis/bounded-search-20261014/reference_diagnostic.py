"""Post-screen ceiling diagnostic; no reference rules/routes enter the solver."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def definitions(path,ns,names):
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)


def main():
    assert json.loads((HERE/'RESULT.json').read_text())['validity']=='VALID'
    screen=module('screen_adapters',HERE/'run_screen.py')
    native=module('native_helpers',ROOT/'analysis/planner-qualification-v2-20261013/run_probe.py')
    helpers,_=native.native_context()
    search=module('shipped_search',ROOT/'solvers/m2_search/src/inference/utils/bounded_search.py').bounded_search
    records_path=ROOT/'analysis/s5i5-inspection-20261012/records.json'
    source=ROOT/'analysis/score-conversion-20261013/run_probe.py'
    ns={'ast':ast,'copy':copy,'re':re,'COLORS':'OSYN'}
    definitions(source,ns,{'extract','functions','reference'})
    ns['records']=json.loads(records_path.read_text())
    board=ns['records']['control-pc']['entry_board']
    good={'OBS':{(r,c) for r,row in enumerate(board) for c,color in enumerate(row)
                 if color==15 or (r>=52 and color!=5)}}
    ns['functions']('control-pc',4862,good);ns['functions']('control-pc',5635,good)
    ns['good']=good
    cases=json.loads((HERE/'cases.json').read_text());rows=[]
    for case in cases:
        start=tuple(case['start'][k]['L'] for k in 'OSYN')
        def successors(state):
            for i,key in enumerate('OSYN'):
                for delta in (3,-3):
                    next_state=list(state);next_state[i]+=delta;next_state=tuple(next_state)
                    if good['ok3'](next_state):
                        yield ('g' if delta>0 else 's',key),next_state
        def heuristic(state):
            r,c=good['marker'](state)
            return (abs(r-31)+abs(c-52)+2)//3
        result=search(start,successors,state_key=lambda s:s,is_goal=lambda s:good['marker'](s)==(31,52),
                      heuristic=heuristic,max_depth=60,max_nodes=100000,max_expanded=100000,max_seconds=5)
        assert result['status']=='FOUND',(case['episode'],result['status'])
        # Reference validates the route; evaluate frozen generated predictions
        # along the same native successors, continuing diagnostically after errors.
        game,frame=screen.replay_prefix(case,helpers)
        _,predict,_,_,_=screen.model(case)
        click={'Og':12,'Os':6,'Sg':27,'Ss':21,'Yg':42,'Ys':36,'Ng':57,'Ns':51}
        transitions=[]
        for action in result['path']:
            before=helpers['state_from_board'](frame.frame[-1])
            expected_reference=ns['reference'](before,action[1],3 if action[0]=='g' else -3)
            predicted=predict(before,action)
            frame=helpers['perform'](game,{'id':'ACTION6','data':{'x':click[action[1]+action[0]],'y':57}})
            actual=helpers['state_from_board'](game.inspection_wins[-1] if frame.levels_completed>1 else frame.frame[-1])
            assert actual==expected_reference
            transitions.append({'step':len(transitions)+1,'action':action,'before':before,
                                'model_predicted':predicted,'actual':actual,'matched':predicted==actual})
        assert frame.levels_completed==2
        mismatches=[r for r in transitions if not r['matched']]
        rows.append({'episode':case['episode'],'reference_route_length':len(transitions),
                     'reference_native_solved':True,'frozen_model_matches':len(transitions)-len(mismatches),
                     'first_model_mismatch':mismatches[0] if mismatches else None,'transitions':transitions})
    report={'scope':'Post-screen assisted ceiling/diagnosis; no reference rules supplied to solver or search-screen arms',
            'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,records_path,HERE/'cases.json')},
            'rows':rows}
    (HERE/'reference-diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    for r in rows:
        print(r['episode'],r['reference_route_length'],r['frozen_model_matches'],
              'first mismatch',r['first_model_mismatch']['action'] if r['first_model_mismatch'] else None)


if __name__=='__main__':
    main()
