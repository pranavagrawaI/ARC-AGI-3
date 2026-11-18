"""Actual adapter, transport, host, dispatcher and CodeAct; deterministic HTTP model."""
import importlib.util
import json
from pathlib import Path
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from test_contingent_controller import relevant_proposal

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]


@pytest.mark.parametrize('controller,changed', [(False,1),(True,1),(True,2),(True,3)])
def test_actual_contingent_worker(monkeypatch,tmp_path,controller,changed):
    unexpected = changed == 3
    sys.path[:0]=[str(ROOT/'deployment/kaggle/tests'),str(ROOT/'deployment/kaggle')]
    import test_my_agent, test_nooa_join, test_join_qualification as qual
    spec=importlib.util.spec_from_file_location('runtime.local_solver',HERE/'shared_local_solver.py')
    runtime=importlib.util.module_from_spec(spec);monkeypatch.setitem(sys.modules,'runtime.local_solver',runtime);spec.loader.exec_module(runtime)
    monkeypatch.setattr(test_my_agent,'AGENT_PATH',HERE/'adapter.py')
    monkeypatch.setattr(test_nooa_join,'CANDIDATE',HERE)
    seen=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_):pass
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append(body)
            if len(seen)>2:
                self.send_error(500, 'fixture call cap exceeded'); return
            with (tmp_path/'requests.jsonl').open('a') as f:f.write(json.dumps(body)+'\n')
            if len(seen)==1:
                code=('self.submit_contingent_plan('+repr(relevant_proposal())+')' if controller else
                      'self.submit_actions(["UP"], rationale="predict: fixture probe")')
            else:
                code='self.submit_actions(["'+('DOWN' if unexpected else ('RIGHT' if changed == 2 else 'LEFT'))+'"], rationale="predict: fixture continuation")'
            reply={'id':f'fixture-{len(seen)}','object':'chat.completion','created':1,'model':'fixture',
                'choices':[{'index':0,'finish_reason':'tool_calls','message':{'role':'assistant','content':None,
                'tool_calls':[{'id':f'c{len(seen)}','type':'function','function':{
                    'name':'execute_python','arguments':json.dumps({'code':code})}}]}}],
                'usage':{'prompt_tokens':100,'completion_tokens':50,'total_tokens':150}}
            with (tmp_path/'responses.jsonl').open('a') as f:f.write(json.dumps(reply)+'\n')
            raw=json.dumps(reply).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    server=HTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    settings={'ARC_DISABLE_DOTENV':'1','ARC_LLM_MODEL':'qwen/qwen3.8-27b','ARC_LLM_API_KEY':'fixture',
        'ARC_LLM_BASE_URL':f'http://127.0.0.1:{server.server_port}/v1',
        'ARC_LLM_EXTRA_BODY':'{"reasoning":{"enabled":false}}','ARC_LLM_MAX_TOKENS':'1024',
        'ARC_SOLVER_ARM':'loop','ARC_PROCEDURAL_DIAGNOSTIC':'0','ARC_DISCOVERY_ARM':'a',
        'NOOA_JOIN_ROOT':str(HERE),'NOOA_AGENT_COMMAND':json.dumps([sys.executable,str(HERE/'headless.py'),'--run-dir','.']),
        'TYCHO_NOOA_WORKER_TIMEOUT_SECONDS':'30','LITELLM_LOCAL_MODEL_COST_MAP':'True',
        'TYCHO_KAGGLE_MAX_ACTIONS':'2','ARC_STALL_REVIEW':'0'}
    for k,v in settings.items():monkeypatch.setenv(k,v)
    frame=lambda grid: test_my_agent.frame(grid, actions=tuple(getattr(test_my_agent.GameAction, f'ACTION{i}') for i in range(1,5)))
    first=frame([[0,0],[0,0]])
    agent=qual.build_agent(monkeypatch,tmp_path,first)
    try:
        action=agent.choose_action(agent.frames,first)
        assert action.name=='ACTION1'
        second=frame([[1,1],[1,0]] if unexpected else ([[1,1],[0,0]] if changed == 2 else [[1,0],[0,0]]))
        agent.action_counter += 1
        agent.frames.append(second)
        action2=agent.choose_action(agent.frames,second)
        assert action2.name==('ACTION2' if unexpected else ('ACTION4' if changed == 2 else 'ACTION3'))
        agent.action_counter += 1
        agent.frames.append(frame([[0,1],[0,0]]))
    finally:
        try: agent.cleanup(scorecard=None)
        finally: server.shutdown();thread.join(timeout=2)
    expected=1 if controller and not unexpected else 2
    assert len(seen)==expected
    actions=[json.loads(l) for p in tmp_path.rglob('actions.jsonl') for l in p.read_text().splitlines()]
    assert len(actions)==2
    assert not list(tmp_path.rglob('ABORT'))
    events=[json.loads(l) for p in tmp_path.rglob('contingent_events.jsonl') for l in p.read_text().splitlines()]
    if controller:
        assert any(e['status']==('stopped' if unexpected else 'continuation_submitted') for e in events)
    (tmp_path/'RESULT.json').write_text(json.dumps({'controller':controller,'unexpected':unexpected,
        'model_calls':len(seen),'actions':actions,'events':events,'claim':'protocol feasibility; scripted model, no Qwen quality evidence'},indent=2)+'\n')
