"""Real ToolAgent and sandbox wiring; supports verification of installed source."""
import json
import os
from pathlib import Path
import sys
import tempfile
import types

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(os.environ['M2_SEARCH_SOURCE'])
sys.path.insert(0,str(SOURCE))
pkg=types.ModuleType('inference.agent');pkg.__path__=[str(SOURCE/'inference/agent')]
sys.modules['inference.agent']=pkg
os.environ.update({'ARC3_MEMORY_SECTIONS':'off','LOCAL_ANALYZER_TOOL_TIMEOUT':'60','ARC3_PERSISTENT_FUNCTIONS':'1'})
from inference.agent import tool_agent as ta
from inference.agent.runtime_state import Frame,write_runtime_state

with tempfile.TemporaryDirectory() as directory:
    state=Path(directory)/'search_p0_runtime_state.json'
    write_runtime_state(state,current_frame=Frame(grid=((0,1),(1,0)),step=5,level=1),history=[])
    a=ta.ToolAgent(model='search-gate',base_url='http://127.0.0.1:9',api_key='unused')
    a._ensure_session(state);a._current_valid_actions=['UP']
    assert 'bounded_search(initial_state' in a._system_prompt
    assert 'HORIZON_EXHAUSTED' in a._system_prompt and 'CALLBACK_ERROR' in a._system_prompt
    code='''
def successors(s):
    if s < 4:
        yield {'action':'advance'}, s+1
def key(s): return s
def goal(s): return s == 4
def plan(s):
    return bounded_search(s, successors, state_key=key, is_goal=goal, max_depth=6)
r=plan(0)
assert r['status']=='FOUND' and len(r['path'])==4
save_data('search_plan',r)
print('SEARCH_WIRED_OK')
'''
    first=a._run_python_tool(state,{'code':code})
    assert 'SEARCH_WIRED_OK' in json.dumps(first.__dict__,default=str),first
    second=a._run_python_tool(state,{'code':"assert plan(2)['depth']==2\nassert load_data('search_plan')['value']['depth']==4\nprint('SEARCH_RESTORED_OK')"})
    assert 'SEARCH_RESTORED_OK' in json.dumps(second.__dict__,default=str),second
    assert 'plan' in a._kept_functions
    print('Search ToolAgent gate PASS: advertised API, planning, saved result, retained callbacks')
