"""Reproduce commit-time eviction that B1's request-time counter misses."""
import importlib
import json
import os
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'solvers/m2_b1/src'))
pkg=types.ModuleType('inference.agent');pkg.__path__=[str(ROOT/'solvers/m2_b1/src/inference/agent')]
sys.modules['inference.agent']=pkg
os.environ.update({'ARC3_MEMORY_SECTIONS':'off','ARC3_PIN_LEVEL_OPENING_TOKENS':'12288',
                   'ARC3_HISTORY_ASSISTANT_TURNS':'8','ARC3_HISTORY_TURN_DRAIN':'0',
                   'LOCAL_ANALYZER_TOOL_TIMEOUT':'60'})
from inference.agent import tool_agent as ta

a=ta.ToolAgent(model='local-counter-audit',base_url='http://127.0.0.1:9',api_key='local-unused')
a._context_budget_tokens=10**6
messages=[{'role':'system','content':'sys'}]
for i in range(16):
    messages += [{'role':'user','content':f'Current state: step {i+1}, level 2.\nTURN{i}'},
                 {'role':'assistant','content':f'Observation/reasoning TURN{i}'}]
messages.append({'role':'user','content':'Current state: step 17, level 2.\nNOW'})
requested=a._trim_messages_for_context(messages)
before=dict(a._b1_stats)
assert len(requested)==len(messages)
history=a._persistent_history_messages(messages)
after=dict(a._b1_stats)
assert len(history)<len(messages)-1
assert after['trims']==0
assert a._context_was_trimmed
assert 'TURN0' in json.dumps(history)
report={'input_history_messages':len(messages)-1,'retained_history_messages':len(history),
        'request_time_counter_before':before,'request_time_counter_after_commit':after,
        'context_was_trimmed':a._context_was_trimmed,'opening_retained':True,
        'source':'solvers/m2_b1/src/inference/agent/tool_agent.py',
        'conclusion':'trims=0 does not establish absence of context eviction. Commit-time assistant-turn-cap trimming bypasses _b1_count.'}
(OUT/'context-counter-reproduction.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
