"""Exercise the fixed donor runner in its own interpreter: lordhansolo or sirikilohit."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT.parents[1]/'analysis/m2-winners-20261001/audit'
name=sys.argv[1]
source=AUDIT/('lord-bundle/src/ARC3-Inference' if name=='lordhansolo' else 'dfranzen-repo/ARC3-Inference')
sys.path.insert(0,str(source))
sys.path.insert(0,str(AUDIT/'bug-audit-20261001'))
package=ModuleType('inference.agent');package.__path__=[str(source/'inference/agent')]
sys.modules['inference.agent']=package
spec=importlib.util.spec_from_file_location('fixed_sandbox',ROOT/'donor_fixes'/f'{name}_sandbox.py')
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
calls=[]
def handler(actions):
    calls.append(actions)
    return {'action_result':{'executed':True,'executed_actions':['UP']},'state':{}}
result=module.run_sandboxed_python(code='r=action("UP");r["executed_actions"].clear();r.clear()',
    timeout_seconds=3,initial_state={},action_handler=handler)
assert len(calls)==1 and result['action_results']==[{'executed':True,'executed_actions':['UP']}],result
print(name,'receipt mutation regression passed')
