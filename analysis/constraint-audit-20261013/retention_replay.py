"""Exercise the shipped retention method with the shipped timeout return shape."""
import ast
import hashlib
import json
from pathlib import Path
import types

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
rel='analysis/m2-winners-20261001/audit/dfranzen-repo/ARC3-Inference/inference/agent/tool_agent.py'
source=(ROOT/rel).read_text()
tree=ast.parse(source)
cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and any(isinstance(y,ast.FunctionDef) and y.name=='_record_retained_functions' for y in x.body))
method=next(y for y in cls.body if isinstance(y,ast.FunctionDef) and y.name=='_record_retained_functions')
ns={'_persistent_functions':lambda:True,'_persistent_functions_scope':lambda:'game',
    '_get_env_bool':lambda *args:False,'function_signature':lambda s:next(x.name for x in ast.parse(s).body if isinstance(x,ast.FunctionDef))+'()'}
exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),rel,'exec'),ns)
helper='def read_board():\n    return 1\n'
obj=types.SimpleNamespace(_kept_functions={'read_board':helper})
payload={}
ns['_record_retained_functions'](obj,{'error':'Tool timed out after 30s','stdout':'','action_results':[]},payload)
result={'method_path':rel,'method_line':method.lineno,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'dependencies_stubbed':['persistent feature/scope env reads','function signature formatter'],
        'input_existing_helpers':['read_board'],'input_timeout_result_has_retention_metadata':False,
        'output_helpers':list(obj._kept_functions),'output_notice':payload,
        'limitation':'CPU replay of audited local source, not a Kaggle execution or exact installed-byte receipt.'}
assert result['output_helpers']==[]
patched_rel='solvers/m2_b1/src/inference/agent/tool_agent.py'
patched_source=(ROOT/patched_rel).read_text()
patched_tree=ast.parse(patched_source)
patched_cls=next(x for x in patched_tree.body if isinstance(x,ast.ClassDef) and any(isinstance(y,ast.FunctionDef) and y.name=='_record_retained_functions' for y in x.body))
patched_method=next(y for y in patched_cls.body if isinstance(y,ast.FunctionDef) and y.name=='_record_retained_functions')
exec(compile(ast.fix_missing_locations(ast.Module(body=[patched_method],type_ignores=[])),patched_rel,'exec'),ns)
obj=types.SimpleNamespace(_kept_functions={'read_board':helper});payload={}
ns['_record_retained_functions'](obj,{'error':'Tool timed out after 60s','stdout':'','action_results':[]},payload)
assert obj._kept_functions=={'read_board':helper}
result['existing_b1_fix']=dict(path=patched_rel,line=patched_method.lineno,
    source_sha256=hashlib.sha256(patched_source.encode()).hexdigest(),after_missing_snapshot=list(obj._kept_functions),notice=payload)
ns['_record_retained_functions'](obj,{'keepable_functions':[],'stdout':'','action_results':[]},{})
assert obj._kept_functions=={}
result['existing_b1_fix']['after_explicit_completed_empty_snapshot']=list(obj._kept_functions)
(HERE/'retention_replay.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
