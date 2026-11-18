"""Execute exact pure functions from the shipped KF source, without model/game calls."""
import ast
import importlib.util
import json
from pathlib import Path
import re
import types

from audit import DUCK, HERE, sections, codes

ROOT = DUCK/'runs/flash-next/kf/source-dataset/src/ARC3-Inference/inference'


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in functions} == set(names)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *functions], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)


def main():
    path = ROOT/'agent/action_names.py'
    spec = importlib.util.spec_from_file_location('kf_action_names', path)
    mapping = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mapping)
    namespace = {'json': json, 'to_engine_action': mapping.to_engine_action}
    load_functions(ROOT/'framework/solver.py', ['_normalize_actions'], namespace)
    # ACTION7 fails before any engine access. This probes the actual normalizer, not a substitute.
    normalized = namespace['_normalize_actions'](None, {'actions': [{'action':'ACTION7'}]})
    load_functions(ROOT/'agent/tool_agent.py', ['_normalize_summary_text', '_extract_labeled_blocks', '_extract_scientist_note', '_estimate_tokens'], namespace)
    sample_notes = {s: namespace['_extract_scientist_note'](s)['world_model'] for s in ['World model: measured fact', 'World model (revised): measured fact', 'World model v2: measured fact']}
    result = {'scope': 'Exact extracted pure functions, no imported runtime, model, or game engine.',
              'action_roundtrip': {x: [mapping.to_model_action(x), mapping.to_engine_action(mapping.to_model_action(x))] for x in [f'ACTION{i}' for i in range(1,8)]},
              'action7_normalizer_result': normalized, 'note_examples': sample_notes,
              'per_game': {}}
    for path in sorted((DUCK/'runs/flash-next/out-kf2/artifacts').glob('*events.jsonl')):
        events = list(map(json.loads, path.open()))
        counts = {'action7_advertised': False, 'python_calls_containing_action7':0,
                  'executed_action7':0, 'assistant_world_model_blocks':0, 'world_model_blocks_lost_by_parser':0}
        for e in events:
            if e['type']=='action':
                counts['executed_action7'] += e.get('action_name')=='ACTION7'
            if e['type']!='analysis': continue
            text=e['transcript']
            counts['action7_advertised'] |= bool(re.search(r'^Valid actions right now:.*ACTION7',text,re.M))
            for label,value in sections(text):
                if label=='MODEL RESPONSE META':
                    counts['python_calls_containing_action7'] += sum('ACTION7' in c for c in codes(value))
            for m in re.finditer(r'^\[ASSISTANT\]\n(.*?)(?=^\[(?:TOOL CALL: python|MODEL RESPONSE META|THINKING|ANALYZER STATUS)\]|\Z)',text,re.M|re.S):
                content=m[1]
                if re.search(r'^\s*World model[^\n:]*:',content,re.M|re.I):
                    counts['assistant_world_model_blocks']+=1
                    counts['world_model_blocks_lost_by_parser']+=not bool(namespace['_extract_scientist_note'](content)['world_model'])
        result['per_game'][path.name[:4]]=counts
    result['totals']={k:sum(int(v[k]) for v in result['per_game'].values()) for k in next(iter(result['per_game'].values()))}
    (HERE/'contract-probes.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='per_game'},indent=2))


if __name__=='__main__':
    main()
