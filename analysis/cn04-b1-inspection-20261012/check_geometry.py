"""Reproduce the failed run's pure geometry code on its recorded input board.

Only the inspected get_pieces/local functions, variant builder, and finite
four-piece search are executed. Tool APIs are replaced by local data stubs.
"""
from pathlib import Path
import ast
import hashlib
import json
import re
from types import SimpleNamespace

OUT = Path(__file__).resolve().parent
records = json.loads((OUT/'records.json').read_text())
rec = records['b1-r1']
board = rec['actions'][2]['board_ascii']  # after selecting yellow; before any connection


def code_at(line):
    f = next(f for b in rec['blocks'] for f in b['fragments'] if f['source_line']==line)
    assert f['type'].startswith('TOOL CALL:')
    return re.search(r'<parameter=code>\s*(.*?)\s*</parameter>', f['text'], re.S)[1]


store = {}
g = {'grid_from_frame':lambda f:[list(row) for row in f.ascii.splitlines()],
     'bbox':lambda cells:(min(r for r,c in cells),min(c for r,c in cells),max(r for r,c in cells),max(c for r,c in cells)),
     'load_data':lambda key:{'value':store[key]},
     'save_data':lambda key,value:store.__setitem__(key,value),
     'current_frame':SimpleNamespace(ascii=board)}
tree = ast.parse(code_at(3833))
functions = ast.Module(body=[node for node in tree.body if isinstance(node,ast.FunctionDef)], type_ignores=[])
exec(compile(functions, 'recorded-get-pieces', 'exec'),g)
local = g['local'](g['get_pieces'](g['current_frame']))
local['Y'] = local.pop('W')
store['L0'] = {k:{'body':[list(x) for x in sorted(v['body'])],
                  'ends':[[list(x) for x in sorted(e)] for e in v['ends']],
                  'pos':list(v['pos'])} for k,v in local.items()}

# Reproduce the actual corrected-for-rotation builder and the search that follows it.
exec(compile(code_at(4784),'recorded-variants','exec'),g)
bad_variants = store['VAR']
exec(compile(code_at(5013),'recorded-search','exec'),g)
bad = {'solutions':len(g['sols']), 'relative_joint_pairs':len(g['rel'])}


def common_frame_variants(piece):
    b = set(map(tuple,piece['body']))
    ends = [set(map(tuple,e)) for e in piece['ends']]
    variants = []
    for k in range(4):
        br,bc = min(r for r,c in b), min(c for r,c in b)
        variants.append([[list(x) for x in sorted((r-br,c-bc) for r,c in b)],
                         [[list(x) for x in sorted((r-br,c-bc) for r,c in e)] for e in ends]])
        b = {(c,-r) for r,c in b}
        ends = [{(c,-r) for r,c in e} for e in ends]
    return variants


good_variants = {k:common_frame_variants(v) for k,v in store['L0'].items()}
store['VAR'] = good_variants
exec(compile(code_at(5013),'same-search-corrected-frames','exec'),g)
corrected_frames = {'solutions':len(g['sols']), 'relative_joint_pairs':len(g['rel']),
                    'note':'The unmodified search still rejects multi-connector joins and remains incomplete.'}
assert bad['solutions']==0 and bad['relative_joint_pairs']==0
assert corrected_frames['relative_joint_pairs']>0
# The successful run joins both purple/white tips to yellow simultaneously.
# Reconcile that observed placement with the model's pair admissibility rule.
p, y = store['L0']['p'], store['L0']['Y']
shift = (-6,-6)  # final p origin (29,11) minus fixed Y origin (35,17)
pb = {(r+shift[0],c+shift[1]) for r,c in p['body']}
yb = set(map(tuple,y['body']))
pe = [{(r+shift[0],c+shift[1]) for r,c in e} for e in p['ends']]
ye = [set(map(tuple,e)) for e in y['ends']]
overlaps = [(i,j) for i,e in enumerate(pe) for j,f in enumerate(ye) if e&f]
assert overlaps == [(0,0),(1,2)]
assert not pb&yb and not any(e&yb for e in pe) and not any(e&pb for e in ye)
double_join = {'observed_in_successful_run_at_local_action':8,
               'body_overlap_pixels':len(pb&yb),'connector_pairs':overlaps,
               'original_exactly_one_pair_filter_accepts':len(overlaps)==1,
               'note':'The recorded search rejects a valid observed two-connector join even after coordinate frames are corrected.'}
rows = []
for name,piece in store['L0'].items():
    original = [list(min(map(tuple,e))) for e in piece['ends']]
    collapsed = [list(min(map(tuple,e))) for e in bad_variants[name][0][1]]
    recovered = [list(min(map(tuple,e))) for e in good_variants[name][0][1]]
    assert original == recovered
    assert all(x==[0,0] for x in collapsed)
    rows.append({'piece':name,'true_connector_offsets':original,
                 'failed_builder_offsets':collapsed,'corrected_common_frame_offsets':recovered})
audit = {'recorded_input_after_local_action':3, 'input_board_sha256':hashlib.sha256(board.encode()).hexdigest(),
         'source_lines':{'piece_extraction':3833,'variant_builder':4784,'search':5013},
         'original_search':bad,'same_search_corrected_coordinate_frames':corrected_frames,
         'valid_double_join_check':double_join,
         'connector_offsets':rows,'input_pieces':store['L0'],
         'original_variants':bad_variants,'corrected_variants':good_variants,
         'scope':'Reproduces a defect in this generated planner. Does not establish its correction would improve gameplay.'}
(OUT/'geometry-check.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps({'original':bad,'coordinate_fix_only':corrected_frames,'connector_offsets':rows},indent=2))
