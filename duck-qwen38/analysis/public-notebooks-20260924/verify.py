"""Recompute source comparisons and public-run totals without executing competitors' code."""
import hashlib
import json
from pathlib import Path
p = Path(__file__).resolve().parent
names = ['keithtyser', 'wuliao0', 'chiakazirim']
code = {}
for name in names:
    notebook = json.loads(next((p / name).glob('*.ipynb')).read_text())
    code[name] = '\n\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
    print(name, 'code_sha256', hashlib.sha256(code[name].encode()).hexdigest())
print('Keith and Wuliao code identical:', code['keithtyser'] == code['wuliao0'])
prompt = Path('src/ARC3-Inference/inference/agent/prompts.py')
print('Keith base prompt identical to local:', (p/'keith-bundle'/prompt).read_bytes() == (p.parent.parent/'ARC3-Inference/inference/agent/prompts.py').read_bytes())
b = json.loads((p/'keith-output/benchmark.json').read_text())
g = b['game_runs']
print(json.dumps({'games':len(g), 'passes':b['n_passes'], 'levels':sum(x['levels_completed'] for x in g), 'actions':sum(len(x['history']) for x in g), 'mean_final_score':sum(x['final_score'] for x in g)/len(g)}, indent=2))
