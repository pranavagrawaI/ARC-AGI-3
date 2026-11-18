"""Resume policy: preserve the launched card verdict and enforce the user's plan."""
import ast
import copy
import importlib.util
import math
import statistics
from pathlib import Path

FLASH = Path('/home/pranav/projects/arc-agi-3-reference/duck-qwen38/runs/flash-next')
BENCH = FLASH / 'bench'


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


original = module(BENCH / 'assess_screen.py', 'frozen_screen_assessor')


def exact_greedy(configs, names):
    """Require all four fixed prompts, both attempts, successful and identical."""
    rows = []
    for name in names:
        greedy = configs.get(name, {}).get('greedy', [])
        mapped = {(r['game'], r['attempt']): r for r in greedy}
        games = {g for g, a in mapped}
        if len(greedy) != 8 or len(games) != 4 or set(mapped) != {(g, a) for g in games for a in (0, 1)}:
            return False
        if any(r.get('status') != 200 or r.get('text') is None or not r.get('prompt_tokens') for r in mapped.values()):
            return False
        rows.append({g: (mapped[g, 0]['text'], mapped[g, 0]['prompt_tokens']) for g in games})
        if any((mapped[g, 1]['text'], mapped[g, 1]['prompt_tokens']) != rows[-1][g] for g in games):
            return False
    return bool(rows) and all(r == rows[0] for r in rows)


def assess_screen(result):
    card = original.assess(result)
    strict = copy.deepcopy(result)
    failures = {}
    for name, ev in strict['configs'].items():
        fails = []
        log = ev.get('log') or {}
        lw, mem = log.get('prune_load_weights', []), log.get('model_gib', [])
        try:
            valid_prune = bool(lw) and all(b == '48' and k == '448' and
                all(ast.literal_eval(t).get(key) == val for key, val in
                    {'dropped': 36864, 'renumbered': 258048, 'routers': 48}.items()) for t, b, k in lw)
        except (ValueError, SyntaxError, TypeError):
            valid_prune = False
        if not valid_prune or not mem or not all(math.isfinite(m) and m < 76 for m in mem):
            fails.append('Missing qualified pruning/memory evidence')
        if ev.get('overrides', {}).get('TAAF_VLLM_ENABLE_PREFIX_CACHING') == '1':
            if log.get('mamba_cache_mode') != ['align']:
                # Multiple identical worker messages are harmless.
                if not log.get('mamba_cache_mode') or set(log['mamba_cache_mode']) != {'align'}:
                    fails.append('align mode not confirmed in worker log')
            names = ['S1', 'S2'] + (['S3'] if name == 'S3' else [])
            if not exact_greedy(strict['configs'], names):
                fails.append('Exact greedy equality missing; the launched noise allowance does not satisfy NEXT-PLAN')
        wins = ev.get('load', {}).get('windows', [])
        if len(wins) != 5 or any(not all(isinstance(w.get(k), (int, float)) and
                math.isfinite(w[k]) and w[k] > 0 for k in ('gen_tok_s', 'requests_s')) for w in wins):
            fails.append('Missing five finite positive throughput windows')
        if fails:
            ev['ok'] = False
            ev['error'] = '; '.join(fails)
            failures[name] = fails
    return {'launched_card': card, 'user_plan': original.assess(strict), 'additional_failures': failures,
            'note': 'Launched source/card preserved. Exact equality is the original user requirement, not a post-result relaxation.'}


def validation_decision(arms):
    issues = [f'{n}: {issue}' for n, a in arms.items() for issue in a['issues']]
    if set(arms) != {'P+1', 'P+2'}:
        issues.append('Two complete passes required')
    elif not issues and set(arms['P+1']['per_game']) != set(arms['P+2']['per_game']):
        issues.append('Validation game sets differ')
    if issues:
        return {'status': 'INVALID', 'candidate': 'P', 'issues': issues}
    levels = statistics.mean(a['levels'] for a in arms.values())
    throughput = {n: a['serving']['busy']['gen_tok_s'] for n, a in arms.items()}
    mechanism = all(v is not None and math.isfinite(v) and v >= 422.4 for v in throughput.values())
    passed = levels >= 48.5 and mechanism
    return {'status': 'PASS' if passed else 'FAIL', 'candidate': 'P+' if passed else 'P',
            'mean_levels': levels, 'level_floor': 48.5, 'busy_gen_tok_s': throughput,
            'throughput_floor': 422.4, 'mechanism_pass': mechanism,
            'interpretation': 'Registered retention rule against historical P, not a statistical noninferiority proof.'}
