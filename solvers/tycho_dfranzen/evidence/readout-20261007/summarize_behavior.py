"""Recompute behavioral counts from hash-checked downloaded traces."""
import json
import hashlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
report = {}
for arm in ('lean-cap4', 'lean-prune'):
    root = ROOT / arm / 'tycho-dfranzen'
    index = {r['path']: r['sha256'] for r in json.loads((root/'evidence-index.json').read_text())['files']}
    games = []
    for game in sorted((root/'hybrid-games').iterdir()):
        llm = rows(game/'llm-trace.jsonl')
        if not llm:
            continue
        for name in ('llm-trace.jsonl','tool-trace.jsonl','host-actions.jsonl','compute.json','context-projection.jsonl'):
            file = game/name
            if file.exists():
                assert hashlib.sha256(file.read_bytes()).hexdigest() == index[str(file.relative_to(ROOT/arm))]
        tools = rows(game/'tool-trace.jsonl')
        host = rows(game/'host-actions.jsonl')
        projections = rows(game/'context-projection.jsonl')
        sequences = Counter(r['sequence'] for r in llm)
        regular = [r for r in llm if r.get('reasoning_tokens', 0)]
        demands = [r for r in llm if not r.get('reasoning_tokens', 0)]
        total = {k:sum(r.get(k,0) or 0 for r in llm) for k in ('completion_tokens','reasoning_tokens','prompt_tokens','cached_tokens')}
        last_sequence = llm[-1]['sequence']
        tail = [r for r in llm if r['sequence']==last_sequence]
        games.append(dict(game=game.name[:4], calls=len(llm), roles=dict(Counter(r['source'] for r in llm)),
            total=total, mean_prompt=total['prompt_tokens']/len(llm),
            reasoning_fraction=total['reasoning_tokens']/total['completion_tokens'],
            cache_fraction=total['cached_tokens']/total['prompt_tokens'],
            calls_with_actions=sum('take_actions' in r['tool_calls'] for r in llm),
            zero_reasoning_calls=len(demands), max_length_calls=sum(r['completion_tokens']>=12288 for r in llm),
            tool_counts=dict(Counter(t['name'] for t in tools)),
            tool_failures=[{k:t.get(k) for k in ('call_index','sequence','name','execution_status','error_stage','result')} for t in tools if t['execution_status']!='complete'],
            longest_decisions=sequences.most_common(5),
            last_decision=dict(sequence=last_sequence,calls=len(tail),completion_tokens=sum(r['completion_tokens'] for r in tail),first_call=tail[0]['call_index'],last_call=tail[-1]['call_index']),
            projection_windows=sorted({r['window'] for r in projections}),
            engine_actions=len(host),level_completions=[dict(sequence=r['sequence'],level=r['after']['levels_completed']) for r in host if r['after']['levels_completed']>r['before']['levels_completed']],
            deaths=sum(r['after']['state']=='GAME_OVER' and r['before']['state']!='GAME_OVER' for r in host)))
    totals = {k:sum(g['total'][k] for g in games) for k in ('completion_tokens','reasoning_tokens','prompt_tokens','cached_tokens')}
    calls = sum(g['calls'] for g in games)
    report[arm] = dict(games=games,coverage=len(games),calls=calls,total=totals,
        mean_prompt=totals['prompt_tokens']/calls if calls else None,
        reasoning_fraction=totals['reasoning_tokens']/totals['completion_tokens'] if calls else None,
        calls_with_actions=sum(g['calls_with_actions'] for g in games))
(ROOT/'behavior-summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({a:{k:v for k,v in r.items() if k!='games'} for a,r in report.items()},indent=2))
