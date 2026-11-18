"""Replay frozen source against the downloaded speed-probe evidence; no inference."""
import ast
import collections
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RAW = ROOT / 'analysis/deployment-probes/20260905-q6-speed-probe/raw'
IPC = next((RAW / 'solver-workspaces').glob('*/ipc'))
sys.path.insert(0, str(OUT / 'frozen/chassis'))
from host_contract import ObservationRecord
from stall_contract import detect_stall


def read(name):
    return [json.loads(line) for line in (IPC / f'{name}.jsonl').read_text().splitlines()]


def main():
    calls, states, batches, transitions, attempts = [read(n) for n in ('model_calls', 'states', 'actions', 'transitions', 'policy_attempts')]
    runtime_path = OUT / 'frozen-runtime.py'
    tree = ast.parse(runtime_path.read_text())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('expected_launcher_solver_requests', 'reconcile_launcher_request_count', 'server_request_facts')]
    assignments = [n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('CHAT_COMPLETION_ACCESS_PREFIX', 'CHAT_COMPLETION_ACCESS_MARKER', 'FREEZE_GC_ACCESS_MARKER', 'SGLANG_READY_MARKER') for t in n.targets)]
    ns = {'Any': Any}
    exec(compile(ast.Module(body=assignments + funcs, type_ignores=[]), str(runtime_path), 'exec'), ns)
    launcher = json.loads((RAW / 'official-launcher-result.json').read_text())
    try:
        ns['expected_launcher_solver_requests'](launcher)
    except RuntimeError as error:
        schema_error = str(error)
    else:
        raise AssertionError('Frozen schema rejection did not reproduce')
    facts = ns['server_request_facts']((RAW / 'server.log').read_text())
    # Diagnostic projection tests the NEXT check, not a proposed production fix.
    synthetic = {'solver_sessions': {'schema': 'tycho-nooa-public-solver-sessions-v1', 'game_count': 1, 'total_actor_calls': len(calls), 'games': {'ls20': {'model_evidence': {'actor_call_count': len(calls), 'actor_calls_by_sequence': dict(collections.Counter(str(r['turn']) for r in calls))}}}}}
    try:
        ns['reconcile_launcher_request_count'](synthetic, json.loads((RAW / 'warmup-facts.json').read_text()), facts)
    except RuntimeError as error:
        count_error = str(error)
    else:
        raise AssertionError('Missing-provider-call mismatch did not reproduce')
    record = ObservationRecord()
    replay = []
    for row in transitions:
        a,b = row['before']['grid'], row['after']['grid']
        record.observe(a,b)
        def bbox(g):
            cells = [(r,c) for r,line in enumerate(g) for c,value in enumerate(line) if value == 12]
            return [min(r for r,c in cells), max(r for r,c in cells), min(c for r,c in cells), max(c for r,c in cells)]
        replay.append({'step':row['step'], 'label':record.classify(a,b), 'hud_rows':record.hud_rows, 'color12_before_bbox':bbox(a), 'color12_after_bbox':bbox(b)})
    step5 = replay[4]
    assert step5['label']=='hud_only' and step5['color12_before_bbox'] != step5['color12_after_bbox']
    detection = []
    for attempt in attempts:
        state = next(s for s in states if s['turn']==attempt['turn'])
        history = [b for b in batches if b['turn']<attempt['turn']]
        found=detect_stall(history,state,attempt['rationale'])
        assert found is None
        detection.append({'turn':state['turn'],'step':state['step'],'prior_batches':len(history),'eligible':len(history)>=6 and state['step']>=25})
    trace = []
    noops=[]
    for i,row in enumerate(calls,1):
        for tool in row['tool_calls']:
            args=json.loads(tool['arguments']) if isinstance(tool['arguments'],str) else tool['arguments']
            code=args.get('code','')
            if row['turn']==4:
                trace.append(f'CALL {i} ({row["wall_seconds"]:.3f}s)\n{code}\n')
                body=ast.parse(code).body
                if len(body)==1 and isinstance(body[0],ast.If) and "self._visual != 'off'" in code:
                    noops.append({'call':i,'wall_seconds':row['wall_seconds']})
    (OUT/'TURN4-CODE.txt').write_text('\n'.join(trace))
    grouped=collections.defaultdict(list)
    for row in calls:grouped[row['turn']].append(row['wall_seconds'])
    timing=[{'turn':t,'calls':len(v),'model_seconds':sum(v)} for t,v in grouped.items()]
    glyph_history = []
    for row in transitions:
        glyph_history.append({'step': row['step'], 'pixels': [[r,c,value] for r,line in enumerate(row['after']['grid']) for c,value in enumerate(line) if value in (0,1)]})
    assert glyph_history[6]['pixels'] == glyph_history[8]['pixels'] and glyph_history[7]['pixels'] == []
    # Execute the frozen history method against its real run directory.
    from types import SimpleNamespace
    solver_tree=ast.parse((OUT/'frozen/chassis/solver_agent.py').read_text())
    cls=next(n for n in solver_tree.body if isinstance(n,ast.ClassDef) and n.name=='MdArcSolverAgent')
    method=next(n for n in ast.walk(solver_tree) if isinstance(n,ast.FunctionDef) and n.name=='trajectory')
    method.decorator_list=[]
    history_ns={'Path':Path,'json':json}
    exec(compile(ast.Module(body=[method],type_ignores=[]),'<frozen-trajectory>','exec'),history_ns)
    trajectory=history_ns['trajectory'](SimpleNamespace(run_dir=IPC.parent))
    assert trajectory == [] and len(transitions)==10
    result={'glyph_history':glyph_history,'frozen_per_step_trajectory_entries':len(trajectory),'schema_reproduction' :schema_error,'server_request_facts':facts,'recorded_actor_calls':len(calls),'unaccounted_successful_requests_after_warmup':facts['successful_chat_completion_accesses']-1-len(calls),'next_count_check_reproduction':count_error,'timing':timing,'model_total_seconds':sum(r['wall_seconds'] for r in calls),'median_call_seconds':statistics.median(r['wall_seconds'] for r in calls),'median_batch_model_seconds':statistics.median(sum(v) for v in grouped.values()),'executed_actions':len(transitions),'submitted_actions':sum(len(b['actions']) for b in batches),'actor_batches':len(batches),'detector_eligibility':detection,'host_replay':replay,'turn4_visual_disabled_noop_calls':noops,'host_skipped_actions_at_step5':3,'caller_unexecuted_tail_at_budget':3,'source_hashes':{str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/'frozen').rglob('*.py')}}
    (OUT/'EVIDENCE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('host_replay','source_hashes','detector_eligibility')},indent=2))


if __name__=='__main__':main()
