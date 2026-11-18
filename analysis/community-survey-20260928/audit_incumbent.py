"""Read-only KF evidence audit. No model, game, network, or pickle execution."""
import hashlib
import json
from pathlib import Path
import re
import runpy
import statistics

ROOT = Path(__file__).resolve().parent
RUN = Path('/home/pranav/projects/arc-agi-3-reference/duck-qwen38/runs/flash-next')
OUT = RUN / 'out-kf2'
SOURCE = RUN / 'kf/source-dataset/src/ARC3-Inference/inference'


def main():
    mapping_path = SOURCE / 'agent/action_names.py'
    mapping = runpy.run_path(str(mapping_path))
    probe = {
        'source': str(mapping_path),
        'sha256': hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
        'probe': {name: {
            'advertised': mapping['to_model_action'](name),
            'executable': mapping['to_engine_action'](name),
        } for name in ['ACTION1', 'UP', 'ACTION6', 'MOUSE', 'ACTION7', 'UNDO', 'RESET']},
        'scope': 'Calls only the local pure action mapping functions; no engine, model, or gameplay.',
    }
    (ROOT / 'raw/action-mapping-probe.json').write_text(json.dumps(probe, indent=2) + '\n')
    benchmark = json.loads((OUT / 'benchmark.json').read_text())
    games = benchmark['game_runs']
    env = json.loads((OUT / 'taaf_setup_env.json').read_text())
    identity = json.loads((OUT / 'vllm-server-identity.json').read_text())
    snapshots = []
    for path in sorted((OUT / 'prompts').glob('*.log')):
        text = path.read_text()
        model_input = text.split('[MODEL INPUT]', 1)[1].split('[TURN TRANSCRIPT SO FAR]', 1)[0]
        users = re.split(r'\n\[USER\]\n', model_input)[1:]
        last = users[-1].split('\n[', 1)[0] if users else ''
        marker = 'Working world model carried from earlier turns:'
        snapshots.append({
            'game': path.stem,
            'retained_user_messages': len(users),
            'last_user_has_world_model': marker in last,
            'retained_history_has_world_model': marker in model_input,
            'retained_reasoning_blocks': model_input.count('[REASONING]'),
        })
    files = [OUT / name for name in ['benchmark.json', 'score.json', 'taaf_setup_env.json', 'vllm-server-identity.json']]
    files += [SOURCE / name for name in ['agent/tool_agent.py', 'agent/vision_context.py', 'agent/action_names.py', 'framework/solver.py']]
    files += list((RUN / 'kf2/kernel').glob('*.ipynb'))
    report = {
        'identity': 'pranavagrawai/duck-flash-next-kf2 v1; local outputs completed 2026-09-27',
        'games': len(games),
        'levels': sum(game['levels_completed'] for game in games),
        'mean_public_score': statistics.mean(game['final_score'] for game in games),
        'games_with_a_level': sum(game['levels_completed'] > 0 for game in games),
        'zero_level_games': [game['game_id'] for game in games if not game['levels_completed']],
        'configuration': {key: env.get(key) for key in ['LOCAL_ANALYZER_CONTEXT_WINDOW', 'LOCAL_ANALYZER_YIELD_SECONDS', 'MULTIMODAL_CONTEXT', 'MULTIMODAL_UPSCALE', 'DUCK_PROMPT_GAMEOVER_RETRY', 'DUCK_PROMPT_LEVELUP_REVIEW']},
        'serving': identity['vllm_tuning'],
        'final_prompt_snapshot_count': len(snapshots),
        'last_user_with_notes': sum(row['last_user_has_world_model'] for row in snapshots),
        'retained_history_with_notes': sum(row['retained_history_has_world_model'] for row in snapshots),
        'retained_user_messages_min_median_max': [min(row['retained_user_messages'] for row in snapshots), statistics.median(row['retained_user_messages'] for row in snapshots), max(row['retained_user_messages'] for row in snapshots)],
        'limitations': ['Final model-call snapshots only; not all requests.', 'Presence of notes is not proof that their contents are correct or used.', 'Local source inspection does not prove a change improves gameplay.'],
        'files': [{'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in files],
    }
    (ROOT / 'raw/kf-prompt-audit.json').write_text(json.dumps(snapshots, indent=2) + '\n')
    (ROOT / 'raw/incumbent-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ['games', 'levels', 'mean_public_score', 'last_user_with_notes', 'retained_user_messages_min_median_max']}, indent=2))


if __name__ == '__main__':
    main()
