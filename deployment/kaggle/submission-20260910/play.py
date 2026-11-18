"""Sequential official Agent loop, shared by public qualification and gateway rerun."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback


def configure(base_url, model, workspace):
    os.environ.update({
        'ARC_DISABLE_DOTENV': '1', 'ARC_LLM_API_KEY': 'local-model-no-key',
        'ARC_LLM_BASE_URL': base_url, 'ARC_LLM_MODEL': model,
        'ARC_LLM_PROVIDER_ENDPOINT': 'local', 'ARC_LLM_USE_RESPONSES': '0',
        'LITELLM_LOCAL_MODEL_COST_MAP': 'True', 'MPLBACKEND': 'agg',
        'PYTHONUNBUFFERED': '1', 'TYCHO_KAGGLE_WORKSPACE_ROOT': str(workspace),
        'TYCHO_NOOA_WORKER_PYTHON': sys.executable, 'TYCHO_SANDBOX_RUNTIME': 'host',
    })
    os.environ.pop('TYCHO_KAGGLE_HARDENED_HOST', None)


def run(args):
    rerun = os.environ.get('KAGGLE_IS_COMPETITION_RERUN', '').lower() in ('1', 'true')
    framework = Path(os.environ.get('SUBMISSION_FRAMEWORK', '/kaggle/working/ARC-AGI-3-Agents'))
    sys.path.insert(0, str(framework))
    configure(args.base_url, args.model, args.workspace_root)
    from agents.templates.my_agent import MyAgent
    from arc_agi import Arcade, OperationMode
    recording_root = args.output.parent / 'recordings'
    os.environ['RECORDINGS_DIR'] = str(recording_root)
    if rerun:
        os.environ['OPERATION_MODE'] = 'online'
        gateway_url = os.environ.get('SUBMISSION_GATEWAY_BASE_URL', 'http://gateway:8001')
        arcade = Arcade(arc_api_key='test-key-123', arc_base_url=gateway_url,
                        operation_mode=OperationMode.ONLINE, recordings_dir=str(recording_root))
    else:
        os.environ['OPERATION_MODE'] = 'offline'
        arcade = Arcade(operation_mode=OperationMode.OFFLINE,
                        environments_dir=os.environ['SUBMISSION_ENVIRONMENTS'],
                        recordings_dir=str(recording_root))
    available = {e.game_id.split('-')[0]: e.game_id for e in arcade.get_environments()}
    if not available:
        raise RuntimeError('No environments were supplied')
    games = list(available.values()) if rerun else [available[g] for g in os.environ.get(
        'SUBMISSION_PUBLIC_GAMES', 'ls20,s5i5,cd82').split(',')]
    started = time.monotonic()
    # Reserve a full worker decision and cleanup after the final admission.
    remaining = args.timeout_s - 180
    if remaining <= 0:
        raise ValueError('play timeout must exceed cleanup reserve')
    game_budget = min(900 if rerun else 1200, remaining / len(games) - 330)
    if game_budget <= 0:
        raise ValueError('Insufficient runtime for the supplied games and worker cleanup reserves')
    os.environ['TYCHO_GAMEPLAY_ADMISSION_SECONDS'] = str(game_budget)
    action_limit = int(os.environ.get('SUBMISSION_ACTION_LIMIT', '200' if rerun else '10'))
    if action_limit <= 0:
        raise ValueError('Submission action limit must be positive')
    os.environ['TYCHO_KAGGLE_MAX_ACTIONS'] = str(action_limit)
    scorecard = arcade.open_scorecard(tags=['first-submission-20260910'])
    progress = []
    try:
        for game in games:
            env = arcade.make(game, scorecard_id=scorecard)
            if env is None:
                raise RuntimeError(f'Cannot create supplied environment {game}')
            agent = MyAgent(card_id=scorecard, game_id=game, agent_name='myagent',
                            ROOT_URL='http://gateway:8001' if rerun else 'http://127.0.0.1',
                            record=True, arc_env=env, tags=['first-submission-20260910'])
            try:
                agent.main()
            finally:
                agent.cleanup()
            item = dict(game_id=game, actions=agent.action_counter,
                        solver_actions=agent._solver_decisions_emitted,
                        levels_completed=agent.frames[-1].levels_completed,
                        state=agent.frames[-1].state.value,
                        elapsed_seconds=time.monotonic()-started)
            progress.append(item)
            with (args.output.parent / 'game-progress.jsonl').open('a') as output:
                output.write(json.dumps(item)+'\n')
            print(json.dumps(item), flush=True)
    finally:
        arcade.close_scorecard(scorecard)
    # Reuse the qualified identity and exact model-call accounting checks.
    path = Path(os.environ['NOOA_JOIN_ROOT']) / 'nooa_public_probe.py'
    os.environ.pop('TYCHO_PUBLIC_GAME_IDS', None)
    spec = importlib.util.spec_from_file_location('submission_accounting', path)
    accounting = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(accounting)
    accounting.EXPECTED_GAMES = games
    sessions = accounting.summarize_nooa_sessions(args.workspace_root, require_complete=True)
    if any(p['solver_actions'] <= 0 for p in progress):
        raise RuntimeError('A game finished without a Solver action')
    result = dict(schema='first-submission-play-v1', status='passed',
                  operation_mode='gateway' if rerun else 'offline',
                  elapsed_seconds=time.monotonic()-started, games=progress,
                  solver_sessions=sessions)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--workspace-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout-s', type=int, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except BaseException:
        args.output.with_suffix('.failure.json').write_text(json.dumps(
            {'traceback': traceback.format_exc()}, indent=2)+'\n')
        raise


if __name__ == '__main__':
    main()
