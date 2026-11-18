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


def admission_budget(timeout_s, game_count, request_timeout_s):
    """Rolling fair share; reserve final output, per-game cleanup and one call.

    Inputs are remaining wall time and remaining games, NOT original totals.
    The 15s tool margin is a policy reserve, not a measured latency bound.
    """
    import math
    if (not all(math.isfinite(x) for x in (timeout_s, game_count, request_timeout_s))
            or game_count <= 0 or int(game_count) != game_count or request_timeout_s <= 0):
        raise ValueError('Finite positive game count and request timeout required')
    allowance = (timeout_s - 180) / game_count - 30
    budget = allowance - request_timeout_s - 15
    if budget < 0:
        raise ValueError('Insufficient runtime for games, one request each, and cleanup reserves')
    return budget


def run(args):
    rerun = os.environ.get('KAGGLE_IS_COMPETITION_RERUN', '').lower() in ('1', 'true')
    framework = Path(os.environ.get('SUBMISSION_FRAMEWORK', '/kaggle/working/ARC-AGI-3-Agents'))
    sys.path.insert(0, str(framework))
    configure(args.base_url, args.model, args.workspace_root)
    from agents.templates.my_agent import MyAgent
    from runtime.local_solver import GameBudgetExhausted
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
    environment_ids = [e.game_id for e in arcade.get_environments()]
    available = {game.split('-')[0]: game for game in environment_ids}
    if not available:
        raise RuntimeError('No environments were supplied')
    games = environment_ids if rerun else [available[g] for g in os.environ.get(
        'SUBMISSION_PUBLIC_GAMES', 'ls20,s5i5,cd82').split(',')]
    started = time.monotonic()
    global_deadline = started + args.timeout_s
    request_timeout = float(os.environ['ARC_LLM_TIMEOUT_SECONDS'])
    # Retries would invalidate the one-request completion reserve.
    if int(os.environ.get('ARC_LLM_RETRIES', '0')) != 0:
        raise ValueError('Budgeted execution requires ARC_LLM_RETRIES=0')
    admission_budget(args.timeout_s, len(games), request_timeout)
    action_limit = int(os.environ.get('SUBMISSION_ACTION_LIMIT', '200' if rerun else '10'))
    if action_limit <= 0:
        raise ValueError('Submission action limit must be positive')
    os.environ['TYCHO_KAGGLE_MAX_ACTIONS'] = str(action_limit)
    scorecard = arcade.open_scorecard(tags=['epistemic-questioner-20260912'])
    progress = []
    print(json.dumps(dict(game_count=len(games), global_deadline=global_deadline)), flush=True)
    try:
        for index, game in enumerate(games):
            game_started = time.monotonic()
            remaining = global_deadline - game_started
            game_budget = admission_budget(remaining, len(games)-index, request_timeout)
            cap = os.environ.get('SUBMISSION_ADMISSION_CAP_SECONDS')
            if cap is not None:
                import math
                cap = float(cap)
                if not math.isfinite(cap) or cap < 0:
                    raise ValueError('Admission cap must be finite and nonnegative')
                if os.environ.get('SUBMISSION_REQUIRE_FULL_CAP') == '1' and game_budget < cap:
                    raise RuntimeError('Remaining notebook budget cannot fund the precommitted admission window')
                game_budget = min(game_budget, cap)
            admission_deadline = game_started + game_budget
            os.environ['ARC_GAME_MODEL_ADMISSION_DEADLINE'] = str(admission_deadline)
            # Host wall stop is secondary; the absolute model boundary includes
            # environment creation, worker startup, tools and game transitions.
            os.environ['TYCHO_GAMEPLAY_ADMISSION_SECONDS'] = str(max(game_budget, 0.001))
            worker_timeout = game_budget + request_timeout + 15
            os.environ['ARC_GAME_WORK_DEADLINE'] = str(game_started + worker_timeout)
            os.environ['TYCHO_NOOA_WORKER_TIMEOUT_SECONDS'] = str(worker_timeout)
            allocation = dict(game_id=game, remaining_games=len(games)-index,
                remaining_seconds=remaining, admission_seconds=game_budget,
                admission_deadline=admission_deadline, request_reserve_seconds=request_timeout,
                tool_margin_seconds=15, cleanup_reserve_seconds=30,
                worker_watchdog_seconds=worker_timeout)
            with (args.output.parent/'budget-allocations.jsonl').open('a') as output:
                output.write(json.dumps(allocation)+'\n')
            env = arcade.make(game, scorecard_id=scorecard)
            if env is None:
                raise RuntimeError(f'Cannot create supplied environment {game}')
            agent = MyAgent(card_id=scorecard, game_id=game, agent_name='myagent',
                            ROOT_URL='http://gateway:8001' if rerun else 'http://127.0.0.1',
                            record=True, arc_env=env, tags=['epistemic-questioner-20260912'])
            failure = None
            try:
                agent.main()
            except GameBudgetExhausted:
                agent._game_stop_reason = 'model_admission_budget'
            except BaseException as error:
                failure = error
                raise
            finally:
                try:
                    agent.cleanup()
                except BaseException as cleanup_error:
                    if failure is None:
                        raise
                    failure.add_note(f'Additional cleanup failure: {cleanup_error!r}')
            item = dict(game_id=game, actions=agent.action_counter,
                        solver_actions=agent._solver_decisions_emitted,
                        levels_completed=agent.frames[-1].levels_completed,
                        state=agent.frames[-1].state.value,
                        stop_reason=agent._game_stop_reason,
                        game_elapsed_seconds=time.monotonic()-game_started,
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
    for item in progress:
        if item['solver_actions'] <= 0 and item['stop_reason'] != 'model_admission_budget':
            raise RuntimeError('Zero-action game without acknowledged model budget stop')
    if time.monotonic() > global_deadline:
        raise RuntimeError('Global gameplay deadline exceeded')
    result = dict(schema='first-submission-play-v1', status='passed',
                  qualification_scope='operational completion; budget exhaustion is not solving success',
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
