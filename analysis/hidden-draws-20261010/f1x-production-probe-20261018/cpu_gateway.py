"""Exercise the existing 110-game simulator and real HTTP client, without a model."""
import ast
import datetime as dt
import json
import os
from pathlib import Path
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FRAMEWORK = ROOT / 'analysis/m2-winners-20261001/audit/dfranzen-repo/tufa-arc-agi-framework/src'
sys.path.insert(0, str(FRAMEWORK))
os.environ['ARC_API_KEY'] = 'test-key-123'
from arcengine import ActionInput, GameAction
from taaf.competition_arcade import CompetitionArcadeServer
from taaf.game import RunSession
from taaf.game_api import GameAPI


def main():
    started = time.monotonic()
    previous_cwd = Path.cwd()
    working = tempfile.TemporaryDirectory(prefix='f1x-gateway-empty-cwd-')
    os.chdir(working.name)
    server = CompetitionArcadeServer.official_110()
    server.environments_dir = str(Path.home() / 'projects/arc-agi-3-reference/competition-env/environment_files')
    games = []
    session = RunSession(record_intermediate_states=False)
    try:
        with server:
            # Run the original notebook's environment-listing function.
            notebook = json.loads((ROOT / 'solvers/m2_f1/experiment/screen-2/f1x/kernel/m2-f1-screen2-f1x.ipynb').read_text())
            cell = next(''.join(c['source']) for c in notebook['cells']
                        if 'def _competition_games()' in ''.join(c['source']))
            function = next(n for n in ast.parse(cell).body
                            if isinstance(n, ast.FunctionDef) and n.name == '_competition_games')
            os.environ['ARC_BASE_URL'] = server.base_url + '/'
            namespace = {'os': os}
            exec(compile(ast.Module(body=[function], type_ignores=[]), '<actual-notebook-game-factory>', 'exec'), namespace)
            games = namespace['_competition_games']()
            ids = [game.env_name for game in games]
            assert len(ids) == len(set(ids)) == 110
            for game in games:
                game.start_game(session)
            assert len({game._scorecard_id for game in games}) == 1
            for game in games:
                action_id = next(a for a in game.current_state.available_actions if a != 0)
                action = GameAction[f'ACTION{action_id}']
                data = {'x': 0, 'y': 0} if action.is_complex() else {}
                game.execute_action(ActionInput(id=action, data=data))
            for game in games:
                game.finish_game()
            assert all(len(game.game_run.history) == 1 for game in games)
            result = {'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                      'status': 'PASS', 'unique_games': len(ids), 'shared_scorecards': 1,
                      'native_http_actions': sum(len(g.game_run.history) for g in games),
                      'notebook_game_factory': True, 'working_directory': 'empty temporary directory',
                      'elapsed_s': time.monotonic() - started,
                      'limits': 'Local installed runtime and simulator only. No model, GPU, F1x tool loop, or Kaggle gateway.'}
            (HERE / 'CPU-GATEWAY.json').write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(result))
    except BaseException as exc:
        (HERE / 'CPU-GATEWAY.json').write_text(json.dumps({
            'status': 'FAIL', 'error': str(exc), 'working_directory': 'empty temporary directory',
        }, indent=2) + '\n')
        raise
    finally:
        for game in games:
            if game.game_run is not None and game.game_run.final_score is None:
                game.finish_game()
        session.close()
        os.chdir(previous_cwd)
        working.cleanup()


if __name__ == '__main__':
    main()
