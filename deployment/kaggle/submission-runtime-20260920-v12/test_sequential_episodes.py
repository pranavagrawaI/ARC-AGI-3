"""Sequential (pass, game) plan: Swarm must not share the 1800 s wall."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nooa_public_probe as probe


def test_episode_plan_is_one_game_per_official_launcher():
    assert probe.episode_plan(['sb26', 'su15'], 2) == [
        (0, 'sb26'), (0, 'su15'), (1, 'sb26'), (1, 'su15'),
    ]
    assert probe.episode_plan(['sb26'], 1) == [(0, 'sb26')]


def test_episode_timeout_caps_a_hung_game_so_later_episodes_remain():
    # 7800 s budget, 3 leftover episodes, 1800 s admission → 1980 s cap, not 7710.
    assert probe._episode_timeout(7800, leftover=3, admission=1800) == 1980
    assert probe._episode_timeout(200, leftover=0, admission=1800) == 200
    assert probe._episode_timeout(40, leftover=1, admission=1800) == 60


def test_list_handler_serves_the_current_game_only():
    previous = list(probe.CURRENT_GAMES)
    try:
        with probe._GAMES_LOCK:
            probe.CURRENT_GAMES[:] = ['su15']
        handler = probe.GameListHandler.__new__(probe.GameListHandler)
        sent = {}
        handler.path = '/api/games'
        handler.send_response = lambda code: sent.__setitem__('code', code)
        handler.send_header = lambda *a: None
        handler.end_headers = lambda: None
        handler.wfile = type('W', (), {'write': lambda self, b: sent.__setitem__('body', b)})()
        handler.do_GET()
        assert sent['code'] == 200
        assert sent['body'] == b'[{"game_id": "su15"}]'
    finally:
        with probe._GAMES_LOCK:
            probe.CURRENT_GAMES[:] = previous
