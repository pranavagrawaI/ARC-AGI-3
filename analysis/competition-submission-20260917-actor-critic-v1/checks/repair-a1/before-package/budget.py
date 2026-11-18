"""Frozen competition wall-time and request envelope (seconds)."""
import math
PROVIDER = 32400
UNCERTAINTY = 900
FINAL_OUTPUT = 300
BOOTSTRAP = 1800
STARTUP_WITH_WARMUP = 1200
OUTER_CLEANUP = 180
GAMES = 110
REQUEST = 90
CRITIC_RESERVE = 90
GAME_CLEANUP = 30
TOOL_MARGIN = 15
ACTOR_CALLS = 12
CRITIC_CALLS = 4
GAMEPLAY = PROVIDER-UNCERTAINTY-FINAL_OUTPUT-BOOTSTRAP-STARTUP_WITH_WARMUP-OUTER_CLEANUP

def admission(timeout, count, request=REQUEST):
    if not all(math.isfinite(x) for x in (timeout,count,request)) or type(count) is not int or count <= 0 or request <= 0:
        raise ValueError('Invalid budget inputs')
    value = (timeout-180)/count-GAME_CLEANUP-2*request-TOOL_MARGIN
    if value <= 0:
        raise ValueError('Impossible all-game budget; refuse before model startup')
    return value

def validate():
    assert PROVIDER <= 9*3600
    assert GAMEPLAY + BOOTSTRAP + STARTUP_WITH_WARMUP + OUTER_CLEANUP + FINAL_OUTPUT + UNCERTAINTY == PROVIDER
    return dict(games=GAMES, gameplay_seconds=GAMEPLAY, admission_seconds=admission(GAMEPLAY,GAMES),
                actor_calls_per_game=ACTOR_CALLS,critic_calls_per_game=CRITIC_CALLS,
                total_model_requests=1+GAMES*(ACTOR_CALLS+CRITIC_CALLS),provider_seconds=PROVIDER)
