"""Matched-pair budgets; embedded verbatim in both arms so neither derives its own."""
from __future__ import annotations

GAMES = ("ls20", "re86", "sc25")
PER_GAME_SECONDS = 3600
CONCURRENCY = 3
SESSION_SECONDS = 5400
SOFT_STOP_SECONDS = 5100


def game_prefix(game):
    return str(game.env_name).split("-")[0]


def apply(bm):
    """Replace the donor's concurrency-derived budget (870 s at three games) with fixed caps."""
    bm.games = [game for game in bm.games if game_prefix(game) in GAMES]
    bm.game_weights = None
    bm.n_passes = 1
    bm.solver.concurrency = CONCURRENCY
    bm.solver.max_runtime_s_per_game = PER_GAME_SECONDS
    bm.solver.max_actions_per_game = None
    return check(bm)


def check(bm):
    names = sorted(game_prefix(game) for game in bm.games)
    observed = {"games": names, "passes": bm.n_passes, "game_weights": bm.game_weights,
                "concurrency": bm.solver.concurrency,
                "per_game_seconds": bm.solver.max_runtime_s_per_game,
                "per_game_action_cap": bm.solver.max_actions_per_game}
    expected = {"games": sorted(GAMES), "passes": 1, "game_weights": None,
                "concurrency": CONCURRENCY, "per_game_seconds": PER_GAME_SECONDS,
                "per_game_action_cap": None}
    if observed != expected:
        raise RuntimeError(f"pair budget mismatch: {observed} != {expected}")
    return {**observed, "session_seconds": SESSION_SECONDS,
            "soft_stop_seconds": SOFT_STOP_SECONDS}
