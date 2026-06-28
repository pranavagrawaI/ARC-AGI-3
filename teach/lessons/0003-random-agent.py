"""Lesson 0003 companion: your first agent — random — and a harness to measure it.

Run from the repo root:  uv run python teach/lessons/0003-random-agent.py

The agent interface every future agent will implement:
    choose(frame) -> GameAction
Everything else (episodes, budgets, scoring) is the harness, and it
never changes. Today's policy is the dumbest possible one: random.
"""

import random
import time

import arc_agi
from arc_agi.base import OperationMode
from arcengine import GameAction

BASELINES = [22, 123, 73, 84, 96, 192, 186]
EPISODES = 5
MAX_ACTIONS = 5000  # per episode — generous; humans needed 776 total


class RandomAgent:
    """Picks uniformly among legal actions. No perception, no memory, no plan."""

    def choose(self, frame) -> GameAction:
        return GameAction.from_id(random.choice(frame.available_actions))


class BlindMomentumAgent:
    """Picks the same action as last time, till a persistence threshold is hit, then picks a new random action."""

    def __init__(self, persistence=8):
        self.persistence = persistence
        self.direction = None
        self.count = 0

    def choose(self, frame) -> GameAction:
        if self.direction is None or self.count >= self.persistence:
            self.direction = GameAction.from_id(random.choice(frame.available_actions))
            self.count = 0
        else:
            self.count += 1
        return self.direction


def run_episode(arc, agent, max_actions: int) -> dict:
    env = arc.make("ls20")  # no renderer, no recording: full speed
    frame = env.reset()
    level_actions = [0] * 7
    level = 0
    actions = 0

    while actions < max_actions and frame.state.value != "WIN":
        if frame.state.value == "GAME_OVER":
            frame = env.reset()
        else:
            frame = env.step(agent.choose(frame))
        actions += 1
        level_actions[min(level, 6)] += 1
        level = max(level, frame.levels_completed)

    scores = [
        min((BASELINES[i] / level_actions[i]) ** 2, 1.15)
        if i < level and level_actions[i]
        else 0.0
        for i in range(7)
    ]
    if frame.state.value == "WIN":  # last level's actions count too
        scores[6] = min((BASELINES[6] / level_actions[6]) ** 2, 1.15)
    game_score = sum((i + 1) * s for i, s in enumerate(scores)) / 28

    return {
        "won": frame.state.value == "WIN",
        "levels": level,
        "actions": actions,
        "game_score": game_score,
    }


def compare_agents(arc, agents, episodes=EPISODES, max_actions=MAX_ACTIONS):
    results = {agent.__class__.__name__: [] for agent in agents}
    for agent in agents:
        print(f"Running {episodes} episodes of {agent.__class__.__name__}...")
        for ep in range(episodes):
            r = run_episode(arc, agent, max_actions)
            results[agent.__class__.__name__].append(r)
            print(
                f"  episode {ep + 1}: levels={r['levels']}/7  "
                f"actions={r['actions']}  game_score={r['game_score']:.1%}"
            )
    return results


if __name__ == "__main__":
    agent1 = RandomAgent()
    agent2 = BlindMomentumAgent(persistence=8)

    compare_agents(
        arc_agi.Arcade(operation_mode=OperationMode.OFFLINE),
        agents=[agent1, agent2],
        episodes=EPISODES,
        max_actions=MAX_ACTIONS,
    )

    # arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    # agent = RandomAgent()

    # print(
    #     f"Running {EPISODES} episodes of RandomAgent on ls20 "
    #     f"(budget {MAX_ACTIONS} actions each)...\n"
    # )
    # results = []
    # t0 = time.time()
    # for ep in range(EPISODES):
    #     r = run_episode(arc, agent, MAX_ACTIONS)
    #     results.append(r)
    #     print(
    #         f"  episode {ep + 1}: levels={r['levels']}/7  "
    #         f"actions={r['actions']}  game_score={r['game_score']:.1%}"
    #     )
    # dt = time.time() - t0

    # mean = sum(r["game_score"] for r in results) / len(results)
    # best_levels = max(r["levels"] for r in results)
    # total_steps = sum(r["actions"] for r in results)
    # print(
    #     f"\nRandomAgent mean game score: {mean:.1%}  (best run reached level {best_levels + 1})"
    # )
    # print("Your human score on this game: 115.0%")
    # print(
    #     f"({total_steps} env steps in {dt:.1f}s — {total_steps / dt:,.0f} steps/sec: "
    #     f"this speed is your laboratory)"
    # )
