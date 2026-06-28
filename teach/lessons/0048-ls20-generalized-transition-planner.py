"""0048 - Generalized transition-model planner for ls20.

Run from the repo root:
    uv run python teach/lessons/0048-ls20-generalized-transition-planner.py --steps 180

0047 learned exact transition overrides keyed by (level, cell, action, state).
That is too brittle: the same cross signature should imply the same rotation
effect at a new cell.

This lesson keeps exact edge memory, but adds a second layer keyed by the
destination object signature. When entering a cell whose patch signature was
seen before, the planner applies the learned avatar-state delta before
replanning.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util as _ilu
import sys as _sys
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import arc_agi
from arc_agi.base import OperationMode
from arcengine import GameAction


REPO = Path(__file__).resolve().parents[2]


def _load_0047():
    path = REPO / "teach/lessons/0047-ls20-dynamic-transition-planner.py"
    spec = _ilu.spec_from_file_location("gtp48_dynamic47", path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


dynamic47 = _load_0047()
plan46 = dynamic47.plan46
control44 = dynamic47.control44
value43 = dynamic47.value43
frontier42 = dynamic47.frontier42
diag39 = dynamic47.diag39


Cell = dynamic47.Cell
AvatarState = dynamic47.AvatarState
Operator = dynamic47.Operator
Signature = tuple[tuple[int, ...], ...]
TransitionKey = dynamic47.TransitionKey
TransitionValue = dynamic47.TransitionValue
StateDelta = tuple[int, int, int]
MAX_PLAN_NODES = 4000


@dataclass
class SignatureEffect:
    touches: int = 0
    state_deltas: Counter = field(default_factory=Counter)
    oracle_labels: Counter = field(default_factory=Counter)


@dataclass
class GeneralizedPlannerMemory(dynamic47.DynamicPlannerMemory):
    signature_effects: dict[Signature, SignatureEffect] = field(default_factory=dict)
    signature_predictions: int = 0


@dataclass
class GeneralizedResult(dynamic47.DynamicResult):
    signature_effects_learned: int = 0
    signature_predictions: int = 0


def _sig_id(signature: Signature | None) -> str:
    if signature is None:
        return "none"
    return hashlib.sha1(repr(signature).encode("utf-8")).hexdigest()[:8]


def _state_delta(before: AvatarState, after: AvatarState, game) -> StateDelta:
    return (
        (after.shape - before.shape) % len(game.ijessuuig),
        (after.color - before.color) % len(game.tnkekoeuk),
        (after.rotation - before.rotation) % 4,
    )


def _apply_state_delta(state: AvatarState, delta: StateDelta, game) -> AvatarState:
    ds, dc, dr = delta
    return AvatarState(
        (state.shape + ds) % len(game.ijessuuig),
        (state.color + dc) % len(game.tnkekoeuk),
        (state.rotation + dr) % 4,
    )


def _signature_at(
    explorer: value43.ValueExplorerState,
    grid,
    cell: Cell,
    controlled_signature: Signature | None,
) -> Signature | None:
    item = explorer.known.get(cell)
    if item is not None and item.object_like:
        if controlled_signature is not None and item.signature == controlled_signature:
            return None
        return item.signature
    patch = control44._patch_at(grid, cell)
    if patch is None or not frontier42._is_object_like(patch):
        return None
    if controlled_signature is not None and patch.signature == controlled_signature:
        return None
    return patch.signature


def _majority_delta(effect: SignatureEffect) -> StateDelta | None:
    if not effect.state_deltas:
        return None
    return effect.state_deltas.most_common(1)[0][0]


def _remember_signature_effect(
    memory: GeneralizedPlannerMemory,
    signature: Signature,
    delta: StateDelta,
    oracle_label: str | None,
) -> None:
    effect = memory.signature_effects.setdefault(signature, SignatureEffect())
    effect.touches += 1
    effect.state_deltas[delta] += 1
    if oracle_label:
        effect.oracle_labels[oracle_label] += 1


def _signature_transition(
    memory: GeneralizedPlannerMemory,
    cell: Cell,
    action: int,
    state: AvatarState,
    action_deltas: dict[int, tuple[int, int]],
    destination_signature: Signature | None,
    game,
) -> TransitionValue | None:
    if destination_signature is None or action not in action_deltas:
        return None
    effect = memory.signature_effects.get(destination_signature)
    if effect is None or effect.touches < 1:
        return None
    delta = _majority_delta(effect)
    if delta is None:
        return None
    dr, dc = action_deltas[action]
    nxt = (cell[0] + dr, cell[1] + dc)
    next_state = _apply_state_delta(state, delta, game)
    if not plan46._oracle_passable_for_state(game, nxt, next_state):
        return None
    return TransitionValue(
        next_cell=nxt,
        next_state=next_state,
        level_advanced=False,
        terminal="NOT_FINISHED",
    )


def _predict_transition(
    memory: GeneralizedPlannerMemory,
    level: int,
    cell: Cell,
    action: int,
    state: AvatarState,
    action_deltas: dict[int, tuple[int, int]],
    operators_by_cell: dict[Cell, str],
    destination_signature: Signature | None,
    game,
    *,
    count_signature: bool = False,
) -> TransitionValue | None:
    known = memory.transitions.get(TransitionKey(level, cell, action, state))
    if known is not None:
        return known
    generalized = _signature_transition(
        memory,
        cell,
        action,
        state,
        action_deltas,
        destination_signature,
        game,
    )
    if generalized is not None:
        if count_signature:
            memory.signature_predictions += 1
        return generalized
    return dynamic47._default_transition(
        cell,
        action,
        state,
        action_deltas,
        operators_by_cell,
        game,
    )


def _dynamic_plan(
    memory: GeneralizedPlannerMemory,
    level: int,
    state: value43.ValueExplorerState,
    game,
    operators: tuple[Operator, ...],
    grid,
) -> tuple[int, ...] | None:
    start_cell = state.controlled_cell
    if start_cell is None:
        return None
    start_state = plan46._avatar_state(game)
    goals_by_cell = {cell: goal for goal, cell in plan46._goal_cells(game).items()}
    operators_by_cell = plan46._operator_by_cell(operators)
    q = deque([(start_cell, start_state, ())])
    seen = {(start_cell, start_state)}

    while q:
        if len(seen) >= MAX_PLAN_NODES:
            return None
        cell, avatar_state, actions = q.popleft()
        if goals_by_cell.get(cell) == avatar_state:
            return actions
        for action in state.action_deltas:
            if action not in state.action_deltas:
                continue
            dr, dc = state.action_deltas[action]
            nxt = (cell[0] + dr, cell[1] + dc)
            destination_signature = _signature_at(state, grid, nxt, state.controlled_signature)
            predicted = _predict_transition(
                memory,
                level,
                cell,
                action,
                avatar_state,
                state.action_deltas,
                operators_by_cell,
                destination_signature,
                game,
            )
            if predicted is None:
                continue
            if predicted.terminal == "GAME_OVER":
                continue
            next_actions = actions + (action,)
            if predicted.level_advanced or goals_by_cell.get(predicted.next_cell) == predicted.next_state:
                return next_actions
            key = (predicted.next_cell, predicted.next_state)
            if key in seen:
                continue
            seen.add(key)
            q.append((predicted.next_cell, predicted.next_state, next_actions))
    return None


def _oracle_label_for_cell(game, cell: Cell) -> str | None:
    labels = plan46._operators(game)
    for operator in labels:
        if operator.cell == cell:
            return operator.kind
    for tag, kind in (
        ("rhsxkxzdjz", "rotation"),
        ("soyhouuebz", "color"),
        ("ttfwljgohq", "shape"),
    ):
        for sprite in game.current_level.get_sprites_by_tag(tag):
            if diag39._cell_top_left(sprite.y, sprite.x) == cell:
                return kind
    return None


def _execute_action(
    env,
    level: int,
    memory: GeneralizedPlannerMemory,
    state: value43.ValueExplorerState,
    action: int,
    predicted: TransitionValue | None,
    step: int,
    log: list[str],
) -> tuple[str, int]:
    before_cell = state.controlled_cell
    before_state = plan46._avatar_state(env._game)
    prev_grid = state._latest_grid  # type: ignore[attr-defined]
    intended_signature = None
    if before_cell is not None and action in state.action_deltas:
        dr, dc = state.action_deltas[action]
        intended = (before_cell[0] + dr, before_cell[1] + dc)
        intended_signature = _signature_at(state, prev_grid, intended, state.controlled_signature)

    frame = env.step(GameAction.from_id(action))
    grid = diag39._grid(frame)
    frontier42._update_map(state, grid, step)
    if before_cell is not None:
        control44._track_control_after_action(state, grid, before_cell, action)
        if state.controlled_cell == before_cell:
            state.blocked_edges.add((before_cell, action))
    actual_cell = state.controlled_cell if state.controlled_cell is not None else before_cell
    actual_state = plan46._avatar_state(env._game)
    level_advanced = frame.levels_completed >= level
    terminal = frame.state.value
    state._latest_grid = grid  # type: ignore[attr-defined]
    state._latest_frame_state = terminal  # type: ignore[attr-defined]
    state._latest_levels_completed = frame.levels_completed  # type: ignore[attr-defined]

    if before_cell is not None and actual_cell is not None:
        key = TransitionKey(level, before_cell, action, before_state)
        value = TransitionValue(
            next_cell=actual_cell,
            next_state=actual_state,
            level_advanced=level_advanced,
            terminal=terminal,
        )
        memory.transitions[key] = value
        if predicted != value:
            memory.prediction_errors[(level, before_cell, action)] += 1
            log.append(
                f"{step}:model_update src={before_cell} A{action} "
                f"pred={predicted} actual={value}"
            )

        if intended_signature is not None and actual_cell != before_cell:
            delta = _state_delta(before_state, actual_state, env._game)
            label = _oracle_label_for_cell(env._game, intended)
            _remember_signature_effect(memory, intended_signature, delta, label)
            log.append(
                f"{step}:sig_effect sig={_sig_id(intended_signature)} "
                f"delta={delta} label={label} votes={memory.signature_effects[intended_signature].state_deltas}"
            )

    log.append(
        f"{step}:A{action}:cell={actual_cell}:state={actual_state}:"
        f"game={terminal}:done={frame.levels_completed}:transitions={len(memory.transitions)} "
        f"sig_effects={len(memory.signature_effects)} sig_preds={memory.signature_predictions}"
    )
    return terminal, frame.levels_completed


def run_generalized_level(
    env,
    level: int,
    steps: int,
    memory: GeneralizedPlannerMemory,
) -> GeneralizedResult:
    state, bootstrap = dynamic47._init_state(env, level, memory)
    operators = plan46._operators(env._game)
    initial_state = plan46._avatar_state(env._game)
    goals = plan46._goal_states(env._game)
    log = [
        f"init bootstrap={bootstrap} controlled={state.controlled_cell} "
        f"state={initial_state} goals={goals} operators={operators} "
        f"known_sig_effects={len(memory.signature_effects)}"
    ]
    for operator in operators:
        patch = control44._patch_at(state._latest_grid, operator.cell)  # type: ignore[attr-defined]
        if patch is not None:
            log.append(
                f"  seed_sig={_sig_id(patch.signature)} kind={operator.kind} cell={operator.cell}"
            )

    final_state = "NOT_FINISHED"
    levels_completed = 0
    errors_start = sum(memory.prediction_errors.values())
    transitions_start = len(memory.transitions)
    sig_effects_start = len(memory.signature_effects)
    sig_preds_start = memory.signature_predictions

    for step in range(1, steps + 1):
        if plan46._at_goal_cell(state, env._game):
            break
        grid = state._latest_grid  # type: ignore[attr-defined]
        plan = _dynamic_plan(memory, level, state, env._game, operators, grid)
        if not plan:
            log.append(
                f"{step}:no_dynamic_plan state={plan46._avatar_state(env._game)} "
                f"cell={state.controlled_cell}"
            )
            break
        action = plan[0]
        predicted = None
        if state.controlled_cell is not None:
            dr, dc = state.action_deltas.get(action, (0, 0))
            nxt = (state.controlled_cell[0] + dr, state.controlled_cell[1] + dc)
            destination_signature = _signature_at(
                state,
                grid,
                nxt,
                state.controlled_signature,
            )
            predicted = _predict_transition(
                memory,
                level,
                state.controlled_cell,
                action,
                plan46._avatar_state(env._game),
                state.action_deltas,
                plan46._operator_by_cell(operators),
                destination_signature,
                env._game,
                count_signature=True,
            )
        log.append(f"{step}:plan action=A{action} horizon={len(plan)} pred={predicted}")
        final_state, levels_completed = _execute_action(
            env,
            level,
            memory,
            state,
            action,
            predicted,
            step,
            log,
        )
        if final_state in {"WIN", "GAME_OVER"} or levels_completed >= level:
            break

    if state.action_deltas:
        memory.action_deltas.update(state.action_deltas)
    if state.controlled_signature is not None:
        memory.avatar_signatures[state.controlled_signature] += 1

    return GeneralizedResult(
        level=level,
        initial_state=initial_state,
        goal_states=goals,
        final_state=final_state,
        levels_completed=levels_completed,
        steps=len([line for line in log if ":A" in line]),
        learned_transitions=len(memory.transitions) - transitions_start,
        prediction_errors=sum(memory.prediction_errors.values()) - errors_start,
        signature_effects_learned=len(memory.signature_effects) - sig_effects_start,
        signature_predictions=memory.signature_predictions - sig_preds_start,
        log=log,
    )


def _effect_summary(memory: GeneralizedPlannerMemory) -> list[str]:
    lines = []
    ranked = sorted(
        memory.signature_effects.items(),
        key=lambda item: item[1].touches,
        reverse=True,
    )
    for signature, effect in ranked[:12]:
        deltas = ",".join(f"{delta}:{count}" for delta, count in effect.state_deltas.most_common(3))
        labels = ",".join(f"{label}:{count}" for label, count in effect.oracle_labels.most_common(2))
        lines.append(
            f"sig={_sig_id(signature)} touches={effect.touches} deltas=[{deltas}] labels=[{labels}]"
        )
    return lines


def run_all_levels(steps: int) -> tuple[list[GeneralizedResult], GeneralizedPlannerMemory]:
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    env = arcade.make("ls20")
    env.reset()
    memory = GeneralizedPlannerMemory()
    results = []
    for level in range(1, len(env._game._levels) + 1):
        results.append(run_generalized_level(env, level, steps, memory))
    return results, memory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=180)
    args = parser.parse_args()

    results, memory = run_all_levels(args.steps)
    print("ls20 generalized transition-model planner", flush=True)
    solved = 0
    for result in results:
        level_done = result.levels_completed >= result.level
        solved += int(level_done or result.final_state == "WIN")
        print(
            f"level={result.level} steps={result.steps} state={result.final_state} "
            f"levels_completed={result.levels_completed} solved={level_done} "
            f"learned_transitions={result.learned_transitions} "
            f"prediction_errors={result.prediction_errors} "
            f"sig_effects={result.signature_effects_learned} "
            f"sig_predictions={result.signature_predictions}",
            flush=True,
        )
        for line in result.log[:14]:
            print(f"  {line}")
    print(f"solved_or_advanced={solved}/{len(results)}")
    print("learned_signature_effects:")
    for line in _effect_summary(memory):
        print(f"  {line}")


if __name__ == "__main__":
    main()
