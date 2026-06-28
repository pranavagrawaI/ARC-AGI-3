"""0045 - Cross-level object-effect memory for ls20.

Run from the repo root:
    uv run python teach/lessons/0045-ls20-cross-level-object-effect-memory.py --steps 120

0044 carried the controllable role and action semantics across levels. This
lesson carries a second kind of hypothesis: object purpose. If touching an
object-like cell changes the controlled patch, future levels start with the
hypothesis that the same object signature is a state changer.

The policy and memory are frame-only. Private sprite labels and private avatar
state are used only in the printed diagnostics to check whether the learned
object-purpose hypotheses match the real game mechanics.
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
import numpy as np
from arc_agi.base import OperationMode
from arcengine import GameAction


REPO = Path(__file__).resolve().parents[2]


def _load_0044():
    path = REPO / "teach/lessons/0044-ls20-cross-level-control-memory.py"
    spec = _ilu.spec_from_file_location("clem45_control44", path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


control44 = _load_0044()
value43 = control44.value43
frontier42 = control44.frontier42
diag39 = control44.diag39
verify41 = control44.verify41


Signature = tuple[tuple[int, ...], ...]
Cell = tuple[int, int]


@dataclass
class ObjectEffect:
    touches: int = 0
    avatar_changes: int = 0
    remote_state_changes: int = 0
    unchanged: int = 0
    level_advances: int = 0
    game_overs: int = 0
    effect_kinds: Counter = field(default_factory=Counter)
    oracle_deltas: Counter = field(default_factory=Counter)
    labels: Counter = field(default_factory=Counter)

    @property
    def useful_score(self) -> int:
        return self.avatar_changes + self.remote_state_changes + self.level_advances

    @property
    def danger_score(self) -> int:
        return self.game_overs


@dataclass
class EffectObservation:
    level: int
    step: int
    cell: Cell
    signature: Signature
    predicted: str
    effect_kind: str
    avatar_changed: bool
    remote_changed_cells: tuple[Cell, ...]
    level_advanced: bool
    game_state: str
    oracle_delta: str
    oracle_labels: tuple[str, ...]


@dataclass
class CrossLevelEffectMemory(control44.CrossLevelMemory):
    object_effects: dict[Signature, ObjectEffect] = field(default_factory=dict)
    observations: list[EffectObservation] = field(default_factory=list)


def _sig_id(signature: Signature | None) -> str:
    if signature is None:
        return "none"
    raw = repr(signature).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:8]


def _patch_at(grid: np.ndarray, cell: Cell):
    return control44._patch_at(grid, cell)


def _object_patch_at(grid: np.ndarray, cell: Cell, controlled_signature: Signature | None):
    patch = _patch_at(grid, cell)
    if patch is None or not frontier42._is_object_like(patch):
        return None
    if controlled_signature is not None and patch.signature == controlled_signature:
        return None
    return patch


def _private_avatar_state(game) -> tuple[int, int, int]:
    return (game.fwckfzsyc, game.hiaauhahz, game.cklxociuu)


def _oracle_delta(before: tuple[int, int, int], after: tuple[int, int, int]) -> str:
    names = ("shape", "color", "rotation")
    changed = [name for name, b, a in zip(names, before, after) if b != a]
    return "+".join(changed) if changed else "none"


def _background_mask(arr: np.ndarray) -> np.ndarray:
    vals, counts = np.unique(arr, return_counts=True)
    bg = vals[counts.argmax()]
    return arr != bg


def _effect_kind(before: Signature | None, after: Signature | None) -> str:
    if before is None or after is None:
        return "unknown"
    if before == after:
        return "none"

    before_arr = np.asarray(before, dtype=np.int16)
    after_arr = np.asarray(after, dtype=np.int16)
    if before_arr.shape != after_arr.shape:
        return "avatar_signature_change"

    for turns in (1, 2, 3):
        if np.array_equal(np.rot90(before_arr, turns), after_arr):
            return "rotation_like"

    before_mask = _background_mask(before_arr)
    after_mask = _background_mask(after_arr)
    if np.array_equal(before_mask, after_mask):
        return "color_like"
    for turns in (1, 2, 3):
        if np.array_equal(np.rot90(before_mask, turns), after_mask):
            return "rotation_or_shape_like"
    return "shape_like"


def _local_change_cells(prev_cell: Cell, target_cell: Cell | None, controlled_cell: Cell | None) -> set[Cell]:
    cells = {prev_cell}
    if target_cell is not None:
        cells.add(target_cell)
    if controlled_cell is not None:
        cells.add(controlled_cell)
    return cells


def _observation_kind(
    avatar_before: Signature | None,
    avatar_after: Signature | None,
    remote_changed_cells: tuple[Cell, ...],
    level_advanced: bool,
    game_state: str,
) -> str:
    if level_advanced:
        return "level_advance"
    if game_state == "GAME_OVER":
        return "game_over"
    if remote_changed_cells:
        return "remote_state_change"
    return _effect_kind(avatar_before, avatar_after)


def _effect_prediction(memory: CrossLevelEffectMemory, signature: Signature | None) -> str:
    if signature is None or signature not in memory.object_effects:
        return "unknown"
    effect = memory.object_effects[signature]
    if effect.danger_score > effect.useful_score:
        return "danger"
    if effect.effect_kinds:
        kind, count = effect.effect_kinds.most_common(1)[0]
        return f"{kind}:{count}/{effect.touches}"
    return f"known:{effect.touches}"


def _remember_observation(memory: CrossLevelEffectMemory, observation: EffectObservation) -> None:
    effect = memory.object_effects.setdefault(observation.signature, ObjectEffect())
    effect.touches += 1
    if observation.avatar_changed:
        effect.avatar_changes += 1
    if observation.remote_changed_cells:
        effect.remote_state_changes += 1
    if not observation.avatar_changed and not observation.remote_changed_cells:
        effect.unchanged += 1
    if observation.level_advanced:
        effect.level_advances += 1
    if observation.game_state == "GAME_OVER":
        effect.game_overs += 1
    effect.effect_kinds[observation.effect_kind] += 1
    effect.oracle_deltas[observation.oracle_delta] += 1
    for label in observation.oracle_labels:
        effect.labels[label.split("[", 1)[0]] += 1
    memory.observations.append(observation)


def _cell_score_with_memory(
    state: value43.ValueExplorerState,
    memory: CrossLevelEffectMemory,
    cell: Cell,
) -> tuple[int, int, int, int, int]:
    item = state.known[cell]
    effect = memory.object_effects.get(item.signature)
    danger = effect.danger_score if effect is not None else 0
    useful = effect.useful_score if effect is not None else 0
    known = 1 if effect is not None else 0
    base = value43._cell_score(state, cell)
    return (-danger, useful, known, *base)


def _object_targets_with_memory(
    state: value43.ValueExplorerState,
    memory: CrossLevelEffectMemory,
) -> list[Cell]:
    targets = value43._object_targets(state)
    targets.sort(key=lambda cell: _cell_score_with_memory(state, memory, cell), reverse=True)
    return targets


def _path_to_cell(state: value43.ValueExplorerState, target: Cell) -> tuple[int, ...] | None:
    start = state.controlled_cell
    if start is None:
        return None
    if start == target:
        return ()
    q = deque([(start, ())])
    seen = {start}
    while q:
        cell, path = q.popleft()
        for action, nxt in frontier42._neighbors(cell, state.action_deltas):
            if not frontier42._in_lattice(nxt) or nxt in seen:
                continue
            if nxt not in state.known:
                continue
            if (cell, action) in state.blocked_edges:
                continue
            npath = path + (action,)
            if nxt == target:
                return npath
            seen.add(nxt)
            q.append((nxt, npath))
    return None


def _choose_effect_action(
    state: value43.ValueExplorerState,
    memory: CrossLevelEffectMemory,
) -> tuple[int, str]:
    if state.controlled_cell is not None and len(state.action_deltas) < len(frontier42.MOVE_ACTIONS):
        for action in frontier42.MOVE_ACTIONS:
            edge = (state.controlled_cell, action)
            if edge not in state.tried_edges and edge not in state.blocked_edges:
                return action, "edge_probe"

    if state.current_target is not None:
        if state.current_target_age > 20:
            state.unreachable_targets.add(state.current_target)
            state.current_target = None
            state.current_target_age = 0
        else:
            state.current_target_age += 1
    if state.current_target is not None:
        path = _path_to_cell(state, state.current_target)
        if path is None:
            state.unreachable_targets.add(state.current_target)
            state.current_target = None
            state.current_target_age = 0
        elif path:
            item = state.known.get(state.current_target)
            pred = _effect_prediction(memory, item.signature if item is not None else None)
            return path[0], f"target:{state.current_target}:pred={pred}"
        else:
            state.current_target = None
            state.current_target_age = 0

    for target in _object_targets_with_memory(state, memory):
        path = _path_to_cell(state, target)
        if path is None:
            state.unreachable_targets.add(target)
            continue
        state.current_target = target
        state.current_target_age = 0
        state.target_history.append((len(state.log), target))
        if path:
            item = state.known.get(target)
            pred = _effect_prediction(memory, item.signature if item is not None else None)
            return path[0], f"target:{target}:pred={pred}"
        state.current_target = None
        state.current_target_age = 0
        break

    return frontier42._choose_explore_action(state), "frontier"


def _oracle_label_by_cell(game) -> dict[Cell, list[str]]:
    return control44._oracle_label_by_cell(game)


def _effect_summary(memory: CrossLevelEffectMemory) -> list[str]:
    lines = []
    ranked = sorted(
        memory.object_effects.items(),
        key=lambda item: (item[1].useful_score, item[1].touches, -item[1].danger_score),
        reverse=True,
    )
    for signature, effect in ranked[:12]:
        labels = ",".join(f"{label}:{count}" for label, count in effect.labels.most_common(3))
        kinds = ",".join(f"{kind}:{count}" for kind, count in effect.effect_kinds.most_common(3))
        deltas = ",".join(f"{delta}:{count}" for delta, count in effect.oracle_deltas.most_common(3))
        lines.append(
            f"sig={_sig_id(signature)} touches={effect.touches} avatar_changes={effect.avatar_changes} "
            f"remote_state_changes={effect.remote_state_changes} "
            f"level_advances={effect.level_advances} game_overs={effect.game_overs} "
            f"kinds=[{kinds}] oracle_deltas=[{deltas}] labels=[{labels}]"
        )
    return lines


def _known_effect_candidates(
    state: value43.ValueExplorerState,
    memory: CrossLevelEffectMemory,
) -> list[tuple[Cell, str, str]]:
    candidates = []
    for cell, item in state.known.items():
        if cell == state.controlled_cell or item.signature not in memory.object_effects:
            continue
        candidates.append((cell, _sig_id(item.signature), _effect_prediction(memory, item.signature)))
    candidates.sort(key=lambda item: item[0])
    return candidates


def run_level_with_effect_memory(
    env,
    level: int,
    steps: int,
    memory: CrossLevelEffectMemory,
) -> dict[str, object]:
    level_index = level - 1
    controlled_sig, controlled_cell, action_deltas, bootstrap = control44._bootstrap_control(
        env,
        level_index,
        memory,
    )
    grid = frontier42._set_level(env, level_index)
    state = value43.ValueExplorerState(
        controlled_signature=controlled_sig,
        action_deltas=dict(action_deltas),
    )
    frontier42._update_map(state, grid, 0)
    if controlled_cell is not None:
        state.controlled_cell = controlled_cell
    state.log.append(
        f"init bootstrap={bootstrap} controlled_cell={state.controlled_cell} "
        f"actions={state.action_deltas} known_effects={len(memory.object_effects)}"
    )
    initial_known_effect_candidates = _known_effect_candidates(state, memory)

    final_state = "NOT_FINISHED"
    levels_completed = 0
    level_observations_start = len(memory.observations)
    for step in range(1, steps + 1):
        if state.controlled_cell is None or not state.action_deltas:
            state.log.append(f"{step}:blocked_no_control_model")
            break
        prev_grid = grid
        prev_cell = state.controlled_cell
        avatar_before = state.controlled_signature
        private_before = _private_avatar_state(env._game)
        labels_before = _oracle_label_by_cell(env._game)
        action, reason = _choose_effect_action(state, memory)
        target_cell = None
        target_patch = None
        if action in state.action_deltas:
            delta = state.action_deltas[action]
            target_cell = (prev_cell[0] + delta[0], prev_cell[1] + delta[1])
            target_patch = _object_patch_at(prev_grid, target_cell, state.controlled_signature)
        target_signature = target_patch.signature if target_patch is not None else None
        predicted = _effect_prediction(memory, target_signature)

        state.tried_edges.add((prev_cell, action))
        frame = env.step(GameAction.from_id(action))
        grid = diag39._grid(frame)
        private_after = _private_avatar_state(env._game)
        changed = frontier42._changed_visible_cells(prev_grid, grid)
        new_cells = frontier42._update_map(state, grid, step)
        control44._track_control_after_action(state, grid, prev_cell, action)
        avatar_after = state.controlled_signature
        moved = state.controlled_cell != prev_cell
        if not moved:
            state.blocked_edges.add((prev_cell, action))

        entered = state.controlled_cell if moved else None
        level_advanced = frame.levels_completed > levels_completed
        if entered in state.known and entered not in state.tested_cells:
            state.tested_cells[entered] = value43.TestRecord(
                cell=entered,
                step=step,
                changed_cells=len(changed),
                new_cells=new_cells,
                level_advanced=level_advanced,
                game_state=frame.state.value,
            )
            if state.current_target == entered:
                state.current_target = None
                state.current_target_age = 0

        if target_signature is not None and target_cell is not None:
            remote_changed = tuple(
                cell
                for cell in changed
                if cell not in _local_change_cells(prev_cell, target_cell, state.controlled_cell)
            )
            effect_kind = _observation_kind(
                avatar_before,
                avatar_after,
                remote_changed,
                level_advanced,
                frame.state.value,
            )
            observation = EffectObservation(
                level=level,
                step=step,
                cell=target_cell,
                signature=target_signature,
                predicted=predicted,
                effect_kind=effect_kind,
                avatar_changed=avatar_before != avatar_after,
                remote_changed_cells=remote_changed,
                level_advanced=level_advanced,
                game_state=frame.state.value,
                oracle_delta=_oracle_delta(private_before, private_after),
                oracle_labels=tuple(labels_before.get(target_cell, ())),
            )
            _remember_observation(memory, observation)

        final_state = frame.state.value
        levels_completed = frame.levels_completed
        state.log.append(
            f"{step}:A{action}:{reason}:moved={moved}:new={new_cells}:changed={len(changed)}:"
            f"known={len(state.known)}:tested={len(state.tested_cells)}:"
            f"cell={state.controlled_cell}:state={final_state}:level_done={levels_completed}"
        )
        if final_state in {"WIN", "GAME_OVER"} or levels_completed >= level:
            break

    if state.action_deltas:
        memory.action_deltas.update(state.action_deltas)
    if state.controlled_signature is not None:
        memory.avatar_signatures[state.controlled_signature] += 1

    oracle_hits, oracle_total, oracle_labels = frontier42._oracle_visible_cells_seen(env._game, state)
    level_observations = memory.observations[level_observations_start:]
    return {
        "level": level,
        "bootstrap": bootstrap,
        "steps": len([line for line in state.log if ":A" in line]),
        "known_cells": len(state.known),
        "tested_cells": len(state.tested_cells),
        "controlled_cell": state.controlled_cell,
        "final_state": final_state,
        "levels_completed": levels_completed,
        "action_deltas": dict(state.action_deltas),
        "oracle_hits": oracle_hits,
        "oracle_total": oracle_total,
        "oracle_labels": oracle_labels,
        "initial_known_effect_candidates": initial_known_effect_candidates,
        "effect_observations": level_observations,
        "log": state.log,
    }


def run_all_levels(steps: int) -> tuple[list[dict[str, object]], CrossLevelEffectMemory]:
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    env = arcade.make("ls20")
    env.reset()
    memory = CrossLevelEffectMemory()
    results = []
    for level in range(1, len(env._game._levels) + 1):
        results.append(run_level_with_effect_memory(env, level, steps, memory))
    return results, memory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()

    results, memory = run_all_levels(args.steps)
    print("ls20 cross-level object-effect memory explorer")
    for result in results:
        print(
            f"level={result['level']} bootstrap={result['bootstrap']} steps={result['steps']} "
            f"state={result['final_state']} known={result['known_cells']} tested={result['tested_cells']} "
            f"effects_seen={len(result['effect_observations'])} "
            f"known_effect_candidates={result['initial_known_effect_candidates'][:6]}"
        )
        for obs in result["effect_observations"][:8]:
            print(
                f"  effect step={obs.step} cell={obs.cell} sig={_sig_id(obs.signature)} "
                f"pred={obs.predicted} kind={obs.effect_kind} oracle_delta={obs.oracle_delta} "
                f"remote={len(obs.remote_changed_cells)} labels={list(obs.oracle_labels)} state={obs.game_state}"
            )
    print("learned_object_effects:")
    for line in _effect_summary(memory):
        print(f"  {line}")


if __name__ == "__main__":
    main()
