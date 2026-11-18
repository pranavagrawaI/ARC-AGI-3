"""Evaluate a proposed action on a hash-recorded, exactly reconstructed public state.

Run with the assembled hybrid runtime importable. No model, GPU, or online Arcade
is used. Changed pixels are reported as observations, never as useful progress.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

_RUNTIME = Path(__file__).resolve().parents[2] / "runtime"
if _RUNTIME.is_dir():
    sys.path.insert(0, str(_RUNTIME))


class ReconstructionMismatch(ValueError):
    """The local engine does not reproduce the packet's recorded public state."""


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _same(actual, expected, where):
    actual = actual.model_dump(mode="json")
    expected = expected.model_dump(mode="json")
    if actual != expected:
        differing = [key for key in expected if actual.get(key) != expected[key]]
        raise ReconstructionMismatch(f"{where}: differing public fields {differing}; "
            f"actual={_digest(actual)} expected={_digest(expected)}")


def _execute(game, decision):
    import arcengine
    from tycho_nooa.protocol import ARC_ACTION_NAME_TO_ID
    game.execute_action(arcengine.ActionInput(
        id=arcengine.GameAction.from_id(ARC_ACTION_NAME_TO_ID[decision.action]),
        data={"x": decision.col, "y": decision.row} if decision.action == "CLICK" else {}))


def _turn(current, sequence, turn_in_level, previous=None):
    from tycho_nooa.models import GameTurn
    prior = (previous.decision if previous is not None and not previous.after.full_reset
             and previous.after.levels_completed == previous.before.levels_completed else None)
    return GameTurn(grid=current.grid, available_actions=current.available_actions,
        sequence=sequence, level=current.levels_completed, turn_in_level=turn_in_level,
        levels_completed=current.levels_completed, state=current.state,
        previous_action=prior.action if prior else "", previous_row=prior.row if prior else None,
        previous_col=prior.col if prior else None)


def _reconstruct(game, rows, expected_turn):
    from hybrid.solver import observation
    current, previous, turn_in_level = observation(game), None, 0
    for index, receipt in enumerate(rows):
        if (receipt.sequence, receipt.level, receipt.turn_in_level) != (
                index, current.levels_completed, turn_in_level):
            raise ReconstructionMismatch(f"prefix receipt {index}: sequence/level/turn mismatch")
        _same(current, receipt.before, f"prefix {index} before")
        if receipt.decision.action not in current.available_actions:
            raise ReconstructionMismatch(f"prefix {index}: recorded action is unavailable")
        _execute(game, receipt.decision)
        after = observation(game, previous_levels=current.levels_completed)
        _same(after, receipt.after, f"prefix {index} after")
        turn_in_level = 0 if after.levels_completed > current.levels_completed else turn_in_level + 1
        current, previous = after, receipt
    actual_turn = _turn(current, len(rows), turn_in_level, previous)
    # The engine cannot recover an externally chosen action cap. Pydantic validates
    # those optional horizon fields against sequence; copy only that host contract.
    horizon = {name: getattr(expected_turn, name) for name in
               ("action_limit", "actions_used", "actions_remaining", "final_permitted_action")}
    _same(actual_turn.model_copy(update=horizon), expected_turn, "packet decision point")
    return current


def _validate_batch(batch, turn):
    from tycho_nooa.action_batch import grid_sha256
    if turn.actions_remaining is not None and len(batch.actions) > turn.actions_remaining:
        raise ValueError("batch exceeds remaining host action allowance")
    if any(step.action not in turn.available_actions for step in batch.actions):
        raise ValueError("batch contains an action unavailable at the decision point")
    expected = batch.actions[0].expected_before_grid_sha256
    if expected is not None and expected != grid_sha256(turn.grid):
        raise ValueError("first action expected-frame guard does not match")


def _apply(game, current, turn, batch, result):
    from hybrid.solver import observation
    from hybrid.policy import interior_changed
    from tycho_nooa.action_batch import PendingActionBatch
    from tycho_nooa.protocol import ActionResultMessage, PROTOCOL_VERSION
    pending = PendingActionBatch(batch, turn.sequence, turn.level, turn.turn_in_level)
    decision = batch.actions[0].decision()
    result["action_legal"] = True
    for offset in range(len(batch.actions)):
        before = current
        _execute(game, decision)
        current = observation(game, previous_levels=before.levels_completed)
        transition = ActionResultMessage(protocol_version=PROTOCOL_VERSION, type="action_result",
            sequence=turn.sequence, level=turn.level, turn_in_level=turn.turn_in_level,
            decision=decision, before=before, after=current)
        result["transitions"].append(transition.model_dump(mode="json"))
        result["frame_changed"] |= before.grid != current.grid
        result["interior_changed"] |= interior_changed(before.grid, current.grid)
        result["levels_completed_after"] = current.levels_completed
        result["levels_gained"] = current.levels_completed - result["levels_completed_before"]
        result["state_after"] = current.state
        if pending.exhausted:
            result["stop_reason"] = "batch_exhausted"
            break
        next_turn_number = 0 if current.levels_completed > before.levels_completed else turn.turn_in_level + 1
        turn = _turn(current, turn.sequence + 1, next_turn_number, transition)
        decision, reason = pending.release(turn, transition)
        if decision is None:
            result["stop_reason"] = reason
            break
    result["level_completed"] = result["levels_gained"] > 0
    result["game_won"] = current.state == "WIN"
    result["game_over"] = current.state == "GAME_OVER"
    result["status"] = "evaluated"


def evaluate(packet_root, action_decision, environments_dir, output_path):
    """Reconstruct a packet, check it exactly, then evaluate one bounded batch.

    Illegal proposals are returned as such. Reconstruction/engine failures write
    an explicit failure artifact and raise; they are never scored as model losses.
    """
    import arc_agi
    from taaf.game import RunSession
    from taaf.game_api import ArcadeSpec, GameAPI
    from tycho_nooa.action_batch import ActionBatchDecision
    from tycho_nooa.models import GameTurn
    from tycho_nooa.protocol import ActionResultMessage
    packet_root, output_path = Path(packet_root), Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    environments_dir = Path(environments_dir).resolve(strict=True)
    if not environments_dir.is_dir():
        raise ValueError("environments_dir must be an explicit offline environment directory")
    manifest = json.loads((packet_root / "manifest.json").read_text())
    turn = GameTurn.model_validate_json((packet_root / "turn.json").read_text())
    prefix_bytes = (packet_root / "host-prefix.jsonl").read_bytes()
    rows = [ActionResultMessage.model_validate_json(line) for line in prefix_bytes.splitlines() if line]
    result = {"schema": "tycho-tractability-action-evaluation-v1", "game_id": manifest["game_id"],
        "packet_manifest_sha256": hashlib.sha256((packet_root / "manifest.json").read_bytes()).hexdigest(),
        "prefix_sha256": hashlib.sha256(prefix_bytes).hexdigest(), "sequence": turn.sequence,
        "prefix_verified": False, "action_legal": None, "transitions": [], "frame_changed": False,
        "interior_changed": False, "levels_completed_before": turn.levels_completed,
        "levels_completed_after": None, "levels_gained": None, "level_completed": None,
        "state_after": None, "game_won": None, "game_over": None,
        "progress_claim": "Only level completion is classified as progress; changed pixels are not.",
        "offline_environments_dir": str(environments_dir)}
    game, session = None, RunSession(record_intermediate_states=False)
    try:
        point = manifest.get("decision_point", {})
        if any(point.get(key, getattr(turn, key)) != getattr(turn, key)
               for key in ("sequence", "level", "turn_in_level")):
            raise ReconstructionMismatch("manifest and turn identify different decision points")
        if len(rows) != turn.sequence:
            raise ReconstructionMismatch("prefix length does not equal decision sequence")
        game = GameAPI(env_name=manifest["game_id"], arcade_spec=ArcadeSpec(
            operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(environments_dir)))
        game.start_game(session)
        current = _reconstruct(game, rows, turn)
        result["prefix_verified"] = True
        result["observation_sha256"] = _digest(current.model_dump(mode="json"))
        try:
            batch = ActionBatchDecision.model_validate(action_decision)
            _validate_batch(batch, turn)
        except (ValueError, TypeError) as error:
            result.update(status="illegal_action", action_legal=False, error=str(error))
            return result
        result["proposed_batch"] = batch.model_dump(mode="json")
        _apply(game, current, turn, batch, result)
        return result
    except Exception as error:
        result.update(status="reconstruction_mismatch" if isinstance(error, ReconstructionMismatch)
                      else "evaluation_error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            if game is not None and game.game_run is not None:
                game.finish_game()
        finally:
            session.close()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("x") as stream:
                json.dump(result, stream, indent=2)
                stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet_root", type=Path)
    parser.add_argument("action_json", type=Path, help="ActionBatchDecision JSON, or result.json containing action")
    parser.add_argument("environments_dir", type=Path)
    parser.add_argument("output_path", type=Path, help="new JSON artifact; existing files are never overwritten")
    args = parser.parse_args()
    action = json.loads(args.action_json.read_text())
    result = evaluate(args.packet_root, action.get("action", action), args.environments_dir, args.output_path)
    print(json.dumps({key: result[key] for key in
        ("status", "prefix_verified", "action_legal", "levels_gained", "level_completed", "state_after")}))


if __name__ == "__main__":
    main()
