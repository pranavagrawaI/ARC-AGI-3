"""Blind human play harness for ARC-AGI-3 games.

The tool intentionally uses only the public frame/action interface. It records
an admissible JSONL trace, a raw ARC recording, a scorecard snapshot, and a notes
template so first-contact human confusion can become research data.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import arc_agi
from arc_agi.base import OperationMode
from arc_agi.rendering import COLOR_MAP, hex_to_rgb
from arcengine import GameAction

from experiments.harness.runner import public_observation
from experiments.harness.trace import JsonlTraceWriter
from experiments.harness.types import Decision, Transition


RGB = {int(value): hex_to_rgb(color) for value, color in COLOR_MAP.items()}
COLOR_HEX = {int(value): color for value, color in COLOR_MAP.items()}
DEFAULT_GAMES = ("ar25", "cd82", "ls20", "tu93", "m0r0", "bp35", "dc22", "lf52")


@dataclass
class PlayStats:
    actions: int = 0
    reset_actions: int = 0
    level_actions: int = 0
    previous_levels_completed: int = 0


@dataclass(frozen=True)
class FinishResult:
    actions: int
    final: Any
    stop_reason: str


def quiet_arc_logging() -> None:
    for name in ("arc_agi", "arc_agi.scorecard"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.WARNING)
        logger.propagate = False
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())


def x_axis(width: int) -> tuple[str, str]:
    tens = "".join(str((x // 10) % 10) if x >= 10 else " " for x in range(width))
    ones = "".join(str(x % 10) for x in range(width))
    return tens, ones


def compact_render(frame_data: Any, *, show_coords: bool = False) -> str:
    """Render the final settled 64x64 frame into 64x32 terminal cells."""

    if not frame_data.frame:
        return ""
    grid = np.asarray(frame_data.frame[-1])
    height, width = grid.shape
    lines: list[str] = []
    if show_coords:
        tens, ones = x_axis(width)
        lines.append(f"      {tens}")
        lines.append(f"      {ones}")
    for y in range(0, height, 2):
        parts: list[str] = []
        for x in range(width):
            tr, tg, tb = RGB.get(int(grid[y, x]), (0, 0, 0))
            if y + 1 < height:
                br, bg, bb = RGB.get(int(grid[y + 1, x]), (0, 0, 0))
            else:
                br, bg, bb = (0, 0, 0)
            parts.append(f"\033[38;2;{tr};{tg};{tb}m\033[48;2;{br};{bg};{bb}m▀")
        prefix = f"{y:02d}/{min(y + 1, height - 1):02d} " if show_coords else ""
        lines.append(prefix + "".join(parts) + "\033[0m")
    return "\n".join(lines)


def clear_and_render(frame_data: Any, *, enabled: bool, show_coords: bool) -> None:
    if not enabled:
        return
    sys.stdout.write("\033[H\033[2J")
    sys.stdout.write(compact_render(frame_data, show_coords=show_coords))
    sys.stdout.write("\n")
    sys.stdout.flush()


def status_line(frame_data: Any, stats: PlayStats) -> str:
    level = min(int(frame_data.levels_completed) + 1, int(frame_data.win_levels))
    actions = ",".join(str(action) for action in frame_data.available_actions)
    return (
        f"state={frame_data.state.value}  "
        f"level={level}/{frame_data.win_levels}  "
        f"completed={frame_data.levels_completed}  "
        f"level_actions={stats.level_actions}  "
        f"total_actions={stats.actions}  "
        f"available=[{actions}]"
    )


def action_help(available_actions: tuple[int, ...]) -> str:
    pieces = [str(action) for action in available_actions]
    if 6 in available_actions:
        pieces.append("click <x> <y>")
    pieces.append("r/reset")
    pieces.extend(["c/coords", "i/inspect", "n <note>", "h/help", "q/quit"])
    return "commands: " + ", ".join(pieces)


def parse_command(raw: str, available_actions: tuple[int, ...]) -> tuple[int, dict[str, Any] | None] | str:
    """Parse one human command into an ARC action or a control command."""

    command = raw.strip()
    lowered = command.lower()
    if lowered in {"q", "quit", "exit"}:
        return "quit"
    if lowered in {"h", "help", "?"}:
        return "help"
    if lowered.startswith("n "):
        return "note"
    if lowered in {"s", "status"}:
        return "status"
    if lowered in {"c", "coords", "coordinates"}:
        return "coords"
    if lowered in {"i", "inspect", "components", "tiles", "palette"}:
        return "inspect"
    if lowered in {"r", "reset"}:
        return (0, None)

    parts = command.split()
    if not parts:
        return "empty"

    action_token = parts[0].upper()
    if action_token in {"CLICK", "SELECT"}:
        action_id = 6
        parts = ["6", *parts[1:]]
    elif action_token.startswith("ACTION"):
        action_id = int(action_token.removeprefix("ACTION"))
    else:
        action_id = int(action_token)

    data: dict[str, Any] | None = None
    if action_id == 6:
        if len(parts) != 3:
            raise ValueError("ACTION6 requires coordinates: 6 <x> <y>")
        data = {"x": int(parts[1]), "y": int(parts[2])}
    elif len(parts) != 1:
        raise ValueError("Only ACTION6 accepts coordinates.")

    if action_id not in available_actions:
        raise ValueError(f"action {action_id} is not available; available={available_actions}")
    return (action_id, data)


def visible_frame_changed(before: Any, after: Any) -> bool:
    return not np.array_equal(before.pixels, after.pixels)


def component_rows(frame_data: Any, *, max_components: int = 30) -> list[str]:
    """Summarize visible same-color components with public display coordinates."""

    if not frame_data.frame:
        return ["no frame available"]
    grid = np.asarray(frame_data.frame[-1])
    height, width = grid.shape
    visited = np.zeros((height, width), dtype=bool)
    components: list[dict[str, Any]] = []

    for y0 in range(height):
        for x0 in range(width):
            if visited[y0, x0]:
                continue
            color = int(grid[y0, x0])
            visited[y0, x0] = True
            if RGB.get(color) == (0, 0, 0):
                continue

            stack = [(x0, y0)]
            cells: list[tuple[int, int]] = []
            while stack:
                x, y = stack.pop()
                cells.append((x, y))
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if nx < 0 or ny < 0 or nx >= width or ny >= height:
                        continue
                    if visited[ny, nx] or int(grid[ny, nx]) != color:
                        continue
                    visited[ny, nx] = True
                    stack.append((nx, ny))

            xs = [x for x, _ in cells]
            ys = [y for _, y in cells]
            mean_x = sum(xs) / len(xs)
            mean_y = sum(ys) / len(ys)
            click_x, click_y = min(
                cells,
                key=lambda cell: (cell[0] - mean_x) ** 2 + (cell[1] - mean_y) ** 2,
            )
            components.append(
                {
                    "color": color,
                    "hex": COLOR_HEX.get(color, "?"),
                    "area": len(cells),
                    "x_min": min(xs),
                    "x_max": max(xs),
                    "y_min": min(ys),
                    "y_max": max(ys),
                    "click_x": click_x,
                    "click_y": click_y,
                }
            )

    components.sort(key=lambda item: (item["y_min"], item["x_min"], -item["area"]))
    rows = [
        "visible components: color hex area bbox -> click command",
        "click coordinates are public 64x64 display pixels",
    ]
    for item in components[:max_components]:
        rows.append(
            f"color={item['color']:>2} {item['hex']:<7} "
            f"area={item['area']:>4} "
            f"x={item['x_min']:02d}..{item['x_max']:02d} "
            f"y={item['y_min']:02d}..{item['y_max']:02d} "
            f"-> click {item['click_x']} {item['click_y']}"
        )
    if len(components) > max_components:
        rows.append(f"... {len(components) - max_components} more components not shown")
    return rows


def write_notes_template(path: Path, *, game: str, started_at: str, protocol: str) -> None:
    path.write_text(
        "\n".join(
            [
                f"# Blind Human Play Notes: {game}",
                "",
                f"- started_at: {started_at}",
                f"- protocol: {protocol}",
                "",
                "## First-Contact Rules",
                "",
                "- Do not inspect Codex world-model files for this game before playing.",
                "- Record what you noticed before theory, not only what you understood after.",
                "- Mark any action that was a deliberate probe.",
                "",
                "## Level Notes",
                "",
                "For each level/failure, fill:",
                "",
                "- What did I think the objects were?",
                "- What did I test?",
                "- What surprised me?",
                "- What did I notice too late?",
                "- What predicate or probe would have saved actions?",
                "",
                "## Live Notes",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def append_note(path: Path, *, frame_data: Any, stats: PlayStats, text: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    level = min(int(frame_data.levels_completed) + 1, int(frame_data.win_levels))
    with path.open("a", encoding="utf-8") as notes:
        notes.write(
            f"- {timestamp} | level {level}/{frame_data.win_levels} | "
            f"total_actions={stats.actions} | level_actions={stats.level_actions}: {text.strip()}\n"
        )


def jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return jsonable(value.dict())
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def level_score(baseline_actions: list[int] | None, completed_before: int, level_actions: int) -> float | None:
    if not baseline_actions or completed_before >= len(baseline_actions) or level_actions <= 0:
        return None
    return min((baseline_actions[completed_before] / level_actions) ** 2, 1.15)


def operation_mode(value: str) -> OperationMode:
    try:
        return OperationMode(value.lower())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "mode must be one of: normal, online, offline, competition"
        ) from error


def default_run_dir(root: Path, game: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / game / stamp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play an ARC-AGI-3 game blind and record the trace.")
    parser.add_argument("--game", help="Game id, e.g. ls20, ar25, cd82.")
    parser.add_argument(
        "--mode",
        type=operation_mode,
        default=OperationMode.NORMAL,
        help="ARC operation mode: normal, online, offline, competition. Default: normal.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=5000)
    parser.add_argument("--output-root", type=Path, default=Path("human-play"))
    parser.add_argument("--no-render", action="store_true", help="Do not draw frames in the terminal.")
    parser.add_argument(
        "--show-coords",
        action="store_true",
        help="Show display-pixel coordinate rulers around the terminal frame.",
    )
    parser.add_argument(
        "--protocol",
        default="blind-first-contact",
        help="Short label stored in notes.md and summary.json.",
    )
    parser.add_argument(
        "--list-suggested-games",
        action="store_true",
        help="Print the initial multi-game suite and exit.",
    )
    parser.add_argument(
        "--list-local-games",
        action="store_true",
        help="Print games currently available under environment_files/ and exit.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    quiet_arc_logging()

    if args.list_suggested_games:
        print("\n".join(DEFAULT_GAMES))
        return 0

    if args.list_local_games:
        arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE, logger=logging.getLogger("play_game"))
        for environment in sorted(arc.available_environments, key=lambda item: item.game_id):
            title = f" ({environment.title})" if environment.title else ""
            print(f"{environment.game_id}{title}")
        return 0

    if not args.game:
        raise ValueError("--game is required unless using --list-suggested-games or --list-local-games")

    run_dir = default_run_dir(args.output_root, args.game)
    run_dir.mkdir(parents=True, exist_ok=False)
    notes_path = run_dir / "notes.md"
    trace_path = run_dir / "trace.jsonl"
    summary_path = run_dir / "summary.json"
    scorecard_path = run_dir / "scorecard.json"
    recordings_dir = run_dir / "raw-recordings"
    started_at = datetime.now(timezone.utc).isoformat()
    write_notes_template(notes_path, game=args.game, started_at=started_at, protocol=args.protocol)

    logger = logging.getLogger("play_game")
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler(sys.stderr))
    logger.setLevel(logging.WARNING)

    arc = arc_agi.Arcade(
        operation_mode=args.mode,
        recordings_dir=str(recordings_dir),
        logger=logger,
    )
    env = arc.make(args.game, seed=args.seed, save_recording=True)
    if env is None:
        raise RuntimeError(f"could not create environment for game {args.game!r}")

    baseline_actions = getattr(env.info, "baseline_actions", None)
    frame = env.reset()
    if frame is None:
        raise RuntimeError(f"initial reset failed for game {args.game!r}")

    stats = PlayStats(previous_levels_completed=int(frame.levels_completed))
    current = public_observation(frame)
    show_coords = bool(args.show_coords)

    with JsonlTraceWriter(trace_path) as trace:
        trace.start(current)
        clear_and_render(frame, enabled=not args.no_render, show_coords=show_coords)
        print(f"Game: {frame.game_id} | levels to win: {frame.win_levels}")
        print(f"Output: {run_dir.resolve()}")
        print(action_help(tuple(int(action) for action in frame.available_actions)))
        print(status_line(frame, stats))

        stop_reason = "user_quit"
        while True:
            if current.state in {"WIN", "GAME_OVER"}:
                stop_reason = "terminal"
                break
            if stats.actions >= args.max_actions:
                stop_reason = "action_budget"
                break

            raw = input("action> ")
            if raw.strip().lower().startswith("n "):
                append_note(notes_path, frame_data=frame, stats=stats, text=raw.strip()[2:])
                print(f"note appended: {notes_path}")
                continue

            try:
                parsed = parse_command(raw, current.available_actions)
            except ValueError as error:
                print(f"invalid command: {error}")
                continue

            if parsed == "quit":
                break
            if parsed == "help":
                print(action_help(current.available_actions))
                print("ACTION6 syntax: 6 <x> <y> or click <x> <y>")
                print("Toggle coordinate rulers with c/coords.")
                print("List visible component click targets with i/inspect.")
                continue
            if parsed == "status":
                print(status_line(frame, stats))
                continue
            if parsed == "coords":
                show_coords = not show_coords
                clear_and_render(frame, enabled=not args.no_render, show_coords=show_coords)
                print(f"coordinate rulers {'on' if show_coords else 'off'}")
                print(status_line(frame, stats))
                continue
            if parsed == "inspect":
                print("\n".join(component_rows(frame)))
                continue
            if parsed == "empty" or parsed == "note":
                continue

            action_id, action_data = parsed
            before = current
            decision = Decision(action=action_id, data=action_data)
            if action_id == 0:
                frame = env.reset()
                stats.reset_actions += 1
            else:
                frame = env.step(GameAction.from_id(action_id), data=action_data)
            if frame is None:
                raise RuntimeError(f"environment returned no frame after action {action_id}")

            stats.actions += 1
            stats.level_actions += 1
            current = public_observation(frame)
            transition = Transition(before=before, decision=decision, after=current)
            trace.append(stats.actions, transition)

            clear_and_render(frame, enabled=not args.no_render, show_coords=show_coords)

            completed_now = int(frame.levels_completed)
            if completed_now > stats.previous_levels_completed:
                score = level_score(
                    baseline_actions,
                    stats.previous_levels_completed,
                    stats.level_actions,
                )
                score_text = "unknown" if score is None else f"{score:.3f}"
                print(
                    f"*** LEVEL {stats.previous_levels_completed + 1} COMPLETE: "
                    f"{stats.level_actions} actions, score={score_text} ***"
                )
                stats.previous_levels_completed = completed_now
                stats.level_actions = 0

            if action_id != 0 and not visible_frame_changed(before, current):
                print(f"no visible frame change after action {action_id}")
            print(status_line(frame, stats))

        # JsonlTraceWriter.finish only reads actions, final, and stop_reason.
        trace.finish(FinishResult(actions=stats.actions, final=current, stop_reason=stop_reason))  # type: ignore[arg-type]

    finished_at = datetime.now(timezone.utc).isoformat()
    scorecard = arc.get_scorecard()
    save_json(scorecard_path, scorecard)
    save_json(
        summary_path,
        {
            "game": args.game,
            "started_at": started_at,
            "finished_at": finished_at,
            "protocol": args.protocol,
            "mode": args.mode.value,
            "seed": args.seed,
            "actions": stats.actions,
            "reset_actions": stats.reset_actions,
            "levels_completed": current.levels_completed,
            "state": current.state,
            "stop_reason": stop_reason,
            "trace": str(trace_path),
            "notes": str(notes_path),
            "scorecard": str(scorecard_path),
            "raw_recordings_dir": str(recordings_dir),
            "baseline_actions": baseline_actions,
        },
    )

    print("\nRun saved:")
    print(f"  notes: {notes_path.resolve()}")
    print(f"  trace: {trace_path.resolve()}")
    print(f"  scorecard: {scorecard_path.resolve()}")
    print(f"  raw recordings: {recordings_dir.resolve()}")
    print(f"  summary: {summary_path.resolve()}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
