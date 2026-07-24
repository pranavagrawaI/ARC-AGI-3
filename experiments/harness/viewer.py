"""Interactive playback for ARC-AGI-3 JSONL play records."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


ARC_PALETTE = (
    "#000000",
    "#0074d9",
    "#ff4136",
    "#2ecc40",
    "#ffdc00",
    "#aaaaaa",
    "#f012be",
    "#ff851b",
    "#7fdbff",
    "#870c25",
    "#ffffff",
    "#39cccc",
    "#b10dc9",
    "#01ff70",
    "#85144b",
    "#001f3f",
)


@dataclass(frozen=True)
class FrameRecord:
    """One replayable public frame with the action that produced it."""

    index: int
    source_step: int | None
    pixels: tuple[tuple[tuple[int, ...], ...], ...]
    action: str | None
    action_data: Mapping[str, Any] | None
    state: str | None
    levels_completed: int | None
    available_actions: tuple[int, ...]
    full_reset: bool | None
    timestamp: str | None = None

    @property
    def layers(self) -> int:
        return len(self.pixels)

    @property
    def height(self) -> int:
        return len(self.pixels[0])

    @property
    def width(self) -> int:
        return len(self.pixels[0][0])


def load_play_record(path: Path) -> tuple[FrameRecord, ...]:
    """Load harness traces or raw ARC recording JSONL files."""

    frames: list[FrameRecord] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSONL record") from error
            frame = _frame_from_record(record, index=len(frames), line_number=line_number)
            if frame is not None:
                frames.append(frame)

    if not frames:
        raise ValueError(f"{path} does not contain any replayable frames")
    return tuple(frames)


def layer_grid(frame: FrameRecord, layer: int) -> tuple[tuple[int, ...], ...]:
    """Return a validated layer grid from a frame."""

    if layer < 0 or layer >= frame.layers:
        raise IndexError(f"layer {layer} is outside frame layer range 0..{frame.layers - 1}")
    return frame.pixels[layer]


def color_for_value(value: int) -> str:
    if 0 <= value < len(ARC_PALETTE):
        return ARC_PALETTE[value]
    mixed = (value * 2654435761) & 0xFFFFFF
    return f"#{mixed:06x}"


def open_viewer(frames: tuple[FrameRecord, ...], *, title: str) -> None:
    """Open a Tk playback window for already-loaded frames."""

    import tkinter as tk
    import tkinter.font as tkfont
    from tkinter import ttk

    class Viewer:
        def __init__(self) -> None:
            self.root = tk.Tk()
            self.root.title(title)
            self.root.configure(background="#f4f5f7")
            self.frame_index = 0
            self.layer_index = 0
            self.playing = False
            self.after_id: str | None = None
            self.delay_ms = 160
            self.updating_slider = False

            self.configure_style()
            first = frames[0]
            self.cell_size = max(6, min(14, 900 // max(first.width, first.height)))

            self.root.columnconfigure(0, weight=1)
            self.root.rowconfigure(1, weight=1)

            header = ttk.Frame(self.root, style="Replay.Surface.TFrame", padding=(18, 16, 18, 10))
            header.grid(row=0, column=0, sticky="ew")
            header.columnconfigure(0, weight=1)

            self.title_label = ttk.Label(
                header,
                text="ARC Replay",
                style="Replay.Title.TLabel",
            )
            self.title_label.grid(row=0, column=0, sticky="w")

            self.status_label = ttk.Label(
                header,
                text="",
                style="Replay.Status.TLabel",
            )
            self.status_label.grid(row=1, column=0, sticky="w", pady=(6, 0))

            canvas_shell = ttk.Frame(self.root, style="Replay.CanvasShell.TFrame", padding=14)
            canvas_shell.grid(row=1, column=0, sticky="nsew", padx=18)
            canvas_shell.columnconfigure(0, weight=1)
            canvas_shell.rowconfigure(0, weight=1)
            self.canvas = tk.Canvas(
                canvas_shell,
                width=first.width * self.cell_size,
                height=first.height * self.cell_size,
                background="#111827",
                highlightthickness=0,
            )
            self.canvas.grid(row=0, column=0)

            controls = ttk.Frame(self.root, style="Replay.Surface.TFrame", padding=(18, 14, 18, 16))
            controls.grid(row=2, column=0, sticky="ew")
            controls.columnconfigure(6, weight=1)

            ttk.Button(
                controls, text="Previous", command=self.previous, style="Replay.TButton"
            ).grid(row=0, column=0, padx=(0, 8), sticky="ew")
            ttk.Button(
                controls, text="Play", command=self.play, style="Replay.Primary.TButton"
            ).grid(row=0, column=1, padx=(0, 8), sticky="ew")
            ttk.Button(
                controls, text="Pause", command=self.pause, style="Replay.TButton"
            ).grid(row=0, column=2, padx=(0, 8), sticky="ew")
            ttk.Button(
                controls, text="Next", command=self.next, style="Replay.TButton"
            ).grid(row=0, column=3, padx=(0, 16), sticky="ew")
            ttk.Button(
                controls, text="Layer -", command=self.previous_layer, style="Replay.TButton"
            ).grid(row=0, column=4, padx=(0, 8), sticky="ew")
            ttk.Button(
                controls, text="Layer +", command=self.next_layer, style="Replay.TButton"
            ).grid(row=0, column=5, padx=(0, 8), sticky="ew")

            self.slider = ttk.Scale(
                controls,
                from_=0,
                to=len(frames) - 1,
                orient="horizontal",
                command=self.seek,
            )
            self.slider.grid(row=1, column=0, columnspan=7, sticky="ew", pady=(16, 0))

            self.meta = ttk.Label(
                controls,
                text="",
                style="Replay.Meta.TLabel",
                anchor="w",
                justify="left",
            )
            self.meta.grid(row=2, column=0, columnspan=7, sticky="ew", pady=(12, 0))

            self.root.bind("<Left>", lambda _: self.previous())
            self.root.bind("<Right>", lambda _: self.next())
            self.root.bind("<space>", lambda _: self.toggle())
            self.root.bind("<Prior>", lambda _: self.previous_layer())
            self.root.bind("<Next>", lambda _: self.next_layer())
            self.root.protocol("WM_DELETE_WINDOW", self.close)
            self.render()

        def configure_style(self) -> None:
            for name, size in (
                ("TkDefaultFont", 12),
                ("TkTextFont", 12),
                ("TkHeadingFont", 16),
            ):
                try:
                    tkfont.nametofont(name).configure(size=size)
                except tk.TclError:
                    pass

            style = ttk.Style(self.root)
            if "clam" in style.theme_names():
                style.theme_use("clam")
            style.configure("Replay.Surface.TFrame", background="#f4f5f7")
            style.configure("Replay.CanvasShell.TFrame", background="#d6dbe3")
            style.configure(
                "Replay.Title.TLabel",
                background="#f4f5f7",
                foreground="#111827",
                font=("TkHeadingFont", 18, "bold"),
            )
            style.configure(
                "Replay.Status.TLabel",
                background="#f4f5f7",
                foreground="#374151",
                font=("TkDefaultFont", 13),
            )
            style.configure(
                "Replay.Meta.TLabel",
                background="#f4f5f7",
                foreground="#374151",
                font=("TkDefaultFont", 12),
            )
            style.configure(
                "Replay.TButton",
                font=("TkDefaultFont", 12),
                padding=(14, 9),
            )
            style.configure(
                "Replay.Primary.TButton",
                font=("TkDefaultFont", 12, "bold"),
                padding=(18, 9),
            )
            style.configure("Horizontal.TScale", background="#f4f5f7")

        def run(self) -> None:
            self.root.mainloop()

        def close(self) -> None:
            self.pause()
            self.root.destroy()

        def toggle(self) -> None:
            if self.playing:
                self.pause()
            else:
                self.play()

        def play(self) -> None:
            if self.playing:
                return
            self.playing = True
            self.schedule_next()

        def pause(self) -> None:
            self.playing = False
            if self.after_id is not None:
                self.root.after_cancel(self.after_id)
                self.after_id = None

        def schedule_next(self) -> None:
            if self.playing:
                self.after_id = self.root.after(self.delay_ms, self.advance_playback)

        def advance_playback(self) -> None:
            self.after_id = None
            if not self.playing:
                return
            if self.frame_index >= len(frames) - 1:
                self.pause()
                return
            self.set_frame(self.frame_index + 1)
            self.schedule_next()

        def previous(self) -> None:
            self.pause()
            self.set_frame(max(0, self.frame_index - 1))

        def next(self) -> None:
            self.pause()
            self.set_frame(min(len(frames) - 1, self.frame_index + 1))

        def previous_layer(self) -> None:
            self.layer_index = max(0, self.layer_index - 1)
            self.render()

        def next_layer(self) -> None:
            self.layer_index = min(frames[self.frame_index].layers - 1, self.layer_index + 1)
            self.render()

        def seek(self, raw_value: str) -> None:
            if self.updating_slider:
                return
            self.pause()
            self.set_frame(round(float(raw_value)))

        def set_frame(self, index: int) -> None:
            index = max(0, min(len(frames) - 1, index))
            self.frame_index = index
            self.layer_index = min(self.layer_index, frames[index].layers - 1)
            self.updating_slider = True
            try:
                self.slider.set(index)
            finally:
                self.updating_slider = False
            self.render()

        def render(self) -> None:
            frame = frames[self.frame_index]
            grid = layer_grid(frame, self.layer_index)
            self.canvas.delete("cell")
            size = self.cell_size
            for y, row in enumerate(grid):
                for x, value in enumerate(row):
                    self.canvas.create_rectangle(
                        x * size,
                        y * size,
                        (x + 1) * size,
                        (y + 1) * size,
                        fill=color_for_value(value),
                        outline="",
                        tags="cell",
                    )
            self.status_label.configure(text=_status_text(frame, self.frame_index, len(frames)))
            self.meta.configure(
                text=_metadata_text(frame, self.frame_index, len(frames), self.layer_index)
            )

    Viewer().run()


def _status_text(frame: FrameRecord, frame_index: int, frame_count: int) -> str:
    action = f"action {frame.action}" if frame.action is not None else "no action"
    state = frame.state if frame.state is not None else "unknown state"
    levels = (
        f"{frame.levels_completed} levels"
        if frame.levels_completed is not None
        else "unknown levels"
    )
    return f"Frame {frame_index + 1} of {frame_count} | {action} | {state} | {levels}"


def _metadata_text(
    frame: FrameRecord, frame_index: int, frame_count: int, layer_index: int
) -> str:
    parts = [
        f"Frame {frame_index + 1}/{frame_count}",
        f"layer {layer_index + 1}/{frame.layers}",
        f"grid {frame.width}x{frame.height}",
    ]
    if frame.source_step is not None:
        parts.append(f"step {frame.source_step}")
    if frame.action is not None:
        parts.append(f"action {frame.action}")
    if frame.state is not None:
        parts.append(f"state {frame.state}")
    if frame.levels_completed is not None:
        parts.append(f"levels {frame.levels_completed}")
    if frame.available_actions:
        parts.append(f"available {list(frame.available_actions)}")
    if frame.full_reset:
        parts.append("full reset")
    if frame.timestamp is not None:
        parts.append(frame.timestamp)
    return " | ".join(parts)


def _frame_from_record(
    record: Mapping[str, Any], *, index: int, line_number: int
) -> FrameRecord | None:
    record_type = record.get("type")
    if record_type == "end":
        return None
    if record_type == "observation":
        return _frame_from_payload(
            record,
            index=index,
            line_number=line_number,
            source_step=_optional_int(record.get("step"), "step", line_number),
            action="START",
            action_data=None,
            timestamp=None,
        )
    if record_type == "transition":
        observation = record.get("observation")
        if not isinstance(observation, Mapping):
            raise ValueError(f"line {line_number}: transition record is missing observation")
        return _frame_from_payload(
            observation,
            index=index,
            line_number=line_number,
            source_step=_optional_int(record.get("step"), "step", line_number),
            action=str(record["action"]) if "action" in record else None,
            action_data=_optional_mapping(record.get("action_data"), "action_data", line_number),
            timestamp=None,
        )

    data = record.get("data")
    if isinstance(data, Mapping) and "frame" in data:
        action_input = data.get("action_input")
        action = None
        action_data = None
        if isinstance(action_input, Mapping):
            raw_action = action_input.get("id")
            action = str(raw_action) if raw_action is not None else None
            action_data = _optional_mapping(action_input.get("data"), "action_input.data", line_number)
        return _frame_from_payload(
            data,
            index=index,
            line_number=line_number,
            source_step=None,
            action=action,
            action_data=action_data,
            timestamp=str(record["timestamp"]) if "timestamp" in record else None,
        )

    return None


def _frame_from_payload(
    payload: Mapping[str, Any],
    *,
    index: int,
    line_number: int,
    source_step: int | None,
    action: str | None,
    action_data: Mapping[str, Any] | None,
    timestamp: str | None,
) -> FrameRecord:
    raw_pixels = payload.get("pixels", payload.get("frame"))
    if raw_pixels is None:
        raise ValueError(f"line {line_number}: frame payload is missing pixels/frame")
    return FrameRecord(
        index=index,
        source_step=source_step,
        pixels=_normalize_pixels(raw_pixels, line_number=line_number),
        action=action,
        action_data=action_data,
        state=str(payload["state"]) if "state" in payload else None,
        levels_completed=_optional_int(
            payload.get("levels_completed"), "levels_completed", line_number
        ),
        available_actions=_normalize_actions(payload.get("available_actions"), line_number),
        full_reset=_optional_bool(payload.get("full_reset"), "full_reset", line_number),
        timestamp=timestamp,
    )


def _normalize_pixels(raw_pixels: Any, *, line_number: int) -> tuple[tuple[tuple[int, ...], ...], ...]:
    if not isinstance(raw_pixels, list) or not raw_pixels:
        raise ValueError(f"line {line_number}: pixels/frame must be a non-empty list")
    if _is_2d_grid(raw_pixels):
        raw_layers = [raw_pixels]
    elif all(_is_2d_grid(layer) for layer in raw_pixels):
        raw_layers = raw_pixels
    else:
        raise ValueError(f"line {line_number}: pixels/frame must be a 2D grid or layer stack")

    layers = tuple(_normalize_grid(layer, line_number=line_number) for layer in raw_layers)
    shapes = {(len(layer), len(layer[0])) for layer in layers}
    if len(shapes) != 1:
        raise ValueError(f"line {line_number}: all frame layers must share one shape")
    return layers


def _normalize_grid(raw_grid: list[Any], *, line_number: int) -> tuple[tuple[int, ...], ...]:
    if not raw_grid:
        raise ValueError(f"line {line_number}: frame grid is empty")
    width = None
    rows = []
    for row in raw_grid:
        if not isinstance(row, list) or not row:
            raise ValueError(f"line {line_number}: frame rows must be non-empty lists")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError(f"line {line_number}: frame rows must be rectangular")
        try:
            rows.append(tuple(int(value) for value in row))
        except (TypeError, ValueError) as error:
            raise ValueError(f"line {line_number}: frame cells must be integers") from error
    return tuple(rows)


def _normalize_actions(raw_actions: Any, line_number: int) -> tuple[int, ...]:
    if raw_actions is None:
        return ()
    if not isinstance(raw_actions, Iterable) or isinstance(raw_actions, (str, bytes)):
        raise ValueError(f"line {line_number}: available_actions must be a list")
    try:
        return tuple(int(action) for action in raw_actions)
    except (TypeError, ValueError) as error:
        raise ValueError(f"line {line_number}: available_actions must contain integers") from error


def _is_2d_grid(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(row, list) for row in value)
        and all(not any(isinstance(cell, list) for cell in row) for row in value)
    )


def _optional_int(value: Any, field_name: str, line_number: int) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"line {line_number}: {field_name} must be an integer") from error


def _optional_bool(value: Any, field_name: str, line_number: int) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise ValueError(f"line {line_number}: {field_name} must be a boolean")


def _optional_mapping(value: Any, field_name: str, line_number: int) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value
    raise ValueError(f"line {line_number}: {field_name} must be an object")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path, help="JSONL play record to inspect")
    args = parser.parse_args(argv)

    frames = load_play_record(args.record)
    open_viewer(frames, title=f"ARC replay: {args.record.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
