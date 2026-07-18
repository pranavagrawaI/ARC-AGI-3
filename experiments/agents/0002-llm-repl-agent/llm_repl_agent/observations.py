"""Public observation transforms and model-facing run workspace artifacts."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
from PIL import Image
from arc_agi.rendering import COLOR_MAP, hex_to_rgb
from arcengine import GameAction

from experiments.harness import Observation, Transition


OBSERVATION_FILE_CONTRACT = (
    "Observation JSON files are flat objects with top-level public metadata, step, and "
    "pixels[layer][row][column]."
)


@dataclass(frozen=True)
class ColorPermutation:
    """A run-local renaming of the sixteen public display colors."""

    mapping: tuple[int, ...]

    @classmethod
    def identity(cls) -> "ColorPermutation":
        return cls(tuple(range(16)))

    @classmethod
    def from_seed(cls, seed: int) -> "ColorPermutation":
        mapping = list(range(16))
        random.Random(seed).shuffle(mapping)
        return cls(tuple(mapping))

    def apply(self, pixels: np.ndarray) -> np.ndarray:
        source = np.asarray(pixels)
        transformed = np.array(source, copy=True)
        mask = (source >= 0) & (source < len(self.mapping))
        lookup = np.asarray(self.mapping, dtype=source.dtype)
        transformed[mask] = lookup[source[mask]]
        transformed.setflags(write=False)
        return transformed


@dataclass(frozen=True)
class ModelObservation:
    step: int
    metadata: Mapping[str, Any]
    pixels: np.ndarray
    png_path: Path
    json_path: Path
    png_data_url: str
    digest: str


def action_requirements(actions: tuple[int, ...]) -> dict[str, Any]:
    requirements: dict[str, Any] = {}
    for action in actions:
        game_action = GameAction.from_id(action)
        requirements[str(action)] = (
            {"data": {"x": "integer column", "y": "integer row"}}
            if game_action.is_complex()
            else {"data": None}
        )
    return requirements


def observation_metadata(observation: Observation) -> dict[str, Any]:
    height, width = observation.pixels.shape[-2:]
    return {
        "available_actions": list(observation.available_actions),
        "action_requirements": action_requirements(observation.available_actions),
        "frame": {
            "layers": int(observation.pixels.shape[0]),
            "height": int(height),
            "width": int(width),
        },
        "levels_completed": observation.levels_completed,
        "state": observation.state,
        "full_reset": observation.full_reset,
    }


def pixels_to_png(pixels: np.ndarray, *, scale: int = 8) -> bytes:
    grid = np.asarray(pixels)[-1]
    rgb = np.zeros((*grid.shape, 3), dtype=np.uint8)
    for value in np.unique(grid):
        color = COLOR_MAP.get(int(value), "#000000")
        rgb[grid == value] = hex_to_rgb(color)
    image = Image.fromarray(rgb, mode="RGB")
    if scale != 1:
        image = image.resize(
            (image.width * scale, image.height * scale),
            resample=Image.Resampling.NEAREST,
        )
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


class ObservationStore:
    """Materialize only public, run-local observations inside the Python workspace."""

    def __init__(self, root: Path, *, colors: ColorPermutation) -> None:
        self.root = root
        self.colors = colors
        self.observations_dir = root / "observations"
        self.observations_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = root / "public_history.jsonl"
        (root / "WORKSPACE.md").write_text(
            f"This workspace is new for this run. {OBSERVATION_FILE_CONTRACT} "
            "current_observation.json contains the latest observation; observations/ "
            "contains every run observation, including the current step. current_frame.png "
            "is the latest rendered frame. public_history.jsonl contains executed actions "
            "and visible transitions. You may create any additional code, notes, models, "
            "or search artifacts here.\n",
            encoding="utf-8",
        )

    def write_observation(self, observation: Observation, *, step: int) -> ModelObservation:
        pixels = self.colors.apply(observation.pixels)
        metadata = observation_metadata(observation)
        record = {**metadata, "pixels": pixels.tolist(), "step": step}
        json_path = self.observations_dir / f"step-{step:05d}.json"
        png_path = self.observations_dir / f"step-{step:05d}.png"
        encoded = json.dumps(record, separators=(",", ":"), sort_keys=True)
        json_path.write_text(encoded + "\n", encoding="utf-8")
        png = pixels_to_png(pixels)
        png_path.write_bytes(png)
        (self.root / "current_observation.json").write_text(
            encoded + "\n", encoding="utf-8"
        )
        (self.root / "current_frame.png").write_bytes(png)
        digest = sha256(pixels.tobytes()).hexdigest()[:16]
        return ModelObservation(
            step=step,
            metadata=metadata,
            pixels=pixels,
            png_path=png_path,
            json_path=json_path,
            png_data_url="data:image/png;base64," + base64.b64encode(png).decode("ascii"),
            digest=digest,
        )

    def append_transition(
        self,
        transition: Transition,
        *,
        step: int,
        before: ModelObservation,
        after: ModelObservation,
    ) -> dict[str, Any]:
        before_grid = np.asarray(before.pixels)[-1]
        after_grid = np.asarray(after.pixels)[-1]
        comparable = before_grid.shape == after_grid.shape
        changed_pixels = int(np.count_nonzero(before_grid != after_grid)) if comparable else None
        record = {
            "step": step,
            "action": transition.decision.action,
            "action_data": transition.decision.data,
            "before_digest": before.digest,
            "after_digest": after.digest,
            "changed_pixels": changed_pixels,
            "levels_before": transition.before.levels_completed,
            "levels_after": transition.after.levels_completed,
            "state_before": transition.before.state,
            "state_after": transition.after.state,
            "full_reset": transition.after.full_reset,
            "available_actions_after": list(transition.after.available_actions),
            "before_file": f"observations/{before.json_path.name}",
            "after_file": f"observations/{after.json_path.name}",
        }
        with self.history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
        return record
