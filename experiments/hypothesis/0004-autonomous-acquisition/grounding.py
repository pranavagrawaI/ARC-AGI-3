"""Public-frame grounding for the closed-loop planning ceiling."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
SUBSTRATE = REPO / "experiments/hypothesis/0001-inverted-authority"
RECORDING = REPO / "recordings/ae20a3e4-ca6e-4f29-b463-042fd0dd6710/ls20-9607627b-b7f0efe3-1ae2-4f66-ad6e-178cf992b58f.jsonl"
SOURCE_SHA256 = "434bd2f66b5e77b6aaa3bc1ed9cb0d5e28b0070ccaf93c5ebb077eacec5bf5b4"
PALETTE = [(0,0,0),(0,116,217),(255,65,54),(46,204,64),(255,220,0),(170,170,170),(240,18,190),(255,133,27),(127,219,255),(135,12,37),(80,80,80),(180,180,255),(255,180,180),(180,255,180),(255,255,180),(220,220,220)]
ACTION_DELTAS = {1: (-5, 0), 2: (5, 0), 3: (0, -5), 4: (0, 5)}


@dataclass(frozen=True)
class PublicGrounder:
    controlled_area: int
    controlled_colours: tuple[tuple[int, int], ...]
    resource_colour: int
    resource_row_band: tuple[int, int]
    resource_usual_delta: int
    target_cell: tuple[int, int] = (50, 54)
    trigger_cell: tuple[int, int] = (10, 49)

    @classmethod
    def from_frozen_prefix(cls) -> "PublicGrounder":
        if sha256(RECORDING.read_bytes()).hexdigest() != SOURCE_SHA256:
            raise RuntimeError("grounding recording drift")
        sys.path.insert(0, str(SUBSTRATE))
        from digest_probe import digest, level_slice, load  # noqa: PLC0415

        prefix = digest(level_slice(load(RECORDING.parent), 2), "colour", horizon=24)
        controlled = prefix["_tracks"]["ctrl"]
        if controlled is None:
            raise RuntimeError("prefix has no controlled track")
        controlled_sprite = controlled.sprites[23]
        if len(prefix["resources"]) != 1:
            raise RuntimeError("prefix resource track is not unique")
        resource = prefix["resources"][0]
        resource_track = next(t for t in prefix["_tracks"]["all"] if t.tid == resource["tid"])
        resource_sprite = resource_track.sprites[23]
        if len(resource_sprite.colors) != 1:
            raise RuntimeError("resource track is not single-colour")
        return cls(
            controlled_area=controlled_sprite.area,
            controlled_colours=controlled_sprite.colors,
            resource_colour=resource_sprite.colors[0][0],
            resource_row_band=(resource_sprite.bbox[0], resource_sprite.bbox[1]),
            resource_usual_delta=resource["step"],
        )

    def observe(
        self,
        grid: Any,
        *,
        previous: dict[str, Any] | None = None,
        action: int | None = None,
    ) -> dict[str, Any]:
        sys.path.insert(0, str(SUBSTRATE))
        from digest_probe import segment  # noqa: PLC0415

        values = np.asarray(grid)
        candidates = [
            sprite for sprite in segment(values, "colour")
            if sprite.area == self.controlled_area and sprite.colors == self.controlled_colours
        ]
        controlled = candidates[0] if len(candidates) == 1 else None
        resource_candidates = [
            sprite for sprite in segment(values, "colour")
            if len(sprite.colors) == 1
            and sprite.colors[0][0] == self.resource_colour
            and sprite.bbox[0] >= self.resource_row_band[0]
        ]
        resource = resource_candidates[0] if len(resource_candidates) == 1 else None
        cell = [controlled.bbox[0], controlled.bbox[2]] if controlled else None
        displacement = None
        expected = list(ACTION_DELTAS[action]) if action is not None else None
        transition = "initial_observation"
        if previous is not None and action is not None and cell is not None and previous.get("controlled_cell") is not None:
            displacement = [cell[i] - previous["controlled_cell"][i] for i in range(2)]
            transition = (
                "matched_action_model" if displacement == expected
                else "blocked_or_no_translation" if displacement == [0, 0]
                else "unexpected_transition"
            )
        resource_area = resource.area if resource else None
        resource_delta = None
        resource_event = "initial_observation"
        if previous is not None and resource_area is not None and previous.get("resource_area") is not None:
            resource_delta = resource_area - previous["resource_area"]
            resource_event = (
                "usual_step" if resource_delta == self.resource_usual_delta
                else "refill_or_discontinuity" if resource_delta * self.resource_usual_delta < 0
                else "irregular_or_unchanged"
            )
        return {
            "schema": "0004-public-grounding-v1",
            "controlled_track_status": "unique" if controlled else "ambiguous_or_lost",
            "controlled_candidate_count": len(candidates),
            "controlled_cell": cell,
            "controlled_bbox": list(controlled.bbox) if controlled else None,
            "requested_action": action,
            "expected_displacement": expected,
            "observed_displacement": displacement,
            "transition_class": transition,
            "resource_track_status": "unique" if resource else "ambiguous_or_lost",
            "resource_candidate_count": len(resource_candidates),
            "resource_area": resource_area,
            "resource_bbox": list(resource.bbox) if resource else None,
            "resource_usual_delta": self.resource_usual_delta,
            "resource_observed_delta": resource_delta,
            "resource_event": resource_event,
            "target_cell": list(self.target_cell),
            "trigger_cell": list(self.trigger_cell),
            "provenance": "current and prior public frames plus prefix-derived track identities",
        }

    def overlay(self, grid: Any, observation: dict[str, Any]) -> bytes:
        values = np.asarray(grid)
        image = Image.new("RGB", (values.shape[1] * 5, values.shape[0] * 5))
        pixels = image.load()
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                rgb = PALETTE[int(values[row, column]) % len(PALETTE)]
                for y in range(row * 5, row * 5 + 5):
                    for x in range(column * 5, column * 5 + 5):
                        pixels[x, y] = rgb
        draw = ImageDraw.Draw(image)

        def box(values: list[int] | None, colour: tuple[int, int, int]) -> None:
            if values is None:
                return
            r0, r1, c0, c1 = values
            draw.rectangle((c0 * 5, r0 * 5, c1 * 5 - 1, r1 * 5 - 1), outline=colour, width=3)

        box(observation.get("controlled_bbox"), (255, 0, 255))
        tr, tc = self.trigger_cell
        box([tr, tr + 5, tc, tc + 5], (255, 255, 0))
        rr, rc = self.target_cell
        box([rr, rr + 5, rc, rc + 5], (0, 255, 255))
        box(observation.get("resource_bbox"), (0, 128, 255))
        stream = BytesIO()
        image.save(stream, format="PNG")
        return stream.getvalue()


def grounding_digest(value: dict[str, Any]) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
