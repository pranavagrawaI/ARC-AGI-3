"""Freeze a prefix-only LS20 planning checkpoint with no route suffix."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
SUBSTRATE = REPO / "experiments/hypothesis/0001-inverted-authority"
RECORDING = REPO / "recordings/ae20a3e4-ca6e-4f29-b463-042fd0dd6710/ls20-9607627b-b7f0efe3-1ae2-4f66-ad6e-178cf992b58f.jsonl"
SOURCE_SHA256 = "434bd2f66b5e77b6aaa3bc1ed9cb0d5e28b0070ccaf93c5ebb077eacec5bf5b4"
PALETTE = [(0,0,0),(0,116,217),(255,65,54),(46,204,64),(255,220,0),(170,170,170),(240,18,190),(255,133,27),(127,219,255),(135,12,37),(80,80,80),(180,180,255),(255,180,180),(180,255,180),(255,255,180),(220,220,220)]


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b"\n")


def render(grid: Any, path: Path) -> list[int]:
    image = Image.new("RGB", (320, 320))
    pixels = image.load()
    for row, values in enumerate(grid):
        for column, color in enumerate(values):
            rgb = PALETTE[int(color) % len(PALETTE)]
            for y in range(row * 5, row * 5 + 5):
                for x in range(column * 5, column * 5 + 5):
                    pixels[x, y] = rgb
    image.save(path)
    return list(image.size)


def main() -> int:
    if sha256(RECORDING.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("recording drift")
    sys.path.insert(0, str(SUBSTRATE))
    from digest_probe import digest, level_slice, load  # noqa: PLC0415

    observations = level_slice(load(RECORDING.parent), 2)
    horizon = 24
    prefix = digest(observations, "colour", horizon=horizon)
    tracks = {track.tid: track for track in prefix["_tracks"]["all"]}
    objective = json.loads((ROOT / "fixtures/case-pruned.json").read_text())["candidate_hypotheses"][0]
    image_dir = ROOT / "fixtures/images/planning-l2-prefix"
    image_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for index in range(horizon - 4, horizon):
        path = image_dir / f"observation-{index + 1}.png"
        size = render(observations[index].grid, path)
        images.append({
            "observation": index + 1,
            "action_that_produced_frame": observations[index].action,
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "size": size,
        })
    positions = {}
    for name, track_id in {
        "controlled": prefix["controlled"]["tid"],
        "target": objective["invariant_track"],
        "workspace": objective["mutable_track"],
    }.items():
        sprite = tracks[track_id].sprites[horizon - 1]
        positions[name] = {"track_id": track_id, "bbox": list(sprite.bbox), "centroid": list(sprite.centroid)}
    case = {
        "schema": "0004-planning-case-v1",
        "case_id": "ls20-prefix24-acquired-objective-planning",
        "authority": "prefix_observations_plus_acquired_objective_no_route",
        "images": images,
        "legal_actions": [1, 2, 3, 4],
        "action_model": prefix["controlled"]["action_model"],
        "tracked_positions": positions,
        "acquired_objective": objective,
        "instruction": "Generate a route of at most 64 actions expected to complete one level.",
    }
    label = {
        "schema": "0004-planning-label-v1",
        "case_id": case["case_id"],
        "offline_level_index": 2,
        "checkpoint_prefix_actions": [int(step.action.removeprefix("ACTION")) for step in observations[1:horizon]],
        "expected_current_grid_sha256": sha256(canonical(observations[horizon - 1].grid.tolist())).hexdigest(),
    }
    write(ROOT / "fixtures/planning-case.json", case)
    write(ROOT / "scorer/planning-label.json", label)
    print(f"planning checkpoint: observations={horizon} prefix_actions={len(label['checkpoint_prefix_actions'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
