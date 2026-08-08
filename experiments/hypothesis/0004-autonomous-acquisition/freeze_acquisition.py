"""Freeze prefix-only acquisition candidates and scorer-only terminal labels."""

from __future__ import annotations

from hashlib import sha256
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
SUBSTRATE = REPO / "experiments/hypothesis/0001-inverted-authority"
SELECTOR = SUBSTRATE / "select_probe.py"
DIGEST = SUBSTRATE / "digest_probe.py"
RECORDING = REPO / "recordings/ae20a3e4-ca6e-4f29-b463-042fd0dd6710/ls20-9607627b-b7f0efe3-1ae2-4f66-ad6e-178cf992b58f.jsonl"
EXPECTED = {
    SELECTOR: "8164eaee3f3e0b0c35d5660bdce30ee5ad6c11d0cb4a19e005ddf8a712f46b67",
    DIGEST: "913dd8dac1845d53c14e20543acacd869570f0b936ca656e7fc0b51fbec93d1a",
    RECORDING: "434bd2f66b5e77b6aaa3bc1ed9cb0d5e28b0070ccaf93c5ebb077eacec5bf5b4",
}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def write(path: Path, value: Any) -> None:
    path.write_bytes(canonical(value) + b"\n")


def verify_sources() -> None:
    for path, expected in EXPECTED.items():
        actual = sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"source drift: {path}: {actual}")


def patch(sprite: Any, frames: dict[int, Any], frame: int) -> list[list[int]]:
    value = sprite.sprites[frame].paint(frames[frame])
    return value.tolist() if hasattr(value, "tolist") else value


def scale_candidates(candidates: list[tuple[Any, Any]]) -> list[tuple[int, Any, Any, int]]:
    rows = []
    for pair_id, (invariant, mutable) in enumerate(candidates):
        left = invariant.sprites[min(invariant.sprites)]
        right = mutable.sprites[min(mutable.sprites)]
        for factor in range(2, 6):
            if (
                right.extent == (left.extent[0] * factor, left.extent[1] * factor)
                and right.area == left.area * factor * factor
            ):
                rows.append((pair_id, invariant, mutable, factor))
                break
    return rows


def change_events(track: Any) -> set[int]:
    return {
        frame for frame in track.sigs
        if frame - 1 in track.sigs and track.sigs[frame] != track.sigs[frame - 1]
    }


def strict_synchronized_subtrack(child: Any, parent: Any) -> bool:
    if set(child.sprites) != set(parent.sprites) or change_events(child) != change_events(parent):
        return False
    strict = False
    for frame in child.sprites:
        child_box = child.sprites[frame].bbox
        parent_box = parent.sprites[frame].bbox
        if not (
            parent_box[0] <= child_box[0] <= child_box[1] <= parent_box[1]
            and parent_box[2] <= child_box[2] <= child_box[3] <= parent_box[3]
        ):
            return False
        strict = strict or child_box != parent_box
    return strict


def prune_derived_mutable_tracks(
    rows: list[tuple[int, Any, Any, int]],
) -> tuple[list[tuple[int, Any, Any, int]], set[int]]:
    mutable_tracks = {mutable.tid: mutable for _, _, mutable, _ in rows}
    derived = {
        track_id
        for track_id, child in mutable_tracks.items()
        if any(
            track_id != parent_id and strict_synchronized_subtrack(child, parent)
            for parent_id, parent in mutable_tracks.items()
        )
    }
    return [row for row in rows if row[2].tid not in derived], derived


def main() -> int:
    verify_sources()
    sys.path.insert(0, str(SUBSTRATE))
    import select_probe as selector  # noqa: PLC0415
    from digest_probe import digest, level_slice, lenses  # noqa: PLC0415

    steps = selector.load(RECORDING.parent)
    transfer = []
    for level in range(1, 5):
        observations = level_slice(steps, level)
        full = digest(observations, "colour", horizon=len(observations))
        truth = selector.oracle_pair(full)
        earliest = None
        deduplicated_earliest = None
        for horizon in range(4, len(observations) + 1):
            prefix = digest(observations, "colour", horizon=horizon)
            candidates = selector.pairs_of(prefix)
            index = {(left.tid, right.tid): pair_id for pair_id, (left, right) in enumerate(candidates)}
            selected = selector.scale_rule(candidates)
            if truth in index and selected == index[truth]:
                earliest = earliest or horizon
            scale_rows = scale_candidates(candidates)
            retained, _ = prune_derived_mutable_tracks(scale_rows)
            if (
                deduplicated_earliest is None
                and len(retained) == 1
                and (retained[0][1].tid, retained[0][2].tid) == truth
            ):
                deduplicated_earliest = horizon
        transfer.append({
            "level": level,
            "observations_total": len(observations),
            "unique_scale_first_observation": earliest,
            "acquired_without_model": earliest is not None,
            "deduplicated_scale_first_observation": deduplicated_earliest,
            "acquired_after_deduplication": deduplicated_earliest is not None,
        })

    level = 2
    observations = level_slice(steps, level)
    horizon = max(4, int(len(observations) * 0.6))
    prefix = digest(observations, "colour", horizon=horizon)
    full = digest(observations, "colour", horizon=len(observations))
    all_candidates = selector.pairs_of(prefix)
    truth = selector.oracle_pair(full)
    frames = prefix["_tracks"]["frames"]
    attribution = {row["tid"]: row for row in prefix.get("attribution", [])}
    visible_candidates = []
    scale_rows = scale_candidates(all_candidates)
    retained_rows, derived_tracks = prune_derived_mutable_tracks(scale_rows)
    for pair_id, invariant, mutable, factor in scale_rows:
        invariant_frame = min(invariant.sprites)
        mutable_first = min(mutable.sprites)
        mutable_current = max(frame for frame in mutable.sprites if frame < horizon)
        left = invariant.sprites[invariant_frame]
        right = mutable.sprites[mutable_first]
        contact = attribution.get(mutable.tid, {})
        visible_candidates.append({
            "pair_id": pair_id,
            "invariant_track": invariant.tid,
            "mutable_track": mutable.tid,
            "program": {"kind": "integer_scale_objective", "factor": factor},
            "invariant": {
                "area": left.area,
                "extent": list(left.extent),
                "colors": [list(row) for row in left.colors],
                "patch": patch(invariant, frames, invariant_frame),
                "observed_frames": len(invariant.sprites),
            },
            "mutable": {
                "initial_area": right.area,
                "initial_extent": list(right.extent),
                "initial_colors": [list(row) for row in right.colors],
                "initial_patch": patch(mutable, frames, mutable_first),
                "current_patch": patch(mutable, frames, mutable_current),
                "observed_frames": len(mutable.sprites),
                "appearance_changes": sum(
                    mutable.sigs[frame] != mutable.sigs[frame - 1]
                    for frame in sorted(mutable.sigs)
                    if frame - 1 in mutable.sigs
                ),
                "contact_change_rows": contact.get("rows", []),
                "background_change_rate": contact.get("baseline_rate"),
            },
            "prefix_replay": {
                "status": "working",
                "checks": ["extent_scaled", "area_scaled_by_factor_squared"],
            },
        })

    truth_index = next(
        index for index, (left, right) in enumerate(all_candidates)
        if (left.tid, right.tid) == truth
    )
    truth_pair = next(
        (left, right) for left, right in selector.pairs_of(full)
        if (left.tid, right.tid) == truth
    )
    terminal_frame = max(truth_pair[1].sprites)
    terminal_match = lenses(
        truth_pair[0].sprites[max(truth_pair[0].sprites)].paint(full["_tracks"]["frames"][max(truth_pair[0].sprites)]),
        truth_pair[1].sprites[terminal_frame].paint(full["_tracks"]["frames"][terminal_frame]),
    )["scale_normalised"]
    if not terminal_match:
        raise RuntimeError("scorer truth does not close under terminal scale normalization")

    case = {
        "schema": "0004-acquisition-case-v1",
        "case_id": "ls20-level-2-scale-ambiguity",
        "authority": "prefix_only_neutral_measurements",
        "observations_seen": horizon,
        "observations_total_hidden": True,
        "selection_policy": "select only if structural and causal evidence supports one compiled hypothesis",
        "candidate_hypotheses": visible_candidates,
        "contract": {
            "tools": {
                "SELECT_RELATION": ["pair_id", "program_kind", "factor", "evidence_refs"],
                "ABSTAIN": ["reason_code", "missing_evidence"],
            },
            "exactly_one_call": True,
        },
    }
    pruned_case = deepcopy(case)
    retained_pair_ids = [row[0] for row in retained_rows]
    pruned_case["case_id"] = "ls20-level-2-deduplicated-scale-objective"
    pruned_case["authority"] = "prefix_only_measurements_with_generic_representation_deduplication"
    pruned_case["candidate_hypotheses"] = [
        row for row in visible_candidates if row["pair_id"] in retained_pair_ids
    ]
    pruned_case["representation_deduplication"] = {
        "rule": "remove_strict_spatial_subtracks_with_identical_observation_and_change_events",
        "removed_mutable_tracks": sorted(derived_tracks),
        "retained_pair_ids": retained_pair_ids,
        "evidence": [{
            "child_track": child_id,
            "parent_track": parent_id,
            "shared_observations": len(next(row[2] for row in scale_rows if row[2].tid == child_id).sprites),
            "change_events": sorted(change_events(next(row[2] for row in scale_rows if row[2].tid == child_id))),
        } for child_id in sorted(derived_tracks) for parent_id, parent in {
            row[2].tid: row[2] for row in scale_rows
        }.items() if child_id != parent_id and strict_synchronized_subtrack(
            next(row[2] for row in scale_rows if row[2].tid == child_id), parent
        )],
    }
    labels = {
        "schema": "0004-acquisition-label-v1",
        "case_id": case["case_id"],
        "valid_pair_id": truth_index,
        "valid_program": {"kind": "integer_scale_objective", "factor": 2},
        "terminal_scale_normalised_match": True,
        "terminal_observation": len(observations),
    }
    write(ROOT / "fixtures/case.json", case)
    write(ROOT / "fixtures/case-pruned.json", pruned_case)
    write(ROOT / "fixtures/machinery-transfer.json", {
        "schema": "0004-machinery-transfer-v1",
        "source_sha256": EXPECTED[RECORDING],
        "levels": transfer,
    })
    write(ROOT / "scorer/label.json", labels)
    print(f"candidates={len(visible_candidates)} truth={truth_index} transfer={transfer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
