"""Zero-call observations over preserved Tycho-NOOA run evidence.

This module deliberately reports narrower, mechanically observable facts than the
architectural risk vocabulary that motivated it. In particular:

* a missing live path is superseded-model *unreachability*, not proof that useful
  learning was erased;
* a declared tool that never appears in a response is unattempted, not proof that
  the model could not use it;
* a goal left at revision one is a frozen lifecycle, not proof that an epoch
  mechanism failed at a consequential boundary.

Scientific corpus tallies use only runs explicitly admitted by a reviewed manifest.
All discovered runs remain visible in the report so exclusions cannot disappear.
No model or ARC environment call is made.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from itertools import pairwise
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
DETECTOR_VERSION = "2026-08-22.1"
CANONICAL_MODEL = "world_model.py"
OBSERVATIONS = (
    "superseded_model_unreachability",
    "declared_tool_activity",
    "goal_lifecycle_activity",
)


def check_corpus(
    run_dirs: list[Path],
    eligibility: dict[Path, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Observe runs and tally only the reviewed, eligible subset."""
    metadata = {path.resolve(): value for path, value in (eligibility or {}).items()}
    runs = []
    for run_dir in _unique_paths(run_dirs):
        resolved = run_dir.resolve()
        admission = metadata.get(
            resolved,
            {
                "reviewed": eligibility is None,
                "eligible": eligibility is None,
                "reason": (
                    "explicit run"
                    if eligibility is None
                    else "not listed in corpus manifest"
                ),
            },
        )
        runs.append(check_run(run_dir, admission=admission))

    inspectable = [run for run in runs if run["inspectable"]]
    eligible = [run for run in inspectable if run["eligibility"]["eligible"]]
    review_required = any(not run["eligibility"]["reviewed"] for run in runs)
    if review_required:
        status = "review_required"
    elif not eligible:
        status = "fail"
    else:
        status = "pass"

    return {
        "schema_version": SCHEMA_VERSION,
        "detector_version": DETECTOR_VERSION,
        "execution": {"model_calls": 0, "arc_environment_calls": 0},
        "corpus": {
            "observed_runs": len(runs),
            "inspectable_runs": len(inspectable),
            "eligible_runs": len(eligible),
            "excluded_runs": sum(
                1
                for run in runs
                if run["eligibility"]["reviewed"]
                and not run["eligibility"]["eligible"]
            ),
            "unreviewed_runs": sum(
                1 for run in runs if not run["eligibility"]["reviewed"]
            ),
        },
        "runs": runs,
        "tallies": {name: _tally(eligible, name) for name in OBSERVATIONS},
        "status": status,
    }


def check_run(run_dir: Path, admission: dict[str, Any] | None = None) -> dict[str, Any]:
    """Observe one run without silently treating checkpoint presence as validity."""
    game = _game_dir(run_dir)
    checkpoints = list(_checkpoints(game)) if game is not None else []
    eligibility = {
        "reviewed": bool((admission or {}).get("reviewed", True)),
        "eligible": bool((admission or {}).get("eligible", True)),
        "reason": str((admission or {}).get("reason", "explicit run")),
    }
    if not checkpoints:
        return {
            "run": str(run_dir.resolve()),
            "inspectable": False,
            "reason": "no decision checkpoints",
            "eligibility": eligibility,
        }
    return {
        "run": str(run_dir.resolve()),
        "inspectable": True,
        "eligibility": eligibility,
        "checkpoint_calls": len(checkpoints),
        "superseded_model_unreachability": detect_superseded_model_unreachability(
            game, checkpoints
        ),
        "declared_tool_activity": detect_declared_tool_activity(game, checkpoints),
        "goal_lifecycle_activity": detect_goal_lifecycle_activity(game, checkpoints),
    }


def detect_superseded_model_unreachability(
    game: Path, checkpoints: list[dict[str, Any]]
) -> dict[str, Any]:
    """Report superseded model bodies with no content-identical final workspace path."""
    history = [
        (entry["call"], sha)
        for entry in checkpoints
        if (sha := _workspace_sha(entry["workspace"], CANONICAL_MODEL)) is not None
    ]
    if not history:
        return {
            "status": "not_applicable",
            "reason": f"no {CANONICAL_MODEL} in any checkpoint",
        }

    final_model = _final_workspace_model(game)
    final_sha = _sha256_file(final_model) if final_model is not None else history[-1][1]
    superseded = {sha for _, sha in history} - {final_sha}
    live_paths = _final_paths_with_hashes(game, superseded)
    blobs = _blob_dir(game)
    return {
        "status": "observed" if superseded and not live_paths else "not_observed",
        "unique_versions": len({sha for _, sha in history} | {final_sha}),
        "version_changes": sum(
            1 for a, b in pairwise(history) if a[1] != b[1]
        ),
        "reverts": [
            {"call": b[0], "to_sha256": b[1]}
            for index, (a, b) in enumerate(pairwise(history))
            if a[1] != b[1]
            and b[1] in {sha for _, sha in history[: index + 1]}
        ],
        "final_sha256": final_sha,
        "superseded_versions": len(superseded),
        "analyst_recoverable": all((blobs / sha).is_file() for sha in superseded),
        "agent_recoverable_paths": sorted(live_paths),
        "value_assessed": False,
        "claim_limit": (
            "This establishes final-path reachability only. It does not establish that a "
            "superseded model was valuable, safe to publish, or useful for gameplay."
        ),
    }


def detect_declared_tool_activity(
    game: Path, checkpoints: list[dict[str, Any]]
) -> dict[str, Any]:
    """Separate offered, attempted, completed, and errored tool activity by source."""
    declared: dict[str, set[str]] = {}
    attempted: dict[str, set[str]] = {}
    for entry in checkpoints:
        source = str(entry["request"].get("source", "unknown"))
        body = entry["request"].get("request_body") or {}
        declared.setdefault(source, set()).update(
            name for tool in (body.get("tools") or []) if (name := _tool_name(tool))
        )
        attempted.setdefault(source, set()).update(
            _invoked_names(body.get("messages") or [])
        )
        attempted[source].update(_response_tool_names(entry.get("response")))

    traces = _tool_trace_activity(game)
    sources = {}
    for source in sorted(set(declared) | set(attempted) | set(traces)):
        offered = declared.get(source, set())
        tried = attempted.get(source, set()) | traces.get(source, {}).get(
            "attempted", set()
        )
        completed = traces.get(source, {}).get("completed", set())
        errors = traces.get(source, {}).get("errors", set())
        sources[source] = {
            "declared": sorted(offered),
            "attempted": sorted(tried),
            "completed": sorted(completed),
            "errors": sorted(errors),
            "offered_but_never_attempted": sorted(offered - tried),
            "attempted_but_not_completed": sorted(tried - completed),
        }

    unused = sorted(
        {
            name
            for row in sources.values()
            for name in row["offered_but_never_attempted"]
        }
    )
    return {
        "status": "observed" if unused else "not_observed",
        "offered_but_never_attempted": unused,
        "by_source": sources,
        "claim_limit": (
            "Unattempted means absent from preserved responses and tool traces. It does not "
            "establish inability, uselessness, comprehension, or causal uptake."
        ),
    }


def detect_goal_lifecycle_activity(
    game: Path, checkpoints: list[dict[str, Any]]
) -> dict[str, Any]:
    """Report actual goal-tool events and the final typed-goal state."""
    goals = _load_candidate_goals(game)
    declared = (
        detect_declared_tool_activity(game, checkpoints)["by_source"]
        .get("actor", {})
        .get("declared", [])
    )
    goal_tools = {
        "propose_goal",
        "list_candidate_goals",
        "update_goal",
        "request_builder",
    }
    exposed = bool(goal_tools.intersection(declared))
    events = [
        record
        for record in _read_jsonl(game / "tool-trace.jsonl")
        if record.get("name") in goal_tools
    ]
    activity = {
        name: {
            "attempted": sum(1 for record in events if record.get("name") == name),
            "completed": sum(
                1
                for record in events
                if record.get("name") == name and record.get("status") == "complete"
            ),
            "errors": sum(
                1
                for record in events
                if record.get("name") == name and record.get("status") != "complete"
            ),
        }
        for name in sorted(goal_tools)
    }

    if goals is None:
        return {
            "status": "no_goal_created" if exposed else "not_applicable",
            "goal_tools_exposed": exposed,
            "activity": activity,
            "reason": "no candidate-goals artifact",
        }
    if not goals:
        return {
            "status": "no_goal_created",
            "goal_tools_exposed": exposed,
            "activity": activity,
            "reason": "candidate-goals artifact is empty",
        }

    revisions = [int(goal.get("revision", 1)) for goal in goals]
    statuses = sorted({str(goal.get("status", "")) for goal in goals})
    frozen = (
        max(revisions, default=0) == 1
        and statuses == ["proposed"]
        and activity["update_goal"]["attempted"] == 0
    )
    return {
        "status": "frozen_after_proposal" if frozen else "advanced",
        "goal_tools_exposed": exposed,
        "goals": len(goals),
        "max_revision": max(revisions, default=0),
        "statuses": statuses,
        "activity": activity,
        "claim_limit": (
            "This establishes typed-goal lifecycle activity only. It does not establish "
            "whether the goal was true, understood, or beneficial."
        ),
    }


def load_corpus_manifest(
    manifest_path: Path, root: Path
) -> tuple[list[Path], dict[Path, dict[str, Any]]]:
    """Load a reviewed run-admission manifest rooted at the repository."""
    data = _read_json(manifest_path)
    if not isinstance(data, dict) or not isinstance(data.get("runs"), list):
        raise TypeError("corpus manifest must be an object with a runs list")
    paths: list[Path] = []
    eligibility: dict[Path, dict[str, Any]] = {}
    for entry in data["runs"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("run"), str):
            raise TypeError("each corpus manifest entry must name a run")
        path = Path(entry["run"])
        path = path if path.is_absolute() else root / path
        resolved = path.resolve()
        if resolved in eligibility:
            raise ValueError(f"duplicate corpus manifest run: {entry['run']}")
        paths.append(resolved)
        eligibility[resolved] = {
            "reviewed": True,
            "eligible": bool(entry.get("eligible", False)),
            "reason": str(entry.get("reason", "no reason recorded")),
        }
    return paths, eligibility


def discover_runs(root: Path) -> list[Path]:
    return sorted(
        path.parent.parent
        for path in (root / "runs").glob("*/game/decision-checkpoints")
        if path.is_dir()
    ) + sorted(
        path.parent.parent
        for path in (root / "runs").glob("*/arms/*/game/decision-checkpoints")
        if path.is_dir()
    )


def _tally(runs: list[dict[str, Any]], name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for run in runs:
        status = run[name]["status"]
        counts[status] = counts.get(status, 0) + 1
    return counts


def _game_dir(run_dir: Path) -> Path | None:
    for candidate in (run_dir / "game", run_dir):
        if (candidate / "decision-checkpoints" / "calls").is_dir():
            return candidate
    return None


def _checkpoints(game: Path) -> Iterator[dict[str, Any]]:
    for call in sorted((game / "decision-checkpoints" / "calls").iterdir()):
        request_path = call / "request.json"
        if not request_path.is_file():
            continue
        yield {
            "call": call.name,
            "request": _read_json(request_path),
            "response": _read_json(call / "response.json"),
            "workspace": _read_json(call / "workspace.json"),
        }


def _blob_dir(game: Path) -> Path:
    return game / "decision-checkpoints" / "blobs" / "sha256"


def _workspace_sha(workspace: dict[str, Any] | None, path: str) -> str | None:
    for entry in (workspace or {}).get("files", []):
        if entry.get("path", "").endswith(path):
            return entry.get("sha256")
    return None


def _final_workspace_model(game: Path) -> Path | None:
    matches = sorted((game / "workspace").glob(f"*/{CANONICAL_MODEL}"))
    return matches[0] if matches else None


def _final_paths_with_hashes(game: Path, hashes: set[str]) -> set[str]:
    if not hashes:
        return set()
    workspace = game / "workspace"
    if not workspace.is_dir():
        return set()
    paths = set()
    for path in workspace.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            digest = _sha256_file(path)
        except OSError:
            continue
        if digest in hashes:
            paths.add(path.relative_to(workspace).as_posix())
    return paths


def _tool_name(tool: dict[str, Any]) -> str | None:
    return tool.get("name") or (tool.get("function") or {}).get("name")


def _invoked_names(messages: list[dict[str, Any]]) -> set[str]:
    return {
        name
        for message in messages
        for call in (message.get("tool_calls") or [])
        if (name := (call.get("function") or {}).get("name"))
    }


def _response_tool_names(response: dict[str, Any] | None) -> set[str]:
    if not isinstance(response, dict):
        return set()
    names = {
        str(call["name"])
        for call in response.get("tool_calls", [])
        if isinstance(call, dict) and call.get("name")
    }
    message = response.get("assistant_message") or {}
    names.update(_invoked_names([message]))
    return names


def _tool_trace_activity(game: Path) -> dict[str, dict[str, set[str]]]:
    activity: dict[str, dict[str, set[str]]] = {}
    sources = (("actor", "tool-trace.jsonl"), ("builder", "builder-tool-trace.jsonl"))
    for source, name in sources:
        for record in _read_jsonl(game / name):
            tool = record.get("name")
            if not isinstance(tool, str):
                continue
            row = activity.setdefault(
                source, {"attempted": set(), "completed": set(), "errors": set()}
            )
            row["attempted"].add(tool)
            if record.get("status") == "complete":
                row["completed"].add(tool)
            else:
                row["errors"].add(tool)
    return activity


def _load_candidate_goals(game: Path) -> list[dict[str, Any]] | None:
    matches = sorted((game / "workspace").glob("*/epistemic/candidate-goals.json"))
    if not matches:
        return None
    goals: list[dict[str, Any]] = []
    for match in matches:
        loaded = _read_json(match)
        if isinstance(loaded, list):
            goals.extend(item for item in loaded if isinstance(item, dict))
    return goals


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        loaded = json.loads(line)
        if isinstance(loaded, dict):
            records.append(loaded)
    return records


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique
