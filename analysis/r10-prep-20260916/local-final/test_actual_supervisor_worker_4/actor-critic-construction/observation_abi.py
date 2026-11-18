"""Canonical constructor-visible observation ABI for this identity."""

from __future__ import annotations

from typing import Any

OBSERVATION_KEY = "grid"
FORBIDDEN_OBSERVATION_KEYS = {"grid_rows", "before_grid_rows", "after_grid_rows"}
FORBIDDEN_SURFACE_KEYS = {
    "suffix",
    "sealed_suffix",
    "evaluator_payload",
    "compatible_depth",
    "first_contradiction_step",
    "measurement",
    "collision",
}


def observation_from_state(state: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(state, dict) or "grid" not in state:
        raise ValueError("state must contain integer grid")
    return {OBSERVATION_KEY: copy_grid(state["grid"])}


def copy_grid(grid: Any) -> list[list[int]]:
    return [list(row) for row in validate_grid(grid)]


def validate_grid(grid: Any) -> list[list[int]]:
    if isinstance(grid, (str, bytes)) or not isinstance(grid, (list, tuple)) or not grid:
        raise ValueError("grid must be a nonempty list of rows")
    width = None
    out: list[list[int]] = []
    for row in grid:
        if isinstance(row, (str, bytes)) or not isinstance(row, (list, tuple)) or not row:
            raise ValueError("each grid row must be a nonempty list")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("grid is not rectangular")
        cells: list[int] = []
        for cell in row:
            if type(cell) is not int:
                raise ValueError("grid cells must be integers")
            cells.append(cell)
        out.append(cells)
    return out


def validate_observation(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("observation must be an object")
    extra = set(value) - {OBSERVATION_KEY}
    if extra:
        raise ValueError("observation has non-canonical keys: " + str(sorted(extra)))
    if OBSERVATION_KEY not in value:
        raise ValueError("observation missing grid")
    if FORBIDDEN_OBSERVATION_KEYS & set(value):
        raise ValueError("observation uses forbidden encoding keys")
    return {OBSERVATION_KEY: copy_grid(value[OBSERVATION_KEY])}


def _walk_forbidden(value: Any, path: str, errors: list[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            here = path + "." + key if path else key
            if key in FORBIDDEN_SURFACE_KEYS or key in FORBIDDEN_OBSERVATION_KEYS:
                errors.append(here)
            if key == OBSERVATION_KEY or (
                key in {"before", "after", "initial_observation", "current_observation"}
                and isinstance(child, dict)
                and OBSERVATION_KEY in child
            ):
                try:
                    if key == OBSERVATION_KEY:
                        validate_grid(child)
                    else:
                        validate_observation(child)
                except ValueError as exc:
                    errors.append(here + ": " + str(exc))
            _walk_forbidden(child, here, errors)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            _walk_forbidden(child, path + f"[{i}]", errors)


def validate_constructor_surface(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("constructor surface must be an object")
    errors: list[str] = []
    _walk_forbidden(payload, "", errors)
    if errors:
        raise ValueError("constructor surface ABI drift: " + "; ".join(errors))
    for name in ("initial_observation", "current_observation"):
        if name in payload:
            validate_observation(payload[name])
    for row in payload.get("visible_transitions") or []:
        if not isinstance(row, dict):
            raise ValueError("visible transition must be an object")
        if "before" in row:
            validate_observation(row["before"])
        if "after" in row:
            validate_observation(row["after"])


def observation_from_host(value: Any) -> dict[str, Any]:
    """Project an arc-host observation onto the canonical candidate ABI."""
    if isinstance(value, dict):
        if FORBIDDEN_OBSERVATION_KEYS & set(value):
            raise ValueError("host observation uses forbidden encoding keys")
        if OBSERVATION_KEY in value and "decision_frame" not in value:
            return validate_observation({OBSERVATION_KEY: value[OBSERVATION_KEY]})
        if "decision_frame" in value:
            return validate_observation({OBSERVATION_KEY: value["decision_frame"]})
        raise ValueError("host observation missing grid")
    grid = getattr(value, "grid", None)
    if grid is None:
        raise ValueError("host observation missing grid")
    return validate_observation({OBSERVATION_KEY: grid})
