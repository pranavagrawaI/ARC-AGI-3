"""Pure boundary checks for the precommitted one-action live gate."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

from .serialization import canonical_bytes


PUBLIC_FIELDS = ("state", "levels_completed", "win_levels", "available_actions")


def configure_remote_step_timeout(environment: Any, seconds: float) -> None:
    """Increase the SDK's fixed POST timeout without adding unsafe action retries."""

    session = getattr(environment, "_session", None)
    if session is None or not callable(getattr(session, "post", None)):
        raise TypeError("online environment has no configurable HTTP session")
    original_post = session.post

    def post(*args: Any, **kwargs: Any) -> Any:
        kwargs["timeout"] = seconds
        return original_post(*args, **kwargs)

    session.post = post


def frame_as_records(frame_data: Any) -> list[list[list[int]]]:
    return [plane.tolist() if hasattr(plane, "tolist") else plane for plane in frame_data.frame]


def frame_sha256(frame_data: Any) -> str:
    return sha256(canonical_bytes(frame_as_records(frame_data))).hexdigest()


def public_state(frame_data: Any) -> dict[str, Any]:
    state = frame_data.state
    return {
        "state": state.value if hasattr(state, "value") else state,
        "levels_completed": frame_data.levels_completed,
        "win_levels": frame_data.win_levels,
        "available_actions": list(frame_data.available_actions),
    }


def frame_record(frame_data: Any) -> dict[str, Any]:
    value = frame_data.model_dump(mode="json")
    value["frame"] = frame_as_records(frame_data)
    return value


def validate_current(
    frame_data: Any, case: dict[str, Any], label: dict[str, Any]
) -> dict[str, Any]:
    actual_public = public_state(frame_data)
    actual_digest = frame_sha256(frame_data)
    checks = {
        "frame_sha256": actual_digest == label["expected_current_frame_sha256"],
        "public_state": actual_public == case["current_public"],
        "legal_actions": actual_public["available_actions"] == case["legal_actions"],
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "actual_frame_sha256": actual_digest,
        "actual_public": actual_public,
    }


def validate_successor(
    frame_data: Any,
    predicted_public_fields: dict[str, int],
    label: dict[str, Any],
) -> dict[str, Any]:
    actual_public = public_state(frame_data)
    actual_digest = frame_sha256(frame_data)
    predicted_levels = predicted_public_fields["public.levels_completed"]
    checks = {
        "frame_sha256": actual_digest == label["expected_successor_frame_sha256"],
        "predicted_levels_completed": actual_public["levels_completed"] == predicted_levels,
        "protected_successor": {
            "public.levels_completed": actual_public["levels_completed"]
        }
        == label["protected_successor"],
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "actual_frame_sha256": actual_digest,
        "actual_public": actual_public,
    }
