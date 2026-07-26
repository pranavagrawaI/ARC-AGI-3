"""Normalized arm comparison: the smallest verifier that catches an unauthorized difference.

Two comparisons run over the same machinery:

* the **declared** comparison, over capability manifests read from constructed arms;
* the **observed** comparison, over the exact request payloads each arm sent while
  being driven by an identical fixture.

Both flatten to dotted paths and both are checked against the same authorized
path set. A difference is authorized only if the path, or one of its ancestors,
was declared in advance by an enabled treatment capability. An authorized path
that never differs is reported too: a capability that changes nothing is a
finding, not a success.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Iterable, Mapping


MISSING = object()


class UnauthorizedArmDifference(AssertionError):
    """Raised when two arms differ on a path no enabled capability declared."""


@dataclass(frozen=True)
class Difference:
    path: str
    status: str
    control: Any
    treatment: Any

    def as_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "status": self.status,
            "control": None if self.control is MISSING else self.control,
            "treatment": None if self.treatment is MISSING else self.treatment,
        }


@dataclass(frozen=True)
class ArmDiffReport:
    compared_paths: int
    shared_paths: int
    authorized_used: tuple[Difference, ...]
    authorized_unused: tuple[str, ...]
    unauthorized: tuple[Difference, ...]

    @property
    def ok(self) -> bool:
        return not self.unauthorized

    def as_json(self) -> dict[str, Any]:
        return {
            "compared_paths": self.compared_paths,
            "shared_paths": self.shared_paths,
            "authorized_used": [item.as_json() for item in self.authorized_used],
            "authorized_unused": list(self.authorized_unused),
            "unauthorized": [item.as_json() for item in self.unauthorized],
            "ok": self.ok,
        }


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten to dotted paths; a list of scalars stays one leaf value."""

    if isinstance(value, Mapping):
        if not value:
            return {prefix: {}} if prefix else {}
        flat: dict[str, Any] = {}
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            flat.update(flatten(item, child))
        return flat
    if isinstance(value, (list, tuple)):
        items = list(value)
        if all(not isinstance(item, (Mapping, list, tuple)) for item in items):
            return {prefix: items}
        flat = {}
        for index, item in enumerate(items):
            flat.update(flatten(item, f"{prefix}.{index}"))
        return flat
    return {prefix: value}


def normalized_diff(control: Mapping[str, Any], treatment: Mapping[str, Any]) -> list[Difference]:
    left = flatten(control)
    right = flatten(treatment)
    differences: list[Difference] = []
    for path in sorted(set(left) | set(right)):
        control_value = left.get(path, MISSING)
        treatment_value = right.get(path, MISSING)
        if control_value is MISSING:
            differences.append(Difference(path, "only_in_treatment", MISSING, treatment_value))
        elif treatment_value is MISSING:
            differences.append(Difference(path, "only_in_control", control_value, MISSING))
        elif control_value != treatment_value:
            differences.append(Difference(path, "changed", control_value, treatment_value))
    return differences


def is_authorized(path: str, authorized: Iterable[str]) -> bool:
    """A path is authorized if a declared pattern matches it or one of its ancestors.

    A pattern segment of ``*`` matches any single segment, so list positions the
    declaration cannot know in advance (message indices) stay declarative.
    """

    parts = path.split(".")
    for pattern in authorized:
        segments = pattern.split(".")
        if len(segments) > len(parts):
            continue
        if all(
            segment == "*" or segment == part
            for segment, part in zip(segments, parts)
        ):
            return True
    return False


def compare_arms(
    control: Mapping[str, Any],
    treatment: Mapping[str, Any],
    *,
    authorized_paths: Iterable[str],
) -> ArmDiffReport:
    authorized = set(authorized_paths)
    differences = normalized_diff(control, treatment)
    used = tuple(item for item in differences if is_authorized(item.path, authorized))
    unauthorized = tuple(item for item in differences if not is_authorized(item.path, authorized))
    exercised = {
        path
        for path in authorized
        if any(is_authorized(item.path, {path}) for item in used)
    }
    compared = len(set(flatten(control)) | set(flatten(treatment)))
    return ArmDiffReport(
        compared_paths=compared,
        shared_paths=compared - len(differences),
        authorized_used=used,
        authorized_unused=tuple(sorted(authorized - exercised)),
        unauthorized=unauthorized,
    )


def assert_authorized(report: ArmDiffReport, *, label: str) -> None:
    if report.unauthorized:
        rendered = "\n".join(
            f"  {item.path}: {item.status} {item.control!r} -> {item.treatment!r}"
            for item in report.unauthorized
        )
        raise UnauthorizedArmDifference(f"{label} has unauthorized differences:\n{rendered}")


def normalize_request_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Make one recorded request comparable across arms.

    Image data URLs collapse to a digest, and the observation payload — which the
    agent sends as a JSON string — is parsed so differences land on structural
    paths instead of one opaque blob.
    """

    normalized = _digest_data_urls(json.loads(json.dumps(payload)))
    messages = normalized.get("messages")
    if isinstance(messages, list):
        for message in messages:
            _parse_embedded_payload(message)
    return normalized


def _parse_embedded_payload(message: Any) -> None:
    if not isinstance(message, dict):
        return
    content = message.get("content")
    if not isinstance(content, list):
        return
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "text":
            continue
        text = part.get("text")
        if not isinstance(text, str) or not text.lstrip().startswith("{"):
            continue
        try:
            part["observation_payload"] = json.loads(text)
        except json.JSONDecodeError:
            continue
        part.pop("text", None)


def _digest_data_urls(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("data:image/"):
        return "sha256:" + sha256(value.encode("utf-8")).hexdigest()
    if isinstance(value, dict):
        return {key: _digest_data_urls(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_digest_data_urls(item) for item in value]
    return value
