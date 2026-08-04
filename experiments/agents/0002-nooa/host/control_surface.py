"""Shared minimal workspace and decision boundary for the paired arms."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from llm_repl_agent.client import ChatResponse
from p9.agent import (
    HISTORY_CONVENTION,
    IMAGE_CONVENTION,
    LAYER_CONVENTION,
    NOTES_CONVENTION,
    PATH_CONVENTION,
    RUNTIME_CONVENTION,
    SCHEMA_CONVENTION,
)


MINIMAL_CONTROL_ARM = "0002-R-minimal"
_WORKSPACE_CONTENTS = (
    "The workspace is new for this run but not empty: it already contains "
    "NOTES.md, WORKSPACE.md, current_frame.png, current_observation.json, "
    "observations, public_history.jsonl. Everything else in it is yours.\n"
)
MINIMAL_WORKSPACE_CONTRACT = (
    _WORKSPACE_CONTENTS
    + IMAGE_CONVENTION
    + NOTES_CONVENTION
    + LAYER_CONVENTION
    + PATH_CONVENTION
    + SCHEMA_CONVENTION
    + HISTORY_CONVENTION
    + RUNTIME_CONVENTION
    + "Declared system packages use normal imports. Workspace files are data and "
    "persistent storage; the workspace is not a Python module search path. Reuse "
    "computed state by defining names in executed cells, which persist until a "
    "reported interpreter restart.\n"
)
MINIMAL_WORKSPACE_CONTRACT_SHA256 = sha256(
    MINIMAL_WORKSPACE_CONTRACT.encode("utf-8")
).hexdigest()

_CONTROL_FRAME_PROMPT = (
    "frames.py in your workspace loads and compares observations for you —\n"
    '`import sys; sys.path.insert(0, "."); import frames` then frames.diff(a, b) — '
    "and it reads\n"
    "the current layer correctly. "
)
_CONTROL_IMPORT_PROMPT = (
    "The interpreter runs isolated, so the workspace directory is not on sys.path. To import a\n"
    'module you wrote, run sys.path.insert(0, ".") first. '
)


def prepare_minimal_workspace(root: Path) -> None:
    """Remove the unsafe optional helper and write the exact shared contract."""

    helper = root / "frames.py"
    if helper.exists():
        helper.unlink()
    (root / "WORKSPACE.md").write_text(
        MINIMAL_WORKSPACE_CONTRACT,
        encoding="utf-8",
    )


def adapt_control_spec(spec: Any) -> Any:
    """Remove helper/module-import promises that 0002-NOOA cannot safely match."""

    prompt = spec.system_prompt
    replacements = (
        ("NOTES.md, frames.py, WORKSPACE.md", "NOTES.md, WORKSPACE.md"),
        (_CONTROL_FRAME_PROMPT, ""),
        (_CONTROL_IMPORT_PROMPT, ""),
    )
    for old, new in replacements:
        if prompt.count(old) != 1:
            raise RuntimeError("control prompt has an unexpected workspace surface")
        prompt = prompt.replace(old, new)
    return replace(spec, system_prompt=prompt)


def adapt_minimal_control_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Identify the P9-derived arm and commit its minimal adapter surface."""

    manifest["source_arm_id"] = manifest["arm_id"]
    manifest["arm_id"] = MINIMAL_CONTROL_ARM
    manifest["compatibility_adapter"] = {
        "workspace_contract_sha256": MINIMAL_WORKSPACE_CONTRACT_SHA256,
        "provided_helper_sha256": None,
        "quoted_null_normalization": "exact-lowercase-string-v1",
    }
    workspace = manifest["model_visible"]["workspace"]
    workspace["provided_helper_sha256"] = None
    return manifest


def normalize_quoted_null_message(response: ChatResponse) -> ChatResponse:
    """Canonicalize only an exact JSON action object's lowercase quoted null."""

    content = response.message.get("content")
    if not isinstance(content, str):
        return response
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return response
    if not isinstance(value, dict) or value.get("data") != "null":
        return response
    value["data"] = None
    return replace(
        response,
        message={
            **response.message,
            "content": json.dumps(value, sort_keys=True, separators=(",", ":")),
        },
    )


class QuotedNullNormalizingClient:
    """Apply the shared final-action canonicalization after transport evidence."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> ChatResponse:
        return normalize_quoted_null_message(
            self.inner.complete(messages=messages, tools=tools)
        )
