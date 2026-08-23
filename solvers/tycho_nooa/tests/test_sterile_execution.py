from __future__ import annotations

import json
import os

import pytest

from tycho_nooa.models import ActionDecision
from tycho_nooa.sterile_execution import (
    ExecutionManifest,
    SterileCellTimeoutError,
    SterileExecutor,
    SterileReturnValidationError,
)


async def test_repl_state_stays_in_child_until_typed_return() -> None:
    manifest = ExecutionManifest(
        invocation_id="actor-decision-7",
        role="actor",
        input_values={"available_actions": ["UP", "DOWN"]},
    )

    async with SterileExecutor(manifest, return_type=ActionDecision) as executor:
        first = await executor.run_cell(
            "assert 'self' not in globals(); "
            "candidates = [action for action in available_actions if action != 'DOWN']"
        )
        second = await executor.run_cell("return_result(action=candidates[0])")

    assert first.status == "complete"
    assert first.return_value is None
    assert first.locals_preview == {"candidates": "list(len=1)"}
    assert second.status == "returned"
    assert second.return_value == ActionDecision(action="UP")
    assert second.locals_preview == {"candidates": "list(len=1)"}
    assert not hasattr(second, "locals")


async def test_valid_return_is_terminal_and_next_invocation_starts_clean() -> None:
    first_manifest = ExecutionManifest(invocation_id="actor-clean-1", role="actor")
    async with SterileExecutor(
        first_manifest, return_type=ActionDecision
    ) as executor:
        await executor.run_cell("private_thought = 42")
        await executor.run_cell("return_result(action='UP')")
        with pytest.raises(RuntimeError, match="already returned"):
            await executor.run_cell("private_thought += 1")

    second_manifest = ExecutionManifest(invocation_id="actor-clean-2", role="actor")
    async with SterileExecutor(
        second_manifest, return_type=ActionDecision
    ) as executor:
        result = await executor.run_cell(
            "assert 'private_thought' not in globals(); return_result(action='DOWN')"
        )

    assert result.return_value == ActionDecision(action="DOWN")
    assert result.locals_preview == {}


async def test_child_has_only_scratch_and_sanitized_process_authority(
    tmp_path, monkeypatch
) -> None:
    protected = tmp_path / "canonical.txt"
    protected.write_text("must not cross")
    monkeypatch.setenv("TYCHO_STERILE_SECRET", "must-not-inherit")
    source_fd = os.open(protected, os.O_RDONLY)
    inherited_fd = 200
    os.dup2(source_fd, inherited_fd, inheritable=True)
    os.close(source_fd)
    manifest = ExecutionManifest(
        invocation_id="actor-isolation-1",
        role="actor",
        input_values={
            "protected_path": str(protected),
            "parent_fd": inherited_fd,
        },
    )

    try:
        async with SterileExecutor(manifest, return_type=ActionDecision) as executor:
            result = await executor.run_cell(
                """
import json, os, socket, subprocess

def denied(fn):
    try:
        fn()
    except (OSError, PermissionError):
        return True
    return False

with open('scratch.txt', 'w') as scratch:
    scratch.write('local only')

checks = {
    'environment_sanitized': os.getenv('TYCHO_STERILE_SECRET') is None,
    'canonical_read_denied': denied(lambda: open(protected_path).read()),
    'symlink_traversal_denied': (
        os.symlink(protected_path, 'canonical-link') is None
        and denied(lambda: open('canonical-link').read())
    ),
    'parent_fd_closed': denied(lambda: os.fstat(parent_fd)),
    'internet_socket_denied': denied(lambda: socket.socket(socket.AF_INET)),
    'unix_socket_denied': denied(lambda: socket.socket(socket.AF_UNIX)),
    'subprocess_denied': denied(lambda: subprocess.run(['/bin/true'], check=True)),
    'scratch_writable': open('scratch.txt').read() == 'local only',
}
print(json.dumps(checks, sort_keys=True))
"""
            )
    finally:
        os.close(inherited_fd)

    assert result.status == "complete"
    assert json.loads(result.stdout) == {
        "canonical_read_denied": True,
        "environment_sanitized": True,
        "internet_socket_denied": True,
        "parent_fd_closed": True,
        "scratch_writable": True,
        "subprocess_denied": True,
        "symlink_traversal_denied": True,
        "unix_socket_denied": True,
    }


async def test_t1_rejects_invalid_return_without_destroying_repl_state() -> None:
    manifest = ExecutionManifest(invocation_id="actor-return-1", role="actor")

    async with SterileExecutor(manifest, return_type=ActionDecision) as executor:
        await executor.run_cell("attempts = 1")
        with pytest.raises(SterileReturnValidationError, match="CLICK requires"):
            await executor.run_cell("return_result(action='CLICK')")
        accepted = await executor.run_cell(
            "attempts += 1; return_result(action='UP')"
        )

    assert accepted.return_value == ActionDecision(action="UP")
    assert accepted.locals_preview == {"attempts": "int"}


async def test_timeout_kills_child_without_changing_protected_state(tmp_path) -> None:
    protected = tmp_path / "canonical.json"
    protected.write_text('{"revision":7}')
    manifest = ExecutionManifest(
        invocation_id="actor-timeout-1",
        role="actor",
        input_values={"protected_path": str(protected)},
        wall_time_seconds=0.1,
    )

    with pytest.raises(SterileCellTimeoutError, match="0.1 seconds"):
        async with SterileExecutor(manifest, return_type=ActionDecision) as executor:
            await executor.run_cell(
                """
try:
    open(protected_path, 'w').write('corrupt')
except OSError:
    pass
while True:
    pass
"""
            )

    assert protected.read_text() == '{"revision":7}'


async def test_output_and_locals_preview_are_bounded() -> None:
    manifest = ExecutionManifest(
        invocation_id="actor-bounds-1",
        role="actor",
        output_bytes=1_024,
        locals_preview_items=3,
    )

    async with SterileExecutor(manifest, return_type=ActionDecision) as executor:
        result = await executor.run_cell(
            "globals().update({f'item_{i}': i for i in range(100)}); print('x' * 10000)"
        )

    assert len(result.stdout.encode("utf-8")) == 1_024
    assert len(result.locals_preview) == 3


async def test_capability_contract_describes_the_executable_query_surface() -> None:
    contract = {
        "kind": "transitions",
        "level": 0,
        "limit": 4,
        "projection": "metadata",
        "max_query_attempts": 2,
        "max_successful_disclosures": 1,
    }
    manifest = ExecutionManifest(
        invocation_id="actor-query-help-1",
        role="actor",
        capabilities=("evidence.query",),
        capability_contracts={"evidence.query": contract},
    )

    async with SterileExecutor(
        manifest,
        return_type=ActionDecision,
        capability_handlers={"evidence.query": lambda arguments: arguments},
    ) as executor:
        result = await executor.run_cell(
            "import inspect; "
            "print(inspect.signature(evidence.query)); "
            "print(evidence.query.__doc__); "
            "help(evidence.query)"
        )

    signature, documentation = result.stdout.split("\n", maxsplit=1)
    assert signature == "(*, kind, level, limit, projection, grid_encoding=None)"
    assert "evidence.query(" in documentation
    assert "kind='transitions'" in documentation
    assert "level=0" in documentation
    assert "limit=4" in documentation
    assert "projection='metadata'" in documentation
    assert "grid_encoding is optional and semantically ignored" in documentation
    assert "2 attempts; at most 1 successful disclosure" in documentation
    assert "Help on _EvidenceQuery" in documentation
    assert "evidence.query(kind='transitions'" in documentation


def test_manifest_rejects_contract_for_ungranted_capability() -> None:
    with pytest.raises(ValueError, match="ungranted capabilities"):
        ExecutionManifest(
            invocation_id="actor-query-help-invalid",
            role="actor",
            capability_contracts={"evidence.query": {"kind": "transitions"}},
        )


def test_manifest_rejects_names_that_break_namespace_contract() -> None:
    for name in (
        "self",
        "help",
        "return_result",
        "__builtins__",
        "not-valid",
        "class",
    ):
        with pytest.raises(ValueError, match="input_values name"):
            ExecutionManifest(
                invocation_id="actor-invalid-manifest",
                role="actor",
                input_values={name: None},
            )
