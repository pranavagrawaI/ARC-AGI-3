from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta

import pytest

from tycho_nooa.context_evidence import (
    ContextEvidencePlane,
    EvidenceAccessContext,
    EvidenceAccessDenied,
    EvidenceAdmissionAuthority,
    EvidenceCache,
    EvidenceGrantRegistry,
    EvidenceIntegrityError,
    EvidenceReadBroker,
    EvidenceViewRequest,
)
from tycho_nooa.context_execution import (
    ContextComputationalSterileExecutor,
    ContextExecutionManifest,
)
from tycho_nooa.execution_events import ExecutionProvenanceLog, InvocationProvenance

NOW = datetime(2026, 8, 24, tzinfo=UTC)
FRONTIER = {"level": 1, "turn": 2}


def _page() -> dict:
    return {
        "kind": "transitions",
        "frontier": FRONTIER,
        "records": [
            {
                "type": "transition",
                "level": 0,
                "turn": 1,
                "ref": "L0:T1",
                "action": "LEFT",
                "prev": ["b:0"],
                "next": ["a:0,1:5"],
            }
        ],
        "next_after": None,
        "truncated": False,
        "grid_encoding": "rle",
        "projection": "full",
    }


def _context(**changes: object) -> EvidenceAccessContext:
    values = {
        "session_id": "actor-session",
        "invocation_id": "actor-turn-7",
        "role": "actor",
        "frontier": FRONTIER,
    }
    values.update(changes)
    return EvidenceAccessContext.model_validate(values)


def _components(*, clock=lambda: NOW, handle_factory=None):
    cache = EvidenceCache()
    grants = EvidenceGrantRegistry()
    admission_kwargs = {"clock": clock}
    if handle_factory is not None:
        admission_kwargs["handle_factory"] = handle_factory
    admission = EvidenceAdmissionAuthority(cache, grants, **admission_kwargs)
    broker = EvidenceReadBroker(cache, grants, clock=clock)
    return cache, admission, broker


def test_receipt_names_but_does_not_copy_exact_evidence() -> None:
    _, admission, broker = _components()

    receipt = admission.admit(
        _page(),
        query={"kind": "transitions", "level": 0},
        context=_context(),
        grant_lifetime=timedelta(minutes=5),
    )

    value = receipt.model_dump(mode="json")
    assert receipt.handle.startswith("ev_")
    assert receipt.refs == ("L0:T1",)
    assert receipt.frontier == FRONTIER
    assert receipt.record_count == 1
    assert receipt.encoding == "rle"
    assert "records" not in value
    assert "LEFT" not in receipt.model_dump_json()
    assert broker.read(receipt.handle, context=_context()) == _page()


def test_admission_and_read_authority_are_separate_surfaces() -> None:
    _, admission, broker = _components()

    assert not hasattr(admission, "read")
    assert not hasattr(admission, "view")
    assert not hasattr(broker, "admit")
    assert not hasattr(broker, "query")
    assert not hasattr(broker, "list")


def test_read_returns_a_copy_and_cannot_mutate_the_cached_value() -> None:
    _, admission, broker = _components()
    receipt = admission.admit(
        _page(), query={"kind": "transitions"}, context=_context()
    )

    first = broker.read(receipt.handle, context=_context())
    first["records"][0]["action"] = "RIGHT"

    assert (
        broker.read(receipt.handle, context=_context())["records"][0]["action"]
        == "LEFT"
    )


@pytest.mark.parametrize(
    "context",
    [
        _context(invocation_id="another-turn"),
        _context(session_id="another-session"),
        _context(role="builder"),
        _context(frontier={"level": 1, "turn": 3}),
    ],
)
def test_handle_is_useless_outside_its_exact_grant(
    context: EvidenceAccessContext,
) -> None:
    _, admission, broker = _components()
    receipt = admission.admit(
        _page(), query={"kind": "transitions"}, context=_context()
    )

    with pytest.raises(EvidenceAccessDenied):
        broker.read(receipt.handle, context=context)


def test_expired_and_revoked_grants_do_not_delete_audit_evidence() -> None:
    clock = [NOW]
    cache, admission, broker = _components(clock=lambda: clock[0])
    receipt = admission.admit(
        _page(),
        query={"kind": "transitions"},
        context=_context(),
        grant_lifetime=timedelta(seconds=1),
    )
    clock[0] += timedelta(seconds=2)

    with pytest.raises(EvidenceAccessDenied, match="expired"):
        broker.read(receipt.handle, context=_context())

    assert cache.read(receipt.content_hash) == _page()

    replacement = admission.admit(
        _page(), query={"kind": "transitions"}, context=_context()
    )
    admission.revoke_invocation(_context())
    with pytest.raises(EvidenceAccessDenied, match="not granted"):
        broker.read(replacement.handle, context=_context())
    assert cache.read(replacement.content_hash) == _page()


def test_grant_clock_must_be_timezone_aware() -> None:
    _, admission, _ = _components(clock=lambda: NOW.replace(tzinfo=None))

    with pytest.raises(ValueError, match="timezone-aware"):
        admission.admit(_page(), query={}, context=_context())


def test_corrupted_cache_entry_fails_hash_verification() -> None:
    cache, admission, broker = _components()
    receipt = admission.admit(
        _page(), query={"kind": "transitions"}, context=_context()
    )
    cache._payloads[receipt.content_hash] = b'{"corrupt":true}'

    with pytest.raises(EvidenceIntegrityError, match="content hash"):
        broker.read(receipt.handle, context=_context())


def test_grant_cannot_follow_a_handle_remapped_to_another_entry() -> None:
    cache, admission, broker = _components()
    receipt = admission.admit(
        _page(), query={"kind": "transitions"}, context=_context()
    )
    other = _page()
    other["records"][0]["action"] = "RIGHT"
    other_entry = cache.store(other, frontier=FRONTIER)
    broker._grants._handles[receipt.handle] = other_entry.content_hash

    with pytest.raises(EvidenceAccessDenied, match="no longer resolves"):
        broker.read(receipt.handle, context=_context())


def test_view_is_bounded_and_cannot_discover_an_ungranted_record() -> None:
    _, admission, broker = _components()
    receipt = admission.admit(
        _page(), query={"kind": "transitions"}, context=_context()
    )

    view = broker.view(
        EvidenceViewRequest(
            handle=receipt.handle,
            refs=("L0:T1",),
            fields=("ref", "action", "next"),
            rows=(0,),
            max_bytes=512,
        ),
        context=_context(),
    )

    assert view == {
        "handle": receipt.handle,
        "records": [{"ref": "L0:T1", "action": "LEFT", "next": ["a:0,1:5"]}],
    }
    with pytest.raises(EvidenceAccessDenied, match="not present"):
        broker.view(
            EvidenceViewRequest(handle=receipt.handle, refs=("L9:T9",)),
            context=_context(),
        )


def test_view_rejects_an_oversized_projection_without_partial_records() -> None:
    page = _page()
    page["records"][0]["large"] = "x" * 1_000
    _, admission, broker = _components()
    receipt = admission.admit(page, query={"kind": "transitions"}, context=_context())

    with pytest.raises(ValueError, match="view exceeds"):
        broker.view(
            EvidenceViewRequest(
                handle=receipt.handle,
                refs=("L0:T1",),
                fields=("ref", "large"),
                max_bytes=128,
            ),
            context=_context(),
        )


class _Source:
    def __init__(self, value: dict):
        self.value = value
        self.queries: list[dict] = []

    def query(self, raw_query: dict) -> dict:
        self.queries.append(raw_query)
        return self.value


async def test_native_and_codeact_computation_cross_the_same_read_broker(
    tmp_path,
) -> None:
    source = _Source(_page())
    _, admission, broker = _components()
    plane = ContextEvidencePlane(source, admission, broker, _context())
    receipt = plane.query({"kind": "transitions", "level": 0})
    manifest = ContextExecutionManifest(
        invocation_id=_context().invocation_id,
        role="actor",
        capabilities=("evidence.read",),
        output_bytes=2_048,
    )

    async with ContextComputationalSterileExecutor(
        manifest,
        return_type=dict,
        capability_handlers={"evidence.read": plane.read},
    ) as executor:
        result = await executor.run_cell(
            "page = evidence_cache.load(" + repr(receipt["handle"]) + ")\n"
            "print(page.records[0].ref, rle.decode_row(page.records[0].next_rows[0])[-1])"
        )

    assert result.stdout == "L0:T1 5\n"
    assert source.queries == [{"kind": "transitions", "level": 0}]


async def test_worker_cannot_reuse_a_handle_after_invocation_close() -> None:
    _, admission, broker = _components()
    plane = ContextEvidencePlane(_Source(_page()), admission, broker, _context())
    receipt = plane.query({"kind": "transitions"})
    plane.close()
    manifest = ContextExecutionManifest(
        invocation_id=_context().invocation_id,
        role="actor",
        capabilities=("evidence.read",),
    )

    async with ContextComputationalSterileExecutor(
        manifest,
        return_type=dict,
        capability_handlers={"evidence.read": plane.read},
    ) as executor:
        result = await executor.run_cell(f"evidence_cache.load({receipt['handle']!r})")

    assert result.status == "error"
    assert result.exception is not None
    assert "not granted" in result.exception.message


async def test_shared_computation_output_is_bounded_with_a_verifiable_marker() -> None:
    manifest = ContextExecutionManifest(
        invocation_id="output-governor",
        role="actor",
        output_bytes=1_024,
    )

    async with ContextComputationalSterileExecutor(
        manifest,
        return_type=dict,
    ) as executor:
        result = await executor.run_cell("print('x' * 5000, end='')")

    assert len(result.stdout.encode()) <= 1_024
    assert "<output truncated: 5000 bytes; sha256=" in result.stdout
    assert "x" * 2_000 not in result.stdout


async def test_context_worker_preserves_sterile_process_isolation(
    tmp_path, monkeypatch
) -> None:
    protected = tmp_path / "canonical.txt"
    protected.write_text("must not cross")
    monkeypatch.setenv("TYCHO_CONTEXT_SECRET", "must-not-inherit")
    source_fd = os.open(protected, os.O_RDONLY)
    inherited_fd = 200
    os.dup2(source_fd, inherited_fd, inheritable=True)
    os.close(source_fd)
    manifest = ContextExecutionManifest(
        invocation_id="context-isolation",
        role="actor",
        input_values={
            "protected_path": str(protected),
            "parent_fd": inherited_fd,
        },
    )

    try:
        async with ContextComputationalSterileExecutor(
            manifest, return_type=dict
        ) as executor:
            result = await executor.run_cell(
                """
import json, os, socket, subprocess

def denied(fn):
    try:
        fn()
    except (OSError, PermissionError):
        return True
    return False

checks = {
    'environment_sanitized': os.getenv('TYCHO_CONTEXT_SECRET') is None,
    'canonical_read_denied': denied(lambda: open(protected_path).read()),
    'parent_fd_closed': denied(lambda: os.fstat(parent_fd)),
    'internet_socket_denied': denied(lambda: socket.socket(socket.AF_INET)),
    'subprocess_denied': denied(lambda: subprocess.run(['/bin/true'], check=True)),
}
print(json.dumps(checks, sort_keys=True))
"""
            )
    finally:
        os.close(inherited_fd)

    assert json.loads(result.stdout) == {
        "canonical_read_denied": True,
        "environment_sanitized": True,
        "internet_socket_denied": True,
        "parent_fd_closed": True,
        "subprocess_denied": True,
    }


async def test_context_worker_start_failure_is_explicit_in_provenance(
    tmp_path, monkeypatch
) -> None:
    async def fail_to_start(*args, **kwargs):
        raise OSError("scripted launch failure")

    monkeypatch.setattr(
        "tycho_nooa.context_execution.asyncio.create_subprocess_exec",
        fail_to_start,
    )
    log = ExecutionProvenanceLog(
        tmp_path / "provenance.jsonl", actor_session_id="context-session"
    )
    provenance = InvocationProvenance(log, invocation_id="context-start-failure")
    manifest = ContextExecutionManifest(
        invocation_id="context-start-failure", role="actor"
    )

    with pytest.raises(OSError, match="scripted launch failure"):
        async with ContextComputationalSterileExecutor(
            manifest,
            return_type=dict,
            provenance=provenance,
        ):
            pass

    assert [event.event_type for event in log.events] == [
        "InvocationStarted",
        "InvocationFailed",
    ]
    assert log.events[-1].payload == {
        "reason": "worker_start_failed",
        "locals_lost": True,
    }
