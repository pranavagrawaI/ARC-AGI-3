"""What 0002-R is, pinned so that changing it has to be deliberate.

``participating_sources()`` in ``arms.py`` digests modules at manifest time. That
proves the two arms of one pair executed identical code, which is what the seam
needed, and it is not a freeze: the digest is recomputed every run and no expected
value is stored anywhere. Under that alone an edit to ``agent.py`` changes what the
control *is* and every check stays green, so a later 0006 result would be compared
against a moving baseline.

Two digests are pinned, and they are not two independent nets:

``runtime_digest()`` covers the code that executes and is the detector. It is
deliberately total — a comment change fails it, because nothing can distinguish a
comment change from a behavioural one without reading it, and a freeze that only
notices the changes it judges important is not a freeze.

``control_manifest_digest()`` covers every model↔harness crossing, and in practice
almost nothing moves it that does not also move the runtime digest. It is kept
because it *classifies* what the runtime digest caught: runtime alone says the
control moved, and the pair says whether it moved somewhere the model can see.
Verified on this pin — a comment appended to ``agent.py`` moves runtime only, while
editing a sentence of the system prompt in the same file moves both. It is also the
artifact a 0006 comparison cites when it names its baseline.

Re-pinning is the intended way to change 0002-R: run ``python -m ...p9.freeze``, read
what moved, and commit the new constants with a reason. The test that enforces this
lives in ``test_p9_shared_arm.py`` so the freeze fails in the ordinary suite.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from llm_repl_agent.experiment_protocol import participating_source_hashes

from .arms import CONTROL_ARM_ID, P9_ROOT


# The modules an arm imports, resolved from arms.py's own import graph. Prototypes,
# the mutation probe and the friction reader are excluded on purpose: they measure
# 0002-R, they are not part of it, and editing a measuring tool must not read as the
# control having changed.
RUNTIME_MODULES = (
    "agent.py",
    "annotated_frame.py",
    "arms.py",
    "capabilities.py",
    "conversation.py",
    "failures.py",
    "frame_tools.py",
    "transport.py",
)

# The 0002 protocol's own frozen set, folded in as a single entry. 0002-R is 0002 plus
# a delta, so a change underneath it genuinely changes the control; collapsing it to
# one digest keeps that true without duplicating 0002's list here, where it would rot.
BASELINE_PROTOCOL_KEY = "0002-protocol"

# Re-pinned 2026-08-02, for two changes taken together.
#
# The learning_loop capability was declared in the registry. That moves runtime only:
# the control does not enable it, so nothing the model sees differs.
#
# It moved the control manifest too, until this pin. ``host_only.capability_registry``
# listed every declared capability, so registering any treatment changed the control's
# manifest and forced a re-pin — the moving baseline this file's opening paragraph
# exists to prevent, arriving through the audit fields rather than the model-visible
# ones. The field is gone from ``arm_manifest``; scoped to what an arm enables it was
# ``host_only.spec.treatment_capabilities``, already recorded beside it.
#
# Verified on this pin: registering learning_loop leaves the control manifest digest
# at the value below and moves runtime alone. That is the pair reading its two digests
# were meant to produce — the control moved, and not where the model can see.
FROZEN_RUNTIME_DIGEST = "b00dbcd9e3eb13b5a3bb3c23caf87c9840574e47940eab7f9b17829d8cb4e9fe"
FROZEN_CONTROL_MANIFEST_DIGEST = (
    "5a6ee9f886a1f2b43374880edda85a2e2b9d88bfc8c72ef2a634a708c8b29ef3"
)

# Per-component digests exist only so a failure names the file that moved. A scalar
# that says "something changed" and nothing else is what makes a freeze annoying
# enough to be disabled.
FROZEN_RUNTIME_COMPONENTS: dict[str, str] = {
    "0002-protocol": "5fc34b7a9927af0905810a6aed747b286da0d958e695bd1cd5a0a0d1cb6a9616",
    "p9/agent.py": "17ed0f1833486d34d76f1b363ec972f68273789133f206b265c4bae1dcabc7a7",
    "p9/annotated_frame.py": "3c9c0627e04ad6f20541bba535bc40b27e889669c9400c7dd2fd4818141084d1",
    "p9/arms.py": "31e5bccb39b521387f7b5a11b885a72c61b7f57972aeb3307c4eaf84c683ef28",
    "p9/capabilities.py": "f7dd816493f7e6ef32ca1ffe8e543fa99e84df857e019d0662a34122df095eab",
    "p9/conversation.py": "4b467dabac9a6c67a504fcde51d1f4eadcec8ca60c1d4fb2a714901e135cafc4",
    "p9/failures.py": "8216b7aae6f97c9044fa2817dd8e774803e906f02c9ea1a5d5a6afecb000fa20",
    "p9/frame_tools.py": "5aa1f87809daba7c3aec012eb9a50b586af1a79907f4382771b466cb89a008e0",
    "p9/transport.py": "d665ce60fe4d92353315df44da096da9d5e55bd00d1a6535e20036a563762926",
}


def runtime_components() -> dict[str, str]:
    """Each frozen input to the control's behaviour, digested separately."""

    components = {
        f"p9/{name}": sha256((P9_ROOT / name).read_bytes()).hexdigest()
        for name in RUNTIME_MODULES
    }
    components[BASELINE_PROTOCOL_KEY] = _digest(participating_source_hashes())
    return dict(sorted(components.items()))


def runtime_digest() -> str:
    """One scalar over everything the control executes."""

    return _digest(runtime_components())


def control_manifest() -> dict[str, Any]:
    """The control's manifest, built through the ordinary factory.

    Construction touches the filesystem, so this builds into a temporary directory
    and drops the fields that vary with where it was built.

    ``host_only.source_sha256`` is dropped for a sharper reason, found by this pin on
    its first run: it globs every p9 module, so it covers the prototypes, the mutation
    probe, and this file. Including it would make the manifest digest change whenever
    a measuring tool changed — and self-referentially, whenever the pin below was
    re-pinned. Code coverage belongs to ``runtime_digest()``, which names its inputs.
    """

    from tempfile import TemporaryDirectory

    from .arms import arm_manifest, build_arm, control_spec

    with TemporaryDirectory() as tmp:
        arm = build_arm(
            control_spec(),
            run_dir=Path(tmp) / "run",
            inner_client=_SilentClient(),
        )
        try:
            manifest = arm_manifest(arm)
        finally:
            arm.close()
    host_only = manifest.get("host_only", {})
    host_only.pop("source_sha256", None)
    host_only.pop("run_dir", None)
    return manifest


def control_manifest_digest() -> str:
    return _digest(control_manifest())


def verify() -> dict[str, Any]:
    """Report what moved. Empty ``drift`` means the control is unchanged."""

    runtime, manifest = runtime_digest(), control_manifest_digest()
    drift = []
    if FROZEN_RUNTIME_DIGEST and runtime != FROZEN_RUNTIME_DIGEST:
        drift.append(
            {
                "digest": "runtime",
                "expected": FROZEN_RUNTIME_DIGEST,
                "actual": runtime,
                "moved": _moved(),
            }
        )
    if FROZEN_CONTROL_MANIFEST_DIGEST and manifest != FROZEN_CONTROL_MANIFEST_DIGEST:
        drift.append(
            {
                "digest": "control_manifest",
                "expected": FROZEN_CONTROL_MANIFEST_DIGEST,
                "actual": manifest,
            }
        )
    return {
        "arm_id": CONTROL_ARM_ID,
        "pinned": bool(FROZEN_RUNTIME_DIGEST and FROZEN_CONTROL_MANIFEST_DIGEST),
        "runtime_digest": runtime,
        "control_manifest_digest": manifest,
        "drift": drift,
    }


def _moved() -> list[str]:
    """Which runtime components differ from the recorded per-component digests."""

    frozen = FROZEN_RUNTIME_COMPONENTS
    if not frozen:
        return []
    current = runtime_components()
    return sorted(
        set(frozen) ^ set(current)
        | {name for name in set(frozen) & set(current) if frozen[name] != current[name]}
    )


def _digest(payload: Any) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class _SilentClient:
    """A client that is never called: the manifest reads structure, not responses."""

    def complete(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise AssertionError("manifest construction must not reach the model")


def main() -> int:
    report = verify()
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pinned"]:
        print("\nNot pinned yet. Paste into freeze.py:")
        print(f'FROZEN_RUNTIME_DIGEST = "{report["runtime_digest"]}"')
        print(
            "FROZEN_CONTROL_MANIFEST_DIGEST = "
            f'"{report["control_manifest_digest"]}"'
        )
        print("FROZEN_RUNTIME_COMPONENTS = " + json.dumps(runtime_components(), indent=4))
        return 1
    return 1 if report["drift"] else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
