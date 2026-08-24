"""Versioned, byte-exact dependencies available to canonical world models."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from tycho.workspace import wmlib_template

MANIFEST_PATH = ".trusted-runtime.json"
SCHEMA_VERSION = 1
_MODULE_CONTRACTS = {
    "wmlib.py": (
        "wmlib",
        "tycho.workspace.wmlib_template",
        ("actor", "builder", "verifier", "planner", "transaction_evaluator"),
    ),
    "symmetry.py": (
        "symmetry",
        "tycho_nooa.symmetry_template",
        ("actor", "builder", "verifier", "planner", "transaction_evaluator"),
    ),
    "host_planner_contract.py": (
        "host_planner_contract",
        "tycho_nooa.probe_gate",
        ("verifier", "planner", "transaction_evaluator"),
    ),
    "theory_probe.py": (
        "theory_probe",
        "tycho_nooa.templates.theory_evaluation_probe",
        ("transaction_evaluator",),
    ),
}
RUNTIME_MODULE_PATHS = frozenset(_MODULE_CONTRACTS)
TRUSTED_PATHS = RUNTIME_MODULE_PATHS | {MANIFEST_PATH}


class TrustedRuntimeError(RuntimeError):
    """The declared runtime is malformed, incomplete, or no longer byte-exact."""


class RuntimeState(StrEnum):
    DECLARED = "declared"
    LEGACY_UNKNOWN = "legacy_unknown"


@dataclass(frozen=True)
class TrustedModule:
    path: str
    import_name: str
    provenance: str
    availability: tuple[str, ...]
    sha256: str
    content: bytes

    def __post_init__(self) -> None:
        contract = _MODULE_CONTRACTS.get(self.path)
        if contract is None or (
            self.import_name,
            self.provenance,
            self.availability,
        ) != contract:
            raise TrustedRuntimeError(f"invalid trusted module identity for {self.path!r}")
        if self.sha256 != hashlib.sha256(self.content).hexdigest():
            raise TrustedRuntimeError(f"trusted module content hash mismatch for {self.path}")

    @classmethod
    def from_source(
        cls,
        path: str,
        import_name: str,
        provenance: str,
        content: bytes,
        availability: tuple[str, ...] | None = None,
    ) -> TrustedModule:
        if availability is None and path in _MODULE_CONTRACTS:
            availability = _MODULE_CONTRACTS[path][2]
        return cls(
            path=path,
            import_name=import_name,
            provenance=provenance,
            availability=availability or (),
            sha256=hashlib.sha256(content).hexdigest(),
            content=content,
        )

    def manifest_entry(self) -> dict[str, object]:
        return {
            "availability": list(self.availability),
            "import_name": self.import_name,
            "path": self.path,
            "provenance": self.provenance,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class TrustedRuntime:
    modules: tuple[TrustedModule, ...]
    sha256: str

    def __post_init__(self) -> None:
        paths = tuple(module.path for module in self.modules)
        if len(paths) != len(set(paths)) or set(paths) != RUNTIME_MODULE_PATHS:
            raise TrustedRuntimeError("trusted runtime must contain each required module once")
        if self.sha256 != _commitment(_manifest_body(self.modules)):
            raise TrustedRuntimeError("trusted runtime commitment does not match its modules")

    @classmethod
    def create(cls, modules: tuple[TrustedModule, ...]) -> TrustedRuntime:
        body = _manifest_body(modules)
        return cls(modules=modules, sha256=_commitment(body))

    @classmethod
    def from_workspace(cls, workspace: Path) -> TrustedRuntime:
        manifest_path = workspace / MANIFEST_PATH
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise TrustedRuntimeError(f"cannot read {MANIFEST_PATH}: {error}") from error
        modules = _load_declared_modules(workspace, manifest)
        runtime = cls.create(modules)
        declared_sha = manifest.get("runtime_sha256")
        if declared_sha != runtime.sha256:
            raise TrustedRuntimeError(
                f"{MANIFEST_PATH} runtime hash mismatch: "
                f"declared {declared_sha!r}, computed {runtime.sha256!r}"
            )
        return runtime

    def manifest_bytes(self) -> bytes:
        manifest = _manifest_body(self.modules) | {"runtime_sha256": self.sha256}
        return _canonical_json(manifest) + b"\n"

    def write_to(self, workspace: Path) -> None:
        workspace.mkdir(parents=True, exist_ok=True)
        for module in self.modules:
            _atomic_write(workspace / module.path, module.content)
        # Publishing the declaration last prevents a partial write from claiming integrity.
        _atomic_write(workspace / MANIFEST_PATH, self.manifest_bytes())


@dataclass(frozen=True)
class RuntimeInspection:
    state: RuntimeState
    runtime: TrustedRuntime | None
    observed_paths: tuple[str, ...] = ()


def current_runtime() -> TrustedRuntime:
    return TrustedRuntime.create(
        (
            TrustedModule.from_source(
                "wmlib.py",
                "wmlib",
                "tycho.workspace.wmlib_template",
                Path(wmlib_template.__file__).read_bytes(),
            ),
            TrustedModule.from_source(
                "symmetry.py",
                "symmetry",
                "tycho_nooa.symmetry_template",
                Path(__file__).with_name("symmetry_template.py").read_bytes(),
            ),
            TrustedModule.from_source(
                "host_planner_contract.py",
                "host_planner_contract",
                "tycho_nooa.probe_gate",
                Path(__file__).with_name("probe_gate.py").read_bytes(),
            ),
            TrustedModule.from_source(
                "theory_probe.py",
                "theory_probe",
                "tycho_nooa.templates.theory_evaluation_probe",
                (
                    Path(__file__).with_name("templates")
                    / "theory_evaluation_probe.py.tmpl"
                ).read_bytes(),
            ),
        )
    )


def inspect_runtime(workspace: Path) -> RuntimeInspection:
    if (workspace / MANIFEST_PATH).is_file():
        return RuntimeInspection(
            RuntimeState.DECLARED, TrustedRuntime.from_workspace(workspace)
        )
    observed = tuple(
        sorted(
            path
            for path in RUNTIME_MODULE_PATHS
            if (workspace / path).is_file()
        )
    )
    return RuntimeInspection(RuntimeState.LEGACY_UNKNOWN, None, observed)


def _load_declared_modules(
    workspace: Path, manifest: object
) -> tuple[TrustedModule, ...]:
    entries = _manifest_entries(manifest)
    modules = tuple(_load_declared_module(workspace, entry) for entry in entries)
    if {module.path for module in modules} != RUNTIME_MODULE_PATHS:
        raise TrustedRuntimeError(f"{MANIFEST_PATH} does not declare the complete runtime")
    return modules


def _manifest_entries(manifest: object) -> list[dict[str, object]]:
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise TrustedRuntimeError(f"unsupported {MANIFEST_PATH} schema")
    entries = manifest.get("modules")
    if not isinstance(entries, list) or not entries:
        raise TrustedRuntimeError(f"{MANIFEST_PATH} has no modules")
    for entry in entries:
        if not isinstance(entry, dict):
            raise TrustedRuntimeError(f"invalid module entry in {MANIFEST_PATH}")
    return entries


def _load_declared_module(
    workspace: Path, entry: dict[str, object]
) -> TrustedModule:
    path = entry.get("path")
    if path not in RUNTIME_MODULE_PATHS:
        raise TrustedRuntimeError(f"unrecognized trusted runtime path {path!r}")
    assert isinstance(path, str)
    try:
        content = (workspace / path).read_bytes()
    except OSError as error:
        raise TrustedRuntimeError(f"cannot read {path}: {error}") from error
    actual_sha = hashlib.sha256(content).hexdigest()
    if entry.get("sha256") != actual_sha:
        raise TrustedRuntimeError(f"trusted runtime hash mismatch for {path}")
    import_name, provenance, availability = _module_metadata(path, entry)
    return TrustedModule(
        path,
        import_name,
        provenance,
        availability,
        actual_sha,
        content,
    )


def _module_metadata(
    path: str, entry: dict[str, object]
) -> tuple[str, str, tuple[str, ...]]:
    import_name = entry.get("import_name")
    provenance = entry.get("provenance")
    availability = entry.get("availability")
    if (
        not isinstance(import_name, str)
        or not isinstance(provenance, str)
        or not isinstance(availability, list)
        or not all(isinstance(item, str) for item in availability)
    ):
        raise TrustedRuntimeError(f"invalid metadata for {path}")
    return import_name, provenance, tuple(availability)


def _manifest_body(modules: tuple[TrustedModule, ...]) -> dict[str, object]:
    return {
        "availability": [
            "actor",
            "builder",
            "verifier",
            "planner",
            "transaction_evaluator",
        ],
        "integrity": "read_only",
        "modules": [module.manifest_entry() for module in modules],
        "restore_policy": "exact_version",
        "schema_version": SCHEMA_VERSION,
    }


def _commitment(body: dict[str, object]) -> str:
    return hashlib.sha256(_canonical_json(body)).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def _atomic_write(path: Path, content: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
