"""Host owns active helper bytes. Bound names are sealed; revision uses a new name."""

import hashlib
import re
import textwrap
from pathlib import Path

from theory import IntegrityError, FormatError


def active_bindings(theory):
    result = {}
    if not isinstance(theory, (dict, type(None))):
        raise FormatError("theory must be an object")
    procedures = (theory or {}).get("procedures", [])
    if not isinstance(procedures, list):
        raise FormatError("procedures must be a list")
    for entry in procedures:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("path"), str)
            or not isinstance(entry.get("sha256"), str)
        ):
            raise FormatError("procedure requires path and sha256")
        match = re.fullmatch(r"helpers/([a-z0-9_]+\.py)@([0-9a-f]{64})", entry["path"])
        if not match or match[2] != entry["sha256"]:
            raise FormatError(
                "procedure path must be an exact registered helper version"
            )
        name = match[1]
        if name in result and result[name] != entry["sha256"]:
            raise FormatError("one active byte identity per helper name")
        result[name] = entry["sha256"]
    return result


class Helpers:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir()
        self.sealed = {}

    def check(self, theory):
        for name, sha in active_bindings(theory).items():
            path = self.root / name
            if (
                path.is_symlink()
                or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != sha
            ):
                raise IntegrityError(
                    "active helper differs from canonical bytes: " + name
                )

    def check_proposal(self, theory, artifacts):
        """Unaccepted references are repairable; accepted-state checks stay strict."""
        for name, sha in active_bindings(theory).items():
            reference = f"helpers/{name}@{sha}"
            if artifacts.get(reference) != sha:
                raise FormatError(
                    f"proposed helper {reference} is unregistered; write the helper, "
                    "register its exact bytes, and revise using the returned path/sha256"
                )
            path = self.root / name
            if (
                path.is_symlink()
                or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != sha
            ):
                raise FormatError(
                    f"proposed helper {reference} does not match active bytes; "
                    "write/register the intended bytes and revise their exact binding"
                )

    def seal(self, theory):
        self.check(theory)
        self.sealed.update(active_bindings(theory))

    def write(self, filename, source):
        # Same syntax/name/AST contract as the incumbent write_helper.
        from solver_agent import _check_helper_ast

        if (
            not isinstance(filename, str)
            or re.fullmatch(r"[a-z0-9_]+\.py", filename) is None
            or not isinstance(source, str)
        ):
            return "REJECTED: invalid helper filename/source"
        source = textwrap.dedent(source)
        error = _check_helper_ast(source)
        if error:
            return "REJECTED: " + error
        raw = source.encode()
        sha = hashlib.sha256(raw).hexdigest()
        if filename in self.sealed and sha != self.sealed[filename]:
            return "REJECTED: canonical helper is sealed; write a new filename and revise the theory to bind it before authoritative use"
        (self.root / filename).write_bytes(raw)
        return f"wrote helpers/{filename} ({len(source)} chars). Call load_helpers() to use it."

    def register(self, store, filename, claimed_source):
        if (
            not isinstance(filename, str)
            or re.fullmatch(r"[a-z0-9_]+\.py", filename) is None
        ):
            return store.feedback("invalid helper filename")
        path = self.root / filename
        if (
            path.is_symlink()
            or not path.is_file()
            or path.read_text() != claimed_source
        ):
            return store.feedback(
                "artifact registration must match active host helper bytes"
            )
        return store.register(filename, path.read_text())
