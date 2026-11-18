"""Deterministic predecessor-chain validator: every node resolves to real bytes.

A node binds a predecessor identity directory and the exact files that must exist
and hash there. The chain must descend one candidate revision per step
(t1rN -> t1r(N-1) -> ... -> t1r1 -> t1), terminate at t1, contain no cycles, and
list its predecessor's own PREDECESSOR pointer at every non-terminal node so the
provenance is tamper-bound end to end. Launch/audit artifacts that live outside a
candidate directory are bound by their real repo-relative paths.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

CANDIDATE = re.compile(r"operational-theory-(t1(?:r\d+)?)-\d{8}")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _token(identity):
    match = CANDIDATE.search(identity)
    return match.group(1) if match else None


def _next_expected(token):
    """The only candidate that may follow `token` in a well-formed chain."""
    if token == "t1":
        return None
    if token == "t1r1":
        return "t1"
    match = re.fullmatch(r"t1r(\d+)", token)
    return f"t1r{int(match.group(1)) - 1}" if match else None


def _check_files(node, base, root, errors, where):
    for name, expected in node.get("files", {}).items():
        target = base / name
        if target.is_symlink() or not target.is_file():
            errors.append(f"{where}: missing file {name} at {base}")
        elif _sha(target) != expected:
            errors.append(f"{where}: hash mismatch {name} at {base}")
    for group in ("launch", "audit"):
        for rel, expected in node.get(group, {}).items():
            target = root / rel
            if target.is_symlink() or not target.is_file():
                errors.append(f"{where}: missing {group} artifact {rel}")
            elif _sha(target) != expected:
                errors.append(f"{where}: hash mismatch {group} artifact {rel}")


def validate_chain(predecessor_path, root):
    """Return (ok, errors, sequence-of-candidate-tokens)."""
    root = Path(root)
    errors = []
    sequence = []
    seen = set()
    node = json.loads(Path(predecessor_path).read_text())
    depth = 0
    while node is not None:
        where = f"node[{depth}]"
        identity = node.get("identity")
        if not isinstance(identity, str):
            errors.append(f"{where}: missing identity")
            break
        if identity in seen:
            errors.append(f"{where}: cycle at {identity}")
            break
        seen.add(identity)
        token = _token(identity)
        if token is None:
            errors.append(f"{where}: identity is not a candidate dir: {identity}")
            break
        sequence.append(token)
        base = root / identity
        if not base.is_dir():
            errors.append(f"{where}: identity dir absent: {identity}")
        _check_files(node, base, root, errors, where)
        child = node.get("earlier_chain")
        if child is not None:
            child_token = _token(child.get("identity", ""))
            if child_token != _next_expected(token):
                errors.append(
                    f"{where}: sequence skip {token} -> {child_token}"
                    f" (expected {_next_expected(token)})"
                )
        else:
            if token != "t1":
                errors.append(f"{where}: chain ends at {token}, not terminal t1")
            if "PREDECESSOR.json" in node.get("files", {}):
                errors.append(f"{where}: terminal t1 must not bind a PREDECESSOR.json")
        if token != "t1" and "PREDECESSOR.json" not in node.get("files", {}):
            errors.append(
                f"{where}: non-terminal {token} does not bind its predecessor pointer"
            )
        node = child
        depth += 1
    return (not errors, errors, sequence)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predecessor", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    ok, errors, sequence = validate_chain(args.predecessor, args.root)
    print("sequence:", " -> ".join(sequence))
    for error in errors:
        print("ERROR:", error)
    print("RESULT:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
