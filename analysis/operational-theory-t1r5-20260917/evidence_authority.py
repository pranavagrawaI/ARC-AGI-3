"""Identity-only evidence custody: pinned bytes/inodes and verified broker snapshots."""

import hashlib
import json
import os
import stat
from pathlib import Path

from experiment import catalog, state
from theory import IntegrityError, encoded


def identity(st):
    # atime is intentionally excluded: a verified read may legitimately change it.
    return (
        st.st_dev,
        st.st_ino,
        st.st_mode,
        st.st_nlink,
        st.st_size,
        st.st_mtime_ns,
        st.st_ctime_ns,
    )


class Evidence:
    def __init__(self, output, rows):
        self.output = Path(output)
        self.root = self.output / "host-evidence/ipc"
        self.root.mkdir(parents=True)
        self.original = encoded(rows)
        self.original_sha = hashlib.sha256(self.original).hexdigest()
        self.pinned = {}
        self.dirs = {
            p: (p.stat().st_dev, p.stat().st_ino)
            for p in (self.output, self.root.parent, self.root)
        }
        self.publish(1)

    def rows(self, count=None):
        if hashlib.sha256(self.original).hexdigest() != self.original_sha:
            raise IntegrityError("evidence source value drift")
        return json.loads(self.original)[:count]

    def publish(self, count):
        if self.pinned:
            self.check()
        rows = self.rows(count)
        states = [state(rows[0]["before"], 0, 0)] + [
            state(r["after"], r["step"], r["step"]) for r in rows
        ]
        blobs = {
            self.root / "states.jsonl": "".join(
                json.dumps(s) + "\n" for s in states
            ).encode(),
            self.root / "transitions.jsonl": "".join(
                json.dumps(r) + "\n" for r in rows
            ).encode(),
            self.output / "EVIDENCE.json": encoded(rows),
        }
        self.expected_catalog = encoded(catalog(rows))
        blobs[self.root / "catalog.json"] = encoded(
            {
                "frontier": count,
                "catalog": catalog(rows),
                "sha256": {
                    p.name: hashlib.sha256(b).hexdigest() for p, b in blobs.items()
                },
            }
        )
        self.pinned = {}
        for path, raw in blobs.items():
            path.write_bytes(raw)
            st = path.stat()
            self.pinned[path] = (
                identity(st),
                hashlib.sha256(raw).hexdigest(),
                raw,
            )
        self.count = count
        self.check()

    def check(self, store=None):
        try:
            self.rows()
            for path, directory_identity in self.dirs.items():
                st = path.lstat()
                if (
                    not stat.S_ISDIR(st.st_mode)
                    or (st.st_dev, st.st_ino) != directory_identity
                ):
                    raise IntegrityError("evidence directory relink: " + str(path))
            if set(self.root.iterdir()) != {
                p for p in self.pinned if p.parent == self.root
            }:
                raise IntegrityError("evidence file set drift")
            captured = {}
            for path, (
                expected_identity,
                expected_sha,
                expected,
            ) in self.pinned.items():
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(fd, "rb") as stream:
                    before = os.fstat(stream.fileno())
                    if (
                        not stat.S_ISREG(before.st_mode)
                        or identity(before) != expected_identity
                    ):
                        raise IntegrityError("evidence replacement: " + str(path))
                    raw = stream.read(len(expected) + 1)
                    after = os.fstat(stream.fileno())
                linked = path.lstat()
                if (
                    identity(linked) != expected_identity
                    or stat.S_ISLNK(linked.st_mode)
                    or identity(before) != identity(after)
                    or raw != expected
                    or hashlib.sha256(raw).hexdigest() != expected_sha
                ):
                    raise IntegrityError("evidence bytes/identity drift: " + str(path))
                captured[path.name] = raw
            if encoded(catalog(self.rows(self.count))) != self.expected_catalog:
                raise IntegrityError("evidence catalog mismatch")
            if store is not None and encoded(store.evidence) != self.expected_catalog:
                raise IntegrityError("host evidence catalog drift")
            return captured
        except OSError as error:
            raise IntegrityError("evidence custody failure: " + str(error)) from error

    def read(self, name, store=None):
        captured = self.check(store)
        if name not in (
            "states.jsonl",
            "transitions.jsonl",
            "observations.json",
            "catalog.json",
        ):
            raise IntegrityError("unknown evidence object")
        # Return the already verified buffer, never reopen after validation.
        raw = captured.get(name)
        return raw.decode() if raw is not None else None
