"""Shared check/report plumbing for the four P9 prototypes.

Each prototype states its checks by name, records what it observed, and fails
loudly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: Any = None

    @property
    def passed(self) -> bool:
        return self.status == "pass"


class PrototypeFailure(AssertionError):
    pass


@dataclass
class Checks:
    prototype: str
    question: str
    items: list[Check] = field(default_factory=list)

    def record(self, name: str, condition: bool, detail: Any = None) -> Check:
        check = Check(name=name, status="pass" if condition else "fail", detail=detail)
        self.items.append(check)
        return check

    @property
    def failed(self) -> list[Check]:
        return [item for item in self.items if not item.passed]

    def report(self, **extra: Any) -> dict[str, Any]:
        return {
            "schema": "p9-prototype-evidence-v1",
            "prototype": self.prototype,
            "question": self.question,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "passed": not self.failed,
            "check_count": len(self.items),
            "failed_count": len(self.failed),
            "checks": [
                {"name": item.name, "status": item.status, "detail": item.detail}
                for item in self.items
            ],
            **extra,
        }

    def finish(self, path: Path, **extra: Any) -> dict[str, Any]:
        report = self.report(**extra)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "prototype": self.prototype,
                    "passed": report["passed"],
                    "checks": report["check_count"],
                    "failed": [item.name for item in self.failed],
                    "evidence": str(path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        if self.failed:
            rendered = "\n".join(
                f"  {item.name}: {item.status} {item.detail!r}" for item in self.failed
            )
            raise PrototypeFailure(f"{self.prototype} failed checks:\n{rendered}")
        return report


def artifacts_root() -> Path:
    """Where prototypes write evidence.

    ``P9_ARTIFACTS_DIR`` redirects it. The mutation probe sets that variable: it runs
    the prototypes against deliberately broken source, and ``finish`` writes the
    evidence file before it raises, so without the redirect every probe run
    overwrites the committed artifacts with mutant output.
    """

    override = os.environ.get("P9_ARTIFACTS_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / "artifacts"
