#!/usr/bin/env python3
"""Authoritative kernel status, straight from the REST endpoint.

`kaggle kernels status` (CLI 2.2.4) printed `KernelWorkerStatus.QUEUED` for two
kernels that were demonstrably running — GPU quota was accruing at two GPU-minutes
per wall-clock minute — while this endpoint returned `running` for the same two.
The CLI string is not trustworthy for terminal detection, so the watcher reads
here instead.

    python kernel_status.py <slug> [<slug> ...]

Prints `<slug> <status> <failureMessage>` per line. Exit 0 if every named kernel
is terminal (complete/error/cancel), 1 otherwise.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

CREDENTIALS = Path.home() / ".kaggle" / "credentials.json"
TERMINAL = {"complete", "error", "cancelacknowledged", "cancelrequested"}


def status(user: str, token: str, slug: str) -> tuple[str, str]:
    url = f"https://www.kaggle.com/api/v1/kernels/status?userName={user}&kernelSlug={slug}"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode())
    return payload.get("status", "unknown"), payload.get("failureMessage", "")


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: kernel_status.py <slug> [<slug> ...]")
    credentials = json.loads(CREDENTIALS.read_text())
    user = credentials["username"]
    # The stored OAuth token expires; a 401 here means re-authenticate, not that
    # the kernel is gone.
    token = credentials["access_token"]
    all_terminal = True
    for slug in sys.argv[1:]:
        try:
            state, failure = status(user, token, slug)
        except Exception as error:  # noqa: BLE001 - a transient poll failure is not terminal
            print(f"{slug} poll-failed {type(error).__name__}: {error}")
            all_terminal = False
            continue
        print(f"{slug} {state} {failure}".rstrip())
        if state.lower() not in TERMINAL:
            all_terminal = False
    return 0 if all_terminal else 1


if __name__ == "__main__":
    raise SystemExit(main())
