"""Rebuild the exact provider request behind a failed crossing, and replay it.

A failed run records the request body, but the audit replaces every image with a
stub -- ``{"bytes", "kind", "media_type", "redacted", "sha256"}`` -- so the
recorded body is not the body that was sent. Replaying it unchanged produces a
*different* provider error, which is easy to mistake for the original. That
mistake cost a wrong diagnosis and nearly cost an unnecessary change to frozen
code: the recorded failure was a 476-character message, and the replayed one was
29 characters, and nothing in the artifact made the difference obvious.

So this tool does two things, and the second matters more than the first:

1. It restores each redacted image by content, verifying the sha256 the stub
   records, and refuses to proceed on any stub it cannot resolve.
2. It compares the replayed provider error against the recorded one and says
   plainly whether the reproduction is faithful. A mismatch means the replay is
   evidence about the replay, not about the run.

Reading a failed run is how the ``return_result`` replay defect was found, after
six live runs had failed in six apparently different ways. Preserve the ability.

No provider call happens without ``--send``.
"""

from __future__ import annotations

import argparse
import json
import sys
from base64 import b64encode
from hashlib import sha256
from pathlib import Path
from typing import Any

TREATMENT_ROOT = Path(__file__).resolve().parent


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def failed_crossing(run_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the request body and the recorded failure for the failed crossing."""

    records = _read_jsonl(run_root / "worker" / "process.jsonl")
    for record in records:
        if record.get("kind") != "model_crossing":
            continue
        for call in (record.get("audit") or {}).get("model_calls") or []:
            outcome = call.get("outcome") or {}
            if outcome.get("kind") != "transport_failure":
                continue
            for attempt in call.get("attempts") or []:
                for crossing in attempt.get("actual_provider_crossings") or []:
                    body = crossing.get("request", {}).get("body")
                    if isinstance(body, dict):
                        return body, outcome
    raise SystemExit("no failed provider crossing is recorded in this run")


def _candidate_images(run_root: Path) -> dict[str, bytes]:
    """Index every preserved image in the run by content hash.

    ``model-frame.png`` is overwritten at each action point, so only the newest
    survives directly. Frames for earlier crossings are recovered by
    regenerating the annotation from the recorded public observation, which is
    deterministic; a hash mismatch simply means that candidate is not used.
    """

    by_digest: dict[str, bytes] = {}
    for path in run_root.rglob("*.png"):
        raw = path.read_bytes()
        by_digest.setdefault(sha256(raw).hexdigest(), raw)

    host_observations = run_root / "worker" / "host-observations"
    if not host_observations.is_dir():
        return by_digest
    sys.path.insert(0, str(TREATMENT_ROOT))
    try:
        import numpy as np
        from p9.annotated_frame import annotate
    except ImportError:
        return by_digest
    for path in sorted(host_observations.glob("*.json")):
        try:
            pixels = json.loads(path.read_text(encoding="utf-8")).get("pixels")
            raw = annotate(np.asarray(pixels, dtype="|i1"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        by_digest.setdefault(sha256(raw).hexdigest(), raw)
    return by_digest


def restore_images(body: dict[str, Any], run_root: Path) -> tuple[dict[str, Any], int]:
    """Replace every redaction stub with the bytes whose hash the stub records."""

    by_digest = _candidate_images(run_root)
    restored = 0
    unresolved: list[str] = []

    def visit(node: Any) -> Any:
        nonlocal restored
        if isinstance(node, list):
            return [visit(item) for item in node]
        if not isinstance(node, dict):
            return node
        image = node.get("image_url")
        if isinstance(image, dict) and image.get("redacted") and image.get("sha256"):
            digest = str(image["sha256"])
            raw = by_digest.get(digest)
            if raw is None:
                unresolved.append(digest)
                return {key: visit(value) for key, value in node.items()}
            media = image.get("media_type", "image/png")
            restored += 1
            return {
                **{k: visit(v) for k, v in node.items() if k != "image_url"},
                "image_url": f"data:{media};base64,{b64encode(raw).decode()}",
            }
        return {key: visit(value) for key, value in node.items()}

    rebuilt = visit(body)
    if unresolved:
        raise SystemExit(
            "cannot rebuild the request: no preserved image matches "
            f"{unresolved}. Replaying without them tests a payload the worker "
            "never sent."
        )
    return rebuilt, restored


def _send(body: dict[str, Any], api_base: str, api_key: str) -> tuple[int, str]:
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        api_base.rstrip("/") + "/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8")


def _report_fidelity(recorded: dict[str, Any], status: int, text: str) -> bool:
    """Say whether the replay reproduced the recorded failure, or something else."""

    try:
        error = json.loads(text).get("error", {})
    except json.JSONDecodeError:
        error = {}
    message = error.get("message", "") if error else text
    print(f"replayed status {status}, provider message {len(message)} chars")
    print(f"  {message[:300]!r}")
    # The outer message is often a wrapper. What diagnosed the return_result
    # defect was the upstream provider's own text, carried under metadata.
    raw = (error.get("metadata") or {}).get("raw")
    if raw:
        print(f"  upstream: {str(raw)[:400]}")
    print(
        f"recorded status {recorded.get('status_code')}, "
        f"{recorded.get('message_chars')} chars, "
        f"sha256 {str(recorded.get('message_sha256'))[:16]}"
    )
    if status != recorded.get("status_code"):
        print("NOT FAITHFUL: different status code. This replay is not the run.")
        return False
    if len(message) != recorded.get("message_chars"):
        print(
            "UNCONFIRMED: the recorded length is of the worker's wrapped exception, "
            "not the raw provider message, so lengths need not match. Compare the "
            "provider error text itself before trusting this reproduction."
        )
        return False
    print("FAITHFUL: status and message length both match the recorded failure.")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="write the rebuilt body here")
    parser.add_argument(
        "--send",
        action="store_true",
        help="replay against the provider; requires ARC_LLM_API_KEY",
    )
    parser.add_argument("--api-base", default="https://openrouter.ai/api/v1")
    args = parser.parse_args(argv)

    body, recorded = failed_crossing(args.run_root.resolve())
    rebuilt, restored = restore_images(body, args.run_root.resolve())
    print(f"rebuilt the failed request, restoring {restored} redacted image(s)")
    if args.out:
        args.out.write_text(json.dumps(rebuilt, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")
    if not args.send:
        print("not sent; pass --send to replay against the provider")
        return 0

    import os

    api_key = os.environ.get("ARC_LLM_API_KEY")
    if not api_key:
        raise SystemExit("ARC_LLM_API_KEY must be present to replay")
    status, text = _send(rebuilt, args.api_base, api_key)
    return 0 if _report_fidelity(recorded, status, text) else 1


if __name__ == "__main__":
    raise SystemExit(main())
