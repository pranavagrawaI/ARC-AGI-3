"""Replacement-pair paid entry; independent T6 and new user release required."""

import argparse
import json
import os
import re
import time
from decimal import Decimal
from pathlib import Path

from bootstrap import HERE, ROOT
from identity import verify, sha
from supervisor import run_arm
from transport import Transport, Stop, initial_ledger, write, exclusive, MODEL, ENDPOINT


def metadata_check(path):
    data = json.loads(Path(path).read_text())["data"]
    if data["id"] != MODEL:
        raise Stop("metadata model drift; no substitution")
    endpoints = [e for e in data["endpoints"] if e["tag"] == ENDPOINT]
    if len(endpoints) != 1:
        raise Stop("pinned endpoint missing/ambiguous")
    endpoint = endpoints[0]
    pinned = json.loads((HERE / "metadata/selected.json").read_text())["endpoint"]
    for key in (
        "quantization",
        "context_length",
        "max_completion_tokens",
        "provider_name",
        "model_id",
    ):
        if endpoint[key] != pinned[key]:
            raise Stop("endpoint metadata drift: " + key)
    for key in ("prompt", "completion"):
        if Decimal(endpoint["pricing"][key]) != Decimal(pinned["pricing"][key]):
            raise Stop("price drift: new identity required")
    if endpoint["status"] != 0 or not {
        "tools",
        "tool_choice",
        "max_tokens",
        "reasoning",
    } <= set(endpoint["supported_parameters"]):
        raise Stop("endpoint inactive or unsupported fields")
    return sha(path)


def gates(review, authorization, metadata):
    manifest = verify()
    for path, role, decision in (
        (review, "independent-reviewer", "PASS"),
        (authorization, "user", "AUTHORIZE_REPLACEMENT_PAIR"),
    ):
        record = json.loads(Path(path).read_text())
        if (
            record.get("role") != role
            or record.get("decision") != decision
            or record.get("manifest_sha256") != manifest
            or record.get("task") != "operational-theory-T1R4"
        ):
            raise Stop("missing exact-byte T6 review or replacement-pair authorization")
    meta_sha = metadata_check(metadata)
    auth = json.loads(Path(authorization).read_text())
    if auth.get("metadata_sha256") != meta_sha:
        raise Stop("replacement release must bind its refreshed metadata")
    return manifest, meta_sha


def reconcile(ledger):
    state = json.loads(Path(ledger).read_text())
    attempts = state["attempts"]
    for arm, totals in state["arms"].items():
        own = [a for a in attempts if a["arm"] == arm]
        if totals["requests"] != len(own):
            raise Stop("request accounting mismatch")
        for key in ("prompt_tokens", "completion_tokens"):
            if totals[key] != sum(a.get("usage", {}).get(key, 0) for a in own):
                raise Stop("receipt accounting mismatch")
    if Decimal(state["known_cost_usd"]) != sum(
        (
            Decimal(a["actual_cost_usd"])
            for a in attempts
            if a["actual_cost_usd"] is not None
        ),
        Decimal(0),
    ):
        raise Stop("cost accounting mismatch")
    unknown = [a for a in attempts if a["actual_cost_usd"] is None]
    return {
        "requests": len(attempts),
        "known_cost_usd": state["known_cost_usd"],
        "unresolved_requests": len(unknown),
        "unresolved_exposure_usd": state["unknown_exposure_usd"],
        "total_cost_usd": None if unknown else state["known_cost_usd"],
        "arms": state["arms"],
        "matched_caps": {
            "requests": 6,
            "completion_tokens": 24576,
            "per_request_completion_tokens": 4096,
            "prompt_tokens": 600000,
            "wall_seconds": 240,
        },
        "extra_consolidator_calls": 0,
    }


def claim_output(output, manifest, meta_sha, review, authorization):
    """One immutable consumption claim per candidate, across output locations."""
    output = Path(output).absolute()
    if (
        output.parent != ROOT / "analysis"
        or not re.fullmatch(
            r"operational-theory-replacement-[0-9]{8}T[0-9]{6}Z", output.name
        )
        or output.exists()
        or output.is_symlink()
    ):
        raise Stop(
            "output must be a fresh analysis/operational-theory-replacement-YYYYMMDDTHHMMSSZ"
        )
    claims = ROOT / "analysis/operational-theory-replacement-claims"
    claims.mkdir(exist_ok=True)
    record = {
        "manifest_sha256": manifest,
        "metadata_sha256": meta_sha,
        "review_sha256": sha(review),
        "authorization_sha256": sha(authorization),
        "output": str(output),
    }
    exclusive(claims / (manifest + ".json"), record)
    output.mkdir(exist_ok=False)
    live = output / "live"
    live.mkdir()
    exclusive(live / "CONSUMED.json", record)
    return live


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    if not args.execute:
        raise Stop(
            "--execute and new authorization required; T5 makes zero provider calls"
        )
    manifest, meta_sha = gates(args.review, args.authorization, args.metadata)
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise Stop("OPENROUTER_API_KEY absent; no credential-file discovery")
    live = claim_output(
        args.output, manifest, meta_sha, args.review, args.authorization
    )
    initial_ledger(live / "ledger.json")
    start = time.monotonic()
    try:
        for arm in ("control", "treatment"):
            if time.monotonic() - start >= 480:
                raise Stop("pair wall cap")
            verify()
            state = json.loads((live / "ledger.json").read_text())
            if state["stopped"] or Decimal(state["unknown_exposure_usd"]) != 0:
                raise Stop("unresolved/stopped pair")
            duration = min(240, 480 - (time.monotonic() - start))
            transport = Transport(
                arm,
                live / f"{arm}-raw",
                live / "ledger.json",
                time.monotonic() + duration,
                key=key,
            )
            run_arm(arm, live / arm, transport, wall_seconds=duration)
            write(live / "ACCOUNTING.json", reconcile(live / "ledger.json"))
        write(
            live / "RESULT.json",
            {
                "status": "pair_completed",
                "elapsed_seconds": time.monotonic() - start,
                "accounting": reconcile(live / "ledger.json"),
                "gameplay": "not run",
            },
        )
    except BaseException as exc:
        state = json.loads((live / "ledger.json").read_text())
        state["stopped"] = True
        write(live / "ledger.json", state)
        write(
            live / "STOP.json",
            {
                "error_type": type(exc).__name__,
                "reason": str(exc)
                if isinstance(exc, Stop)
                else "local/integrity failure",
                "elapsed_seconds": time.monotonic() - start,
                "accounting": reconcile(live / "ledger.json"),
            },
        )
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
