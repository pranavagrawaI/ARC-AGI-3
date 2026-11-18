"""Strict input contract for a frozen, full-request Actor replay."""

import hashlib
import json


def validate(value):
    if value.get("schema") != "actor-replay-v2":
        raise ValueError("Expected schema actor-replay-v2")
    records = value.get("records", [])
    if not 1 <= len(records) <= 14:
        raise ValueError("Expected 1..14 captured cases")
    if not isinstance(value.get("generation_config"), dict):
        raise ValueError("Captured generation defaults required")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Case IDs must be unique")
    for record in records:
        request = record["request"]
        if not record.get("provenance"):
            raise ValueError(
                "Every case needs capture provenance; reconstructed histories must be labelled"
            )
        if not request.get("messages") or not request.get("tools"):
            raise ValueError(
                "Full messages and tool schemas required, not response/token-count records"
            )
        if request.get("stream", False) or request.get("n", 1) != 1:
            raise ValueError(
                "This replay requires the captured non-streaming single-response Actor path"
            )
        # Omitted sampling/output-limit fields are part of the captured contract.
        # Bound the trial externally, not by silently rewriting model requests.
        if "request_body" in record:
            raw = record["request_body"].encode()
            if (
                json.loads(raw) != request
                or hashlib.sha256(raw).hexdigest()
                != record["provenance"]["request_sha256"]
            ):
                raise ValueError("Captured request identity mismatch")
        if record["kind"] == "tool-continuation" and not any(
            m["role"] == "tool" for m in request["messages"]
        ):
            raise ValueError("Tool continuation must contain the actual tool result")
        if record["kind"] == "forced-action" and request.get("tool_choice") in (
            None,
            "auto",
            "none",
        ):
            raise ValueError(
                "Forced-action case needs the captured enforcing tool_choice"
            )
        if any(k in request for k in ("api_key", "api_base", "base_url", "headers")):
            raise ValueError("Do not embed transport credentials")
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    if len(encoded) > 12_000_000:
        raise ValueError("Corpus exceeds 12 MB notebook budget")
    return hashlib.sha256(encoded).hexdigest()
