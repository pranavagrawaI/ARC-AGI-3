"""Credential-free provider request assembly and response normalization."""

from __future__ import annotations

from base64 import b64encode
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .contract import TOOL_ARGUMENT_FIELDS
from .requests import ARMS, visible_case
from .serialization import canonical_bytes, digest


GEMMA4_VISUAL_BUDGETS = {
    70: (8, 8),
    140: (11, 11),
    280: (16, 16),
    560: (23, 23),
}
GEMMA4_VISION_REFERENCE = "https://ai.google.dev/gemma/docs/capabilities/vision/image"

SYSTEM_PROMPT = """You make one bounded decision from public offline game evidence. Preserve unresolved executable alternatives. Use exactly one supplied function tool. EXECUTE only a supplied replay-working plan, PROBE only a supplied discriminator, or ABSTAIN with concrete missing evidence. Never invent an action, plan, outcome, or evidence reference."""


def _json_type(field: str) -> str:
    return {
        "plan_id": "string",
        "first_action": "integer",
        "predicted_public_fields": "object",
        "discriminator_id": "string",
        "action": "integer",
        "predicted_outcomes": "object",
        "reason_code": "string",
        "missing_evidence": "array",
    }[field]


def tool_definitions() -> list[dict[str, Any]]:
    descriptions = {
        "EXECUTE": "Select a supplied replay-working plan and its legal first action.",
        "PROBE": "Select a supplied discriminator and state all of its predicted outcomes.",
        "ABSTAIN": "Decline when named evidence required for a safe decision is missing.",
    }
    tools = []
    for tool, fields in TOOL_ARGUMENT_FIELDS.items():
        properties: dict[str, Any] = {}
        for field in fields:
            schema: dict[str, Any] = {"type": _json_type(field)}
            if field == "missing_evidence":
                schema.update({"items": {"type": "string"}, "minItems": 1})
            elif schema["type"] == "object":
                schema["additionalProperties"] = True
            properties[field] = schema
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool,
                    "description": descriptions[tool],
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": list(fields),
                        "additionalProperties": False,
                    },
                },
            }
        )
    return tools


def _delivered_png(path: Path, step: int, labelled: bool) -> tuple[bytes, tuple[int, int]]:
    with Image.open(path) as source:
        image = source.convert("RGB")
    if labelled:
        draw = ImageDraw.Draw(image)
        text = f"STEP {step}"
        font = ImageFont.load_default(size=18)
        box = draw.textbbox((0, 0), text, font=font)
        padding = 4
        draw.rectangle(
            (0, 0, box[2] + padding * 2, box[3] + padding * 2), fill=(255, 255, 255)
        )
        draw.text((padding, padding), text, fill=(0, 0, 0), font=font)
    output = BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue(), image.size


def assemble_provider_request(
    case: dict[str, Any], arm: str, experiment_root: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    logical = visible_case(case, arm)
    visual_budget = int(config["visual_token_budget"])
    if visual_budget not in GEMMA4_VISUAL_BUDGETS:
        raise ValueError(f"unsupported Gemma 4 visual token budget {visual_budget}")
    labelled = arm in {"C", "D", "E", "F", "G", "H"}
    content: list[dict[str, Any]] = []
    provenance = []
    for image_ref in logical.pop("images"):
        source = experiment_root / image_ref["path"]
        source_bytes = source.read_bytes()
        if sha256(source_bytes).hexdigest() != image_ref["sha256"]:
            raise ValueError(f"image digest mismatch: {source}")
        delivered, size = _delivered_png(source, int(image_ref["step"]), labelled)
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64," + b64encode(delivered).decode("ascii")
                },
            }
        )
        grid = GEMMA4_VISUAL_BUDGETS[visual_budget]
        provenance.append(
            {
                "step": image_ref["step"],
                "source_path": image_ref["path"],
                "source_sha256": image_ref["sha256"],
                "source_pixel_size": image_ref["size"],
                "chronology_label_rendered": labelled,
                "delivered_sha256": sha256(delivered).hexdigest(),
                "delivered_pixel_size": list(size),
                "requested_visual_token_budget": visual_budget,
                "requested_visual_token_grid": list(grid),
                "reference_realized_visual_token_grid": list(grid),
                "reference_realized_visual_tokens": grid[0] * grid[1],
                "provider_reported_visual_token_grid": None,
                "reference": GEMMA4_VISION_REFERENCE,
            }
        )
    content.append(
        {
            "type": "text",
            "text": json.dumps(logical, sort_keys=True, indent=2, ensure_ascii=False),
        }
    )
    request: dict[str, Any] = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": config.get("system_prompt", SYSTEM_PROMPT)},
            {"role": "user", "content": content},
        ],
        "tools": tool_definitions(),
        "tool_choice": "required",
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "top_k": config["top_k"],
        "max_tokens": config["max_completion_tokens"],
        "reasoning": deepcopy(config["reasoning"]),
        "seed": config["seed"],
    }
    if config.get("provider"):
        request["provider"] = deepcopy(config["provider"])
    audit = {
        "schema": "0003-request-audit-v1",
        "case_id": case["case_id"],
        "arm": arm,
        "logical_request_sha256": digest(visible_case(case, arm)),
        "provider_request_sha256": digest(request),
        "images": provenance,
    }
    return request, audit


def prepare_requests(
    batch_path: Path, experiment_root: Path, config: dict[str, Any]
) -> list[dict[str, Any]]:
    batch = json.loads(batch_path.read_text())
    rows = []
    for case in batch["cases"]:
        for arm in ARMS:
            request, audit = assemble_provider_request(case, arm, experiment_root, config)
            rows.append(
                {
                    "case_id": case["case_id"],
                    "arm": arm,
                    "request": request,
                    "audit": audit,
                }
            )
    return rows


def prepared_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(canonical_bytes(row) + b"\n" for row in rows)


def normalize_response(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("provider response must contain exactly one choice")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("provider response choice has no message")
    calls = message.get("tool_calls")
    if isinstance(calls, list) and len(calls) == 1:
        function = calls[0].get("function", {})
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        normalized = {"tool": function.get("name"), "arguments": arguments}
    elif not calls and isinstance(message.get("content"), str):
        normalized = json.loads(message["content"])
    else:
        raise ValueError("provider response must contain exactly one tool call")
    return {
        "decision": normalized,
        "model": payload.get("model"),
        "provider": payload.get("provider"),
        "usage": payload.get("usage", {}),
        "finish_reason": choices[0].get("finish_reason"),
        "reasoning": message.get("reasoning") or message.get("reasoning_content"),
    }
