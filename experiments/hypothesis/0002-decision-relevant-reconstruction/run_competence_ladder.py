"""Two-rung Gemma competence ladder using the frozen LS20 L1 selector cell."""

from __future__ import annotations

import importlib.util
import json
import os
import random
import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SELECTOR = ROOT / "experiments/hypothesis/0001-inverted-authority/select_probe.py"
MODEL = "google/gemma-4-31b-it"
SYSTEM = """You perform a bounded symbolic selection task. Use only the supplied rule and measurements. Reply with one JSON object and nothing else: {"pair": <index>, "calculation": "<brief calculation>"}."""


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def module():
    spec = importlib.util.spec_from_file_location("ladder_selector", SELECTOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen selector")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def candidates(cell: dict) -> list[dict]:
    rows = []
    for index, (left, right) in enumerate(cell["cands"]):
        a = left.sprites[min(left.sprites)]
        b = right.sprites[min(right.sprites)]
        factors = [
            f for f in range(2, 6)
            if b.extent == (a.extent[0] * f, a.extent[1] * f)
            and b.area == a.area * f * f
        ]
        rows.append({
            "pair": index,
            "left": {"area": a.area, "extent": list(a.extent)},
            "right": {"area": b.area, "extent": list(b.extent)},
            "integer_scale_factors": factors,
        })
    return rows


def prompts(rows: list[dict]) -> list[tuple[str, str]]:
    common = "Candidate order is frozen. Select exactly one pair.\n" + json.dumps(
        [{k: v for k, v in row.items() if k != "integer_scale_factors"} for row in rows]
    )
    return [
        (
            "L5_verified_values",
            "The applicable relation is integer scale normalization. The verifier has already "
            "computed every satisfying factor. Select the unique row whose "
            "integer_scale_factors list is non-empty.\n" + json.dumps(rows),
        ),
        (
            "L4_explicit_rule",
            "The applicable relation is integer scale normalization. A pair satisfies it iff "
            "there is one integer f in [2,5] such that right.extent equals left.extent times f "
            "in both dimensions AND right.area equals left.area times f squared. Compute this "
            "for every candidate and select the unique satisfying pair.\n" + common,
        ),
    ]


def call(key: str, user: str) -> tuple[dict, dict]:
    request = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": 6000,
        "reasoning": {"enabled": False},
    }
    req = Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(request, separators=(",", ":")).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urlopen(req, timeout=300) as response:
        return request, json.loads(response.read())


def main() -> int:
    key = os.environ.get("ARC_LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("source .env before running")
    selector = module()
    cell = selector.cell(selector.load(selector.resolve(selector.TRACES["A"][0])), 1, .6, "colour")
    if not cell or cell.get("answer") is None:
        raise RuntimeError("frozen cell unavailable")
    rows = candidates(cell)
    if selector.scale_rule(cell["cands"]) != cell["answer"]:
        raise RuntimeError("frozen scale rule no longer matches held-out answer")
    out = HERE / "runs" / ("ladder-ls20-l1-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir(parents=True)
    summary = []
    for rung, prompt in prompts(rows):
        if any(term in prompt.lower() for term in ("ls20", "oracle", "answer", "win")):
            raise RuntimeError(f"forbidden request term in {rung}")
        request, response = call(key, prompt)
        write(out / f"{rung}-request.json", request)
        write(out / f"{rung}-response.json", response)
        text = response["choices"][0]["message"].get("content") or ""
        matches = [int(value) for value in re.findall(r'"pair"\s*:\s*(\d+)', text)]
        pick = matches[0] if len(matches) == 1 else None
        result = {
            "rung": rung, "pair": pick, "answer": cell["answer"],
            "hit": pick == cell["answer"], "format_valid": len(matches) == 1,
            "observed_pairs": matches, "usage": response.get("usage", {}),
        }
        summary.append(result)
        print(rung, "pair", pick, "hit", result["hit"])
        if not result["format_valid"] or not result["hit"]:
            break
    write(out / "score.json", {"results": summary, "provider_calls": len(summary), "environment_actions": 0})
    print("run:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
