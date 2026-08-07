"""Four-call symbolic transformation competence microbattery."""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
MODEL = "google/gemma-4-31b-it"
RULES = {
    "integer_scale": "replace every cell by an f-by-f block of itself",
    "tiled_repetition": "repeat the whole input grid r times vertically and c times horizontally",
    "dihedral_congruence": "rotate by 0/90/180/270 degrees and optionally reflect; values stay unchanged",
    "colour_permutation": "keep all positions but apply one consistent bijection to cell values",
    "embedded_subpattern": "the input occurs exactly as one contiguous rectangle inside the output",
    "advance_mod_n": "each next integer equals the previous integer plus delta modulo n",
}
CASES = [
    ("scale_vs_tile", {"input": [[1, 2], [3, 4]], "output": [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]}, {"relation": "integer_scale", "parameters": {"f": 2}}),
    ("congruence", {"input": [[1, 2, 3], [4, 5, 6]], "output": [[4, 1], [5, 2], [6, 3]]}, {"relation": "dihedral_congruence", "parameters": {"rotation_degrees": 90}}),
    ("colour_or_embed", {"input": [[1, 2], [1, 1]], "output": [[7, 8], [7, 7]]}, {"relation": "colour_permutation", "parameters": {"mapping": {"1": 7, "2": 8}}}),
    ("modular_progression", {"sequence": [3, 0, 1, 2, 3, 0]}, {"relation": "advance_mod_n", "parameters": {"n": 4, "delta": 1}}),
]
SYSTEM = "Use the closed rule inventory exactly. Return one JSON object only with relation and parameters."


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def final_object(text: str) -> dict | None:
    objects = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.S)
    for candidate in reversed(objects):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "relation" in value:
            return value
    return None


def main() -> int:
    key = os.environ.get("ARC_LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("source .env before running")
    out = HERE / "runs" / ("transformation-microbattery-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir(parents=True)
    results = []
    for name, evidence, expected in CASES:
        user = "Rule inventory:\n" + json.dumps(RULES, sort_keys=True) + "\nEvidence:\n" + json.dumps(evidence)
        request = {"model": MODEL, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], "temperature": 0, "max_tokens": 2000, "reasoning": {"enabled": False}}
        if json.dumps(expected, sort_keys=True) in json.dumps(request, sort_keys=True):
            raise RuntimeError("expected answer leaked into request")
        write(out / f"{name}-request.json", request)
        req = Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(request, separators=(",", ":")).encode(), headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        with urlopen(req, timeout=300) as response:
            payload = json.loads(response.read())
        write(out / f"{name}-response.json", payload)
        observed = final_object(payload["choices"][0]["message"].get("content") or "")
        hit = observed == expected
        results.append({"case": name, "expected": expected, "observed": observed, "hit": hit, "usage": payload.get("usage", {})})
        print(name, "PASS" if hit else "FAIL", observed)
    write(out / "score.json", {"results": results, "passed": sum(row["hit"] for row in results), "total": len(results), "provider_calls": 4, "environment_actions": 0})
    print("run:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
