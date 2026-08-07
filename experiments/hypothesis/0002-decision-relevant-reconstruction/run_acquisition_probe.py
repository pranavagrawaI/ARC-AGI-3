"""One-call prefix-only acquisition gate; no environment actions."""

from __future__ import annotations

import importlib.util
import json
import os
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "experiments/hypothesis/0001-inverted-authority/select_probe.py"
MODEL = "google/gemma-4-31b-it"


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write(path: Path, value: object) -> None:
    path.write_bytes(canonical(value))


def load_selector():
    spec = importlib.util.spec_from_file_location("neutral_selector", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen selector")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    key = os.environ.get("ARC_LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("source .env before running")
    selector = load_selector()
    steps = selector.load(selector.resolve(selector.TRACES["A"][0]))
    cell = selector.cell(steps, 1, 0.6, "colour")
    if not cell or cell.get("answer") is None:
        raise RuntimeError("frozen cell is unavailable")

    run_id = "acquisition-ls20-l1-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = HERE / "runs" / run_id
    out.mkdir(parents=True)
    request = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": selector.SYSTEM},
            {"role": "user", "content": cell["prompt"]},
        ],
        "temperature": 0,
        "max_tokens": 6000,
        "reasoning": {"enabled": False},
    }
    visible = json.dumps(request).lower()
    forbidden = ["ls20", "oracle", "levels_completed", "win", "action 1"]
    leaks = [term for term in forbidden if term in visible]
    if leaks:
        write(out / "invalid.json", {"forbidden_terms": leaks})
        raise RuntimeError(f"model request failed leakage scan: {leaks}")
    write(out / "request.json", request)
    body = json.dumps(request, separators=(",", ":")).encode()
    req = Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urlopen(req, timeout=300) as response:
        payload = json.loads(response.read())
    write(out / "response.json", payload)
    text = payload["choices"][0]["message"].get("content") or ""
    parsed = selector.re.search(r'"pair"\s*:\s*(\d+)', text)
    pick = int(parsed.group(1)) if parsed else None
    bases = selector.baselines(cell["cands"], selector.random.Random(0))
    score = {
        "schema": "0002-neutral-acquisition-score-v1",
        "authority": "model_selection_scored_post_commitment",
        "pair": pick,
        "answer": cell["answer"],
        "hit": pick == cell["answer"],
        "candidate_count": cell["n"],
        "frames_seen": cell["frames_seen"],
        "frames_total": cell["frames_total"],
        "baselines": bases,
        "provider_calls": 1,
        "environment_actions": 0,
        "request_sha256": sha256(canonical(request)).hexdigest(),
        "usage": payload.get("usage", {}),
    }
    write(out / "score.json", score)
    print(f"run: {out}")
    print(f"pair: {pick} answer: {cell['answer']} hit: {score['hit']}")
    print(f"candidates: {cell['n']} provider_calls: 1 environment_actions: 0")
    return 0 if pick is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
