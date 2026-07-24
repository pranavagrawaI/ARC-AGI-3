"""Generate exact frozen prompt counts with the pinned official model tokenizer."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
from urllib.request import urlopen

try:
    from tokenizers import Tokenizer
except ImportError as error:  # pragma: no cover - exercised by invocation guard
    raise SystemExit(
        "tokenizers is required; run with `uv run --with tokenizers python generate_prompt_token_counts.py`"
    ) from error

AGENT_ROOT = Path(__file__).resolve().parent
REPO = AGENT_ROOT.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from llm_repl_agent.experiment_protocol import (
    EXPERIMENT_FILES,
    ExperimentCondition,
    compose_condition_prompt,
)


MODEL_REPOSITORY = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
TOKENIZER_URL = (
    "https://huggingface.co/"
    f"{MODEL_REPOSITORY}/resolve/{MODEL_REVISION}/tokenizer.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_FILES / "prompt-token-counts.json",
    )
    args = parser.parse_args()

    with urlopen(TOKENIZER_URL, timeout=60.0) as response:
        tokenizer_bytes = response.read()
    with tempfile.NamedTemporaryFile(suffix=".json") as temporary:
        temporary.write(tokenizer_bytes)
        temporary.flush()
        tokenizer = Tokenizer.from_file(temporary.name)

    conditions: dict[str, dict[str, object]] = {}
    for condition in ExperimentCondition:
        path = EXPERIMENT_FILES / f"reconstruction-{condition.value}.txt"
        block_bytes = path.read_bytes()
        block_text = block_bytes.decode("utf-8")
        full_text = compose_condition_prompt(condition)
        conditions[condition.value] = {
            "sha256": sha256(block_bytes).hexdigest(),
            "bytes": len(block_bytes),
            "tokens": len(tokenizer.encode(block_text, add_special_tokens=False).ids),
            "full_prompt_sha256": sha256(full_text.encode("utf-8")).hexdigest(),
            "full_prompt_bytes": len(full_text.encode("utf-8")),
            "full_prompt_tokens": len(
                tokenizer.encode(full_text, add_special_tokens=False).ids
            ),
        }
    output = {
        "schema": "scientist-game-designer-player-prompt-token-counts-v1",
        "tokenizer": {
            "model_repository": MODEL_REPOSITORY,
            "model_revision": MODEL_REVISION,
            "tokenizer_url": TOKENIZER_URL,
            "tokenizer_json_sha256": sha256(tokenizer_bytes).hexdigest(),
            "add_special_tokens": False,
        },
        "conditions": conditions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
