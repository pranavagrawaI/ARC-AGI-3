# 0003 — Temporal executable Game Judgment

Offline matched-arm interface experiment implementing `NEXT-LOOP-PLAN.md` from 0002.
It never calls the ARC environment.  Oracle labels live only under `scorer/`; deleting that
directory cannot change a request assembled from `fixtures/`.

## Commands

```bash
uv run python experiments/hypothesis/0003-temporal-game-judgment/freeze_batch.py
uv run python -m unittest discover -s experiments/hypothesis/0003-temporal-game-judgment/tests -v
uv run python experiments/hypothesis/0003-temporal-game-judgment/run_ladder.py \
  --responses responses.jsonl --output runs/offline
```

`freeze_batch.py` deterministically extracts the three frozen cases and four-frame PNGs
from the hash-bound public recording. `run_ladder.py` assembles credential-free requests,
records their hashes, parses supplied raw responses, and scores them. Provider transport is
deliberately outside this module: calls must not begin until provider, model revision,
decoding, visual budget, repetitions, cost cap, and stop rule are frozen.

The accepted response is one JSON object representing exactly one native-tool-equivalent
call: `EXECUTE`, `PROBE`, or `ABSTAIN`. Invalid submissions fail closed.
