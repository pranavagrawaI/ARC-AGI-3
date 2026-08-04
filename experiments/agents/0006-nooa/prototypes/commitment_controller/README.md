# Commitment-preserving controller prototype

**Question.** Gemma's LS20 failure is claimed to be executive, not perceptual:
it cannot hold one provisional abstraction across an action. Before spending on
live trials, does a fixed eight-field state block with commitment-preserving
rules actually work — that is, does it admit play that won, and refuse the drift
that lost?

Run it. No provider call, no spend:

```bash
python experiments/agents/0006-nooa/prototypes/commitment_controller/probe.py --replay both
```

Interactive mode, for the sequences that are hard to argue about on paper —
blocked move, reroute, changer event, contradiction left open:

```bash
python experiments/agents/0006-nooa/prototypes/commitment_controller/probe.py
```

## What it replays

- `--replay luna` — the recorded 28-action trajectory from
  `known-world-rules-treatment-luna-openai-l1-44-v3`, the only run that has
  completed LS20 level 1. Every turn must be admitted. A rejection means the
  rules would have blocked competent play.
- `--replay drift` — six state updates built from lines in the recorded Gemma
  rounds of `known-world-rules-treatment-open-l1-44-v1`. Each must be rejected
  by its named rule.

`fixtures/luna_win.json` is extracted from that run's `competence.jsonl`:
per-action commitment, working state, and the pixel-diff cell count.

## Files

- `../../host/commitment_controller.py` — the state machine and the rules.
- `STATE_BLOCK.md` — the model-facing protocol, i.e. the `NOTES.md` seed.
- `probe.py` — the driver.
- `NOTES.md` — the verdict.

Throwaway. When it has answered the question, delete it or absorb the rules
into the treatment.
