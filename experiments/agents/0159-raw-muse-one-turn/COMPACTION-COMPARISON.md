# 0159 context-compaction comparison

## Question

Can a resumed Muse agent recover immediate orientation and exact older evidence
from a compact checkpoint plus a searchable immutable archive?

The exact v14 NOOA retry prompt was not preserved, so the control is a
reconstructed explicit recovery header plus a contiguous six-turn suffix. No
comparison run executed an ARC action.

## Results

### Immediate orientation (v4)

| Condition | Scores |
|---|---:|
| suffix | 13, 13, 13 out of 14 |
| checkpoint | 13, 13, 13 out of 14 |
| checkpoint + search | 13, 13, 13 out of 14 |

The suffix already contained almost every tested fact, so the probe reached a
ceiling. Search was never invoked. The model-generated checkpoint passed schema,
size, hash, and reference validation but missed consequential contradictions and
blurred the turn-15 model-window failure with the separate turn-16 account
allowance failure.

### Exact older evidence

| Retrieval surface | Suffix | Checkpoint | Checkpoint + retrieval |
|---|---:|---:|---:|
| v5 opaque lexical tools | 1/15 | 1/15 | invalid: four failed searches |
| v6 legible flat/raw tools | 1,1,1 | 1,1,1 | 12,12,11 |
| v7 structured views | 1,1,1 | 1,1,1 | **15,15,15** |

V7 used five stable calls per search repeat:

1. `history_turn(turn=3)`;
2. `history_note(name="hypotheses.md")`;
3. `history_provider_episode(sequence=15, phase="context_recovery")`;
4. `history_provider_episode(sequence=15, phase="commitment")`;
5. `history_provider_episode(sequence=16, phase="failures")`.

It recovered the controlled object's turn-3 DOWN motion, exact 72-message
initial request, exact 27-message recovered request, incorrect
`execute_python` commitment followed by `return_result`, stale note state, and
the distinction between the two failures.

## Efficiency

| Run | Calls | Prompt tokens | Completion tokens | Cost |
|---|---:|---:|---:|---:|
| v6 flat/raw retrieval | 17 | 655067 | 25513 | $0.098192 |
| v7 structured retrieval | 12 | 203420 | 17977 | $0.037404 |

Structured views reduced prompt tokens by 69%, calls by 29%, and cost by 62%
while improving recovery from 11.67/15 to 15/15. Provider latency variance means
the runs do not establish a latency improvement.

## Decision

The next live candidate should use:

```text
immutable archive
  + compact semantic checkpoint
  + recent contiguous suffix
  + decision-shaped retrieval
```

Do not expose flat raw turn bundles or generic lexical search as the primary
surface. Offline evidence does not yet prove autonomous retrieval during
gameplay, safe proactive triggering, or better environment actions.

## Local evidence commitments

Raw artifacts remain ignored under `artifacts/`. These hashes bind the evidence
used by this tracked summary:

| Artifact | SHA-256 |
|---|---|
| v4 `summary.json` | `2747cf7eefe958d086e95f6f34b5de06f42ed5b98f1e364c628361fcd4fa4f70` |
| v4 `probe-results.jsonl` | `b1896dc7dedbe8056fafaa9db6405e471c435ae33cbeb9c3c0991713c7973adf` |
| v5 `summary.json` | `f4f16d2508c1ee9556c92a109f0b0ce1bcc5a431d8297d6055dc968bc0427631` |
| v6 `summary.json` | `3834027b419d477765a604e3cc8acd502266be29253fe01c48c1681fdb3ae5e1` |
| v6 `probe-results.jsonl` | `9ce19e60b146bf5ee97d23ece421a21bab479ebf03b84365757ab17570dc7a55` |
| v7 `summary.json` | `3e39dbc5d690e23df44d5b02a6ea89cf543f89a7a432a98f37fa0df271f35af6` |
| v7 `probe-results.jsonl` | `e84395b91c784fce619b2e3a441c6f8276c205dc2a51263cb038f26543fe302c` |

Local run directories:

- `artifacts/20260815T120734Z-compaction-comparison-v4/`
- `artifacts/20260815T121350Z-compaction-retrieval-v5/`
- `artifacts/20260815T121544Z-compaction-retrieval-v6/`
- `artifacts/20260815T123201Z-compaction-retrieval-v7/`
