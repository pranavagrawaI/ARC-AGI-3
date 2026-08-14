# 0158 Qwen3.6 autonomous 44-action condition

## Answer

The condition successfully supported autonomous multi-action revision, but
Qwen did **not** reach the 44-action cap or complete LS20 level 1.

| Candidate | Confirmed actions | Calls | Counted tokens | Terminal cause |
|---|---:|---:|---:|---|
| v18 | 1 | 16 | 186,034 | second-claim 8,192-token exhaustion |
| v19 | 6 | 80 | 566,372 | ARC action 7 uncertain after 10s read timeout |
| v20 | 1 | 34 | 241,548 | second-commitment 12-iteration exhaustion |
| v21 | 8 | 116 | 1,688,851 | ninth-claim submission-grammar exhaustion |

All attempts were fresh and non-resumable. No actions or score were combined
across them.

## Strongest behavioral run

v21 produced successors for actions `1, 1, 1, 1, 2, 1, 1, 1`. Its verdict sequence
was:

```text
contradicted
supported
supported
supported  -> G2 recorded, no new API, autonomous reset
contradicted
supported
contradicted
supported
```

Every supported verdict was `PRIMARY_REPLACEMENT_EXACT`. This is the first
run to exceed six confirmed successors. Atomic commitment closed all eight
commitments and v19's action-7 transport boundary did not recur. The ninth
claim then exhausted 12 iterations because the hard gate accepted only a
single-expression submit cell while Qwen repeatedly used a constructor
assignment followed by submission.

## Conclusion

Qwen demonstrated **bounded multi-action adaptation and G2 grounding evidence**,
not long-horizon task success. Atomic commitment corrected v20's commitment
closure failure. The current boundary is narrower: the submission-only gate
mistook deterministic constructor assignments for further exploration.
`levels_completed >= 1` remains false. See
`LIVE-RUN-QWEN36-ATOMIC-44-V21-20260812.md`.
