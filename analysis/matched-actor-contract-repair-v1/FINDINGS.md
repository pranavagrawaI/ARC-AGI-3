# Matched Actor contract repair v1

## Judgment

The five diagnosed contract seams are repaired offline. Native and CodeAct now
derive evidence-query syntax from the same bounded query model, receipts expose
canonical continuation metadata without copying records, and Native states its
one-tool scheduling rule.

## Repairs

1. Native advertises and validates `1 <= limit <= 32`.
2. `after` and `frontier` require complete `{level, turn}` references.
3. CodeAct receives a manifest-derived general query signature, constraints,
   example, and receipt schema.
4. Receipt v2 exposes `next_after`, `truncated`, and `truncation_reason` while
   retaining bulk evidence outside provider context.
5. The Native prompt states the controller's exactly-one-tool-per-turn rule.

## Golden paths

The same five answer-key-independent paths remain correct. Both arms now have
documented, derivable operations. Four packets fit within six calls in both
arms. `remote-failure-composition` still requires 32 Native operations and two
CodeAct operations; the repair does not disguise that benchmark/architecture
mismatch.

This checkpoint authorizes no paid inference. It is the zero-cost prerequisite
for selecting a fair next treatment rather than repeating the incomplete V5
pilot unchanged.
