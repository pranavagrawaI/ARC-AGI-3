# Matched Actor V5 golden-path findings

## Judgment

The paid pilot did not primarily measure Native versus CodeAct reasoning. Four
of the five incomplete packets have correct runtime paths inside six provider
turns for both arms, but the model-facing contracts make those paths needlessly
hard to discover. The fifth packet, `remote-failure-composition`, is
structurally impossible for Native under the six-call ceiling and remains a
two-turn CodeAct task.

## Minimum valid provider turns

| Packet | Native | CodeAct | Paid outcome |
| --- | ---: | ---: | --- |
| `action-success-grounded` | 2 | 2 | both exhausted at 6 |
| `history-success-level1-noop` | 3 | 2 | both exhausted at 6 |
| `objective-success-level-boundary` | 4 | 2 | Native returned an incorrect answer at 6; CodeAct exhausted |
| `cross-level-success-action-response` | 6 | 2 | both exhausted at 6 |
| `remote-failure-composition` | 32 | 2 | both exhausted at 6 |

The CodeAct paths each use one persistent Python cell followed by one typed
return. The remote cell successfully performs all 30 bounded evidence
acquisitions, compares 59 transitions, and returns the four correct references
within the frozen five-second cell limit. Native requires one provider turn per
acquisition; 30 pages plus computation and return establish the 32-turn lower
bound.

## Contract defects exposed

1. **CodeAct has no query contract.** The matched packet omits
   `evidence_contract`. Its manifest consequently supplies only
   `result_schema`, and the worker renders the leaf as `<evidence.query>`.
   Every successful scripted query therefore relies on undocumented keywords.
2. **Native advertises invalid limits.** Its provider schema imposes no maximum,
   while T1 rejects values greater than 32. The paid model's 50, 60, and 100
   limits were schema-valid provider actions followed by host rejection.
3. **The cursor shape is under-specified.** `after` is merely an integer map;
   neither surface states that both `level` and `turn` are required.
4. **Receipts hide pagination state.** `next_after`, `truncated`, and
   `truncation_reason` exist in the immutable cached page but not its receipt.
   A provider must spend computation to recover the canonical cursor.
5. **Native's one-tool policy is hidden.** The controller rejects multi-tool
   turns, but its system prompt does not state that constraint.

## Classification

- `action-success-grounded`, `history-success-level1-noop`, and
  `objective-success-level-boundary`: avoidable model repair or reasoning loops;
  the Native contract contains enough information for the golden path.
- `cross-level-success-action-response`: zero-slack Native path plus an
  under-specified cursor; any repair consumes the entire six-call allowance.
- `remote-failure-composition`: benchmark/architecture mismatch for Native, not
  Native model failure. CodeAct has ample provider-turn capacity but lacks the
  query documentation required to discover its path reliably.

## Consequence

Do not rerun V5 unchanged. Repair contract congruence first, then decide whether
to remove the remote packet from a six-call matched comparison or give Native a
semantically equivalent bulk-computation operation. Raising the call ceiling to
32 would compare orchestration cost rather than cognition.

Machine-readable proof: `proof/report.json`.
