# Matched Actor V3 offline replay after-action

## Outcome

The zero-provider replay of `history-success-level1-noop`, a packet on which both
V2 arms exhausted their paid call budgets, passed through both real V3 controller
paths. Native and CodeAct each disclosed the target record once, decoded the RLE
representation, returned the correct verdict and falsifier with full evidence
recall, and incurred zero invalid interactions. No external provider or ARC call
was available.

## Failed attempt preserved

Attempt 1 correctly exercised Native but its scripted CodeAct cell referenced
`task["query"]`, a value the frozen task never contains. It therefore measured a
script defect, not the adapter. The failed summary and full pair trace remain in
`offline-replay-attempt-1*`. Attempt 2 embedded the same explicit query value used
by Native and passed.

## Licensed conclusion

The observed V2 class-identity defect is repaired for this real Native path, and
both treatments can attain the same evidence/computation/answer on this packet.
This establishes readiness for a paid V3 pilot, not relative cognitive quality.
