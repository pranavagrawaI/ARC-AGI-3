# Matched Actor V3 Pilot protocol

V3 retains the V2 six-packet sample, counterbalanced order, resource ceilings,
provider wrapper, durable ledger, scoring, advancement rule, and zero ARC
authority. The only cognitive-treatment change is an explicit adapter at the
legacy-controller/V2-evidence boundary:

```text
legacy EvidenceQuery -> model_dump(mode="json") -> V2 EvidenceQuery validation
CodeAct query dict    ---------------------------> V2 EvidenceQuery validation
```

The disabled manifest commits every executable and analytical input. Paid
execution requires a separately committed authorization binding an enabled copy
of that exact manifest. Partial and failed runs are preserved without reruns.
