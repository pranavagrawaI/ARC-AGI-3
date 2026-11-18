# Bounded trial15 consumer evidence review

Run from repository root, choosing an unused output directory:

```bash
python analysis/serving-alternatives-20261021/coordination-20261024/trial15-binding-review/witness.py /tmp/trial15-consumer-review-unique
```

The source is the preserved actual CPU lifecycle in `qualified-r2/cases/cpu-evidence-joins`. Baseline final-resource helper and all its referenced raw artifacts pass. The GPU mutation models a producer/parser inconsistency: raw5000 with correctly recomputed artifact hash, stored eligibility/usage left unchanged. The same final-resource helper and artifact verifier still pass. `gpu.before.raw` / `gpu.after.raw` preserve both raw variants; the copied baseline fixture is restored after the GPU case. The variant JSON therefore needs `gpu.after.raw` installed at its stdout reference to repeat its verification in isolation. The deadline and PGID cases only mutate their named fields/trace joins, leaving actual candidate binding unchanged.

These witnesses prove limits of consumer independence, not a reachable bypass through the exact reviewed producer. The current pinned runtime calculates the correct shared deadline, cleans the actual candidate group, and parses raw GPU/apps/port/disk/native evidence before eligibility. No frozen source was edited and no API/GPU/launch was used. This is not a full production pair success fixture; collection tests elsewhere explicitly substitute package provenance. Final terminal acceptance should directly recompute raw resource predicates and verify configured deadline and candidate PGID, rather than claiming the current offline helper does all three independently.
