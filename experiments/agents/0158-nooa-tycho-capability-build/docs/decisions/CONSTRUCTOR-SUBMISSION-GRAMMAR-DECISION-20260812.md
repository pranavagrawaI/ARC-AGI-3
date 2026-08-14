# Constructor-only submission grammar decision — 2026-08-12

The v21 live run's ninth claim did not fail because Qwen emitted invalid Python
or an invalid claim. After its three permitted explorations, Qwen repeatedly
used this ordinary two-statement form:

```python
primary = make_mask(...)
return_result(submit_grounding_claim(primary_region=primary, alternatives=[]))
```

The host classified every multi-statement cell as further exploration before
executing it. That made deterministic artifact construction indistinguishable
from semantic evidence gathering and caused iteration exhaustion.

The corrected post-limit grammar permits:

- assignments to fresh local names;
- literals and previously assigned construction values;
- `make_mask` in the claim phase;
- `make_mask` and `cell_after` in the commitment phase; and
- one final `return_result(...)` call around the phase-appropriate atomic
  submitter.

It still rejects evidence reads, imports, control flow, comprehensions,
attribute/subscript access, arbitrary function calls, and all other
computation before sandbox execution. Thus the model can express its already
formed semantic decision using its natural Python convention, while a fourth
semantic exploration remains physically unavailable.

The public-boundary regression drives the real `GroundingAgent.produce()` and
NOOA `execute_python` sandbox with Qwen's constructor-assignment shape. A
paired regression proves that inserting `current_frame()` after the limit is
rejected and only a later constructor-only submission can complete. This is a
host correction only; no provider or ARC run is part of its qualification.
