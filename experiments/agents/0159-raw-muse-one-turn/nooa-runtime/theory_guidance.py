"""Frozen game-general practice guidance for WM2 executable conjectures."""

from __future__ import annotations

import hashlib

GUIDE_SCHEMA = "0159-theory-practice-guide-v1"

NARROW_PARTIAL_SOURCE = '''import numpy as np

UNKNOWN = -1

def predict(grid: np.ndarray, action: dict) -> np.ndarray | None:
    """Synthetic example: claim one cell only for one action."""
    if action["action"] != "LEFT":
        return None
    prediction = np.full_like(grid, UNKNOWN)
    prediction.ravel()[:1] = grid.ravel()[:1]
    return prediction
'''

MOORE_PROMOTION_SOURCE = '''from dataclasses import dataclass
import numpy as np

@dataclass
class State:
    grid: np.ndarray
    phase: int

def init_state(grid0: np.ndarray, level: int) -> State:
    return State(grid0.copy(), 0)

def transition(state: State, action: dict) -> State:
    return State(state.grid.copy(), state.phase + 1)

def render(state: State) -> np.ndarray:
    return state.grid

def outcome(state: State) -> str:
    return "level_complete" if state.phase >= 2 else "ongoing"
'''

COMPLETE_PRACTICE_EXAMPLES = {
    "narrow_partial_predictor": NARROW_PARTIAL_SOURCE,
    "abstention": (
        "Return `None` when the function makes no prediction for an action. "
        "Return a same-shaped grid containing `-1` at every unclaimed cell "
        "when only part of a successor is predictable."
    ),
    "contradiction_and_revision": (
        "An operationally valid candidate becomes current even when replay "
        "contradicts it. Read the first divergence, decide what it means, and "
        "author another edit only if your theory should change."
    ),
    "moore_promotion": MOORE_PROMOTION_SOURCE,
    "legitimate_nonuse": (
        "When evidence is insufficient, the rule is irrelevant to the next "
        "decision, or competing hypotheses are still live, keep them in prose "
        "or continue investigating instead of manufacturing a commitment."
    ),
}

_PRACTICE_GUIDE = f"""## Executable theory practice

An executable conjecture is useful when its prospective value justifies its
construction and maintenance cost. Possible uses include:

- preserve a rule you would otherwise reconstruct repeatedly;
- make one hypothesis falsifiable: claim only cells you can defend;
- compare a predicted consequence with later public evidence;
- use replay's first divergence to locate a failed claim, without assuming what
  the failure means;
- promote to a Moore machine only when history or hidden state is needed.

Smallest useful pattern:

```python
{NARROW_PARTIAL_SOURCE.rstrip()}
```

You may return `None` for whole-action abstention or `-1` for an unclaimed cell.
Either can be the honest result; greater coverage is not necessarily better.

Situation -> smallest useful commitment -> mechanical feedback -> your choice
to retain, revise, promote, or defer. Operational validity does not mean a
candidate is supported, and empirical contradiction does not replace it
automatically. Competing hypotheses remain first-class in prose.

You may defer. An executable commitment is useful only when its prospective
value justifies its construction and maintenance cost."""


def render_practice_guide() -> str:
    """Return the frozen always-visible WM2-A guide."""

    rendered = _PRACTICE_GUIDE
    if len(rendered.encode("utf-8")) > 3_072:
        raise RuntimeError("theory practice guide exceeds its frozen byte bound")
    return rendered


def practice_guide_sha256() -> str:
    """Commit the exact model-facing guide and complete example catalogue."""

    payload = render_practice_guide() + "\n" + repr(COMPLETE_PRACTICE_EXAMPLES)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
