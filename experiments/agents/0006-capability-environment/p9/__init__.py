"""P9: the shared arm seam for issue #9, built as four small prototypes.

Import path note: ``llm_repl_agent`` ships inside ``0002-llm-repl-agent``, whose
directory name is not an identifier, so it is added to ``sys.path`` here — the
same bootstrap the 0006 spike launcher uses. Nothing in ``llm_repl_agent`` is
modified; every file there is inside ``participating_source_hashes()`` and an
edit would invalidate the 0002 release gate.
"""

from __future__ import annotations

from pathlib import Path
import sys


P9_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = P9_ROOT.parent
REPO = EXPERIMENT_ROOT.parents[2]
BASELINE_ROOT = REPO / "experiments" / "agents" / "0002-llm-repl-agent"

for _path in (REPO, BASELINE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
