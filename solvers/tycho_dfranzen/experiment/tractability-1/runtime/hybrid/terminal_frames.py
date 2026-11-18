"""Present observed winning transitions beside the next decision, when enabled."""
from __future__ import annotations

import json
import os


def context(workspace, turn):
    if os.environ.get("TYCHO_TERMINAL_FRAMES") != "1":
        return ""
    from tycho.workspace.workspace import grid_text

    events = []
    for level in range(turn.levels_completed):
        path = workspace.dir / f"level_{level}" / "terminal.json"
        if path.exists():
            event = json.loads(path.read_text())
            if event["level"] != level or event["outcome"] != "win":
                raise ValueError(f"Invalid recorded winning transition: {path}")
            events.append((level, event))
    if not events:
        return ""
    parts = ["\n\nOBSERVED WINNING TRANSITIONS (zero-based levels). "
             "Ordinary turn_*.txt frames are decision observations BEFORE their next action. "
             "The terminal grid below was observed AFTER the winning action. "
             "The current observation belongs to the current level; it is not this terminal grid. "
             "These are observations, not inferred winning rules. "
             "Full grids for the latest two recorded completions follow; earlier terminal.json files remain available."]
    for level, event in events[-2:]:
        parts.append(f"\nCompleted level {level}: level_{level}/turn_{event['pre_turn']:03d}.txt "
                     f"-- winning action {json.dumps(event['action'], sort_keys=True)} --> "
                     f"level_{level}/terminal.json (AFTER winning action):\n"
                     + grid_text(event["terminal_grid"]))
    return "\n".join(parts)
