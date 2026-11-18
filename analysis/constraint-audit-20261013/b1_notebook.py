%%writefile /kaggle/harness-changes.patch
diff --git a/ARC3-Inference/inference/agent/action_names.py b/ARC3-Inference/inference/agent/action_names.py
index 4b8dfa3..1ef0773 100644
--- a/ARC3-Inference/inference/agent/action_names.py
+++ b/ARC3-Inference/inference/agent/action_names.py
@@ -1,6 +1,7 @@
 """Shared action-name mapping between model-facing labels and engine actions."""
 from __future__ import annotations
 
+import os
 from typing import Iterable
 
 
@@ -17,8 +18,32 @@ ENGINE_TO_MODEL_ACTION = {
 MODEL_TO_ENGINE_ACTION = {value: key for key, value in ENGINE_TO_MODEL_ACTION.items()}
 
 
+def reset_exposed() -> bool:
+    """Opt-in RESET advertising and snippet guards; off preserves legacy use."""
+    return os.environ.get("EXPOSE_RESET", "off").strip().lower() in {
+        "on", "1", "true", "yes",
+    }
+
+
+def undo_exposure_mode() -> str:
+    """Read at call time; unset/invalid values preserve the original wiring.
+
+    tufa: advertise ACTION7 verbatim but reject execution (legacy behavior).
+    off: hide ACTION7/UNDO and reject execution.
+    on: advertise UNDO and accept either name for engine ACTION7.
+    """
+    mode = os.environ.get("EXPOSE_UNDO", "tufa").strip().lower()
+    return mode if mode in {"tufa", "off", "on"} else "tufa"
+
+
 def to_model_action(name: str | None) -> str:
     raw = str(name or "").strip().upper()
+    if raw in {"ACTION7", "UNDO"}:
+        mode = undo_exposure_mode()
+        if mode == "off":
+            return ""
+        if mode == "on":
+            return "UNDO"
     return ENGINE_TO_MODEL_ACTION.get(raw, raw)
 
 
@@ -26,6 +51,8 @@ def to_engine_action(name: str | None) -> str | None:
     raw = str(name or "").strip().upper()
     if not raw:
         return None
+    if raw in {"ACTION7", "UNDO"}:
+        return "ACTION7" if undo_exposure_mode() == "on" else None
     if raw in ENGINE_TO_MODEL_ACTION:
         return raw
     return MODEL_TO_ENGINE_ACTION.get(raw)
diff --git a/ARC3-Inference/inference/agent/noop_repeat_guard.py b/ARC3-Inference/inference/agent/noop_repeat_guard.py
new file mode 100644
index 0000000..314e47a
--- /dev/null
+++ b/ARC3-Inference/inference/agent/noop_repeat_guard.py
@@ -0,0 +1,178 @@
+"""Known-no-op repeat guard.
+
+Records ``(level, interior_state_hash, action_signature)`` triples that were
+observed to change nothing outside the board border, and refuses to execute
+them again in a bit-identical state.
+
+Two deliberate choices:
+
+* The state key uses ``_interior_state_hash`` - the same border-cropped hash
+  the death ledger uses - so "the same state" means the same thing everywhere
+  in the harness, and a ticking HUD timer does not make every state unique.
+
+* Blocking is overridable. A blocked action arms a single slot; if the model
+  immediately re-issues that same action, it executes. This avoids the
+  self-sealing failure of a hard block: once an action is on the list, a hard
+  block guarantees no contradicting evidence can ever arrive, so a mechanic
+  gated on hidden state (a counter, a cooldown, an off-screen timer) would be
+  foreclosed forever. Each insistence costs exactly one action and the entry
+  survives, so the model can always insist again.
+
+Known limitation: the harness does not receive per-action frame counts, so an
+action whose effect appears only in intermediate animation frames (a rejected
+click, a bounce) looks identical to one that did nothing, and will be
+recorded. The override path is the only recourse there, which is why this
+guard defaults to off.
+"""
+from __future__ import annotations
+
+import re
+from collections import OrderedDict
+from typing import Any, Callable
+
+
+_DISPLAY_COORDS_RE = re.compile(
+    r"^([A-Z0-9_]+)\(\s*(?:ROW\s*=\s*)?(-?\d+)\s*,\s*(?:COL\s*=\s*)?(-?\d+)\s*\)$"
+)
+
+
+def action_signature(action: Any) -> str:
+    """Stable signature for one action. Mouse coordinates are part of the
+    identity: a click is only "the same action" at the same cell.
+
+    Accepts both shapes the harness uses - the dict form the sandbox passes
+    (`{"action": "MOUSE", "row": 4, "col": 7}`) and the display form the death
+    ledger stores (`"MOUSE(row=4, col=7)"`) - and normalizes them to the same
+    string, so a verdict recorded through one path is found through the other.
+    Callers must pass MODEL-facing names (UP/DOWN/...), not engine names
+    (ACTION1/ACTION2/...).
+    """
+    if isinstance(action, dict):
+        name = str(action.get("action") or action.get("name") or "").strip().upper()
+        row = action.get("row")
+        col = action.get("col")
+        if row is not None or col is not None:
+            return f"{name}({row},{col})"
+        return name
+    text = " ".join(str(action or "").split()).upper()
+    match = _DISPLAY_COORDS_RE.match(text)
+    if match:
+        return f"{match.group(1)}({match.group(2)},{match.group(3)})"
+    return text
+
+
+class NoopRepeatGuard:
+    """Per-level store of state+action combinations known to do nothing."""
+
+    def __init__(self, *, max_states_per_level: int = 512, max_actions_per_state: int = 16) -> None:
+        self._max_states = max(1, int(max_states_per_level))
+        self._max_actions = max(1, int(max_actions_per_state))
+        self._states: "OrderedDict[str, OrderedDict[str, None]]" = OrderedDict()
+        self._level: Any = None
+        # One slot per guard kind. A shared slot let whichever guard ran first
+        # consume the other's override: the death check runs before the no-op
+        # check, cleared the arm on every non-match, and re-armed it, so the
+        # no-op guard's "issue it again and it will execute" never fired. A
+        # mouse-only game observed 291 replies and one executed action.
+        self._armed_noop: tuple[str, str] | None = None
+        self._armed_death: tuple[str, str] | None = None
+
+    # -- level scoping -------------------------------------------------------
+    def note_level(self, level: Any) -> None:
+        """Drop everything when the level changes: a new layout means old
+        state hashes describe boards that no longer exist. Deaths and resets
+        deliberately do NOT clear the store - the board returns to the level
+        start, so recorded no-ops still hold and carry across attempts."""
+        if level != self._level:
+            self._level = level
+            self._states.clear()
+            self._armed_noop = None
+            self._armed_death = None
+
+    # -- recording -----------------------------------------------------------
+    def observe(self, state_hash: str, action_sig: str, *, gameplay_changed: Any) -> None:
+        if not state_hash or not action_sig:
+            return
+        if gameplay_changed is False:
+            entry = self._states.get(state_hash)
+            if entry is None:
+                entry = OrderedDict()
+                self._states[state_hash] = entry
+                while len(self._states) > self._max_states:
+                    self._states.popitem(last=False)
+            entry[action_sig] = None
+            while len(entry) > self._max_actions:
+                entry.popitem(last=False)
+        elif gameplay_changed is True:
+            # Evidence contradicts a recorded no-op: drop it. Reachable via the
+            # override path, where a blocked action is executed after all.
+            entry = self._states.get(state_hash)
+            if entry is not None and action_sig in entry:
+                del entry[action_sig]
+                if not entry:
+                    del self._states[state_hash]
+
+    # -- blocking ------------------------------------------------------------
+    def should_block(self, state_hash: str, action_sig: str) -> bool:
+        """True when this combination is a known no-op AND the model has not
+        just been told so. Arms the override slot when it blocks."""
+        if not state_hash or not action_sig:
+            return False
+        key = (state_hash, action_sig)
+        if self._armed_noop == key:
+            # the model saw the refusal and insisted: let it through, and
+            # re-arm nothing so a third attempt is blocked again
+            self._armed_noop = None
+            return False
+        entry = self._states.get(state_hash)
+        if not entry or action_sig not in entry:
+            self._armed_noop = None
+            return False
+        self._armed_noop = key
+        return True
+
+    def note_executed(self, state_hash: str, action_sig: str) -> None:
+        """Any executed action that is not the armed one clears the slot, so
+        the override only applies to an immediate repeat."""
+        key = (state_hash, action_sig)
+        if self._armed_noop and self._armed_noop != key:
+            self._armed_noop = None
+        if self._armed_death and self._armed_death != key:
+            self._armed_death = None
+
+    # -- death verdicts ------------------------------------------------------
+    def should_block_death(
+        self,
+        state_hash: str,
+        action_sig: str,
+        is_known_fatal: "Callable[[str, str], bool]",
+    ) -> bool:
+        """Same block-then-insist contract as `should_block`, but the verdict
+        comes from the death ledger rather than this store.
+
+        The ledger already keeps, per recorded attempt, an `actions` list and
+        an index-aligned `hashes` list of interior state hashes taken BEFORE
+        each action - so "this action, from this exact board, killed us" is a
+        lookup, not a sequence match. Matching per position rather than by
+        sequence prefix is what makes an interleaved or differently-routed
+        approach to the same fatal move still get caught, while a batch whose
+        earlier actions changed the board correctly does NOT match.
+
+        Deaths are more likely than no-ops to hinge on state the board does
+        not show (a hazard's facing, a timer), so blocking must stay
+        overridable: under permadeath a hard block could foreclose the only
+        path through a level with no way to ever discover the error.
+        """
+        if not state_hash or not action_sig:
+            return False
+        key = (state_hash, action_sig)
+        if self._armed_death == key:
+            self._armed_death = None
+            return False
+        if not is_known_fatal(state_hash, action_sig):
+            # deliberately does NOT clear the slot: an action this guard has no
+            # verdict on is none of its business, and clearing here would drop
+            # an override armed for a different action
+            return False
+        self._armed_death = key
+        return True
diff --git a/ARC3-Inference/inference/agent/priority_pace_reference.json b/ARC3-Inference/inference/agent/priority_pace_reference.json
new file mode 100644
index 0000000..9c624f8
--- /dev/null
+++ b/ARC3-Inference/inference/agent/priority_pace_reference.json
@@ -0,0 +1,16 @@
+{
+  "fallback_tokens": 45924.98353694247,
+  "by_level": {
+    "1": 27011.05963663083,
+    "2": 51072.88442280241,
+    "3": 48932.24773612239,
+    "4": 56352.32702962253,
+    "5": 55796.95578535604,
+    "6": 67264.12383313512,
+    "7": 40699.37351375245,
+    "8": 22696.312661663567
+  },
+  "method": "Kaplan-Meier restricted mean token cost, capped at 120000 tokens",
+  "training": "1417 reached-level episodes from historical transcripts; pooled by level number, no game identity at inference",
+  "limitations": "Historical config differs from current config. Reference is a bounded cost scale, not an unbiased uncapped completion time. Last survival value carried to cap beyond observed support."
+}
diff --git a/ARC3-Inference/inference/agent/priority_scheduler.py b/ARC3-Inference/inference/agent/priority_scheduler.py
new file mode 100644
index 0000000..14bb787
--- /dev/null
+++ b/ARC3-Inference/inference/agent/priority_scheduler.py
@@ -0,0 +1,134 @@
+"""Pure scheduler inputs and scoring; never reads hidden game metadata."""
+from __future__ import annotations
+
+import math
+from dataclasses import dataclass
+
+
+# Expected future level-weight sum, conditional on completing the current level.
+# 80k tokens are shared by ALL future levels; efficiency is NOT baked in.
+# Source: arc3-levelup-chances, pooled_late_provisional, N6..10_H80000 / 0.8.
+# KM duration distributions: separate levels 1-4, pooled observed levels 5-8
+# for levels 5 onward. Levels 9-10 are extrapolated. Costs rounded up to 250
+# tokens. 336 historical passes / 1,417 reached levels; not a contest fit.
+# Tuples are indexed by current_level - 1. The final level has no future tail.
+TAIL_LOOKUP_80K: dict[int, tuple[float, ...]] = {
+    6: (3.9082328014, 5.0756939323, 5.3532912917, 5.6080111343,
+        4.0898867400, 0.0),
+    7: (3.9262240771, 5.1669653335, 5.6943514417, 6.6500041602,
+        6.6562876217, 4.7715345300, 0.0),
+    8: (3.9291260268, 5.1855629875, 5.7851444897, 7.0053702980,
+        7.8471367942, 7.7045641090, 5.4531823200, 0.0),
+    9: (3.9295002558, 5.1885168225, 5.8035897108, 7.0958580947,
+        8.2469236991, 9.0442694281, 8.7528405964, 6.1348301100, 0.0),
+    10: (3.9295397677, 5.1888936715, 5.8065473797, 7.1136691211,
+         8.3474656955, 9.4884771003, 10.2414020621, 9.8011170838,
+         6.8164779000, 0.0),
+}
+
+
+# Heuristic continuation value by levels remaining AFTER the current level:
+# 3 or more -> 8; 2 -> 7; 1 -> 5; 0 -> 0. These are final B values,
+# so neither score normalization nor the empirical efficiency discount applies.
+TAIL_LOOKUP_REMAINING: dict[int, tuple[float, ...]] = {
+    n: (8.0,) * (n - 3) + (7.0, 5.0, 0.0) for n in range(6, 11)
+}
+
+
+def parse_total_levels(value: object) -> int | None:
+    """Validate public level-count metadata, then clip to the table domain."""
+    if value is None or isinstance(value, bool):
+        return None
+    try:
+        number = float(value)
+    except (TypeError, ValueError, OverflowError):
+        return None
+    if not math.isfinite(number) or number <= 0 or not number.is_integer():
+        return None
+    return min(10, max(6, int(number)))
+
+
+@dataclass
+class ProgressPace:
+    completed: int = 0
+    log_cost_ratio: float = 0.0
+
+    def record_completion(self, tokens_used: float, reference_tokens: float) -> None:
+        observation = math.log(max(1.0, tokens_used) / max(1.0, reference_tokens))
+        self.log_cost_ratio = (
+            observation if self.completed == 0
+            else 0.6 * self.log_cost_ratio + 0.4 * observation
+        )
+        self.completed += 1
+
+    def cost_multiplier(self) -> float:
+        confidence = self.completed / (self.completed + 3.0)
+        return math.exp(min(math.log(2.0), max(math.log(0.5),
+                                             confidence * self.log_cost_ratio)))
+
+
+@dataclass(frozen=True)
+class PrioritySnapshot:
+    level: int
+    actions: int
+    tokens: float
+    cost_multiplier: float = 1.0
+    # Appended to preserve four-argument callers and older pickled snapshots.
+    total_levels: int | None = None
+
+
+def priority_value(
+    state: PrioritySnapshot, *, endgame: bool = False,
+    tail_fraction: float | None = None, human_actions: float = 25.0,
+    action_scale: float = 115.0, token_scale: float = 62000.0,
+    action_weight: float = 0.25, tail_base: float = 6.0, tail_cap: float = 6.0,
+    normalize_score: bool = False, tail_lookup: bool | str = False,
+    tail_efficiency: float = 0.8,
+) -> int:
+    """No phase offset: all candidates must be evaluated at the same time.
+
+    tail_fraction=None retains the endgame rule. Supplying a fraction
+    keeps the likelihood proxy active and fades the tail instead of switching.
+    The likelihood proxy is heuristic, not a calibrated completion probability.
+
+    tail_lookup=True uses the empirical 80k table; "remaining" uses the
+    fixed 8/7/5/0 table without an efficiency discount. False keeps the capped
+    tail. normalize_score scales only A, never B or the endgame's 0.25 floor.
+    """
+    level = max(1, state.level)
+    total_levels = parse_total_levels(getattr(state, "total_levels", None)) or 10
+    actions, tokens = max(0, state.actions), max(0.0, state.tokens)
+    pace = min(2.0, max(0.5, state.cost_multiplier))
+    efficiency = (human_actions / (human_actions + actions)) ** 2
+    current_value = level * efficiency
+    if normalize_score:
+        # Normalize immediate reward A to a ten-level reference (sum 1..10).
+        # B is a separately chosen continuation value, not normalized reward.
+        current_value *= 55.0 / (total_levels * (total_levels + 1) / 2)
+    if endgame and tail_fraction is None:
+        # Preserve the original arithmetic when normalization is disabled.
+        value = (
+            level * 0.25 + current_value if normalize_score
+            else level * (0.25 + efficiency)
+        ) / pace * 100
+    else:
+        if tail_lookup == "remaining":
+            index = min(total_levels, level) - 1
+            tail = TAIL_LOOKUP_REMAINING[total_levels][index]
+        elif tail_lookup:
+            future_efficiency = (
+                min(1.0, max(0.0, tail_efficiency))
+                if math.isfinite(tail_efficiency) else 0.8
+            )
+            index = min(total_levels, level) - 1
+            tail = TAIL_LOOKUP_80K[total_levels][index] * future_efficiency
+        else:
+            tail = min(tail_cap, tail_base + level - 1)
+        if tail_fraction is not None:
+            tail *= min(1.0, max(0.0, tail_fraction))
+        chance = (
+            action_weight * 0.5 ** ((actions / action_scale) ** 2)
+            + (1.0 - action_weight) * 0.5 ** ((tokens / (token_scale * pace)) ** 2)
+        )
+        value = (current_value + tail) * chance / pace * 100
+    return max(1, int(value))
diff --git a/ARC3-Inference/inference/agent/prompts.py b/ARC3-Inference/inference/agent/prompts.py
index 2e59574..52351db 100644
--- a/ARC3-Inference/inference/agent/prompts.py
+++ b/ARC3-Inference/inference/agent/prompts.py
@@ -2,6 +2,62 @@
 
 from inference.utils.grid_utils import ARC_COLOR_LEGEND
 
+LEVEL_TRANSFER_SYSTEM_GUIDANCE = (
+    "- Levels usually build on mechanics learned in earlier levels, especially the most "
+    "recent one. Carry forward supported knowledge as a starting hypothesis, while "
+    "re-checking anything contradicted by new evidence. New levels often introduce "
+    "additional mechanics, sometimes through unfamiliar board elements. These additions "
+    "are often important for solving the level. The goal may remain the same but require "
+    "new mechanics to reach it, or the goal itself may change.\n"
+)
+
+LEVEL_START_USER_PROMPT = (
+    "You have completed the previous level. `current_frame` now contains the starting "
+    "board of the next level; any accompanying current-grid image shows this new board. "
+    "Build a new plan for this layout rather than continuing the previous level's action "
+    "sequence.\n\n"
+    "Start from the mechanics you established on the previous level; do not rediscover "
+    "them without reason. Inspect the new board for unfamiliar elements, changed "
+    "arrangements, or interactions your previous understanding does not explain. New "
+    "elements often introduce mechanics needed to solve this level, so prioritize small, "
+    "informative tests when their behavior is unclear.\n\n"
+    "Reassess the goal: does the previous objective still apply, now requiring the new "
+    "mechanics, or does the evidence suggest a different objective? Combine retained "
+    "knowledge with new findings to plan for this board."
+)
+
+
+def apply_level_transfer_system_guidance(prompt: str) -> str:
+    """Opt-in system wording; the caller gates ARC3_LEVEL_TRANSFER_GUIDANCE.
+
+    Preserve the original constants for the default arm. The first user prompt
+    after completion gives the new-board instructions; the system retains flag
+    definitions and stop instructions for failure and run completion.
+    """
+    return (
+        prompt.replace(
+            "- Levels often build on earlier mechanics, but layouts and interactions can still change between levels.\n",
+            LEVEL_TRANSFER_SYSTEM_GUIDANCE,
+        )
+        .replace(
+            "- Strategies may transfer loosely across levels, but layouts and mechanics can change. Re-check the new board before repeating a plan.\n",
+            "",
+        )
+        .replace(
+            "- Re-ground on the newest frame after any score increase or abrupt scene change; the returned board may already be the next level.\n",
+            "",
+        )
+        .replace(
+            "- `WIN` means the whole game is solved. Mid-run level completion is more likely to appear as a score increase while play continues.\n",
+            "- `WIN` means the whole game is solved.\n",
+        )
+        .replace(
+            "- If an action result reports `game_over`, `run_complete`, `level_completed`, or `done`, stop acting immediately and re-ground on the next turn.\n",
+            "- If an action result reports `game_over`, `run_complete`, or `done`, stop acting immediately and re-ground on the next turn.\n",
+        )
+    )
+
+
 TOOL_CALL_FORMAT_GUIDANCE = (
     "When calling `python`, emit exactly the tool-call format shown elsewhere in this prompt for this model. "
     "Use only that format; do not add markdown fences, prose wrappers, or alternate tool-call syntax. "
@@ -34,13 +90,50 @@ VISUAL_GAME_ADDENDUM = (
     "- For `MOUSE`, pass `row` and `col` integer arguments. `row` is vertical position, `col` is horizontal position.\n"
 )
 
+ACTION_INFO_ADDENDUM = (
+    "\n\nAction meanings (use only actions currently available):\n"
+    "- `UP`, `DOWN`, `LEFT`, and `RIGHT` are directional controls; what they affect "
+    "depends on the game.\n"
+    "- When available, `SPACE` performs a game-specific action, such as interacting, "
+    "selecting, rotating, attaching/detaching, or executing. Test its effect rather "
+    "than assuming what it does.\n"
+    "- When available, `MOUSE` clicks a board location. Pass integer `row` and `col` "
+    "fields from 0 to 63. Coordinates are zero-based from the top-left: `row` "
+    "increases downward and `col` increases rightward.\n"
+)
+
+UNDO_INFO_ADDENDUM = (
+    "- When available, `UNDO` reverses a previous action, usually the last turn. "
+    "Check what it restores. It cannot recover a failed attempt after game over.\n"
+)
+
+RESET_INFO_ADDENDUM = (
+    "- When available, `RESET` usually restores the current level to its starting "
+    "state, including the remaining-action/time bar, while keeping completed levels. "
+    "Use it to recover from an unrecoverable position or start a different approach. "
+    "RESET itself counts as one action, and actions already spent still count toward "
+    "your score. RESET must be the first game action in a Python snippet; inspection "
+    "or computation may precede it.\n"
+)
+
+GAMEPLAY_CHANGED_ADDENDUM = (
+    # One region, named once and referred back to twice. "outside that margin"
+    # and "inside that border" both read as possibly including the border
+    # cells themselves; "ignores those edge cells" states the rule before any
+    # geometry, and "further in" cannot be read as the border.
+    "- `board_changed` is true when ANY cell differs, including cells within {border} of the "
+    "edge where a timer or remaining-steps bar usually sits; `gameplay_changed` ignores those "
+    "edge cells and is true only when a cell further in differs. An action that only shrinks "
+    "a timer bar in those edge cells sets `board_changed` but not `gameplay_changed`.\n"
+)
+
 STRUCTURED_RUNTIME_STATE_ADDENDUM = (
     "\n\nRuntime variables inside every `python` tool call:\n"
     "- `current_frame` is a lightweight frame view for the latest environment state.\n"
     "- `current_frame` exposes only `.ascii`, `.step`, `.level`, `.shape`, and `.segmentation`.\n"
     "- `current_frame.ascii` is a single newline-delimited string containing the latest board rendered with the letter-coded ARC color symbols.\n"
     "- `current_frame.segmentation` parses the board into objects. It returns `{'nodes': [...], 'adjacency_list': [...]}`.\n"
-    "- Each node in `segmentation['nodes']` is one 4-connected same-color object with: `id` (index, ordered top-most-left-most), `color` (ARC color character), `hash` (a signature of the object's color and shape that ignores its position -- equal hashes mean the same object regardless of where it is, so use it to track an object across frames or to spot multiple identical objects in one frame), `pixels` (cell count), `boundary` (clockwise outer-perimeter corner points as `[row, col]`), and `children` (ids of objects fully enclosed by this one).\n"
+    "- Each node in `segmentation['nodes']` is one 4-connected same-color object with: `id` (index, ordered top-most-left-most), `color` (ARC color character), `hash` (a signature of the object's color and shape that ignores position AND rotation -- equal hashes indicate matching color and shape up to rotation; they do not establish object identity), `rotation` (degrees clockwise from the object's canonical orientation: 0/90/180/270, reduced to 0/90 for 2-fold symmetric shapes, `None` for fully symmetric ones), `rotational_symmetry` (1, 2, or 4), `pose_hash` (the rotation-SENSITIVE variant, if you need to match exact orientation), `shape_hash` (color-independent pose shape signature), `pixels` (cell count), `boundary` (clockwise outer-perimeter corner points as `[row, col]` cell coordinates), and `children` (ids of objects fully enclosed by this one).\n"
     "- `segmentation['adjacency_list']` is a list of `[i, j]` node-id pairs whose objects share an edge.\n"
     
     "- `current_frame.step` is the current environment step count.\n"
@@ -50,20 +143,48 @@ STRUCTURED_RUNTIME_STATE_ADDENDUM = (
 
     "- `history` is a chronological list of action/frame snapshots.\n"
     "- `history` is a Python list of objects, not a dict.\n"
-    "- Each history entry exposes only `.action` and `.frame`; entries are not subscriptable like `entry['action']`.\n"
+    "- Each history entry exposes `.action`, `.frame`, and `.result`; entries are not subscriptable like `entry['action']`.\n"
     "- Each `history[i].frame` is the frame after `history[i].action`; each frame exposes only `.ascii`, `.step`, `.level`, `.shape`, and `.segmentation`.\n"
     "- Important history semantics: when `history` is non-empty, `history[-1].frame` is the same latest/post-action board as `current_frame`. It is not the previous board. To inspect the state before the latest action, use `previous_frame` or `history[-2].frame` when available.\n"
     "- `previous_frame` is the frame before the most recent real environment action, or `None` if no previous frame is available.\n"
     "- `last_action` is the most recent real environment action name/display, or `None` before any real action.\n"
     "- `last_action_frame` is the post-action frame for `last_action`; it matches `current_frame` after a real action.\n"
-    "- `transitions` is a chronological list of actual action transitions, excluding the initial seeded frame. Each transition exposes `.action`, `.before_frame`, `.after_frame`, `.frame` (alias of `.after_frame`), and `.result`.\n"
-    "- `last_transition` is `transitions[-1]` or `None`. Its `.result` mirrors `last_action_result`; older transitions may have an empty `.result`. For before/after diffs, compare `last_transition.before_frame` to `last_transition.after_frame`; do not compare `current_frame` to `history[-1].frame`.\n"
-    "- `last_action_result` is the persisted result dict from the most recent `action(...)` call. It remains available across later Python inspection calls that do not call `action(...)`, and is `{}` before any action result exists. Read transition metadata from fields/keys such as `last_action_result['board_changed']`, `last_action_result['done']`, `last_action_result['level_completed']`, `last_action_result['game_over']`, `last_action_result['run_complete']`, `last_action_result['reward']`, and `last_action_result['valid_actions']`.\n"
+    "- `transitions` is a chronological list of actual action transitions, excluding the initial seeded frame. Each transition exposes `.action`, `.before_frame`, `.after_frame`, `.frame` (alias of `.after_frame`), and `.result`. Each `.result` describes that individual action and is retained with its history entry.\n"
+    "- `last_transition` is `transitions[-1]` or `None`. A refused action creates no transition and does not overwrite existing transition results. For before/after diffs, compare `last_transition.before_frame` to `last_transition.after_frame`; do not compare `current_frame` to `history[-1].frame`.\n"
+    "- `last_action_call_result` contains the result of the latest model-issued `action(...)` call, including batch totals, executed/skipped actions, and stop reasons. `action(...)` returns this result. It remains available across later Python inspection calls and analyzer turns, and is `{}` before any action call result exists.\n"
+    "- On the next analyzer turn after automatic RESET, `current_frame` shows the restarted level. The fatal action and RESET are separate transitions. Each transition result has `automatic=True` for a harness-initiated action and `automatic=False` for a model-issued action. Automatic RESET does not overwrite `last_action_call_result`: it still describes the call that caused death.\n"
     "- `valid_actions` is the current list of valid action names.\n"
     "- Call `action(actions)` to execute one or more real environment actions from Python.\n"
     "- Pass `action(actions)` a list like `['LEFT']` or `[{'action': 'MOUSE', 'row': 4, 'col': 7}]`.\n"
     "- One action usually returns one frame, but a single action can result in a short multi-frame animation.\n"
-    "- After `action(actions)` returns, `current_frame`, `previous_frame`, `history`, `transitions`, `valid_actions`, and `last_action_result` are refreshed.\n"
+    "- After `action(actions)` returns, `current_frame`, `previous_frame`, `history`, `transitions`, `valid_actions`, and `last_action_call_result` are refreshed.\n"
+)
+
+ANIMATION_ADDENDUM_TIMELINE = (
+    "- `last_animation_timeline` is a diff timeline of those frames: it "
+    "shows which cells changed at each step. It is a dict with `action`, `frames` and "
+    "`steps`, one entry per step that changed anything. Each entry has `step`, "
+    "`changed`, `bbox`, and "
+    "either `changes` (the cells, as `old>new @ (row,col) ...`) or `transitions` (a "
+    "count per colour change, when there were too many cells to list).\n"
+)
+
+ANIMATION_ADDENDUM = (
+    "- One action can return a short animation. `current_frame` is its final frame.\n"
+    "- When the final executed action of the latest model-issued call animated, `last_action_call_result['animation']` "
+    "reports `frames`, `transient_pixels` and `transient_bbox`. Transient cells "
+    "changed and then changed BACK during the animation, so they appear in no frame "
+    "you can otherwise reach - not in `current_frame`, not in `previous_frame`, not "
+    "in `history`.\n"
+    "- `last_animation_frames` is the frames of the last executed action that "
+    "animated, as views with "
+    "`.ascii`, `.shape` and `.segmentation`, the same as `current_frame`. Index 0 is "
+    "the first frame of the animation and the last entry is the board it settled on. "
+    "After a death these are the frames of the FATAL action, not of the automatic "
+    "reset that followed it. Crop "
+    "their `.ascii` yourself; printing a whole 64x64 frame will not fit the tool "
+    "budget. Empty when the action did not animate, and reading them costs no "
+    "in-game action.\n"
 )
 
 MULTIMODAL_CONTEXT_ADDENDUM = (
@@ -73,7 +194,7 @@ MULTIMODAL_CONTEXT_ADDENDUM = (
     "- You can use images and other tools to understand the game state and guide your strategy, each may be useful depending on the current uncertainty.\n"
 )
 
-PYTHON_ADDENDUM = (
+PYTHON_ADDENDUM_HEAD = (
     "\n\nPython tool guidance:\n"
     "- Use `current_frame.segmentation` as your primary view of the board -- objects, colors, containment, adjacency, and cross-frame object hashes.\n"
     "- Use `current_frame.ascii` only to read a small, specific region of the board when `segmentation` is not enough; never use it to scan or summarize the whole board.\n"
@@ -82,9 +203,13 @@ PYTHON_ADDENDUM = (
     "- The only tool is `python`; call it with one ephemeral `code` string.\n"
     "- Always inspect `current_frame`, `history`, and `valid_actions` from Python instead of reasoning from the raw board by eye.\n"
     "- For the most recent change, compare `previous_frame` to `current_frame`, or `last_transition.before_frame` to `last_transition.after_frame`. `history[-1].frame` is the current frame, so comparing it to `current_frame` only compares the board to itself.\n"
-    "- Maintain a compact working world model: what entities or regions exist, what actions seem to do, what the goal likely is, what remains uncertain, and what plan best fits the evidence so far.\n"
+)
+
+# split so the world-model lines can be swapped in at their original
+# position - immediately before the IMPORTANT search guidance - rather than
+# appended after the whole addendum
+PYTHON_ADDENDUM_TAIL = (
     "- IMPORTANT: Especially when the game is about making an agent navigate to a target, it is usually safer to write an explicit search algorithm such as BFS. More generally, when the objective is understood but the best action order is unclear, pathfinding, flood fill, BFS, DFS, beam search, shortest-path search, limited action-sequence search, or custom heuristics are all valid.\n"
-    "- Optimize for the shortest reliable sequence that advances the current goal as described by your world model. If confidence is low, program a discriminating probe and revise the world model from the result.\n"
     "- Once the important state variables and action effects are sufficiently understood, stop probing and search in the inferred state space.\n"
     "- Inspect current and history frames from Python instead of describing frames freehand.\n"
     "- Never print or echo full board frames. Return only compact derived summaries such as object lists, diffs, coordinates, counts, or tiny local crops.\n"
@@ -98,17 +223,124 @@ PYTHON_ADDENDUM = (
     "- `action(...)` accepts an ordered list of one or more actions. Once your code has selected a reliable sequence, it is often useful to batch it.\n"
     "- You can also call `action(...)` multiple times in one Python snippet, including inside loops. Each call updates the preloaded variables before execution continues.\n"
     "- If an action result reports `game_over`, `run_complete`, `level_completed`, or `done`, stop acting immediately and re-ground on the next turn.\n"
+    "- Flag semantics: `game_over` = this attempt FAILED (death or a limit ran out) — the level auto-resets to its initial state and completed levels are kept; it never means the run is won. `level_completed` = advanced one level. `run_complete`/`done` = the whole game is won. Deaths come either from the action itself (e.g., a hazard cell) or from a step/time budget depleting — watch for a bar at the grid border that shrinks each action, and count remaining budget into your plans.\n"
+)
+
+FRAME_DIFF_HINT_ADDENDUM = (
+    "- `frame_diff(before=None, after=None)` compares two frames. Omitted arguments default to `previous_frame` and `current_frame`, respectively.\n"
+    "- It returns a dictionary containing `changed_cell_count` (an integer) and lists named `moved`, `rotated`, `appeared`, `disappeared`, `changed_color`, and `resized`. Unchanged objects are omitted.\n"
+    "- Object-level changes are inferred by matching same-color, 4-connected components between frames: cells connect through shared edges, not diagonally. A game object may therefore be split into several components, or touching same-color objects may form one component.\n"
+    "- `moved` describes components matched at different positions. Matching does not establish persistent object identity: with multiple similar components, the reported correspondence may be ambiguous or incorrect. Movement combined with rotation includes `rotated_by` in degrees clockwise; `rotated` describes rotation in place.\n"
+    "- `changed_color` requires identical occupied cells with a different color. Entries contain `at`, `pixels`, `before_color`, `after_color`, `before_hash`, and `after_hash`.\n"
+    "- `resized` describes same-color components with overlapping footprints whose size changed. This can include backgrounds and budget bars. `at_before` and `at_after` can differ.\n"
+    "- `appeared` and `disappeared` entries include `bbox` in `[r0, c0, r1, c1]` order. Object hashes use the same rotation-invariant shape-and-color representation as segmentation; they are not unique object identities, and multiple objects can share a hash.\n"
+    "- Examples: `frame_diff()` compares the latest action’s before/after frames; `frame_diff(history[-3].frame, history[-2].frame)` compares two explicitly selected historical frames, when available.\n"
+)
+
+STEP_VERIFICATION_ADDENDUM = (
+    "- During a multi-action plan, verify important effects by comparing the actual game "
+    "state with your predictions—for example, changed cells, object properties, positions, "
+    "or counters. Make these comparisons in code so the batch can stop when a mismatch "
+    "undermines the remaining plan. Test uncertain interactions before committing to long "
+    "batches.\n"
+    "- If your search finds no solution under your current model of the game, remember "
+    "that the game is solvable. Reconsider your mechanics, goal, search implementation, "
+    "or search limits, including interactions with new elements. Take a targeted action "
+    "to test an uncertain rule or overlooked interaction, then update your model from "
+    "the result.\n"
+)
+
+WORLD_MODEL_ADDENDUM = (
+    "- Maintain a compact working world model: what entities or regions exist, what actions "
+    "seem to do, what the goal likely is, what remains uncertain, and what plan best fits the "
+    "evidence so far.\n"
+    "- Optimize for the shortest reliable sequence that advances the current goal as described "
+    "by your world model. If confidence is low, program a discriminating probe and revise the "
+    "world model from the result.\n"
+)
+
+WORLD_MODEL_FREE_ADDENDUM = (
+    "- Optimize for the shortest reliable sequence that advances the current goal. If "
+    "confidence is low, program a discriminating probe and revise your approach from the "
+    "result.\n"
+)
+
+DEATH_GUARD_ADDENDUM = (
+    "- An action that ended a previous attempt from this EXACT board state is refused before "
+    "execution: it raises `KnownDeathActionError`, nothing runs, and no attempt is lost. "
+    "`stop_reason` is `known_death`. The guard checks every position of a batch, not just the "
+    "first, so a fatal move is caught whichever route reached it. To repeat such an action "
+    "deliberately, issue it again as your very next action.\n"
 )
 
+NOOP_GUARD_ADDENDUM = (
+    "- An action already proven to change nothing in this EXACT board state is refused before "
+    "execution: it raises `KnownNoOpActionError`, which is NOT catchable and ends the snippet - "
+    "nothing runs, "
+    "and no action budget is spent. `stop_reason` is `known_noop`. If you have a specific reason "
+    "to believe the action will behave differently now, issue that same action again as your very "
+    "next action and it will execute.\n"
+)
+
+PREFER_TOOL_CALLS_LINE = (
+    # The closing clause used to warn that a long chain "wastes the whole turn".
+    # That overstated it - a truncated reply keeps its reasoning in history, and
+    # what is lost is only that nothing executed - and it aimed at the tail,
+    # since truncation touched about 4% of replies. The speed claim is the
+    # stronger one and it is true: decode runs at tens of tokens per second per
+    # stream, while a tool call is a short generation plus a mostly-cached
+    # prefill, so 2,000 tokens of further reasoning costs far more wall-clock
+    # than running the code that would settle the question.
+    "- Reading and computing cost nothing; only `action(...)` spends the level budget. When you "
+    "are unsure, prefer another tool call over more reasoning: the code answers what the "
+    "reasoning would only guess at, and running it is faster than thinking your way to the "
+    "same answer.\n"
+)
+
+EPHEMERAL_FUNCTIONS_LINE = (
+    "- The `python` tool code is not saved between calls, so rewrite any custom utility logic you still need.\n"
+)
+
+PERSISTENT_FUNCTIONS_LINE = (
+    "- Python variables reset between tool calls. Eligible functions you define are "
+    "retained automatically {lifetime}. Call your retained functions directly without "
+    "repeating their definitions. Redefine a function when you need to change it.\n"
+    "- For a function to be retained, its dependencies must also be available next call. "
+    "Pass snippet-specific data as arguments. You may use Python builtins, provided "
+    "globals such as `current_frame`, and other retained functions. Provided globals "
+    "are refreshed for each call. {import_guidance} Use plain "
+    "top-level definitions without decorators or annotations, and literal defaults. "
+    "Tool results report which functions were retained and explain any rejection.\n"
+)
+
+
 COMPACT_TOOL_SESSION_ADDENDUM = (
     "\n\nTool session rules:\n"
     "- You have exactly one tool: `python`.\n"
     f"- {TOOL_CALL_FORMAT_GUIDANCE}\n"
-    "- The `python` tool code is not saved between calls, so rewrite any custom utility logic you still need.\n"
+    "{persistent_functions}"
     "- You can call the `python` tool as many times as you want per step. Investigate until your code has a clear probe or plan.\n"
     "- Do not ration tool calls when the state is unclear. Spend extra tool calls to confirm what changed between frames and whether the last action affected gameplay state or only HUD elements such as countdown bars.\n"
-    "- After `action(...)` returns, the structured runtime state is refreshed before the next Python statement and before the next tool call. Inspection-only Python calls do not clear `last_action_result`.\n"
+    "{prefer_tool_calls}"
+    "- After `action(...)` returns, the structured runtime state is refreshed before the next Python statement and before the next tool call. Inspection-only Python calls do not clear `last_action_call_result`.\n"
     "- Each `python` tool call has a hard time limit of 30 seconds.\n"
     "- Tool responses are capped to about {tool_output_tokens} tokens. If a response is cut off, the tool result will tell you that.\n"
     "- Keep code snippets short and purpose-built rather than dumping large frameworks into one call.\n"
 )
+
+
+SUMMARY_REQUEST_PROMPT = (
+    "Summarize everything above: what you have established about this game, what you have "
+    "ruled out, what you were in the middle of, and anything else a later turn would need. "
+    "Keep uncertain ideas tentative, and preserve unresolved alternatives. "
+    "Do not act and do not call any tool - write the summary only.\n"
+    # No "the turns after this one stay visible": the summary is appended at
+    # the end of history, so nothing follows it when it is written. That
+    # sentence belonged to an earlier design where the summary was inserted
+    # ahead of a retained tail, and here it points at messages that do not
+    # exist yet.
+    "Write it to stand completely on its own. Everything above this point will eventually be "
+    "dropped and replaced by this summary, and earlier summaries go with it, so anything you "
+    "leave out is gone.\n"
+    "Length is not a problem here; leaving something out is."
+)
diff --git a/ARC3-Inference/inference/agent/python_tool_sandbox.py b/ARC3-Inference/inference/agent/python_tool_sandbox.py
index ac24747..58a94b3 100644
--- a/ARC3-Inference/inference/agent/python_tool_sandbox.py
+++ b/ARC3-Inference/inference/agent/python_tool_sandbox.py
@@ -3,6 +3,7 @@ from __future__ import annotations
 
 import inspect
 import json
+import logging
 import os
 import queue
 import signal
@@ -14,12 +15,15 @@ import textwrap
 import time
 from typing import Any, Callable
 
+from inference.utils import frame_diff as _frame_diff_module
 from inference.utils import segmentation as _segmentation
+from inference.utils import retained_functions as _retained_functions
 from inference.utils.grid_utils import ARC_COLOR_CHARS
 
 
 _SANDBOX_BOOTSTRAP = textwrap.dedent(
     r"""
+    import ast
     import builtins
     import contextlib
     import io
@@ -37,12 +41,22 @@ _SANDBOX_BOOTSTRAP = textwrap.dedent(
 
     __SEGMENTATION_SOURCE__
 
+    __FRAME_DIFF_SOURCE__
+
+    __RETAINED_FUNCTIONS_SOURCE__
+
     HOST_STDOUT = sys.stdout
 
     SAFE_MODULES = {
         "bisect",
         "collections",
         "copy",
+        # pure-Python sequence comparison, no I/O. Measured across 32
+        # transcripts it was imported 9 times and USED none of them - the model
+        # reaches for it on "compare two things" and then writes the comparison
+        # by hand. Allowing it therefore buys robustness, not capability: a
+        # stray import line no longer costs the whole snippet.
+        "difflib",
         "fractions",
         "functools",
         "heapq",
@@ -56,6 +70,25 @@ _SANDBOX_BOOTSTRAP = textwrap.dedent(
         "string",
     }
     SAFE_BUILTINS = {
+        # Exception types. Only four were present, so anything narrower than
+        # `except Exception` failed at the except clause and took the snippet
+        # with it. These are inert classes and expose nothing; SystemExit and
+        # KeyboardInterrupt stay out because the sandbox uses them for its own
+        # timeout and shutdown handling.
+        "ArithmeticError",
+        "AssertionError",
+        "AttributeError",
+        "EOFError",
+        "IndexError",
+        "KeyError",
+        "LookupError",
+        "NameError",
+        "NotImplementedError",
+        "OverflowError",
+        "RecursionError",
+        "StopIteration",
+        "UnboundLocalError",
+        "ZeroDivisionError",
         "abs",
         "all",
         "any",
@@ -147,9 +180,10 @@ _SANDBOX_BOOTSTRAP = textwrap.dedent(
 
 
     class HistoryEntryView:
-        def __init__(self, *, action, frame):
+        def __init__(self, *, action, frame, result=None):
             self.action = action
             self.frame = frame
+            self.result = dict(result) if isinstance(result, dict) else {}
 
         def __str__(self):
             return f"AsciiHistoryEntryView(action={self.action!r}, frame={self.frame})"
@@ -197,12 +231,13 @@ _SANDBOX_BOOTSTRAP = textwrap.dedent(
                 HistoryEntryView(
                     action=str(entry.get("action", "")),
                     frame=_frame_from_payload(entry.get("frame")),
+                    result=entry.get("result"),
                 )
             )
         return items
 
 
-    def _transitions_from_history(history, last_action_result):
+    def _transitions_from_history(history):
         transitions = []
         for index, entry in enumerate(history):
             action = str(getattr(entry, "action", "") or "").strip()
@@ -214,11 +249,9 @@ _SANDBOX_BOOTSTRAP = textwrap.dedent(
                     action=action,
                     before_frame=before_frame,
                     after_frame=entry.frame,
-                    result={},
+                    result=entry.result,
                 )
             )
-        if transitions and isinstance(last_action_result, dict):
-            transitions[-1].result = dict(last_action_result)
         return transitions
 
 
@@ -327,15 +360,38 @@ _SANDBOX_BOOTSTRAP = textwrap.dedent(
             "result": None,
         }
         runtime_globals["__builtins__"]["__import__"] = _safe_import
+        _animation_views = []
+
+        def frame_diff(before=None, after=None):
+            # Object-level diff between two frames. Defaults: previous_frame -> current_frame.
+            # Returns only what changed; unchanged objects are omitted.
+            if before is None:
+                before = runtime_globals.get("previous_frame")
+            if after is None:
+                after = runtime_globals.get("current_frame")
+            if before is None or after is None:
+                return {"error": "need two frames (no previous_frame available?)"}
+            if before.shape != after.shape:
+                return {"changed_cell_count": -1, "note": "frame shapes differ; full redraw",
+                        "before_shape": before.shape, "after_shape": after.shape}
+            return compute_frame_diff(
+                before._grid,
+                after._grid,
+                before.segmentation.get("nodes", []),
+                after.segmentation.get("nodes", []),
+            )
 
         def _refresh_state(state_payload):
             current_frame = _frame_from_payload(state_payload.get("current_frame"))
             history = _history_from_payload(state_payload.get("history"))
-            last_action_result = state_payload.get("last_action_result")
+            last_action_call_result = state_payload.get("last_action_call_result")
             action_result = (
-                dict(last_action_result) if isinstance(last_action_result, dict) else {}
+                dict(last_action_call_result) if isinstance(last_action_call_result, dict) else {}
             )
-            transitions = _transitions_from_history(history, action_result)
+            # action_echo is printed by action(), then removed from its return value.
+            # Keep the exposed call result identical across subsequent inspections.
+            action_result.pop("action_echo", None)
+            transitions = _transitions_from_history(history)
             last_transition = transitions[-1] if transitions else None
 
             runtime_globals["current_frame"] = current_frame
@@ -350,12 +406,98 @@ _SANDBOX_BOOTSTRAP = textwrap.dedent(
                 last_transition.after_frame if last_transition is not None else None
             )
             runtime_globals["last_action"] = last_transition.action if last_transition is not None else None
+            runtime_globals["frame_diff"] = frame_diff
             runtime_globals["valid_actions"] = [str(item) for item in state_payload.get("valid_actions", [])]
-            runtime_globals["last_action_result"] = action_result
+            runtime_globals["death_ledger"] = state_payload.get("death_ledger") or {"attempts": []}
+            runtime_globals["last_action_call_result"] = action_result
+
+        # Both refusals derive from BaseException, not Exception, so a
+        # defensive `except Exception:` around action(...) - which models write
+        # routinely - cannot swallow them. A refusal means the action did not
+        # run at all, so the snippet's assumed state is already wrong and every
+        # line after it rests on a false premise. One run treated 1500
+        # consecutive refusals as proof that every cell on the board was inert.
+        # StaleStateActionError below is deliberately catchable: it reports a
+        # belief that MAY be stale rather than a verdict on the action.
+        class KnownNoOpActionError(BaseException):
+            pass  # raised INSTEAD of executing an action already proven inert here
+
+        class KnownDeathActionError(BaseException):
+            pass  # raised INSTEAD of executing an action that ended a prior attempt
+
+        # Raised INSTEAD of executing an action this snippet has already run
+        # from this exact board. Buggy loop code can spend a whole life bar
+        # getting nowhere - a cursor that wraps turns RIGHT into a short cycle,
+        # a mistaken `while pos < j: LEFT` walks away from its target forever -
+        # and the terminal latch only stops that once the game actually ends,
+        # measured at 20 to 30 wasted actions. BaseException like the two above:
+        # the shape that produces these loops is `try: action(...) except
+        # Exception:` inside a while, which would otherwise swallow the verdict
+        # and keep going. No override slot: the snippet is not reading the
+        # refusal, its code is running, and writing a new snippet IS the
+        # response - the map is rebuilt per execution.
+        class RepeatedActionInStateError(BaseException):
+            pass
+
+        # Raised when a PREVIOUS action in this snippet changed nothing, and the
+        # snippet then tries to act again. Deliberately a plain Exception, not
+        # BaseException: unlike the two refusals above this is a warning about
+        # the snippet's beliefs rather than a verdict on the action, and a model
+        # that has re-read current_frame may legitimately want to proceed.
+        class StaleStateActionError(Exception):
+            pass
+
+        runtime_globals["KnownNoOpActionError"] = KnownNoOpActionError
+        runtime_globals["KnownDeathActionError"] = KnownDeathActionError
+        runtime_globals["StaleStateActionError"] = StaleStateActionError
+        runtime_globals["RepeatedActionInStateError"] = RepeatedActionInStateError
+        # armed when an action changes nothing; scoped to THIS snippet because
+        # the closure is rebuilt per execution, so no reset path can be missed
+        _stale_state: list[str] = []
+
+        # Raised when a snippet tries to act after the attempt or level ended.
+        # Derives from BaseException, not Exception, so a defensive
+        # `try: action(...) except Exception:` around the call - which models
+        # write routinely - cannot swallow it. Continuing to execute a path
+        # computed against a board that no longer exists is never correct.
+        class TerminalStateActionError(BaseException):
+            pass
+
+        runtime_globals["TerminalStateActionError"] = TerminalStateActionError
+        # latched when an action reports a terminal result; scoped to THIS
+        # snippet because these closures are rebuilt for every execution, so
+        # no reset path can be missed and the next snippet always starts clear
+        _terminal_state = []
+
+        _TERMINAL_DETAIL = {
+            "game_over": (
+                "the previous action reported game_over. The level has been reset to its "
+                "start, so the board you planned against is gone and this action was not "
+                "executed. The next turn begins from the new state."
+            ),
+            "level_completed": (
+                "the previous action reported level_completed. A new level has been loaded, "
+                "so the board you planned against is gone and this action was not executed. "
+                "The next turn begins from the new state."
+            ),
+            "run_complete": (
+                "the previous action reported run_complete. The run is finished and this "
+                "action was not executed."
+            ),
+        }
 
         def action(actions):
+            if _terminal_state:
+                raise TerminalStateActionError(_TERMINAL_DETAIL[_terminal_state[0]])
             normalized_actions = _normalize_actions(actions)
-            _send({"type": "action", "actions": normalized_actions})
+            _send({
+                "type": "action",
+                "actions": normalized_actions,
+                # the solver decides precedence: a known death or a known no-op
+                # is a sharper message than "your state may be stale", and it
+                # carries a concrete override
+                "stale_after": _stale_state[0] if _stale_state else None,
+            })
             reply = _recv()
             if reply.get("type") == "action_error":
                 raise RuntimeError(str(reply.get("error", "action failed")))
@@ -363,28 +505,225 @@ _SANDBOX_BOOTSTRAP = textwrap.dedent(
                 raise RuntimeError("Invalid action response from sandbox host.")
             action_result = reply.get("action_result") or {}
             action_results.append(action_result)
+            # Cached animations describe the previous action. Invalidate the
+            # existing views so aliases also refresh, even if a guard raises.
+            for view in _animation_views:
+                view.invalidate()
             _refresh_state(reply.get("state") or {})
+            if action_result.get("stop_reason") == "reset_not_first":
+                raise RuntimeError(str(action_result.get("stop_detail")))
+            if action_result.get("stop_reason") == "stale_state":
+                raise StaleStateActionError(
+                    str(action_result.get("stop_detail") or "stale state")
+                )
+            if action_result.get("stop_reason") == "known_death":
+                raise KnownDeathActionError(
+                    str(action_result.get("stop_detail") or "known fatal action was not executed.")
+                )
+            if action_result.get("stop_reason") == "repeated_action_in_state":
+                raise RepeatedActionInStateError(
+                    str(action_result.get("stop_detail") or "")
+                    or "this action was already run from this board in this snippet"
+                )
+            if action_result.get("stop_reason") == "known_noop":
+                # nothing executed and no action was spent; halt the snippet so
+                # the model re-plans instead of continuing on a false premise
+                raise KnownNoOpActionError(
+                    str(action_result.get("stop_detail") or "known no-op action was not executed.")
+                )
+            for _flag in ("run_complete", "done", "level_completed", "game_over"):
+                if action_result.get(_flag):
+                    # `done` means the run was won; report it as run_complete
+                    _terminal_state.append(
+                        "run_complete" if _flag == "done" else _flag
+                    )
+                    break
+            _echo = action_result.pop("action_echo", None)
+            if _echo:
+                # into the snippet's own stdout, so a loop can see what its
+                # call did without waiting for the next turn's opener
+                print(_echo)
+            gameplay_changed = action_result.get("gameplay_changed")
+            if gameplay_changed is True:
+                # a real change means the snippet's beliefs are current again
+                del _stale_state[:]
+            if gameplay_changed is False and action_result.get("executed"):
+                _executed = action_result.get("executed_actions")
+                _name = (
+                    _executed[-1]
+                    if isinstance(_executed, list) and _executed
+                    else action_result.get("action_display")
+                    or action_result.get("action_name")
+                    or "?"
+                )
+                # Arm rather than raise. The action executed and changed
+                # nothing, which is often exactly what the snippet went looking
+                # for - it printed the board before, and wants to print it
+                # after. Killing it here throws away the second half of the
+                # measurement AND the batch stop's own trace, which is computed
+                # and then discarded. So let the snippet finish observing, and
+                # stop it only if it tries to ACT on the stale belief.
+                if not _stale_state:
+                    _stale_state.append(str(_name))
             return action_result
 
+        def _animation_fetch(kind):
+            _send({"type": "animation", "request": {"kind": kind}})
+            reply = _recv()
+            if reply.get("type") == "animation_error":
+                raise RuntimeError(str(reply.get("error", "animation failed")))
+            if reply.get("type") != "animation_result":
+                raise RuntimeError("Invalid animation response from sandbox host.")
+            payload = reply.get("animation")
+            if kind != "frames":
+                return payload or {}
+            return [
+                FrameView(
+                    ascii=item.get("ascii", ""),
+                    step=item.get("step", 0),
+                    level=item.get("level", 0),
+                    shape=item.get("shape", (0, 0)),
+                    grid=item.get("grid", ()),
+                )
+                for item in (payload or [])
+            ]
+
         runtime_globals["action"] = action
+        if initial.get("animation_enabled"):
+            # Globals rather than attributes on last_transition: after a death
+            # the harness appends a RESET entry, so last_transition is the
+            # reset and not the action that animated. The record is simply
+            # "the last executed action that animated", and naming it that way
+            # removes the mismatch. Empty when the last executed action
+            # returned a single frame.
+            class _LazyAnimation:
+                def __init__(self, kind, empty):
+                    self._kind = kind
+                    self._empty = empty
+                    self._value = None
+
+                def _get(self):
+                    if self._value is None:
+                        self._value = _animation_fetch(self._kind) or self._empty
+                    return self._value
+
+                def invalidate(self):
+                    self._value = None
+
+                def __len__(self):
+                    return len(self._get())
+
+                def __iter__(self):
+                    return iter(self._get())
+
+                def __getitem__(self, key):
+                    return self._get()[key]
+
+                def __bool__(self):
+                    return bool(self._get())
+
+                def __contains__(self, key):
+                    return key in self._get()
+
+                def keys(self):
+                    return self._get().keys()
+
+                def get(self, key, default=None):
+                    return self._get().get(key, default)
+
+                def __repr__(self):
+                    return repr(self._get())
+
+            runtime_globals["last_animation_frames"] = _LazyAnimation("frames", [])
+            runtime_globals["last_animation_timeline"] = _LazyAnimation("timeline", {})
+            _animation_views.extend((runtime_globals["last_animation_frames"],
+                                     runtime_globals["last_animation_timeline"]))
+
         _refresh_state(initial.get("state") or {})
 
+        _persistence_enabled = bool(initial.get("persistence_enabled"))
+        _retention_options = {
+            "retain_imports": bool(initial.get("retain_imports")),
+            "repair_hints": bool(initial.get("repair_hints")),
+        }
+        _kept_pool = {}
+        _retention_rejected = []
+        _available_names = set(runtime_globals) | set(runtime_globals["__builtins__"])
+        if _persistence_enabled:
+            _before_restore = dict(runtime_globals)
+            try:
+                _kept_pool, _retention_rejected = restore_functions(
+                    initial.get("kept_functions") or [], runtime_globals, _available_names,
+                    **_retention_options)
+            except Exception:
+                # Optional cache maintenance must never prevent this snippet running.
+                runtime_globals.clear()
+                runtime_globals.update(_before_restore)
+                _kept_pool = {}
+                _retention_rejected = [{"name": "all", "reason": "restoration failed; redefine needed functions"}]
+
+        def _collect_retention_payload():
+            if not _persistence_enabled:
+                return {}
+            try:
+                kept, rejected = collect_functions(
+                    _code_text, _kept_pool, runtime_globals, _available_names,
+                    **_retention_options)
+                return {
+                    "keepable_functions": [{"name": n, "source": src} for n, src in kept.items()],
+                    "retention_rejected": _retention_rejected + rejected,
+                }
+            except Exception:
+                # Bookkeeping failures may drop helpers, but must never replace
+                # the snippet's original error/output or replay its actions.
+                return {"keepable_functions": [], "retention_rejected": [
+                    {"name": "all", "reason": "retention failed; redefine needed functions"}]}
+
         try:
-            compiled = compile(str(initial.get("code", "")), "<python_tool>", "exec")
+            _code_text = str(initial.get("code", ""))
+            _tree = ast.parse(_code_text, "<python_tool>")
+            _last_expr = None
+            if _tree.body and isinstance(_tree.body[-1], ast.Expr):
+                _last_expr = ast.Expression(_tree.body[-1].value)
+                ast.copy_location(_last_expr, _tree.body[-1])
+                _tree.body = _tree.body[:-1]
             with contextlib.redirect_stdout(stdout):
-                exec(compiled, runtime_globals, runtime_globals)
+                if _tree.body:
+                    exec(compile(_tree, "<python_tool>", "exec"), runtime_globals, runtime_globals)
+                if _last_expr is not None:
+                    _value = eval(compile(_last_expr, "<python_tool>", "eval"), runtime_globals, runtime_globals)
+                    if _value is not None:
+                        print(repr(_value))
+                        if runtime_globals.get("result") is None:
+                            runtime_globals["result"] = _value
             _send(
                 {
                     "type": "final",
+                    **_collect_retention_payload(),
                     "stdout": stdout.getvalue(),
                     "result": _json_safe(runtime_globals.get("result")),
                     "action_results": _json_safe(action_results),
                 }
             )
-        except Exception as exc:
+        except (
+            KnownNoOpActionError,
+            RepeatedActionInStateError,
+            KnownDeathActionError,
+            TerminalStateActionError,
+            Exception,
+        ) as exc:
+            # The three refusals derive from BaseException so a model's own
+            # `except Exception:` cannot swallow them - but THIS handler is the
+            # same construct, so without naming them explicitly they escaped it
+            # too, killed the child process, and the host reported a bare
+            # "Sandbox process exited unexpectedly". The refusal messages - the
+            # action name, that nothing was spent, how to override - were built
+            # and then never delivered. They stop the snippet either way; the
+            # point is that the model is told why.
             _send(
                 {
                     "type": "error",
+                    **_collect_retention_payload(),
                     "error": _sanitize_exception(exc),
                     "stdout": stdout.getvalue(),
                     "action_results": _json_safe(action_results),
@@ -395,12 +734,31 @@ _SANDBOX_BOOTSTRAP = textwrap.dedent(
     if __name__ == "__main__":
         main()
     """
-).replace("__SEGMENTATION_SOURCE__\n", inspect.getsource(_segmentation))
+).replace("__RETAINED_FUNCTIONS_SOURCE__\n", inspect.getsource(_retained_functions)).replace(
+    "__SEGMENTATION_SOURCE__\n", inspect.getsource(_segmentation)
+).replace(
+    "__FRAME_DIFF_SOURCE__\n", inspect.getsource(_frame_diff_module)
+)
+
+
+log = logging.getLogger(__name__)
 
 
 def _sanitize_host_error_text(text: str) -> str:
-    if not str(text or "").strip():
-        return "Sandbox process exited unexpectedly."
+    """What the MODEL is told when the sandbox process dies.
+
+    Deliberately fixed: the child's traceback carries host paths and internals
+    that have no business in the model's context. But the text was previously
+    read and thrown away entirely, so a crash left no trace anywhere - the one
+    thing that knows what went wrong said nothing. It now goes to the harness
+    log instead, where it is useful and nobody is prompted by it.
+    """
+    detail = str(text or "").strip()
+    if detail:
+        log.warning(
+            "sandbox process died; stderr follows:\n%s",
+            detail[-4000:],
+        )
     return "Sandbox process exited unexpectedly."
 
 
@@ -450,7 +808,11 @@ def run_sandboxed_python(
     code: str,
     timeout_seconds: int,
     initial_state: dict[str, Any],
-    action_handler: Callable[[list[dict[str, Any]]], dict[str, Any]],
+    action_handler: Callable[..., dict[str, Any]],
+    animation_handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
+    kept_functions: list[str] | None = None,
+    retain_imports: bool = False,
+    repair_hints: bool = False,
 ) -> dict[str, Any]:
     with tempfile.TemporaryDirectory(prefix="rgb_python_tool_") as sandbox_dir:
         host_action_results: list[dict[str, Any]] = []
@@ -489,6 +851,11 @@ def run_sandboxed_python(
             process.stdin,
             {
                 "code": code,
+                "kept_functions": list(kept_functions or []),
+                "persistence_enabled": kept_functions is not None,
+                "retain_imports": retain_imports,
+                "repair_hints": repair_hints,
+                "animation_enabled": animation_handler is not None,
                 "timeout_seconds": timeout_seconds,
                 "sandbox_cwd": sandbox_dir,
                 "state": initial_state,
@@ -534,9 +901,37 @@ def run_sandboxed_python(
                 }
 
             msg_type = str(message.get("type", "")).strip()
+            if msg_type == "animation":
+                if animation_handler is None:
+                    _send_json_line(
+                        process.stdin,
+                        {"type": "animation_error",
+                         "error": "Animation frames are not available in this session."},
+                    )
+                    continue
+                try:
+                    animation_payload = animation_handler(dict(message.get("request") or {}))
+                except Exception:  # noqa: BLE001
+                    _send_json_line(
+                        process.stdin,
+                        {"type": "animation_error",
+                         "error": "animation failed in sandbox host."},
+                    )
+                    continue
+                _send_json_line(
+                    process.stdin,
+                    {"type": "animation_result", "animation": animation_payload},
+                )
+                continue
             if msg_type == "action":
                 try:
-                    action_result_payload = action_handler(list(message.get("actions") or []))
+                    # the message carries more than the action list: stale_after
+                    # tells the solver a previous action in this snippet changed
+                    # nothing, which it weighs after its own guards
+                    action_result_payload = action_handler(
+                        list(message.get("actions") or []),
+                        stale_after=message.get("stale_after"),
+                    )
                 except Exception:  # noqa: BLE001
                     _send_json_line(
                         process.stdin,
@@ -566,6 +961,9 @@ def run_sandboxed_python(
                     "result": message.get("result"),
                     "error": str(message.get("error", "") or ""),
                     "action_results": list(message.get("action_results") or host_action_results),
+                    "keepable_functions": list(message.get("keepable_functions") or []),
+                    **({"retention_rejected": message.get("retention_rejected", [])}
+                       if kept_functions is not None else {}),
                 }
 
             _wait_for_process_exit(process)
diff --git a/ARC3-Inference/inference/agent/runtime_state.py b/ARC3-Inference/inference/agent/runtime_state.py
index 03edebb..5a00139 100644
--- a/ARC3-Inference/inference/agent/runtime_state.py
+++ b/ARC3-Inference/inference/agent/runtime_state.py
@@ -2,7 +2,7 @@
 from __future__ import annotations
 
 import json
-from dataclasses import dataclass
+from dataclasses import dataclass, field
 from pathlib import Path
 from typing import Any
 
@@ -42,6 +42,7 @@ class Frame:
 class HistoryEntry:
     action: str
     frame: Frame
+    result: dict[str, Any] = field(default_factory=dict)
 
 
 def normalize_grid(raw: Any) -> tuple[tuple[int, ...], ...]:
@@ -95,13 +96,17 @@ def history_entry_from_payload(payload: Any) -> HistoryEntry | None:
     frame = frame_from_payload(payload.get("frame"))
     if frame is None:
         return None
-    return HistoryEntry(action=str(payload.get("action", "")).strip(), frame=frame)
+    return HistoryEntry(
+        action=str(payload.get("action", "")).strip(), frame=frame,
+        result=dict(payload["result"]) if isinstance(payload.get("result"), dict) else {},
+    )
 
 
 def history_entry_to_payload(entry: HistoryEntry) -> dict[str, Any]:
     return {
         "action": entry.action,
         "frame": frame_to_payload(entry.frame),
+        "result": dict(getattr(entry, "result", {}) or {}),
     }
 
 
diff --git a/ARC3-Inference/inference/agent/tool_agent.py b/ARC3-Inference/inference/agent/tool_agent.py
index 59655e9..84a9cdd 100644
--- a/ARC3-Inference/inference/agent/tool_agent.py
+++ b/ARC3-Inference/inference/agent/tool_agent.py
@@ -1,11 +1,16 @@
 """Direct OpenAI-compatible tool-calling analyzer for ARC puzzle runs."""
 from __future__ import annotations
 
+import base64
+import heapq
+import threading
 import json
 import logging
 import os
+import random
 import re
 import time
+from functools import lru_cache
 from dataclasses import dataclass
 from pathlib import Path
 from typing import Any, Callable, Optional
@@ -13,25 +18,61 @@ from urllib.parse import urlparse, urlunparse
 
 import requests
 
-from inference.agent.action_names import to_engine_action, to_model_action
+from inference.agent.noop_repeat_guard import NoopRepeatGuard, action_signature
+from inference.agent.priority_scheduler import (
+    ProgressPace, PrioritySnapshot, parse_total_levels, priority_value,
+)
+from inference.utils.animation import (
+    _transient_cells as _animation_transient_cells,
+    build_animation_frames,
+    build_animation_view,
+    describe_animation,
+)
+from inference.agent.action_names import reset_exposed, to_engine_action, to_model_action, undo_exposure_mode
+from inference.utils.retained_functions import function_signature
 from inference.agent.prompts import (
+    apply_level_transfer_system_guidance,
+    LEVEL_START_USER_PROMPT,
     COMPACT_TOOL_SESSION_ADDENDUM,
+    PERSISTENT_FUNCTIONS_LINE,
+    EPHEMERAL_FUNCTIONS_LINE,
+    SUMMARY_REQUEST_PROMPT,
+    PREFER_TOOL_CALLS_LINE,
     GAME_OVERVIEW_ADDENDUM,
-    PYTHON_ADDENDUM,
+    WORLD_MODEL_ADDENDUM,
+    WORLD_MODEL_FREE_ADDENDUM,
+    FRAME_DIFF_HINT_ADDENDUM,
+    STEP_VERIFICATION_ADDENDUM,
+    PYTHON_ADDENDUM_HEAD,
+    PYTHON_ADDENDUM_TAIL,
+    GAMEPLAY_CHANGED_ADDENDUM,
     STRUCTURED_RUNTIME_STATE_ADDENDUM,
+    ANIMATION_ADDENDUM,
+    ANIMATION_ADDENDUM_TIMELINE,
     MULTIMODAL_CONTEXT_ADDENDUM,
     TOOL_CALL_FORMAT_GUIDANCE,
     VISUAL_GAME_ADDENDUM,
+    ACTION_INFO_ADDENDUM,
+    UNDO_INFO_ADDENDUM,
+    RESET_INFO_ADDENDUM,
 )
 
 from inference.agent.vision_context import (
+    animation_composite_part,
+    animation_death_image_mode,
+    animation_image_mode,
+    animation_peak_part,
     current_grid_image_enabled,
     current_grid_image_part,
 )
 
 from inference.agent.python_tool_sandbox import run_sandboxed_python
 from inference.agent.runtime_state import Frame, HistoryEntry, RUNTIME_STATE_FILENAME, load_runtime_state
-from inference.utils.openai_compat import build_chat_payload, build_headers
+from inference.utils.openai_compat import (
+    assemble_streamed_chat_response,
+    build_chat_payload,
+    build_headers,
+)
 
 log = logging.getLogger(__name__)
 
@@ -73,6 +114,14 @@ def _get_env_bool(name: str, default: bool) -> bool:
     return default
 
 
+def _level_transfer_guidance_enabled() -> bool:
+    """Opt-in level-transfer prompts; set before constructing the agent.
+
+    ARC3_LEVEL_TRANSFER_GUIDANCE defaults off, preserving the original prompts.
+    """
+    return _get_env_bool("ARC3_LEVEL_TRANSFER_GUIDANCE", False)
+
+
 def _contains_tool_call_markup(*chunks: str) -> bool:
     for chunk in chunks:
         lowered = chunk.lower()
@@ -141,6 +190,13 @@ _LOCAL_ANALYZER_TOOL_STEPS = _get_env_int("LOCAL_ANALYZER_TOOL_STEPS", 12)
 _LOCAL_ANALYZER_TOOL_TIMEOUT = _get_env_int("LOCAL_ANALYZER_TOOL_TIMEOUT", 30)
 _LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS = _get_env_int("LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS", 1024)
 _LOCAL_ANALYZER_YIELD_SECONDS = _get_env_float("LOCAL_ANALYZER_YIELD_SECONDS", 0.0)
+# A wall-clock budget is really a token budget with the throughput baked in:
+# the same 45 seconds bought 2,140 generated tokens on one backend and 3,829 on
+# a faster one, so every serving change silently retuned it. Measured in
+# tokens the setting means the same thing whatever the hardware does. Zero
+# disables, as with the seconds; if both are set, whichever comes first ends
+# the turn.
+_LOCAL_ANALYZER_YIELD_TOKENS = _get_env_int("LOCAL_ANALYZER_YIELD_TOKENS", 0)
 _LOCAL_ANALYZER_ENABLE_THINKING = _get_env_bool("LOCAL_ANALYZER_ENABLE_THINKING", True)
 _LOCAL_ANALYZER_TEMPERATURE = _get_env_float("LOCAL_ANALYZER_TEMPERATURE", 0.6)
 _LOCAL_ANALYZER_TOP_P = _get_env_float("LOCAL_ANALYZER_TOP_P", 0.95)
@@ -148,15 +204,102 @@ _LOCAL_ANALYZER_TOP_K = _get_env_int("LOCAL_ANALYZER_TOP_K", 20)
 _LOCAL_ANALYZER_SEED = _get_env_int("LOCAL_ANALYZER_SEED", -1)
 _REQUEST_SAFETY_MARGIN_TOKENS = 512
 _CONTEXT_OVERFLOW_RETRY_TRIM_TOKENS = 512
-_PERSISTENT_HISTORY_ASSISTANT_TURNS = 30
+def _context_drain_floor_mode() -> bool:
+    """Whether the drain stops ABOVE the target rather than below it.
+
+    History is dropped in whole blocks, so trimming "until under the target"
+    normally overshoots by most of a block. With a 40k budget and a 12k drain
+    the band is nominally 28k-40k, but simulation with ~6k blocks gives an
+    effective 14k-38k: the average sits ~3k below the nominal midpoint and the
+    minimum is far below the target.
+
+    Floor mode drops the mandatory blocks to get under budget, then stops
+    before a drop would cross below the target - so the retained content keeps
+    a real floor and the band means what it says. The cost is trim frequency:
+    the overshoot is what lengthens the interval between trims, so removing it
+    trims about half again as often, which shortens the stable prefix and cuts
+    the prefix-cache hit rate. Off by default for that reason; the same average
+    depth is available by raising the budget, which does not touch trim
+    frequency."""
+    return _get_env_bool("ARC3_CONTEXT_DRAIN_FLOOR", False)
+
+
+def _context_drain_tokens() -> int:
+    """Hysteresis for context trimming, in tokens.
+
+    Without it, trimming drops the oldest exchange the moment the estimate
+    exceeds the budget and stops the instant it fits - so once history reaches
+    the ceiling, nearly every turn evicts a little. Each eviction is cheap in
+    tokens but total in cache cost: prefix caching keys on the token sequence
+    from position zero, so removing anything from the FRONT shifts every
+    surviving token and invalidates the whole cached prefix, not just the part
+    removed. Frequent small evictions are therefore the worst case.
+
+    With a drain set, trimming still TRIGGERS at the budget but keeps dropping
+    until it is `drain` tokens below it, so evictions become rare and large:
+    the same tokens discarded overall, a fraction of the invalidations. The
+    cost is average history depth - the window sits below the budget most of
+    the time rather than pressed against it.
+
+    Size it against tokens per turn: at ~4-5k per turn including the opener,
+    8-12k buys two or three clean turns between evictions. 0 (default) keeps
+    the original drop-until-it-fits behavior."""
+    return max(0, _get_env_int("ARC3_CONTEXT_DRAIN_TOKENS", 0))
+
+
+def _history_turn_drain() -> int:
+    """Hysteresis for the assistant-turn cap, in assistant messages.
+
+    The token trimmer got hysteresis in ARC3_CONTEXT_DRAIN_TOKENS, but the
+    turn cap runs AFTER it and had none: once history sits at the cap, every
+    commit shaves back to exactly N, so the prefix is invalidated on every
+    single turn. With a 46k window and a cap of 30 assistant messages the cap
+    is the binding limit, which is why a token drain alone changed nothing.
+
+    Set this and the cap triggers at N but drains to N - drain, so evictions
+    become periodic instead of continuous."""
+    return max(0, _get_env_int("ARC3_HISTORY_TURN_DRAIN", 0))
+
+
+def _history_drain_coalesce() -> bool:
+    """Take the deep turn cut whenever the token trimmer already dropped.
+
+    Invalidation is all-or-nothing: removing anything from the front of
+    history shifts every surviving token, so a commit that has already
+    dropped blocks for token reasons has already paid the full cache cost.
+    Dropping turns in that same commit is therefore free, whereas letting the
+    cap fire on its own a turn later costs a second invalidation. This makes
+    the two mechanisms evict together rather than alternately."""
+    return _get_env_bool("ARC3_HISTORY_DRAIN_COALESCE", False)
+
+
+def _persistent_history_assistant_turns() -> int:
+    """Cap on assistant messages kept in persistent history.
+
+    Counts ASSISTANT MESSAGES, not analyzer turns: a turn that inspects, then
+    computes, then acts spends three of them, so 30 is closer to ten or
+    fifteen turns of real history than to thirty. The cap is independent of
+    the token budget, so with a large context window it can be the binding
+    constraint while thousands of tokens sit unused - which is what the
+    default 30 does at 98k, having been chosen when the window was 32k.
+
+    0 or negative disables the cap entirely, leaving the token budget as the
+    only limit."""
+    return _get_env_int("ARC3_HISTORY_ASSISTANT_TURNS", 30)
+_WM_NUDGE_TURNS = _get_env_int("ARC3_WM_NUDGE_TURNS", 0)
+_WM_WIPE_ON_GAME_OVER = _get_env_int("ARC3_WM_WIPE_ON_GAME_OVER", 1)
+_AUTO_FRAME_DIFF = _get_env_bool("ARC3_AUTO_FRAME_DIFF", False)
+_AUTO_FRAME_DIFF_BUDGET = _get_env_int("ARC3_AUTO_FRAME_DIFF_BUDGET", 300)
+_AUTO_FRAME_DIFF_MAX_GROUP = 40
+_DIFF_IMAGE = _get_env_bool("ARC3_DIFF_IMAGE", False)
 _RESPONSE_META_MAX_CHARS = 4000
 
 _PYTHON_TOOL_DESCRIPTION = (
     "Run one ephemeral Python snippet against preloaded ASCII game state. Available globals: "
     "`current_frame`, `previous_frame`, `history`, `transitions`, `last_transition`, "
-    "`valid_actions`, `last_action_result`, "
+    "`valid_actions`, `last_action_call_result`, "
     "and `action(actions)` for executing one or more real environment actions. "
-    "`current_frame` and each `history[*].frame` expose only `.ascii`, `.segmentation`, `.step`, and `.level`; "
+    "`current_frame` and each `history[*].frame` expose only `.ascii`, `.segmentation`, `.step`, `.level`, and `.shape` (a `(rows, cols)` tuple); "
     "`history[-1].frame` is the current post-action frame, not the previous frame. "
     "For before/after diffs, compare `previous_frame` to `current_frame` or use `last_transition.before_frame` and `.after_frame`. "
     "For MOUSE, pass `row` and `col` integer fields; legacy x/y fields are rejected. "
@@ -198,8 +341,10 @@ def _terminal_action_stop_detail(reason: str | None) -> str:
         return "No further actions were executed because the run is already complete."
     if reason == "game_over":
         return (
-            "No further actions were executed because the previous action reached GAME_OVER; "
-            "the runner will auto-reset before the next analyzer turn."
+            "No further actions were executed because the previous action reached GAME_OVER "
+            "(attempt failed; the run is NOT finished). The runner will auto-reset the level "
+            "before your next turn. Cause is either the action itself (hazard) or a depleted "
+            "step/time budget (usually a shrinking bar at the grid border)."
         )
     if reason == "level_completed":
         return (
@@ -215,6 +360,1324 @@ def _display_action_number(action_num: int) -> int:
     return max(1, int(action_num) + 1)
 
 
+def _stale_summary_lines(summary: dict[str, Any]) -> list[str]:
+    """Rendering for a step summary carried over from an earlier exchange of
+    the same turn. States that explicitly instead of re-narrating the old
+    outcome as if it had just happened. Two ways a turn can reach here:
+    a snippet ran and executed nothing (default wording), or the exchange
+    produced no tool call at all (stale_reason='no_tool_call') - in which case
+    claiming a snippet ran would be false."""
+    if summary.get("stale_reason") == "no_tool_call":
+        lines = [
+            "Your previous exchange this turn produced no tool call and "
+            "executed no actions."
+        ]
+    else:
+        lines = [
+            "Your last python snippet executed NO actions (it only computed or "
+            "inspected state - that is fine)."
+        ]
+    executed_actions = summary.get("executed_actions")
+    rendered = ""
+    if isinstance(executed_actions, list):
+        rendered = ", ".join(
+            str(name).strip() for name in executed_actions[:10] if str(name).strip()
+        )
+    if summary.get("game_over"):
+        fatal = str(summary.get("fatal_action") or "")
+        lines.append(
+            "Reminder of the still-pending outcome from earlier this turn: the "
+            f"executed sequence ({rendered or 'see earlier message'}) ended in "
+            "GAME OVER"
+            + (f" after '{fatal}'" if fatal else "")
+            + "; the level was auto-reset and you are planning the fresh attempt "
+            "now. Do NOT re-submit that sequence - the full diagnosis "
+            "instructions were given in an earlier message this turn."
+        )
+    elif summary.get("level_transition"):
+        lines.append(
+            "Reminder: the earlier sequence this turn completed the level; you "
+            "are now on the new level."
+        )
+    elif rendered:
+        lines.append(
+            "For reference, the most recent executed sequence earlier this turn "
+            f"was: {rendered}."
+        )
+    return lines
+
+
+def _state_identity_border() -> int:
+    """Border cropped when deciding whether two boards are THE SAME STATE.
+
+    Distinct from ARC3_NOOP_GUARD_BORDER, which asks a different question. The
+    batch stop asks "did this action accomplish anything meaningful", and there
+    cropping the HUD is right: a timer ticking down is not progress. The repeat
+    guards and the death ledger ask "have I been here before", and there
+    cropping is a liability - two boards with identical interiors but different
+    HUD are treated as one state, so a mechanic gated on anything the HUD
+    encodes (a countdown, a phase, a collected count) can be mis-recorded. The
+    resulting block is permanent and self-sealing: the action never executes,
+    so no evidence can arrive to correct it.
+
+    Defaults to 0 - exact whole-frame matching, which is safe by construction
+    and matches the reference implementation. Raise it to trade that guarantee
+    for more matches."""
+    return max(0, _get_env_int("ARC3_STATE_IDENTITY_BORDER", 0))
+
+
+def _interior_state_hash(grid: list[list[int]]) -> str:
+    """Hash identifying whether two boards are the same state, for the repeat
+    guards and the death ledger. Crops ARC3_STATE_IDENTITY_BORDER cells, which
+    defaults to 0 - the whole frame, HUD included. Small grids hash whole."""
+    import hashlib
+
+    border = _state_identity_border()
+    rows = len(grid)
+    cols = max((len(row) for row in grid), default=0)
+    if rows > 2 * border and cols > 2 * border:
+        core = [tuple(row[border:cols - border]) for row in grid[border:rows - border]]
+    else:
+        core = [tuple(row) for row in grid]
+    return hashlib.sha1(repr(core).encode("utf-8")).hexdigest()
+
+
+_LEAKED_TOOLCALL_RE = None
+
+
+def _extract_leaked_tool_calls(message: dict[str, Any]) -> dict[str, Any]:
+    """Fallback for servers without a tool-call parser for the model's
+    dialect: when a response carries no structured tool_calls but the content
+    contains Nemotron-style <TOOLCALL>[{...}]</TOOLCALL> markup, parse it and
+    synthesize the OpenAI-shaped tool_calls the harness expects. Malformed
+    payloads leave the message untouched. Disable with
+    ARC3_TOOLCALL_TEXT_FALLBACK=0."""
+    global _LEAKED_TOOLCALL_RE
+    if not _get_env_bool("ARC3_TOOLCALL_TEXT_FALLBACK", True):
+        return message
+    if not isinstance(message, dict) or message.get("tool_calls"):
+        return message
+    content = message.get("content")
+    if not isinstance(content, str) or "<TOOLCALL>" not in content:
+        return message
+    import json as _json
+    import re as _re
+    import uuid as _uuid
+
+    if _LEAKED_TOOLCALL_RE is None:
+        _LEAKED_TOOLCALL_RE = _re.compile(
+            r"<TOOLCALL>\s*(\[.*?\])\s*</TOOLCALL>", _re.DOTALL
+        )
+    match = _LEAKED_TOOLCALL_RE.search(content)
+    if match is None:
+        return message
+    try:
+        parsed = _json.loads(match.group(1))
+    except _json.JSONDecodeError:
+        return message
+    if not isinstance(parsed, list):
+        parsed = [parsed]
+    calls: list[dict[str, Any]] = []
+    for item in parsed:
+        if not isinstance(item, dict) or not item.get("name"):
+            continue
+        arguments = item.get("arguments", {})
+        if not isinstance(arguments, str):
+            arguments = _json.dumps(arguments, ensure_ascii=False)
+        calls.append(
+            {
+                "id": f"call_{_uuid.uuid4().hex}",
+                "type": "function",
+                "function": {
+                    "name": str(item["name"]),
+                    "arguments": arguments,
+                },
+            }
+        )
+    if not calls:
+        return message
+    remainder = (content[: match.start()] + content[match.end():]).strip()
+    updated = dict(message)
+    updated["tool_calls"] = calls
+    updated["content"] = remainder or None
+    return updated
+
+
+def _env_int(name: str, default: int) -> int:
+    """A numeric override that cannot take the run down on a typo."""
+    try:
+        return int(os.environ.get(name, "") or default)
+    except (TypeError, ValueError):
+        return default
+
+
+def _env_float(name: str, default: float) -> float:
+    try:
+        return float(os.environ.get(name, "") or default)
+    except (TypeError, ValueError):
+        return default
+
+
+_RETRYABLE_HTTP_STATUSES = {429, 500, 502, 503, 504, 520, 522, 524}
+# Establishing a TCP connection is fast or it is not happening: a server that
+# has exited refuses instantly, and one that is unreachable will not become
+# reachable inside the read budget. Separate from the read timeout so a dead
+# endpoint fails in seconds rather than waiting out the generation budget.
+_CONNECT_TIMEOUT_SECONDS = 10.0
+
+
+def _http_retry_delay_seconds(
+    attempt: int,
+    retry_after: str | None,
+    base_seconds: float,
+    max_seconds: float = 60.0,
+) -> float:
+    """Exponential backoff with jitter, or the server's Retry-After when it
+    parses as a number and asks for longer. Capped at 60s per wait.
+
+    Retry-After raises the wait, never lowers it. A provider under load can
+    answer `Retry-After: 1` to every request - observed at 100 consecutive
+    retries roughly 1.3s apart - and honouring that literally means hammering
+    an endpoint that has already said no, which neither clears the rate limit
+    nor helps the run. The configured base is the floor; the server can still
+    ask for more.
+    """
+    backoff = min(max_seconds, base_seconds * (2 ** attempt)) + random.uniform(0.0, 1.0)
+    if retry_after:
+        try:
+            # max_seconds caps the exponential, not an explicit request from the
+            # server: a Retry-After of 30 against a 5s ceiling still waits 30,
+            # because the endpoint said so and hammering it earlier helps no one
+            return max(backoff, min(60.0, float(retry_after)))
+        except ValueError:
+            pass
+    return backoff
+
+
+def _post_with_retries(
+    post_fn,
+    *,
+    retries: int,
+    base_seconds: float,
+    max_seconds: float = 60.0,
+    initial_seconds: float = 0.0,
+    sleep=time.sleep,
+    monotonic=time.monotonic,
+):
+    """Retry transient HTTP failures (rate limits, gateway errors) in place,
+    BEFORE they escalate to turn abort + history rollback. Returns the final
+    response either way; non-retryable statuses return immediately and flow
+    into the existing error handling. Rate-limit retries here preserve the
+    turn's partial tool work, which the abort path deliberately discards."""
+    # A negative count means keep trying. Zero keeps its obvious meaning - do
+    # not retry - so a typo disables retries rather than hanging the run.
+    unlimited = retries < 0
+    budget = max(0, retries)
+    # A floor on total retry time, used once. On Kaggle the benchmark is
+    # released before the server has finished loading, so the first request of
+    # every game meets a refused connection and has to wait it out - minutes,
+    # not the seconds the ordinary budget allows. Expressed as time rather than
+    # a count because the count that covers a 15-minute load would be absurd at
+    # any sensible delay, and would then apply to every later failure too.
+    started_at = monotonic()
+
+    def _budget_label(attempt_index: int) -> str:
+        """What the log should call the limit. Only reached outside a grace
+        period now - inside one the retry lines are suppressed entirely, since
+        "1/3" describes a budget that is not the one in force."""
+        if unlimited:
+            return "unlimited"
+        if attempt_index >= budget:
+            return f"{initial_seconds:.0f}s grace"
+        return str(budget)
+
+    def _exhausted(attempt_index: int) -> bool:
+        if unlimited:
+            return False
+        if attempt_index < budget:
+            return False
+        return (monotonic() - started_at) >= initial_seconds
+
+    def _announce_grace(reason: str) -> None:
+        """Said once, on the first failure, so the grace is visible from the
+        start. Without it the log shows three ordinary-looking retries before
+        the "Ns grace" label appears, and a reader watching a cold start cannot
+        tell a long wait from a broken one."""
+        nonlocal announced
+        if announced or initial_seconds <= 0:
+            return
+        announced = True
+        log.warning(
+            "first analyzer request failed (%s); retrying for up to %.0fs while "
+            "the endpoint comes up, then falling back to %d retries per request",
+            reason, initial_seconds, budget,
+        )
+
+    announced = False
+    response = None
+    attempt = -1
+    while True:
+        attempt += 1
+        try:
+            response = post_fn()
+        except requests.exceptions.RequestException as exc:
+            # The server was never reached - refused, DNS, a socket error, or a
+            # connect timeout. Without this the exception leaves the turn
+            # entirely and the analyzer-level retry starts it over, discarding
+            # whatever tool work it had already done; retrying the request in
+            # place keeps it. A ReadTimeout is deliberately not caught: the
+            # response body stalled, which has its own path that can yield with
+            # history preserved.
+            if _is_read_timeout(exc) or _exhausted(attempt):
+                raise
+            _announce_grace(type(exc).__name__)
+            delay = _http_retry_delay_seconds(attempt, None, base_seconds, max_seconds)
+            # Silent while the grace is carrying the loop: at 14 games polling a
+            # cold endpoint every 5s for 15 minutes this line alone would be
+            # ~2,500 entries, burying everything else. The announcement above
+            # already said a wait is in progress, and the outcome is logged when
+            # the loop ends either way.
+            if initial_seconds <= 0:
+                log.warning(
+                    "analyzer endpoint unreachable (%s); retry %d/%s in %.1fs",
+                    type(exc).__name__, attempt + 1, _budget_label(attempt), delay,
+                )
+            sleep(delay)
+            continue
+        status = getattr(response, "status_code", 200)
+        if status not in _RETRYABLE_HTTP_STATUSES or _exhausted(attempt):
+            return response
+        _announce_grace(f"HTTP {status}")
+        retry_after = None
+        headers = getattr(response, "headers", None)
+        if headers is not None:
+            retry_after = headers.get("Retry-After")
+        delay = _http_retry_delay_seconds(attempt, retry_after, base_seconds, max_seconds)
+        if initial_seconds <= 0:
+            log.warning(
+                "analyzer endpoint returned HTTP %s; retry %d/%s in %.1fs",
+                status, attempt + 1, _budget_label(attempt), delay,
+            )
+        sleep(delay)
+    return response
+
+
+_CONTROL_MESSAGE_KEY = "_arc3_control"
+
+_YIELD_RESUME_PROMPT_WITH_TOOLS = (
+    "You yielded control on the turn time budget. Your tool results above are "
+    "still valid - do NOT restart your analysis or re-inspect the board from "
+    "scratch. Continue from them and call `action(actions)` with your best "
+    "next action or short batch."
+)
+
+_YIELD_RESUME_PROMPT_WITH_TOOLS_NEUTRAL = (
+    "You yielded control on the turn time budget. This is a scheduling "
+    "interruption, not a signal that your analysis is complete. Your tool "
+    "results above are still valid - do NOT restart or re-inspect the board "
+    "from scratch. Pick up where you left off: keep investigating if you still "
+    "need to, and call `action(actions)` once your evidence supports a choice."
+)
+
+
+def _yield_resume_tone() -> str:
+    """Wording for the commonest resumption message.
+
+    'commit' (default) is the original text, which reads as though the model
+    has finished: it asks for the best next action. At a 60s budget that fires
+    after very little deliberation, so it can push a commitment the evidence
+    does not yet support.
+
+    'neutral' keeps the load-bearing part - do not restart, your results are
+    valid - while naming the interruption for what it is and leaving the
+    decision to continue investigating open. The anti-stall backstops are
+    untouched: the no-tool-call nudge, the tool-step-exhaustion prompt and the
+    timeout prompt all still push to act, and they fire on exhausted work
+    rather than on elapsed time.
+
+    'minimal' goes further and does not mention the yield at all. A yield that
+    executed nothing and a snippet that only inspected are the same situation
+    from the model's side - nothing happened, carry on - and the scheduling
+    reason is harness detail it cannot act on. The wording echoes the mid-turn
+    stale opener so a resumption reads as a continuation rather than as its own
+    kind of event, keeping only the part that is load-bearing: the earlier
+    results are still good.
+
+    Read the arm as actions performed against tokens generated: the change
+    should raise tokens per action if it is working as intended, and the
+    question is whether the actions get better.
+    """
+    tone = os.environ.get("ARC3_YIELD_RESUME_TONE", "").strip().lower()
+    return tone if tone in ("neutral", "minimal") else "commit"
+
+_YIELD_RESUME_PROMPT_NO_TOOLS = (
+    "You spent the entire turn budget reasoning without calling a tool. Stop "
+    "deliberating and call the `python` tool NOW: inspect "
+    "`current_frame.segmentation`, then call `action(actions)` with the best "
+    "available action. Acting on a partial model beats another turn of "
+    "analysis."
+)
+
+_YIELD_RESUME_PROMPT_WITH_TOOLS_MINIMAL = (
+    # Says nothing about yielding. From the model's side a yield with nothing
+    # executed and an inspect-only snippet are the same situation - nothing
+    # happened, carry on - and the scheduling reason is harness detail it
+    # cannot act on. Deliberately echoes the mid-turn stale opener it already
+    # sees often, so a resumption reads as a continuation rather than as a
+    # distinct kind of message. Kept: the earlier results are still good, so
+    # do not start over.
+    "Your last python snippet executed no actions (it only computed or "
+    "inspected state - that is fine). Your tool results above are still "
+    "valid: continue from them rather than re-inspecting the board from "
+    "scratch."
+)
+
+_YIELD_RESUME_PROMPT_NO_TOOLS_MINIMAL = (
+    # The with-tools wording would be false here: there was no snippet at all,
+    # so there are no results to continue from and the useful instruction is
+    # where to start instead.
+    "Your previous exchange produced no tool call and executed no actions. "
+    "Start with the `python` tool: inspect `current_frame.segmentation`, then "
+    "call `action(actions)` when your evidence supports a choice."
+)
+
+_YIELD_RESUME_PROMPT_NO_TOOLS_NEUTRAL = (
+    # The same edit as the with-tools pair: keep what is load-bearing - that
+    # nothing executed, and the concrete first step - and drop the pressure to
+    # commit. This fires when the yield budget expired on a turn that never
+    # called a tool, so the model really did spend the turn reasoning; what is
+    # unwarranted is telling it that acting on a partial model is better, when
+    # the budget expiring says nothing about whether its analysis was close.
+    "You yielded control on the turn time budget without calling a tool. This "
+    "is a scheduling interruption, not a signal that your analysis is complete "
+    "- but nothing has been executed and the board is unchanged. Start with "
+    "the `python` tool: inspect `current_frame.segmentation`, then call "
+    "`action(actions)` once your evidence supports a choice."
+)
+
+_DEAD_REASONING_STUB = "(previous attempt produced no tool call)"
+
+_DEGENERATE_CONTENT_STUB = (
+    "(previous reply was highly repetitive and has been omitted; do not "
+    "continue that pattern - write concise plain prose)"
+)
+
+_STATE_ONLY_CAPTION = (
+    # Leads with the assurance rather than with "Current board state":
+    # _build_user_message appends its own "Current grid image:" immediately
+    # below, so opening with a second caption put two labels for one picture
+    # next to each other. The load-bearing part is that the opener has not
+    # gone stale, not that a board follows.
+    "Nothing has been executed since the turn opener above, so the step, "
+    "level and valid actions stated there still hold."
+)
+
+
+def _yield_resume_prompt_mode() -> str:
+    """How a resumed turn re-opens.
+
+    'full'       (default) rebuild the complete opener - 2-3k tokens restating
+                 state the retained opener already carries.
+    'short'      a brief continuation sentence and nothing else.
+    'state_only' the continuation sentence plus a one-line caption and the
+                 CURRENT board image(s). Measurements showed 'short' holding up
+                 while images were attached and degrading without them, which
+                 is consistent with the omitted text (batch summary, state
+                 line, diffs, ledger) being what the image was covering for.
+                 This mode makes that explicit: drop the boilerplate, keep the
+                 state, in the cheapest form available.
+    """
+    mode = os.environ.get("ARC3_YIELD_RESUME_PROMPT", "").strip().lower()
+    return mode if mode in ("short", "state_only") else "full"
+
+
+_RESUME_PROMPT_TIMEOUT = (
+    "The previous request to the model timed out and was abandoned. Your tool "
+    "results above are still valid - do NOT restart your analysis. Continue "
+    "from them and call `action(actions)` with your best next action, keeping "
+    "your reasoning short so the next request completes."
+)
+
+_RESUME_PROMPT_TOOL_STEPS_EXHAUSTED = (
+    "You used every available tool step this turn without calling "
+    "`action(actions)`. Your tool results above are still valid - do NOT begin "
+    "a new investigation. Act NOW with the best option you already have."
+)
+
+
+def _yield_resume_prompt(reason: str) -> str:
+    """Continuation text for a resumed turn, chosen by why it ended.
+
+    'timeout'      - the request exceeded the analyzer read timeout: the work
+                     already done this turn is intact, the model just needs to
+                     wrap up quickly.
+    'tool_steps'   - the step budget ran out mid-investigation: results exist,
+                     the missing thing is the action.
+    'yield_tools'  - the time budget ran out with tool work done: same premise,
+                     softer framing.
+    'yield_bare'   - the turn produced only reasoning, so there is nothing
+                     above to continue from and the useful instruction is to
+                     stop deliberating.
+    """
+    if reason == "timeout":
+        return _RESUME_PROMPT_TIMEOUT
+    if reason == "tool_steps":
+        return _RESUME_PROMPT_TOOL_STEPS_EXHAUSTED
+    tone = _yield_resume_tone()
+    if reason == "yield_tools":
+        if tone == "minimal":
+            return _YIELD_RESUME_PROMPT_WITH_TOOLS_MINIMAL
+        return (
+            _YIELD_RESUME_PROMPT_WITH_TOOLS_NEUTRAL
+            if tone == "neutral"
+            else _YIELD_RESUME_PROMPT_WITH_TOOLS
+        )
+    if tone == "minimal":
+        return _YIELD_RESUME_PROMPT_NO_TOOLS_MINIMAL
+    return (
+        _YIELD_RESUME_PROMPT_NO_TOOLS_NEUTRAL
+        if tone == "neutral"
+        else _YIELD_RESUME_PROMPT_NO_TOOLS
+    )
+
+
+def _fatal_lookahead_enabled() -> bool:
+    """Block a batch that replays a recorded fatal path BEFORE spending any of
+    its actions, rather than running the harmless prefix and refusing only the
+    final move.
+
+    Default off keeps the per-position behaviour: the prefix executes and the
+    model ends up at the brink, which costs those actions but leaves it
+    somewhere it can still choose differently. With lookahead on, nothing is
+    spent and the model must resubmit a shorter batch to reach that point."""
+    return _get_env_bool("ARC3_DEATH_GUARD_LOOKAHEAD", False)
+
+
+def _yield_on_timeout_enabled() -> bool:
+    """Treat an analyzer read timeout as a yield instead of a failed turn.
+
+    The default path clears preserve_history, so the whole invocation's tool
+    calls and results are rolled back and the retry re-derives them from
+    scratch - on Kaggle that is a ~900s round trip discarded. Nothing can have
+    executed when a timeout fires (an executing dispatch breaks the loop before
+    another request is issued), so preserving the exchanges is safe: history
+    ends on a tool message and the resumed invocation appends a user message,
+    exactly the shape every yielded turn already produces.
+
+    Scoped to read timeouts only. Connection errors and HTTP failures keep the
+    rollback path, and context-overflow handling is untouched - preserving an
+    over-long context would just reproduce the failure."""
+    return _get_env_bool("ARC3_YIELD_ON_TIMEOUT", True)
+
+
+def _is_read_timeout(exc: BaseException) -> bool:
+    """requests raises ReadTimeout for a stalled response body; ConnectTimeout
+    (a sibling of the same base) means the server was never reached, so it is
+    deliberately excluded."""
+    return isinstance(exc, requests.exceptions.ReadTimeout)
+
+
+def _stub_dead_reasoning_enabled() -> bool:
+    """Replace reasoning-only assistant replies (no tool calls, no content)
+    with a short stub IN-TURN. Without this the dead trace is re-sent on every
+    later request of the same turn - observed at ~35k characters, truncated
+    mid-sentence by finish_reason=length, i.e. a broken thought replayed as
+    context."""
+    return _get_env_bool("ARC3_STUB_DEAD_REASONING", False)
+
+
+_HISTORY_IMAGE_PLACEHOLDER = "[grid image omitted from history]"
+
+
+def _history_image_keep() -> int | None:
+    """How many of the most recent image-bearing history MESSAGES keep their
+    images. None (unset) preserves today's behavior - images accumulate for as
+    long as their message is retained, ~400 tokens each at
+    MULTIMODAL_UPSCALE=10. 0 strips every image from history, so only the
+    current turn is visual. Counting per message rather than per image keeps a
+    multi-image turn (game-over visuals) together in one slot."""
+    raw = os.environ.get("ARC3_HISTORY_IMAGE_KEEP", "").strip()
+    if not raw:
+        return None
+    try:
+        return max(0, int(raw))
+    except ValueError:
+        return None
+
+
+def _resume_prompt_images_enabled() -> bool:
+    """Whether a resumption message re-attaches the grid image. A resumption
+    only happens when nothing executed, so the board is identical to the one
+    the retained opener already showed; skipping it saves an image per
+    resumption and stops resumptions from occupying history retention slots.
+    Default True = today's behavior."""
+    return _get_env_bool("ARC3_RESUME_PROMPT_IMAGES", True)
+
+
+def _message_has_image(message: dict[str, Any]) -> bool:
+    content = message.get("content")
+    if not isinstance(content, list):
+        return False
+    return any(
+        isinstance(part, dict) and part.get("type") == "image_url"
+        for part in content
+    )
+
+
+def _strip_message_images(message: dict[str, Any]) -> dict[str, Any]:
+    """Replace image parts with a short text marker. The marker keeps the
+    message well formed (a message whose only part was an image would
+    otherwise be empty) and tells the model a visual was there rather than
+    silently showing a turn that never had one."""
+    content = message.get("content")
+    if not isinstance(content, list):
+        return message
+    parts: list[Any] = []
+    for part in content:
+        if isinstance(part, dict) and part.get("type") == "image_url":
+            parts.append({"type": "text", "text": _HISTORY_IMAGE_PLACEHOLDER})
+        else:
+            parts.append(part)
+    return {**message, "content": parts}
+
+
+def _apply_history_image_window(
+    messages: list[dict[str, Any]], keep: int
+) -> list[dict[str, Any]]:
+    """Keep images on the `keep` most recent image-bearing messages, strip the
+    rest. Walks backwards so recency wins; originals are never mutated."""
+    seen = 0
+    result: list[dict[str, Any]] = []
+    for message in reversed(messages):
+        if _message_has_image(message):
+            seen += 1
+            if seen > keep:
+                message = _strip_message_images(message)
+        result.append(message)
+    result.reverse()
+    return result
+
+
+# Tag for the pair of messages a rolling summary is made of: the request that
+# asks for it and the reply that is it. Deliberately absent from
+# _CONTROL_KIND_ENV, so no pruning knob can remove them - the trimmer's
+# summary-aware drain navigates by these, and losing one would silently return
+# it to ordinary block dropping.
+_SUMMARY_CONTROL_KIND = "summary"
+# Below this a reply that also called a tool is a preamble rather than a
+# summary. Real ones measured 4,385 to 6,248 characters; an introduction to a
+# tool call measured nine.
+_SUMMARY_MIN_USABLE_CHARS = 500
+
+
+def _summary_interval_tokens() -> int:
+    """New context tokens between summary attempts, successful or not. 0 disables.
+
+    Count newly appended prompts, assistant messages (including reasoning), tool
+    results and images, excluding the summary exchange itself. Re-sending or
+    evicting existing history does not change this counter.
+
+    The model is asked for a summary on a schedule rather than when it feels
+    like one. Measured across a run where the prompt asked for a note every
+    turn: one game in fourteen complied consistently, and the rest wrote
+    something on roughly half their replies - so a mechanism that depends on
+    the model choosing to summarise does not get summaries.
+
+    Each summary is generated from live context, which keeps both the KV prefix
+    and the recurrent state intact - a summary built from a truncated prefix
+    would force a full prefill, and recurrent state cannot be rewound the way a
+    KV prefix can be reused.
+
+    The summary is appended when it is made, so every later turn is generated
+    with it already present. Nothing is ever inserted underneath a message that
+    was produced without it.
+    """
+    return max(0, _get_env_int("ARC3_SUMMARY_INTERVAL_TOKENS", 0))
+
+
+def _drain_stop_at_summaries() -> bool:
+    """Whether the drain may stop early on a summary.
+
+    The drain normally evicts from the budget down to budget-minus-drain in one
+    go. With this on it stops sooner when the head is already a summary request
+    and the request is back under budget - so the retained history starts with a
+    digest of everything dropped, without evicting a token more than the
+    ordinary drain would.
+
+    The earlier design made summaries the ONLY boundary, which could not
+    overshoot in the good case but removed two intervals at once whenever a
+    summary had failed - 38% of a 42k window at an 8k interval. Here the drain
+    target still bounds the drop, so the worst case is exactly today's
+    behaviour and the summary is an early exit rather than the sole exit.
+
+    The trade is coverage: when no summary falls inside the drain range the drop
+    stops mid-span with the previous summary already gone, and that stretch has
+    nothing standing in for it. So ARC3_SUMMARY_INTERVAL_TOKENS wants to be
+    comfortably SMALLER than ARC3_CONTEXT_DRAIN_TOKENS - several candidates per
+    drop - which is the opposite of sizing the interval to replace the drain.
+    """
+    return _get_env_bool("ARC3_DRAIN_STOP_AT_SUMMARIES", False)
+
+
+def _hide_followup_summaries() -> bool:
+    """Send only the first summary in history, hiding the rest.
+
+    Each summary covers everything before it, so a later one restates an
+    earlier one's material - measured at 1 or 2 carried per request, 4,496
+    characters each, so roughly 6% of a 42k window at worst.
+
+    Applied to the wire and to the estimate, never to the stored history: the
+    hidden ones stay so that when eviction removes the first, the next becomes
+    the first and appears. Dropping them at commit would delete what it meant
+    to hide.
+
+    The cost is the thing the rolling design otherwise avoids. A hidden summary
+    is absent while the turns after it are generated and becomes their head
+    later, so those messages end up under something that was not there when
+    they were written - which is exactly the rearrangement the append-only
+    scheme was built to prevent.
+    """
+    return _get_env_bool("ARC3_HIDE_FOLLOWUP_SUMMARIES", False)
+
+
+def _apply_summary_visibility(
+    messages: list[dict[str, Any]], *, evicted: bool
+) -> list[dict[str, Any]]:
+    """Hide completed summaries, preserving a pending final summary request.
+
+    Before the first eviction no completed summaries are shown. What they
+    cover is still in context verbatim, so a summary there is pure duplication.
+    Hiding them all means appending one does not change the prompt, leaving the fill
+    phase a perfect append with a prefix that never moves. The first summary
+    appears when eviction starts, which invalidates the prefix anyway.
+    """
+    if not _hide_followup_summaries():
+        return messages
+
+    # The request currently asking for a summary has no reply yet. It must
+    # reach the model even when completed summaries are hidden. Split it off
+    # without mutating the input; only stored pairs participate in visibility.
+    pending_request: list[dict[str, Any]] = []
+    history = messages
+    if (
+        messages
+        and _message_is_summary(messages[-1])
+        and str(messages[-1].get("role", "")).strip() == "user"
+    ):
+        pending_request = messages[-1:]
+        history = messages[:-1]
+    if not evicted:
+        return [m for m in history if not _message_is_summary(m)] + pending_request
+    kept: list[dict[str, Any]] = []
+    seen_pair = False
+    in_first_pair = False
+    for message in history:
+        if not _message_is_summary(message):
+            kept.append(message)
+            continue
+        is_request = str(message.get("role", "")).strip() == "user"
+        if is_request:
+            # the request opens a pair; the first one and its reply stay
+            in_first_pair = not seen_pair
+            seen_pair = True
+        if in_first_pair:
+            kept.append(message)
+    return kept + pending_request
+
+
+def _summary_replaces_history() -> bool:
+    """Whether a successful summary discards the history it summarised.
+
+    The opposite end of the design from drain sizing: instead of evicting a
+    fixed span and hoping a summary falls inside it, the summary IS the
+    history. Everything above it goes the moment it lands, leaving system
+    prompt, request, summary, and whatever the turn appends after.
+
+    Safe in the sense that matters - nothing is dropped unless a usable summary
+    exists, so a failed or rejected one leaves ordinary eviction in charge and
+    the next turn tries again. But it is the more aggressive option, not the
+    safer one: a large drain still leaves a turn or two of verbatim history,
+    where this leaves none.
+
+    Worth trying because the summaries turn out to be good. One measured at
+    7,107 characters carried the board layout, the movement lattice, which
+    actions did nothing and why, an unresolved two-branch interpretation of the
+    last result, and the next probes to run - which is most of what the evicted
+    turns were holding. And the raw frames are not in the history anyway:
+    current_frame, previous_frame, history and frame_diff are runtime globals
+    rebuilt every turn, so what eviction destroys is the interpretation, which
+    is precisely what the summary keeps.
+
+    Costs: the prefix is invalidated on every summary rather than on every
+    eviction, and the turn immediately after has no verbatim record of the turn
+    before it beyond what the opener carries.
+    """
+    return _get_env_bool("ARC3_SUMMARY_REPLACES_HISTORY", False)
+
+
+def _summary_turn_context() -> bool:
+    """Whether the summary request is told what just happened.
+
+    The request goes out at turn start, BEFORE this turn's opener exists, so
+    the newest thing in context is the previous turn's tool results. A level-up
+    is visible there only as a `level_completed` flag inside a JSON payload -
+    so a summary written at that moment can describe the level just finished as
+    if it were still current, which is exactly the material a later turn needs
+    to be right about.
+
+    What goes in: what executed, whether the board changed, why a sequence
+    stopped early, whether the level changed, and the step and level now.
+
+    What stays out: the valid actions, which are state for the next decision
+    and go stale the moment the summary outlives the turn; the tool inventory
+    and the world-model demands, which are instructions that compete with
+    "write the summary"; and the animation line, which is a pointer to go and
+    read an accessor that will describe a different action by the time anyone
+    reads the summary.
+    """
+    return _get_env_bool("ARC3_SUMMARY_TURN_CONTEXT", False)
+
+
+def _summary_enable_thinking() -> bool:
+    """Whether the summary request lets the model think first.
+
+    OFF by default. Thinking about a summary is not obviously worth it - the
+    material is all in front of the model - and it is expensive twice over:
+    observed at 18,303 characters of reasoning that reached the output limit
+    with no summary at all, and it doubles the generation for a request that
+    already runs once per interval.
+
+    The template must be able to suppress it without moving the prefix. In the
+    Qwen3-family templates the `enable_thinking` conditional wraps BOTH the
+    empty think block at the tail AND the "Reasoning effort is set to ..."
+    sentence at the top of the system message, so the stock template diverges
+    from the first tokens and every summary prefills the whole context -
+    measured as prefix reuse falling from 75% to 45%. Forcing that conditional
+    true (`{%- if true %}`) keeps the sentence unconditional and leaves only
+    the tail block varying, which restores a byte-identical prefix. Check that
+    before assuming this default is free.
+    """
+    return _get_env_bool("ARC3_SUMMARY_ENABLE_THINKING", False)
+
+
+def _summary_max_gen_tokens(default: int | None) -> int:
+    """Output limit: the normal cap, or 8192 when normal output is unlimited.
+
+    The summary is asked for everything worth keeping, so it legitimately wants
+    more room than a turn reply - and it is one request per interval, so a
+    larger ceiling costs little. Raising this keeps the thinking;
+    ARC3_SUMMARY_ENABLE_THINKING=0 removes it.
+    """
+    fallback = default if default is not None and default > 0 else 8192
+    return max(1, _get_env_int("ARC3_SUMMARY_MAX_GEN", fallback))
+
+
+def _message_is_summary(message: dict[str, Any]) -> bool:
+    return message.get(_CONTROL_MESSAGE_KEY) == _SUMMARY_CONTROL_KIND
+
+
+_CONTROL_KIND_ENV = {
+    "nudge": "ARC3_PRUNE_NUDGES",
+    "resume": "ARC3_PRUNE_RESUME_PROMPTS",
+    "stub": "ARC3_PRUNE_DEAD_STUBS",
+}
+
+
+def _prune_control_context_enabled() -> bool:
+    """Master switch: drop every kind of tagged control message when
+    committing turn messages to persistent history. The per-kind knobs
+    (ARC3_PRUNE_NUDGES / _RESUME_PROMPTS / _DEAD_STUBS) turn individual kinds
+    on without it, so an arm can attribute each effect separately."""
+    return _get_env_bool("ARC3_PRUNE_CONTROL_CONTEXT", False)
+
+
+def _prune_control_kind_enabled(kind: str) -> bool:
+    """A kind is pruned if the master switch is on OR its own knob is set.
+
+    - 'nudge'  : the "you have not acted yet" corrective (a few hundred tokens)
+    - 'resume' : the message appended when re-entering a non-executing turn
+                 (a full opener in `full` mode, the short continuation in
+                 `short` mode) - a restatement of unchanged state
+    - 'stub'   : the placeholder that ARC3_STUB_DEAD_REASONING leaves behind
+    """
+    if _prune_control_context_enabled():
+        return True
+    env_name = _CONTROL_KIND_ENV.get(kind)
+    return bool(env_name) and _get_env_bool(env_name, False)
+
+
+def _any_control_pruning_enabled() -> bool:
+    return _prune_control_context_enabled() or any(
+        _get_env_bool(name, False) for name in _CONTROL_KIND_ENV.values()
+    )
+
+
+def _mark_control_message(message: dict[str, Any], kind: str = "resume") -> dict[str, Any]:
+    message[_CONTROL_MESSAGE_KEY] = kind
+    return message
+
+
+def _strip_control_keys(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
+    """Wire-safe copy: the private tag never leaves the process."""
+    cleaned: list[dict[str, Any]] = []
+    for message in messages:
+        if _CONTROL_MESSAGE_KEY in message:
+            copy = dict(message)
+            copy.pop(_CONTROL_MESSAGE_KEY, None)
+            cleaned.append(copy)
+        else:
+            cleaned.append(message)
+    return cleaned
+
+
+def _prune_control_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
+    """Remove tagged control messages, then strip trailing user messages.
+
+    The trailing-user strip is alternation repair, not a preference: pruning a
+    spin-out turn (opener + dead reasoning, no tool calls) would otherwise
+    leave the opener dangling while the next invocation appends another user
+    message. A turn with no tool calls executed nothing, so the next opener
+    regenerates identical state."""
+    kept = [
+        m
+        for m in messages
+        if not (
+            m.get(_CONTROL_MESSAGE_KEY)
+            and _prune_control_kind_enabled(str(m.get(_CONTROL_MESSAGE_KEY)))
+        )
+    ]
+    while kept and str(kept[-1].get("role", "")).strip() == "user":
+        kept.pop()
+    return kept
+
+
+def _reasoning_history_keys() -> tuple[str, ...]:
+    """Message key(s) under which prior reasoning is sent back to the server.
+
+    Backends disagree: OpenRouter returns and reads `reasoning`, while
+    llama.cpp returns `reasoning_content` and its chat template reads ONLY
+    that - so reasoning stored under `reasoning` is silently discarded there
+    (measured: 69 prompt tokens vs 3670 for the same text under the two keys).
+    A dropped reasoning block is invisible except as prompt growth that falls
+    short of the tokens generated.
+
+    Default `reasoning` preserves existing behaviour. Set to
+    `reasoning_content` for llama.cpp, or to a comma list to send both - which
+    is backend-agnostic but risks a template that renders both keys duplicating
+    the text."""
+    raw = os.environ.get("ARC3_REASONING_HISTORY_KEY", "").strip()
+    keys = tuple(k.strip() for k in raw.split(",") if k.strip())
+    return keys or ("reasoning",)
+
+
+def _level_inventory_enabled() -> bool:
+    """The START-vs-START object inventory shown on the first turn of a new
+    level.
+
+    It has its own switch because it answers a different question from the
+    turn-to-turn diff: not "what did my last actions do" but "what is new
+    about this level", with `appeared` types flagged as likely new mechanics.
+    Previously it rode on the auto-diff machinery and was suppressed only when
+    BOTH ARC3_AUTO_FRAME_DIFF and ARC3_DIFF_IMAGE were off - a coupling that
+    was invisible from the knob names."""
+    return _get_env_bool("ARC3_LEVEL_INVENTORY", False)
+
+
+def _resume_diff_image_enabled() -> bool:
+    """Re-attach the current turn's diff image to resumption messages.
+
+    The diff image is produced only while building a full turn opener, so a
+    resumption (short/state_only) carries the current grid but no diff. That
+    is normally harmless - a resumption happens when nothing executed, so the
+    diff would be identical to the one the turn's original opener already
+    showed. It stops being harmless when ARC3_HISTORY_IMAGE_KEEP has stripped
+    that opener's images: the model has then lost the diff entirely, and the
+    resumption is its only chance to see what changed.
+
+    Costs roughly one image per resumption. Off by default."""
+    return _get_env_bool("ARC3_RESUME_DIFF_IMAGE", False)
+
+
+def _reasoning_effort_ladder() -> list[str]:
+    """Reduced reasoning-effort levels to step down through after a truncated
+    generation, e.g. "medium,low".
+
+    A reply cut off at the output ceiling is wasted twice over: the tokens are
+    spent, and if the cut lands inside a tool call the turn produces nothing
+    callable. Rather than accommodating long generations with a bigger ceiling,
+    this asks the model for less deliberation until it completes a turn -
+    stepping one rung down per truncation and resetting once an action
+    executes, which is the evidence that the turn recovered.
+
+    Empty (default) disables the behaviour entirely."""
+    raw = os.environ.get("ARC3_REASONING_EFFORT_LADDER", "").strip()
+    return [level.strip() for level in raw.split(",") if level.strip()]
+
+
+def _preserve_thinking_kwarg() -> bool | None:
+    """Value for the `preserve_thinking` chat-template kwarg, or None to omit.
+
+    The payload builder sets chat_template_kwargs wholesale for vLLM, which
+    replaces rather than merges - so a server-side
+    --default-chat-template-kwargs carrying preserve_thinking can be silently
+    dropped by a request. Defaults to on so the request asserts it rather than
+    depending on server-side merge semantics; set to an empty string to omit
+    the key entirely and leave the server default in charge."""
+    raw = os.environ.get("ARC3_PRESERVE_THINKING", "1").strip().lower()
+    if not raw:
+        return None
+    return raw in ("1", "true", "yes", "on")
+
+
+def _openrouter_provider_prefs(provider: str) -> dict[str, Any] | None:
+    """OpenRouter request-level provider pinning. ARC3_OPENROUTER_PROVIDER
+    holds a provider name (or a comma-separated preference order); routing is
+    then restricted to that list with fallbacks disabled, so every request in
+    a run is served by a known engine. Under :nitro-style routing, providers
+    differ in media handling (some silently drop video/image parts),
+    quantization, and sampling behavior - a silent confound for any controlled
+    comparison. Unset or non-OpenRouter: no change to the request."""
+    if str(provider or "").strip().lower() != "openrouter":
+        return None
+    raw = os.environ.get("ARC3_OPENROUTER_PROVIDER", "").strip()
+    if not raw:
+        return None
+    order = [name.strip() for name in raw.split(",") if name.strip()]
+    if not order:
+        return None
+    return {"order": order, "allow_fallbacks": False}
+
+
+_SLIM_MEMORY_SECTIONS = ("World model", "Plan")
+_MEDIUM_MEMORY_SECTIONS = ("World model", "Plan", "Action model")
+
+_MEMORY_SECTION_GUIDANCE_FULL = (
+                "If you include assistant text before a tool call, use it to update your persistent memory sections. Section meanings: `World model:` = durable RULES of the environment that will still be true many actions from now (mechanics, what objects/colors mean, movement rules, win conditions) — never current positions, events, or what just happened. `Goal model:` = what winning requires. `Action model:` = what each action does mechanically. `Recent findings:` = this turn's observations and events (positions, action results, surprises). `Open questions:` = unresolved hypotheses. `Plan:` = intended next steps. `Cross-level notes:` = knowledge that carries across levels. Example — GOOD: `World model: Red cells are walls; each move shifts the player 2 cells; yellow keys open doors of the same color.` BAD: `World model: Batch stopped after 2 DOWNs, player now at (4,2).` (that belongs in `Recent findings:`). IMPORTANT: each labeled section you write REPLACES the stored version shown under 'Working world model carried from earlier turns' — write the COMPLETE updated section; any fact you omit is permanently lost."
+)
+
+_MEMORY_SECTION_MEANINGS = {
+    "World model": (
+        "durable RULES of the environment that will still be true many actions from now "
+        "(mechanics, what objects/colors mean, movement rules, win conditions) - never "
+        "current positions or what just happened"
+    ),
+    "Goal model": "what winning requires",
+    "Action model": "what each action does mechanically",
+    "Recent findings": "this turn's observations",
+    "Open questions": "unresolved hypotheses",
+    "Plan": "intended next steps",
+    "Cross-level notes": "knowledge that carries across levels",
+}
+
+
+def _advertised_section_labels() -> tuple[str, ...]:
+    """The sections the model should be TOLD about: exactly those the current
+    mode carries.
+
+    Advertising all seven while a mode drops five invites the model to write
+    into sections it will never see again - the write is parsed, stored, and
+    silently discarded from the prompt. It also confounds a slim/medium arm,
+    because a poor result could mean either that fewer sections hurt or that
+    output was wasted on sections that went nowhere."""
+    labels = _memory_section_labels()
+    if labels is None:
+        return tuple(_MEMORY_SECTION_MEANINGS)
+    # canonical order, so the advertisement matches the order the carried block
+    # renders them in
+    chosen = set(labels)
+    return tuple(name for name in _MEMORY_SECTION_MEANINGS if name in chosen)
+
+
+def _section_list_phrase(labels: tuple[str, ...], conjunction: str = "and") -> str:
+    """Render the labels as prose. The conjunction is a parameter because the
+    two call sites word it differently: the opener lists what is available
+    ("and"), the no-tool-call followup offers alternatives ("or")."""
+    quoted = [f"`{name}:`" for name in labels]
+    if len(quoted) == 1:
+        return quoted[0]
+    if len(quoted) == 2:
+        # no serial comma before a two-item conjunction
+        return f" {conjunction} ".join(quoted)
+    return ", ".join(quoted[:-1]) + f", {conjunction} " + quoted[-1]
+
+
+def _memory_section_guidance_upstream() -> str:
+    return (
+        "If you include assistant text before a tool call, keep it short and use it to "
+        "update the world model. Helpful optional prefixes are "
+        f"{_section_list_phrase(_advertised_section_labels())}."
+    )
+
+
+def _memory_section_guidance_empty() -> str:
+    labels = _advertised_section_labels()
+    meanings = " ".join(
+        f"`{name}:` = {_MEMORY_SECTION_MEANINGS[name]}." for name in labels
+    )
+    return (
+        "You have no stored memory sections yet. If you include assistant text before a "
+        f"tool call, use it to start them: {meanings}"
+    )
+
+
+_MEMORY_SECTION_GUIDANCE_UPSTREAM = (
+    "If you include assistant text before a tool call, keep it short and use it to update "
+    "the world model. Helpful optional prefixes are `World model:`, `Goal model:`, "
+    "`Action model:`, `Recent findings:`, `Open questions:`, `Plan:`, and "
+    "`Cross-level notes:`."
+)
+
+
+def _wm_section_explanation_mode() -> str:
+    """Which wording explains the memory sections in the turn opener.
+
+    'tufa' (also spelled 'upstream', the default) is the terse original: one
+    sentence naming the prefixes, framed
+    as optional. 'full' (default) is ours - a glossary of what each section
+    means, a GOOD/BAD example, and the warning that a written section REPLACES
+    the stored one so an omitted fact is lost. Roughly 1100 characters against
+    250, every turn, and it asks for complete rewrites rather than short
+    updates - so it is a candidate for longer generations as well as better
+    memory."""
+    mode = os.environ.get("ARC3_WM_SECTION_EXPLANATION", "tufa").strip().lower()
+    return "full" if mode == "full" else "upstream"
+
+
+_MEMORY_SECTION_GUIDANCE_EMPTY = (
+    "You have no stored memory sections yet. If you include assistant text before a tool "
+    "call, use it to start them: `World model:` = durable RULES of the environment that will "
+    "still be true many actions from now (mechanics, what objects/colors mean, movement "
+    "rules, win conditions) - never current positions or what just happened. `Goal model:` = "
+    "what winning requires. `Action model:` = what each action does mechanically. `Recent "
+    "findings:` = this turn's observations. `Open questions:` = unresolved hypotheses. "
+    "`Plan:` = intended next steps. `Cross-level notes:` = knowledge that carries across "
+    "levels."
+)
+
+_MEMORY_SECTION_GUIDANCE_SLIM = (
+                "If you include assistant text before a tool call, use it to update your persistent memory sections. Section meanings: `World model:` = durable RULES of the environment that will still be true many actions from now (mechanics, what objects/colors mean, movement rules, win conditions) \u2014 never current positions, events, or what just happened. `Plan:` = intended next steps. Example \u2014 GOOD: `World model: Red cells are walls; each move shifts the player 2 cells; yellow keys open doors of the same color.` BAD: `World model: Batch stopped after 2 DOWNs, player now at (4,2).` (that is a passing event, not a rule). IMPORTANT: each labeled section you write REPLACES the stored version shown under 'Working world model carried from earlier turns' \u2014 write the COMPLETE updated section; any fact you omit is permanently lost."
+)
+
+
+def _wm_revise_nudges_enabled() -> bool:
+    """The sticky 'LEVEL CHANGED / ATTEMPT RESET - revise your sections' hint.
+
+    Built for an observed failure (the champion drifting without revising),
+    but a later model maintained its sections diligently with no prompting at
+    all, and the mechanism has needed two corrections since. Turning it off
+    changes ONLY the pressure: sections are still parsed, stored and
+    re-injected, so a null result means the nagging was not earning its
+    prompt weight."""
+    return not _memory_sections_disabled() and _get_env_bool(
+        "ARC3_WM_REVISE_NUDGES", False
+    )
+
+
+def _wm_rebuild_nudge_enabled() -> bool:
+    """The post-wipe "Rebuild them NOW: write fresh World model:, Goal model:,
+    Action model:" demand, which repeats every turn until the model complies.
+
+    An unmodified run wipes the sections silently - the carried block simply
+    vanishes and reappears when the model next writes something. This nudge
+    applies its strongest pressure exactly when the model has least to say: it
+    has just lost its notes and has not yet explored the new level. Default off
+    for that reason; the wipe itself is unaffected."""
+    return not _memory_sections_disabled() and _get_env_bool(
+        "ARC3_WM_REBUILD_NUDGE", False
+    )
+
+
+def _memory_sections_disabled() -> bool:
+    """ARC3_MEMORY_SECTIONS=off removes the persistent-memory mechanic entirely:
+    no carried block, no section guidance, no nudges, no parsing, and the
+    world-model lines drop out of the system prompt too. The model is left to
+    reason from history alone, which is what an unmodified run does before any
+    of this machinery existed."""
+    return os.environ.get("ARC3_MEMORY_SECTIONS", "").strip().lower() == "off"
+
+
+def _memory_section_labels() -> tuple[str, ...] | None:
+    """Which memory sections are advertised and carried, or None for all.
+
+    'slim' keeps World model and Plan - the two the model actually maintains.
+    Measured over 25 games: World model 351 writes, Plan 255, Open questions
+    25, Recent findings 11, Action model 10, Goal model 9, Cross-level notes
+    ZERO. The dropped sections are also the stale ones: a section written once
+    and never revised keeps being re-injected as current, which was observed
+    holding an answered question open for a whole game.
+
+    'medium' additionally keeps Action model. It is written in a third of
+    games and describes MECHANICS rather than transient state, so it ages far
+    better than `Recent findings:` or `Open questions:` - on a game whose
+    difficulty is working out what each control does, it is the section least
+    worth dropping."""
+    mode = os.environ.get("ARC3_MEMORY_SECTIONS", "").strip().lower()
+    if mode == "slim":
+        return _SLIM_MEMORY_SECTIONS
+    if mode == "medium":
+        return _MEDIUM_MEMORY_SECTIONS
+    return None
+
+
+def _message_display_text(message: dict[str, Any]) -> str:
+    """Transcript rendering of a built user message. Content-parts messages
+    carry their text in parts; images are noted rather than dumped."""
+    content = message.get("content")
+    if isinstance(content, str):
+        return content
+    if not isinstance(content, list):
+        return ""
+    pieces: list[str] = []
+    for part in content:
+        if not isinstance(part, dict):
+            continue
+        if part.get("type") == "image_url":
+            pieces.append("[image attached]")
+        else:
+            text = str(part.get("text") or "").strip()
+            if text:
+                pieces.append(text)
+    return "\n\n".join(pieces)
+
+
+def _wm_revise_hint(reason: str, already_shown: bool) -> str:
+    """Text for the sticky revise-nudge. Full event text on first showing;
+    accurate reminder wording on repeats (the event is no longer 'just now',
+    and the adjacent step summary may correctly say the level is unchanged)."""
+    if reason == "level":
+        if not already_shown:
+            return (
+                "LEVEL CHANGED - your memory sections were KEPT, not wiped. Learned "
+                "mechanics, action semantics, and the general goal usually carry over "
+                "to the new level; the map/layout is new, and levels often introduce "
+                "new elements or mechanics. On this first turn, scan the grid for "
+                "unfamiliar elements (compare objects against your World/Action "
+                "model), then revise your `World model:` and `Plan:` sections - "
+                "update the map, keep what still holds, and note anything new."
+            )
+        return (
+            "Reminder: a level change occurred earlier and you have STILL not "
+            "revised your `World model:` or `Plan:` sections since. Do it now - "
+            "update the map for this level, keep what still holds, and note "
+            "anything new."
+        )
+    if reason == "death":
+        if not already_shown:
+            return (
+                "ATTEMPT RESET - your memory sections were KEPT, not wiped. The "
+                "level is unchanged, so your map and mechanics knowledge still "
+                "hold. Amend your sections NOW with what this death taught you "
+                "(the fatal action or sequence and its cause), and revise your "
+                "`Plan:` to avoid repeating it."
+            )
+        return (
+            "Reminder: this attempt was reset by an earlier death and you have "
+            "STILL not amended your memory sections with what it taught you. Do "
+            "it now - record the fatal action or sequence and its cause, and "
+            "revise your `Plan:` to avoid repeating it."
+        )
+    return ""
+
+
+def _suppress_board_dumps(text: str) -> tuple[str, int]:
+    """Collapse runs of full-width board rows in tool output. Enforces the
+    existing 'never print full boards' policy mechanically for models that
+    ignore it: keeps the first rows of each run as orientation, replaces the
+    rest with a stub, and appends a compute-don't-count nudge. Small crops
+    (fewer than 6 rows) pass through untouched. ARC3_BOARD_DUMP_GUARD=0
+    disables."""
+    if not _get_env_bool("ARC3_BOARD_DUMP_GUARD", False):
+        return text, 0
+
+    def _board_like(line: str) -> bool:
+        stripped = line.strip()
+        return (
+            len(stripped) >= 24
+            and stripped.isalnum()
+            and len(set(stripped)) <= 10
+        )
+
+    lines = text.split("\n")
+    out: list[str] = []
+    suppressed = 0
+    i = 0
+    while i < len(lines):
+        if _board_like(lines[i]):
+            j = i
+            while j < len(lines) and _board_like(lines[j]):
+                j += 1
+            run = j - i
+            if run >= 6:
+                out.extend(lines[i:i + 2])
+                out.append(f"[... {run - 2} full-width board rows suppressed ...]")
+                suppressed += run - 2
+            else:
+                out.extend(lines[i:j])
+            i = j
+        else:
+            out.append(lines[i])
+            i += 1
+    if suppressed:
+        out.append(
+            "[BOARD PRINT GUARD] Printing full boards is disallowed and "
+            "character counting is unreliable for you. Compute over the grid "
+            "in python (segmentation nodes, slicing, run-length checks, BFS) "
+            "and print only coordinates, counts, and small crops."
+        )
+    return "\n".join(out), suppressed
+
+
+def _memory_section_max_chars() -> int:
+    """Cap on a single stored memory section, in characters.
+
+    These sections are re-injected into EVERY later turn opener, so an
+    oversized one is paid for repeatedly and grows the prompt without bound.
+    The extraction path used to store them uncapped, which let a degenerate
+    75k-character `World model:` into every subsequent prompt until the
+    request exceeded the model's context entirely. Legitimate sections run a
+    few hundred characters, so the default leaves ample room while making the
+    runaway case impossible. 0 disables the cap (the old behaviour)."""
+    return max(0, _get_env_int("ARC3_MEMORY_SECTION_MAX_CHARS", 8000))
+
+
+def _degeneracy_ratio_threshold() -> float:
+    """Compression ratio below which text is treated as degenerate.
+
+    Measured on representative text: varied prose 0.20-0.28, a coordinate
+    list 0.28, an ASCII board map 0.016, one sentence repeated 300 times
+    0.008, the field mojibake spiral 0.0016. The default sits below the board
+    map so a pasted grid is not falsely rejected, while still catching genuine
+    repetition loops by a wide margin. Rejecting on compressibility catches
+    those generally rather than one encoding artifact, and does not penalise
+    legitimate unicode. 0 disables the check."""
+    try:
+        return max(0.0, float(os.environ.get("ARC3_DEGENERACY_RATIO", "0.01")))
+    except ValueError:
+        return 0.01
+
+
+def _degeneracy_min_chars() -> int:
+    """Length floor for the degeneracy check: short strings compress badly for
+    uninteresting reasons, so testing them would produce false positives."""
+    return max(1, _get_env_int("ARC3_DEGENERACY_MIN_CHARS", 500))
+
+
+def _is_degenerate_text(value: str) -> bool:
+    threshold = _degeneracy_ratio_threshold()
+    if threshold <= 0:
+        return False
+    text = str(value or "")
+    if len(text) < _degeneracy_min_chars():
+        return False
+    try:
+        import zlib
+
+        encoded = text.encode("utf-8", "replace")
+        return len(zlib.compress(encoded, 6)) / max(1, len(encoded)) < threshold
+    except Exception:
+        return False
+
+
 def _normalize_summary_text(value: Any, *, max_chars: int | None = 280) -> str:
     text = " ".join(str(value or "").split())
     if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
@@ -223,9 +1686,100 @@ def _normalize_summary_text(value: Any, *, max_chars: int | None = 280) -> str:
     return f"{text[:max_chars].rstrip()}... [{omitted} chars omitted]"
 
 
+_HEADER_PARENTHETICAL_RE = re.compile(r"\([^)]*\)")
+# Only SHORT decorations are stripped. A long parenthetical or emphasised aside
+# means the line is a sentence rather than a header, and removing it could turn
+# prose into a spurious match - so the header is left intact and simply fails to
+# match, which loses the section rather than mis-capturing it.
+_HEADER_DECORATION_MAX_CHARS = 30
+
+
+def _normalize_section_header(head: str) -> str:
+    """Reduce a candidate section header to its bare label.
+
+    Models decorate these constantly. Observed in a single game: `Revised
+    world model`, `World model update`, `World model updates`, `World model
+    revision`, `Updated world model`, `World model (revised)`, `World model
+    (L4)`, `Action model (from run 1)`, `Plan (19 moves)`, and `Revised world
+    model - **important finding**`. Under exact matching, thirteen of
+    twenty-seven memory writes in that game were silently discarded, including
+    the one that correctly derived the level's kill rule.
+    """
+    def drop_short(match: "re.Match[str]") -> str:
+        return " " if len(match.group(0)) <= _HEADER_DECORATION_MAX_CHARS else match.group(0)
+
+    text = _HEADER_PARENTHETICAL_RE.sub(drop_short, head)
+    # an emphasised trailing aside is dropped on the same terms; the em-dash is
+    # deliberately NOT a cut point - it would discard text AFTER the label, which
+    # can change what the header means, and it earned nothing in practice
+    if "**" in text:
+        head_part, _, rest = text.partition("**")
+        if len(rest) <= _HEADER_DECORATION_MAX_CHARS:
+            text = head_part
+    text = text.replace("*", " ").replace("_", " ").replace("#", " ")
+    # dangling dashes left behind by a removed decoration are punctuation, not
+    # content: stripping them discards nothing, unlike cutting AT a dash
+    return " ".join(text.split()).strip(" \u2014\u2013-")
+
+
+def _tolerant_section_headers_enabled() -> bool:
+    """Accept decorated section headers (`Updated world model:`, `Plan (19
+    moves):`) rather than only the exact label.
+
+    Doubled memory capture on a measured game (12 sections -> 24), which also
+    doubles the carried world-model block the model is asked to rewrite in
+    full each turn - so it is a candidate for longer generations, not only for
+    better memory. Off restores exact matching."""
+    return _get_env_bool("ARC3_TOLERANT_SECTION_HEADERS", False)
+
+
+def _match_section_label(head: str, labels: list[str]) -> str | None:
+    """The canonical label a header refers to, or None.
+
+    Accepts the bare label, or the label with at most ONE extra word before or
+    after it. One word rather than more is deliberate: it covers every
+    decoration seen in practice while keeping prose from matching. Labels are
+    tried longest-first so `Cross-level notes` is not shadowed by `Plan`."""
+    tolerant = _tolerant_section_headers_enabled()
+    # with the knob off, compare the RAW head: normalising would still strip
+    # parentheticals and emphasis, which is itself part of the tolerance
+    normalized = (
+        _normalize_section_header(head).lower() if tolerant else " ".join(head.split()).lower()
+    )
+    if not normalized:
+        return None
+    if tolerant:
+        # A trailing parenthetical is decoration, not part of the label. The
+        # model versions and annotates its own sections - "World model
+        # (revised)", "(major revision)", "World model v34 (level 3)" - and
+        # those are the same section wearing a hat. Measured on one run: the
+        # one-extra-word rule alone caught 89% of section writes, stripping a
+        # trailing parenthetical first took it to 98%.
+        #
+        # Stripping rather than widening the word allowance to two: a
+        # parenthetical is a bounded construct, whereas two free words start
+        # matching prose. The 60-character cap in _extract_labeled_blocks is
+        # applied to the UNDECORATED head, so a long decoration is rejected
+        # before it can be stripped down to a match.
+        without_paren = re.sub(r"\s*\([^)]*\)\s*$", "", normalized).strip()
+        if without_paren:
+            normalized = without_paren
+    for label in sorted(labels, key=len, reverse=True):
+        lowered = label.lower()
+        if normalized == lowered:
+            return label
+        if not tolerant:
+            continue
+        prefix = lowered + " "
+        if normalized.startswith(prefix) and len(normalized[len(prefix):].split()) == 1:
+            return label
+        suffix = " " + lowered
+        if normalized.endswith(suffix) and len(normalized[: -len(suffix)].split()) == 1:
+            return label
+    return None
+
+
 def _extract_labeled_blocks(content: str, labels: list[str]) -> dict[str, str]:
-    normalized_labels = {label.lower(): label for label in labels}
-    targets = tuple(f"{label.lower()}:" for label in labels)
     extracted: dict[str, list[str]] = {label: [] for label in labels}
     current_label: str | None = None
 
@@ -234,15 +1788,17 @@ def _extract_labeled_blocks(content: str, labels: list[str]) -> dict[str, str]:
         candidate = stripped
         while candidate.startswith(("-", "*")):
             candidate = candidate[1:].lstrip()
-        lowered = candidate.lower()
 
         matched_label: str | None = None
         inline_value = ""
-        for target in targets:
-            if lowered.startswith(target):
-                matched_label = normalized_labels[target[:-1]]
-                inline_value = candidate[len(target):].strip()
-                break
+        if ":" in candidate:
+            head, _, rest = candidate.partition(":")
+            # a header is a short line-initial phrase, not a sentence that
+            # happens to contain a colon
+            if len(head) <= 60:
+                matched_label = _match_section_label(head, labels)
+                if matched_label is not None:
+                    inline_value = rest.strip()
 
         if matched_label is not None:
             current_label = matched_label
@@ -254,13 +1810,641 @@ def _extract_labeled_blocks(content: str, labels: list[str]) -> dict[str, str]:
             extracted[current_label].append(stripped)
 
     return {
-        label: _normalize_summary_text("\n".join(lines).strip(), max_chars=None)
+        label: _normalize_summary_text(
+            "\n".join(lines).strip(), max_chars=_memory_section_max_chars() or None
+        )
         for label, lines in extracted.items()
         if "\n".join(lines).strip()
     }
 
 
-def _extract_scientist_note(content: str) -> dict[str, str]:
+def _opener_commit_hint() -> bool:
+    """Ask the turn opener for a commitment, the way a resumption already does.
+
+    Every piece of commitment pressure in the harness lives in the resumption
+    path - "call action(actions) with your best next action or short batch",
+    "Stop deliberating", "Act NOW". The fresh opener says nothing about acting
+    at all. Measured across two runs, sequences following a resumption batch
+    6.73 and 6.91 actions; those following a fresh opener batch 5.43 and 3.91,
+    and the opener figure is what collapsed when resumptions became rarer.
+
+    Deliberately asymmetric: probing stays available but has to be justified,
+    while batching is the default. An earlier prompt that told the model to
+    "re-issue actions one at a time" produced a 200-iteration single-step loop,
+    so the wording avoids prescribing a size while naming the batch.
+    """
+    return _get_env_bool("ARC3_OPENER_COMMIT_HINT", False)
+
+
+def _new_noop_wording() -> bool:
+    """Whether the patch-110 rewordings are used.
+
+    They replaced "changed nothing" with "no change inside the board area"
+    across every guard message. Off restores the wording of the run that scored
+    4.05, so the rewordings can be judged apart from the outcome line they
+    shipped with."""
+    return _get_env_bool("ARC3_NEW_CHANGED_PROMPTS", False)
+
+
+def _animation_timeline_enabled() -> bool:
+    """Whether the diff timeline is offered alongside the raw frames.
+
+    The timeline is a cheap first look, but a summary that reads coherently
+    invites stopping at it: on one level it produced a confident and wrong
+    mechanic that the model only corrected after going to the frames. Off, the
+    model computes its own comparison from `animation_frames` - a few lines it
+    writes readily - at the cost of a snippet before it learns anything, which
+    is wasted on animations that turn out to be irrelevant.
+
+    Only an experiment settles which costs more.
+    """
+    return _get_env_bool("ARC3_ANIMATION_TIMELINE", True)
+
+
+def _dedupe_multicall_line() -> bool:
+    """Drop the duplicated multi-call permission and its stale caveat.
+
+    Upstream states "you may call action() more than once" twice per opener,
+    about thirty lines apart, the second time without the caveat - so the last
+    word the model gets is the permission stripped of its qualifier.
+
+    The caveat itself is now enforced three ways: the batch loop breaks on a
+    terminal result and skips the rest, the per-snippet latch refuses the next
+    action() call, and an executed action ends the turn so the next snippet
+    always opens on a fresh board. On is the change; off keeps upstream.
+    """
+    return _get_env_bool("ARC3_DEDUPE_MULTICALL_LINE", False)
+
+
+def _prefer_tool_calls() -> bool:
+    """State the cost asymmetry: tool calls are free, actions are not.
+
+    The prompt already says not to ration tool calls, but only ever compares
+    them with other tool calls - never with reasoning, which is the trade the
+    model actually makes when it writes twelve thousand characters of
+    hypothesis before its first snippet. Observed: a turn spent 18,012
+    characters reasoning, hit the output cap, and executed nothing.
+    """
+    return _get_env_bool("ARC3_PREFER_TOOL_CALLS", False)
+
+
+# Three bands, a million apart, so no calculated value can cross between them.
+# The maximum _game_priority is about 1,325.
+#
+#   ~2,000,000   never trimmed: the original prefix is whole and this game has
+#                never paid a prefill, so it is the cheapest of all to continue
+#   ~1,000,000   warm again: trimmed once, prefix rebuilt. Ordered within the
+#                band by the estimate at that trim, so a game that was doing
+#                well sits above one that was stalling
+#      0..1,325  awaiting prefill: the trim invalidated the prefix and the next
+#                request has to pay for it anyway, which is the one moment
+#                yielding the slot is free
+#
+# The old scheme sent a flat 999999 while the prefix held, so every game tied at
+# startup and the server broke ties by arrival. Subtracting the queue position
+# gives a deterministic order instead.
+# ---------------------------------------------------------------- TEMPORARY
+# Diagnostic for "the backend never shows more than 13 concurrent requests with
+# 14 games". Records where each game thread is and prints a census once a
+# second, so the missing one can be located rather than guessed at.
+# Remove once the cause is known.
+_GATE_STATS = {"enqueued": 0, "blocked": 0, "blocked_ms": 0, "handover_ms": 0}
+_DIAG_LOCK = threading.Lock()
+_DIAG_STATE: dict[str, str] = {}
+_DIAG_LAST = [0.0]
+
+
+def _diag(agent: Any, where: str) -> None:
+    if not _get_env_bool("ARC3_DIAG_CONCURRENCY", False):
+        return
+    name = getattr(agent, "_diag_name", "?")
+    now = time.monotonic()
+    with _DIAG_LOCK:
+        _DIAG_STATE[name] = where
+        if now - _DIAG_LAST[0] < 1.0:
+            return
+        _DIAG_LAST[0] = now
+        counts: dict[str, int] = {}
+        for value in _DIAG_STATE.values():
+            counts[value] = counts.get(value, 0) + 1
+        total = len(_DIAG_STATE)
+        inflight = counts.get("request", 0)
+        summary = "  ".join(
+            f"{k}={v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])
+        )
+        log.warning(
+            "DIAG %d games, %d in flight | %s | gate: %d handovers, %d blocked, "
+            "%dms blocked, %dms total",
+            total, inflight, summary,
+            _GATE_STATS["enqueued"], _GATE_STATS["blocked"],
+            _GATE_STATS["blocked_ms"], _GATE_STATS["handover_ms"],
+        )
+        if inflight < total:
+            others = sorted(
+                f"{g}:{w}" for g, w in _DIAG_STATE.items() if w != "request"
+            )
+            log.warning("DIAG   elsewhere: %s", ", ".join(others))
+
+
+def _priority_refresh_enabled() -> bool:
+    return any(_get_env_bool(name, False) for name in (
+        "ARC3_PRIORITY_REFRESH_QUEUE", "ARC3_PRIORITY_PACE", "ARC3_PRIORITY_TAIL_FADE",
+    ))
+
+
+def _priority_action_count(summary: dict[str, Any]) -> int:
+    return int(summary.get("end_action_num") or 0)
+
+
+@lru_cache(maxsize=8)
+def _load_pace_reference(path: str) -> dict[str, Any]:
+    data = json.loads(Path(path).read_text())
+    values = [data["fallback_tokens"], *data["by_level"].values()]
+    if not all(isinstance(v, (int, float)) and 0 < v < float("inf") for v in values):
+        raise ValueError("Pace reference costs must be finite positive numbers")
+    return data
+
+
+def _pace_reference_tokens(level: int) -> float:
+    path = os.environ.get("ARC3_PRIORITY_PACE_REFERENCE", "").strip()
+    if not path:
+        path = str(Path(__file__).with_name("priority_pace_reference.json"))
+    reference = _load_pace_reference(path)
+    return float(reference["by_level"].get(str(level), reference["fallback_tokens"]))
+
+
+class _PriorityGate:
+    """Harness-side admission control, for when the server ignores `priority`.
+
+    The request field is a hint a backend may drop; this enforces the same
+    ordering locally by limiting how many games hold a slot at once and
+    choosing who gets a freed one.
+
+    A game keeps its slot while its prefix is intact, because interrupting it
+    there throws away a cache entry about to be reused. When the trimmer drops
+    a message the prefix is invalid anyway and the next request must be
+    prefilled from the divergence point - the one moment when yielding is free -
+    so that is where the handover happens.
+
+    Release and re-enqueue are one critical section. Releasing first would leave
+    a window in which a lower-priority waiter takes the slot the caller was
+    about to compete for.
+    """
+
+    def __init__(self, slots: int) -> None:
+        self._cond = threading.Condition()
+        self._free = max(1, int(slots))
+        self._waiting: list[tuple[int, int]] = []
+        self._seq = 0
+        self._admitted: set[int] = set()
+        self._snapshots: dict[int, PrioritySnapshot] = {}
+        self._clock_start: float | None = None
+        self._clock_deadline: float | None = None
+        self._tail_fade_announced = False
+
+    def configure_clock(self, started_at: float, deadline: float | None) -> None:
+        """One shared monotonic clock, configured before initial admission.
+
+        Earliest session start/deadline wins; appropriate for simultaneously
+        launched sessions. No clock is inferred from the last action payload.
+        """
+        if _get_env_bool("ARC3_PRIORITY_TAIL_FADE", False) and deadline is None:
+            raise ValueError("ARC3_PRIORITY_TAIL_FADE requires a finite runtime or soft deadline")
+        with self._cond:
+            self._clock_start = min(self._clock_start, started_at) if self._clock_start is not None else started_at
+            if deadline is not None:
+                self._clock_deadline = min(self._clock_deadline, deadline) if self._clock_deadline is not None else deadline
+
+    def _snapshot_priority(self, snapshot: PrioritySnapshot, now: float) -> int:
+        elapsed = max(0.0, now - self._clock_start) if self._clock_start is not None else 0.0
+        endgame = _endgame_start_minutes() > 0 and elapsed >= 60 * _endgame_start_minutes()
+        tail_fraction = None
+        if _get_env_bool("ARC3_PRIORITY_TAIL_FADE", False):
+            if self._clock_deadline is None or self._clock_start is None:
+                raise ValueError("ARC3_PRIORITY_TAIL_FADE requires a finite runtime or soft deadline")
+            fraction_setting = os.environ.get("ARC3_PRIORITY_TAIL_FADE_FRACTION", "").strip()
+            if fraction_setting:
+                try:
+                    fraction = float(fraction_setting)
+                except ValueError:
+                    raise ValueError("ARC3_PRIORITY_TAIL_FADE_FRACTION must be > 0 and <= 1") from None
+                if not 0.0 < fraction <= 1.0:
+                    raise ValueError("ARC3_PRIORITY_TAIL_FADE_FRACTION must be > 0 and <= 1")
+                # Use the full shared run duration, not the shrinking time left.
+                window = fraction * max(0.0, self._clock_deadline - self._clock_start)
+            else:
+                window = max(1.0, 60 * _get_env_float("ARC3_PRIORITY_TAIL_FADE_MINUTES", 90.0))
+            remaining = max(0.0, self._clock_deadline - now)
+            tail_fraction = remaining / window if window > 0.0 else 0.0
+            if remaining <= window:
+                # Snapshot scoring also happens outside _pump's lock. Announce
+                # once per shared gate, even when several games trim together.
+                with self._cond:
+                    if not self._tail_fade_announced:
+                        self._tail_fade_announced = True
+                        log.warning(
+                            "tail fade phase after %.1f minutes: %.1f minutes "
+                            "remaining; fading tail B to zero over the final "
+                            "%.1f minutes, retaining hazard C",
+                            elapsed / 60.0, remaining / 60.0, window / 60.0,
+                        )
+        return priority_value(
+            snapshot, endgame=endgame, tail_fraction=tail_fraction,
+            human_actions=_priority_human_actions(), action_scale=_priority_action_scale(),
+            token_scale=_priority_token_scale(), action_weight=_priority_action_weight(),
+            tail_base=_priority_tail_base(), tail_cap=_priority_tail_cap(),
+            **_priority_level_options(),
+        )
+
+    def acquire(self, priority: int) -> None:
+        with self._cond:
+            self._enqueue_and_wait(priority)
+
+    def release(self) -> None:
+        with self._cond:
+            self._free += 1
+            self._pump()
+
+    def handover(self, priority: int, snapshot: PrioritySnapshot | None = None) -> None:  # noqa: D401
+        """Give up the slot and immediately compete for it again.
+
+        The caller may win it straight back, which is the right outcome when
+        nothing else is waiting or everything else is further along.
+        """
+        _t0 = time.monotonic()
+        with self._cond:
+            self._free += 1
+            self._enqueue_and_wait(priority, snapshot)
+        _GATE_STATS["handover_ms"] += int(1000 * (time.monotonic() - _t0))
+
+    def _enqueue_and_wait(self, priority: int, snapshot: PrioritySnapshot | None = None) -> None:
+        # TEMPORARY: with slots == games nobody should ever block here
+        _GATE_STATS["enqueued"] += 1
+        self._seq += 1
+        token = self._seq
+        if snapshot is not None:
+            self._snapshots[token] = snapshot
+        # negated so the heap pops the HIGHEST priority; the sequence number
+        # breaks ties by arrival
+        heapq.heappush(self._waiting, (-int(priority), token))
+        self._pump()
+        if token not in self._admitted:
+            _GATE_STATS["blocked"] += 1
+            t0 = time.monotonic()
+            while token not in self._admitted:
+                self._cond.wait()
+            _GATE_STATS["blocked_ms"] += int(1000 * (time.monotonic() - t0))
+        self._admitted.discard(token)
+
+    def _pump(self) -> None:
+        if _priority_refresh_enabled() and self._free > 0:
+            now = time.monotonic()
+            self._waiting = [
+                (-self._snapshot_priority(self._snapshots[token], now), token)
+                if token in self._snapshots else (priority, token)
+                for priority, token in self._waiting
+            ]
+            heapq.heapify(self._waiting)
+        woke = False
+        while self._free > 0 and self._waiting:
+            _, token = heapq.heappop(self._waiting)
+            self._snapshots.pop(token, None)
+            self._admitted.add(token)
+            self._free -= 1
+            woke = True
+        if woke:
+            self._cond.notify_all()
+
+
+def _max_active_streams() -> int:
+    """Games allowed to hold a slot at once. 0 disables the gate entirely.
+
+    Below the game count this staggers the start - a slot only frees when a
+    game's context is first trimmed, which measurements put 30 to 50 turns in -
+    so early games run alone for a while and late ones begin much later. That is
+    the point: a prefix is reused for as long as it lasts, instead of fourteen
+    games each progressing at a fourteenth of the rate.
+    """
+    return max(0, _get_env_int("ARC3_MAX_ACTIVE_STREAMS", 0))
+
+
+_PRIORITY_GATE: _PriorityGate | None = None
+_PRIORITY_GATE_LOCK = threading.Lock()
+
+
+def _priority_gate() -> _PriorityGate | None:
+    """Process-wide, built on first use. Games run as threads in one process,
+    so a module-level gate needs no plumbing and nothing to pickle."""
+    global _PRIORITY_GATE
+    slots = _max_active_streams()
+    if slots <= 0:
+        return None
+    with _PRIORITY_GATE_LOCK:
+        if _PRIORITY_GATE is None:
+            _PRIORITY_GATE = _PriorityGate(slots)
+            log.warning("priority gate active: %d concurrent streams", slots)
+        return _PRIORITY_GATE
+
+
+_PRIORITY_BAND = 1_000_000
+_PRIORITY_UNTRIMMED_BASE = 2_000_000
+
+
+def _priority_level_options() -> dict[str, Any]:
+    # Keep boolean lookup settings backward compatible. The explicit
+    # "remaining" mode selects the 8/7/5/0 tail, already at its final scale.
+    lookup = os.environ.get("ARC3_PRIORITY_TAIL_LOOKUP", "").strip().lower()
+    return {
+        "normalize_score": _get_env_bool("ARC3_PRIORITY_SCORE_NORMALIZATION", False),
+        "tail_lookup": (
+            "remaining" if lookup == "remaining"
+            else _get_env_bool("ARC3_PRIORITY_TAIL_LOOKUP", False)
+        ),
+        "tail_efficiency": _get_env_float("ARC3_PRIORITY_TAIL_EFFICIENCY", 0.8),
+    }
+
+
+def _priority_action_scale() -> float:
+    return max(1.0, _get_env_float("ARC3_PRIORITY_ACTION_SCALE", 115.0))
+
+
+def _priority_token_scale() -> float:
+    return max(1.0, _get_env_float("ARC3_PRIORITY_TOKEN_SCALE", 62000.0))
+
+
+def _priority_action_weight() -> float:
+    """How much of `c` comes from the action curve rather than the token one.
+
+    A blend rather than max(actions, tokens): max sits below both curves, so it
+    demotes harder than either measure alone warrants. Replaying four runs
+    against their own level-up costs, max was the weakest of every combination
+    tried, while the weight itself barely mattered - 0.25 to 1.0 all landed
+    within a few percent of each other.
+    """
+    return min(1.0, max(0.0, _get_env_float("ARC3_PRIORITY_ACTION_WEIGHT", 0.25)))
+
+
+def _priority_tail_cap() -> float:
+    """Ceiling on `b`, the value of every level after the current one.
+
+    min(cap, level + 1) rather than a constant. Chaining the measured
+    completion rates makes `b` nearly flat, which is why a constant looked
+    right - but that chain prices expected future score, and what the scheduler
+    needs is the term that lets depth outrank freshness. Replayed, a flat tail
+    was the worst option tested at every constrained budget; the cap matters
+    more than the slope, and a lower cap is better because it leaves more of
+    the ordering to `a`.
+    """
+    return max(0.0, _get_env_float("ARC3_PRIORITY_TAIL_CAP", 4.0))
+
+
+def _priority_tail_base() -> float:
+    """Value of `b` at level 1, before the per-level rise.
+
+    b = min(cap, base + level - 1). Base 2 with cap 4 gives the fitted form,
+    2 3 4 4 4...; base 6 with cap 6 gives a flat tail, which is what this
+    replaced and is worth being able to get back for a comparison.
+    """
+    return max(0.0, _get_env_float("ARC3_PRIORITY_TAIL_BASE", 2.0))
+
+
+def _priority_human_actions() -> float:
+    """Denominator offset in the efficiency term.
+
+    The benchmark scores a level by min(1, (human/used)^2). The level is not
+    finished yet, so the term belongs at the expected completion point rather
+    than the current count - hence an offset instead of a bare division.
+    """
+    return max(1.0, _get_env_float("ARC3_PRIORITY_HUMAN_ACTIONS", 25.0))
+
+
+def _endgame_start_minutes() -> float:
+    """Wall-clock minutes from a game's start before it switches phase.
+
+    0 disables the endgame phase entirely.
+    """
+    return max(0.0, _get_env_float("ARC3_ENDGAME_START_MINUTES", 0.0))
+
+
+# Endgame priorities sit in the upper half of whichever warmth band the game is
+# in: the cache argument for the bands is orthogonal to the phase, so a game
+# about to pay a prefill still yields to one that has a warm prefix. The widest
+# value either phase can produce is under 2,000 even at level 15, so half a
+# band is ample.
+_PRIORITY_ENDGAME_OFFSET = 500_000
+
+
+def _game_priority(
+    level: int,
+    actions_on_level: int,
+    tokens_on_level: int = 0,
+    endgame: bool = False,
+    total_levels: int | None = None,
+) -> int:
+    """Expected remaining score for a game on `level` after this much spend.
+
+    Two phases. In the main one, (a + b) * c:
+
+      a  what finishing THIS level banks - the score becomes `level` levels
+         completed, discounted by the benchmark's efficiency term
+      b  every level after it, min(cap, base + level - 1)
+      c  whether this level still yields anything, as a blend of two measured
+         decay curves
+
+    In the endgame there is no time to reach a further level, so the tail and
+    the hazard stop meaning anything and what remains is simply what completing
+    this level would bank. The 0.25 floor lets efficiency separate games at the
+    same depth without letting a shallower one overtake a deeper one.
+
+    Replaying four runs against their own level-up costs, the endgame switch was
+    worth 8 to 12% of total score at constrained budgets and nothing at all once
+    the budget covered what the games actually spent - which is where real runs
+    sit, so this matters only if the competition budget is tighter.
+    """
+    level = max(1, int(level))
+    actions = max(0, int(actions_on_level))
+    tokens = max(0, int(tokens_on_level))
+    options = _priority_level_options()
+    if options["normalize_score"] or options["tail_lookup"]:
+        # Same scoring as the refreshed queue, retaining this legacy path's
+        # phase offset. With both flags off, execute the original path below.
+        value = priority_value(
+            PrioritySnapshot(level, actions, tokens, total_levels=total_levels),
+            endgame=endgame, human_actions=_priority_human_actions(),
+            action_scale=_priority_action_scale(), token_scale=_priority_token_scale(),
+            action_weight=_priority_action_weight(), tail_base=_priority_tail_base(),
+            tail_cap=_priority_tail_cap(), **options,
+        )
+        return value + (_PRIORITY_ENDGAME_OFFSET if endgame else 0)
+    human = _priority_human_actions()
+    efficiency = min(1.0, (human / (human + actions)) ** 2)
+
+    if endgame:
+        value = level * (0.25 + efficiency)
+        return _PRIORITY_ENDGAME_OFFSET + max(1, int(value * 100))
+
+    a = level * efficiency
+    b = min(_priority_tail_cap(), _priority_tail_base() + level - 1.0)
+    weight = _priority_action_weight()
+    by_actions = actions / _priority_action_scale()
+    by_tokens = tokens / _priority_token_scale()
+    c = (
+        weight * 0.5 ** (by_actions * by_actions)
+        + (1.0 - weight) * 0.5 ** (by_tokens * by_tokens)
+    )
+    return max(1, int((a + b) * c * 100))
+
+
+def _persistent_functions_scope() -> str:
+    """How long a kept helper lives: 'levelup' (default), 'turn', or 'game'.
+
+    'turn' clears at every analyzer invocation that is not a resumption, so a
+    turn that inspects, acts and yields loses its toolkit even though the
+    investigation continues. That is conservative but costly: the helpers the
+    keepability check lets through are pure transforms taking the board as an
+    argument, and those do not care that an action happened.
+
+    'levelup' keeps them until the level changes, which is where a helper can
+    genuinely go wrong - not by holding stale data, which the check already
+    prevents, but by encoding an assumption in its code, like a glyph parser
+    with row offsets from the previous layout.
+
+    'game' keeps helpers across turns and level transitions. The model is
+    reminded to recheck level-specific assumptions. New game sessions still
+    start with an empty pool; snippet variables remain ephemeral in every scope.
+    """
+    scope = os.environ.get("ARC3_PERSISTENT_FUNCTIONS_SCOPE", "").strip().lower()
+    return scope if scope in ("turn", "levelup", "game") else "levelup"
+
+
+def _repeat_state_guard_enabled() -> bool:
+    """Refuse an action this snippet has already run from this exact board.
+
+    Buggy loop code can spend a whole life bar getting nowhere: a cursor that
+    wraps turns RIGHT into a five-state cycle, and `while pos < j: LEFT` walks
+    away from its target forever. The terminal latch does stop these, but only
+    once the game ends - 20 to 30 actions later.
+
+    Keyed on (board, action) rather than on repetition counts, because the
+    loops mostly alternate: LEFT, RIGHT, LEFT, RIGHT defeats a same-action rule
+    but revisits its first state on the third action.
+
+    Per snippet, and no override slot. A repeat ACROSS snippets is the model
+    choosing, having seen the board and the results; a repeat inside one is
+    code executing a plan fixed before any of it ran, which is the only case
+    that can be a bug the model has not had a chance to notice. It is also not
+    reading the refusal - its code is running - so writing a new snippet is the
+    response, and that clears the map.
+    """
+    return _get_env_bool("ARC3_REPEAT_STATE_GUARD", False)
+
+
+def _guards_active(level: Any) -> bool:
+    """Whether the per-action guards apply at this level.
+
+    Level 1 pays the smallest share of the score - one part in N - and it is
+    where the model knows least, so it probes hardest: repeating an action to
+    see whether the repeat differs, retrying what killed it to learn what
+    killed it. The guards read that as waste. ARC3_GUARDS_FROM_LEVEL=2 lets
+    level 1 run unguarded.
+
+    Covers the stale block and both repeat guards. NOT the batch no-op stop,
+    which costs one action of a planned sequence rather than refusing a
+    deliberate probe, and not the terminal latch, which stops a snippet acting
+    against a board that no longer exists.
+    """
+    try:
+        floor = max(1, int(os.environ.get("ARC3_GUARDS_FROM_LEVEL", "1") or 1))
+    except ValueError:
+        floor = 1
+    try:
+        return int(level) >= floor
+    except (TypeError, ValueError):
+        return True  # unknown level: guard rather than not
+
+
+def _persistent_functions() -> bool:
+    """Retain eligible function sources across calls; off by default.
+
+    Scope is controlled by ARC3_PERSISTENT_FUNCTIONS_SCOPE. State and variables
+    are never serialized. Definitions are validated and recreated in each fresh
+    sandbox, against its current harness globals.
+    """
+    return _get_env_bool("ARC3_PERSISTENT_FUNCTIONS", False)
+
+
+def _animation_enabled() -> bool:
+    """Whether intermediate animation frames are captured and offered.
+
+    Off by default: this is new information for the model rather than a fix to
+    something broken, so it needs to be measurable on its own."""
+    return _get_env_bool("ARC3_ANIMATION", False)
+
+
+def _report_gameplay_changed() -> bool:
+    """Tell the model WHICH kind of change an action produced.
+
+    The harness computes two flags - `board_changed` over the whole frame and
+    `gameplay_changed` over the interior only - but the turn opener reports the
+    first and the system prompt names only the first. On a game with a
+    remaining-steps bar the whole frame differs after every action, so the
+    opener says a change occurred every time and hedges: "verify that it
+    affected gameplay objects rather than only HUD elements". Meanwhile the
+    per-action trace, built from the second flag, can say the same action
+    changed nothing - so the two lines of the same message contradict each
+    other, and on a single-action turn only the misleading one appears.
+
+    On (default) the opener reports the three states that actually occur and
+    the system prompt explains both flags. The wording stays geometric -
+    "cells inside the board area" rather than "gameplay objects" - because the
+    test is a border crop, not a semantic one."""
+    return _get_env_bool("ARC3_REPORT_GAMEPLAY_CHANGED", False)
+
+
+def _explain_gameplay_changed() -> bool:
+    """Whether the system prompt explains the two change flags.
+
+    Split from the per-turn line because the two halves have very different
+    reach and very different arguments. The sentence is paid once and states
+    something the model cannot work out for itself: `last_action_call_result`
+    exposes both flags and nothing says what distinguishes them, so a game with
+    a timer bar makes `board_changed` true after every action.
+
+    The opener line is paid on every executing turn - on one measured run,
+    thousands of times against thirteen batch-guard firings - and only
+    summarises a comparison the model can make with frame_diff. It is also the
+    half with evidence against it: the one run that carried it scored 2.30 on
+    competition where the run without it scored 4.05, though that run changed
+    guard wording at the same time.
+
+    Defaults to following ARC3_REPORT_GAMEPLAY_CHANGED so the pair behaves as
+    before unless it is set explicitly.
+    """
+    raw = os.environ.get("ARC3_EXPLAIN_GAMEPLAY_CHANGED", "").strip()
+    if not raw:
+        return _report_gameplay_changed()
+    return raw.lower() not in ("0", "false", "no", "off")
+
+
+def _alias_respects_stored() -> bool:
+    """Whether `Hypothesis:` may replace a world model that already exists.
+
+    The alias condition checks only THIS reply's extraction: a reply carrying a
+    hypothesis and no `World model:` has its hypothesis stored as the world
+    model, overwriting whatever was there. That is right on a first turn and
+    destructive later - `Hypothesis` connotes untested, so a speculation
+    silently replaces tested content. Measured on 25 games: 7 such overwrites
+    against 351 world-model writes with tolerant headers on, 16 against 311 in
+    an unpatched run.
+
+    On (default) the aliases apply only when the STORED value is also empty.
+    Off restores the original behaviour."""
+    return _get_env_bool("ARC3_ALIAS_RESPECTS_STORED", True)
+
+
+def _extract_scientist_note(
+    content: str, stored: dict[str, Any] | None = None
+) -> dict[str, str]:
     if not content.strip():
         return {}
     extracted = _extract_labeled_blocks(
@@ -287,12 +2471,19 @@ def _extract_scientist_note(content: str) -> dict[str, str]:
         "current_plan": extracted.get("Plan", ""),
         "cross_level_notes": extracted.get("Cross-level notes", ""),
     }
-    if not result["world_model"]:
-        result["world_model"] = extracted.get("Hypothesis", "")
-    if not result["recent_findings"]:
-        result["recent_findings"] = extracted.get("History check", "")
-    if not result["current_plan"]:
-        result["current_plan"] = extracted.get("Next test", "")
+    respect_stored = _alias_respects_stored()
+    existing = stored or {}
+
+    def apply_alias(key: str, label: str) -> None:
+        if result[key]:
+            return
+        if respect_stored and str(existing.get(key) or "").strip():
+            return
+        result[key] = extracted.get(label, "")
+
+    apply_alias("world_model", "Hypothesis")
+    apply_alias("recent_findings", "History check")
+    apply_alias("current_plan", "Next test")
     return result
 
 
@@ -309,7 +2500,17 @@ def _empty_world_model() -> dict[str, str]:
 
 
 def _request_tool_choice(tools: list[dict[str, Any]] | None) -> str | None:
-    return "auto" if tools else None
+    """Tool-choice sent with tool-bearing requests. LOCAL_ANALYZER_TOOL_CHOICE
+    overrides the default "auto": set "omit"/"none" to drop the field for
+    servers whose validation rejects "auto" without a tool-call parser (tools
+    are still sent, so chat templates still render tool instructions); any
+    other value (e.g. "required") passes through verbatim."""
+    if not tools:
+        return None
+    choice = os.environ.get("LOCAL_ANALYZER_TOOL_CHOICE", "auto").strip().lower()
+    if choice in ("", "omit", "none"):
+        return None
+    return choice
 
 
 def _trim_log_text(text: str, *, max_chars: int = _RESPONSE_META_MAX_CHARS) -> str:
@@ -323,6 +2524,7 @@ def _trim_log_text(text: str, *, max_chars: int = _RESPONSE_META_MAX_CHARS) -> s
 def _format_model_response_meta(
     *,
     finish_reason: str,
+    served_by: str,
     reasoning: str,
     content: str,
     tool_calls: list[dict[str, Any]],
@@ -332,6 +2534,7 @@ def _format_model_response_meta(
 ) -> str:
     lines = [
         f"finish_reason: {finish_reason or '(empty)'}",
+        f"served_by: {served_by or '(unknown)'}",
         f"tool_call_count: {len(tool_calls)}",
         f"content_chars: {len(content)}",
         f"reasoning_chars: {len(reasoning)}",
@@ -347,15 +2550,203 @@ def _format_model_response_meta(
     return "\n".join(lines)
 
 
+def _estimate_diff_tokens(text: str) -> int:
+    return len(text) // 3
+
+
+def _diff_entry_line(bucket: str, entry: dict[str, Any]) -> str:
+    color = entry.get("color")
+    pixels = entry.get("pixels")
+    if bucket == "moved":
+        line = f"moved: color {color} {pixels}px {entry.get('from')}->{entry.get('to')}"
+        if entry.get("rotated_by"):
+            line += f" rotated_by {entry['rotated_by']}"
+        return line
+    if bucket == "rotated":
+        return (
+            f"rotated: color {color} {pixels}px by {entry.get('rotated_by')} "
+            f"at {entry.get('at')}"
+        )
+    if bucket == "changed_color":
+        return (
+            f"recolored at {entry.get('at')}: color "
+            f"{entry.get('before_color')}->{entry.get('after_color')} "
+            f"({entry.get('pixels')}px)"
+        )
+    if bucket == "resized":
+        return (
+            f"resized: color {entry.get('color')} "
+            f"{entry.get('before_pixels')}->{entry.get('after_pixels')}px "
+            f"at {entry.get('at_before')}"
+        )
+    return f"{bucket}: color {color} {pixels}px at {entry.get('at')}"
+
+
+_DIFF_BUCKETS = ("moved", "rotated", "changed_color", "resized", "appeared", "disappeared")
+
+
+def _render_auto_frame_diff(
+    diff: dict[str, Any], budget_tokens: int, total_cells: int | None = None
+) -> list[str]:
+    """Render a frame diff as prompt lines, degrading to stay under budget:
+    full listing -> aggregated by (bucket, color, pixels) -> counts only.
+    Resized entries covering most of the grid (the background reshaping as
+    collateral of other objects moving) are elided as noise; the callable
+    frame_diff() keeps them."""
+    if total_cells:
+        resized = diff.get("resized") or []
+        kept = [e for e in resized if (e.get("before_pixels") or 0) * 2 < total_cells]
+        if len(kept) != len(resized):
+            diff = dict(diff)
+            diff["resized"] = kept
+    count = diff.get("changed_cell_count")
+    if count == -1:
+        return [f"grid shape changed ({diff.get('before_shape')} -> {diff.get('after_shape')}); per-object diff skipped."]
+    header = f"{count} cells changed."
+    # rung 1: full listing
+    full = [header]
+    for bucket in _DIFF_BUCKETS:
+        for entry in diff.get(bucket) or []:
+            full.append("- " + _diff_entry_line(bucket, entry))
+    if _estimate_diff_tokens("\n".join(full)) <= budget_tokens:
+        return full
+    # rung 2: aggregate identical entries, largest objects first, capped
+    lines = [header]
+    for bucket in _DIFF_BUCKETS:
+        entries = list(diff.get(bucket) or [])
+        if not entries:
+            continue
+        entries.sort(key=lambda e: -(e.get("pixels") or e.get("after_pixels") or 0))
+        groups: dict[tuple, int] = {}
+        for entry in entries:
+            key = (entry.get("color"), entry.get("pixels"))
+            groups[key] = groups.get(key, 0) + 1
+        shown = 0
+        for entry in entries:
+            if shown >= 3:
+                break
+            key = (entry.get("color"), entry.get("pixels"))
+            if groups.get(key, 0) > 3:
+                continue
+            lines.append("- " + _diff_entry_line(bucket, entry))
+            shown += 1
+        swarm = {k: n for k, n in groups.items() if n > 3}
+        for (color, pixels), n in sorted(swarm.items(), key=lambda kv: -kv[1]):
+            lines.append(f"- {bucket}: {n}x color {color} {pixels}px (positions elided)")
+        remaining = len(entries) - shown - sum(swarm.values())
+        if remaining > 0:
+            lines.append(f"- {bucket}: +{remaining} more")
+    lines.append("(call frame_diff() in python for the full object list)")
+    if _estimate_diff_tokens("\n".join(lines)) <= budget_tokens:
+        return lines
+    # rung 3: counts only, cannot be large
+    parts = [
+        f"{len(diff.get(bucket) or [])} {bucket}"
+        for bucket in _DIFF_BUCKETS
+        if diff.get(bucket)
+    ]
+    return [
+        header + " " + ", ".join(parts) + " (details omitted; call frame_diff() in python)."
+    ]
+
+
 def _build_system_prompt(*, tool_output_tokens: int) -> str:
     prompt = "You are a coding agent solving a grid-based puzzle game."
+    # ARC3_SYSTEM_PROMPT_PREFIX (read at call time): prepended verbatim as the
+    # first line(s) of the system prompt. Intended for model-specific control
+    # directives that live in the system prompt, e.g. Nemotron's "/think"
+    # reasoning toggle. Empty/unset leaves the prompt unchanged.
+    _prefix = os.environ.get("ARC3_SYSTEM_PROMPT_PREFIX", "").strip()
+    if _prefix:
+        prompt = f"{_prefix}\n\n{prompt}"
     prompt += GAME_OVERVIEW_ADDENDUM
     prompt += STRUCTURED_RUNTIME_STATE_ADDENDUM
+    if _explain_gameplay_changed():
+        prompt += GAMEPLAY_CHANGED_ADDENDUM.format(
+            border=f"{_get_env_int('ARC3_NOOP_GUARD_BORDER', 4)} cells"
+        )
+    if _animation_enabled():
+        prompt += ANIMATION_ADDENDUM
+        if _animation_timeline_enabled():
+            prompt += ANIMATION_ADDENDUM_TIMELINE
     if current_grid_image_enabled():
         prompt += MULTIMODAL_CONTEXT_ADDENDUM
+    if _AUTO_FRAME_DIFF:
+        prompt += (
+            " Each turn's message includes an 'Auto frame diff' section summarizing "
+            "object-level changes since your previous analyzer turn (it spans all "
+            "actions executed in between, not just the last one). Treat it as ground "
+            "truth for what changed, consult it before acting, and call `frame_diff()` "
+            "in python when you need per-action or full object detail."
+        )
+    if _DIFF_IMAGE and current_grid_image_enabled():
+        prompt += (
+            " Alongside the current grid image you receive a diff image: cells that "
+            "changed since your previous analyzer turn are shown in their new true "
+            "color; unchanged cells are dimmed to a dark navy that is not a game "
+            "color. Use it to locate change at a glance."
+        )
     prompt += VISUAL_GAME_ADDENDUM
-    prompt += PYTHON_ADDENDUM
-    prompt += COMPACT_TOOL_SESSION_ADDENDUM.format(tool_output_tokens=tool_output_tokens)
+    if _get_env_bool("ARC3_ACTION_INFO", False):
+        # The action-info section replaces the earlier coordinate-only guidance.
+        prompt = prompt.replace(
+            "- For `MOUSE`, pass `row` and `col` integer arguments. `row` is vertical position, `col` is horizontal position.\n",
+            "",
+        )
+        prompt += ACTION_INFO_ADDENDUM
+        if undo_exposure_mode() == "on":
+            prompt += UNDO_INFO_ADDENDUM
+        if reset_exposed():
+            prompt += RESET_INFO_ADDENDUM
+    prompt += (
+        PYTHON_ADDENDUM_HEAD.replace(
+            "- Every `python` tool call starts fresh. Re-import modules or re-define any custom utility logic you need.\n",
+            "- Python variables reset between tool calls. Re-import modules as needed; "
+            "eligible functions are retained as described in the tool session rules below.\n",
+        ) if _persistent_functions() else PYTHON_ADDENDUM_HEAD
+    )
+    prompt += (
+        WORLD_MODEL_FREE_ADDENDUM if _memory_sections_disabled() else WORLD_MODEL_ADDENDUM
+    )
+    prompt += PYTHON_ADDENDUM_TAIL
+    if _get_env_bool("ARC3_FRAME_DIFF_HINT", False):
+        # documents `frame_diff(before, after)`. The function stays callable
+        # either way - this only controls whether the model is told about it,
+        # so the arm measures whether knowing costs more than it returns.
+        prompt += FRAME_DIFF_HINT_ADDENDUM
+    if _get_env_bool("ARC3_STEP_VERIFICATION_HINT", False):
+        # General checks during action sequences and targeted probes when search fails.
+        prompt += STEP_VERIFICATION_ADDENDUM
+    # only describe the guards that are actually armed: prompt weight spent on a
+    # mechanism that cannot fire is weight the model reads and cannot use, and an
+    # exception it is told about but never sees is worse than silence
+    # The guards are no longer described up front. Every claim the addenda made -
+    # not executed, nothing spent, the batch is checked throughout, how to
+    # override - is repeated in the refusal message itself, which arrives with
+    # the concrete action attached and only when it is relevant. Describing a
+    # mechanism the model may never meet is prompt weight spent on nothing.
+    prompt += COMPACT_TOOL_SESSION_ADDENDUM.format(
+        tool_output_tokens=tool_output_tokens,
+        prefer_tool_calls=PREFER_TOOL_CALLS_LINE if _prefer_tool_calls() else "",
+        persistent_functions=(
+            PERSISTENT_FUNCTIONS_LINE.format(lifetime=(
+                "for later calls throughout this game, including across level changes; "
+                "they are cleared when a new game starts"
+                if _persistent_functions_scope() == "game" else
+                "for later calls on the same level; they are cleared when the level changes"
+                if _persistent_functions_scope() == "levelup" else
+                "for later calls in this analyzer turn, including yield resumptions; "
+                "they are cleared at the start of the next turn"
+            ), import_guidance=(
+                "Supported explicit top-level imports used by retained functions are restored "
+                "with them; imports inside a function also work."
+                if _get_env_bool("ARC3_PERSISTENT_FUNCTIONS_IMPORTS", False) else
+                "Put needed imports inside the function."
+            )) if _persistent_functions() else EPHEMERAL_FUNCTIONS_LINE
+        ),
+    )
+    if _level_transfer_guidance_enabled():
+        prompt = apply_level_transfer_system_guidance(prompt)
     return prompt
 
 
@@ -378,6 +2769,12 @@ class AnalyzerTurnResult:
 class _ToolDispatchResult:
     content: str
     step_executed: bool = False
+    # True when an executed action changed the board outside the HUD border, or
+    # ended the level or the attempt. "Executed" alone is too weak a signal for
+    # the reasoning-effort ladder: bumping a wall executes and achieves nothing,
+    # and resetting on it lets the model oscillate between truncating at full
+    # effort and acting inertly at reduced effort, never settling.
+    made_progress: bool = False
 
 
 @dataclass(frozen=True)
@@ -398,6 +2795,7 @@ class _AsciiFrameView:
 class _AsciiHistoryEntryView:
     action: str
     frame: _AsciiFrameView
+    result: dict[str, Any]
 
     def __str__(self) -> str:
         return f"AsciiHistoryEntryView(action={self.action!r}, frame={self.frame})"
@@ -422,7 +2820,10 @@ def _to_ascii_history_views(history_entries: list[HistoryEntry]) -> list[_AsciiH
         frame_view = _to_ascii_frame_view(entry.frame)
         if frame_view is None:
             continue
-        views.append(_AsciiHistoryEntryView(action=entry.action, frame=frame_view))
+        views.append(_AsciiHistoryEntryView(
+            action=entry.action, frame=frame_view,
+            result=dict(getattr(entry, "result", {}) or {}),
+        ))
     return views
 
 
@@ -445,26 +2846,167 @@ def _ascii_history_view_payload(history_entries: list[HistoryEntry]) -> list[dic
         frame_payload = _ascii_frame_view_payload(entry.frame)
         if frame_payload is None:
             continue
-        payload.append({"action": entry.action, "frame": frame_payload})
+        payload.append({
+            "action": entry.action, "frame": frame_payload,
+            "result": dict(getattr(entry, "result", {}) or {}),
+        })
     return payload
 
 
-def _format_action_span(start_action_num: int | None, end_action_num: int | None) -> str | None:
-    if start_action_num is None or end_action_num is None:
+_VISION_PATCH_PIXELS = 32          # Qwen3.6: 16px patches, 2x2 merge
+_VISION_SENTINEL_TOKENS = 2        # vision start/end markers
+_IMAGE_TOKENS_FALLBACK = 402       # 64-cell board at MULTIMODAL_UPSCALE=10
+_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
+
+
+def _text_token_chars() -> int:
+    """Characters per token for the text heuristic - the SEED value only.
+
+    Measured against a server-reported prompt_tokens on a real request: 25,189
+    characters of system prompt plus opener tokenised to 5,892, i.e. about 4.3
+    characters per token. The 3 default therefore over-counts text by roughly
+    40%, so the trimmer evicts history the budget could have held. This value
+    is used until the first response supplies a measurement; after that
+    ARC3_CALIBRATE_TEXT_TOKENS drives the divisor from the server's own
+    figures."""
+    return max(1, _get_env_int("ARC3_TEXT_TOKEN_CHARS", 3))
+
+
+def _calibrate_text_tokens() -> bool:
+    """Derive the characters-per-token divisor from the server's reported
+    prompt_tokens rather than a fixed guess.
+
+    Every response carries prompt_tokens for the request that produced it - an
+    exact count including the chat template. Subtracting the vision cost of the
+    request's images leaves the text, and dividing by the text characters gives
+    the divisor for the next estimate. No extra requests, no tokenizer, and it
+    tracks content: code-heavy turns tokenise differently from prose.
+
+    The last measurement is used rather than an average: history carries
+    forward, so the prompt just measured is mostly the prompt about to be
+    built, and averaging would blend in content already evicted."""
+    return _get_env_bool("ARC3_CALIBRATE_TEXT_TOKENS", True)
+
+
+# Asymmetric on purpose. Over-counting trims history and degrades one game;
+# under-counting overflows the context, and an overflowing request holds a
+# slot and can pressure the others. Measured across 22 games: 2.75-3.69,
+# mean 3.12. The 3.3 ceiling therefore binds on the sparsest games by
+# design - a deliberate haircut - while the floor is set low enough that
+# only an arithmetic failure reaches it, and such a game is likely lost
+# anyway.
+def _text_token_chars_min() -> float:
+    """Floor for the calibrated divisor. Low enough that only an arithmetic
+    failure reaches it, and a game in that state is likely lost anyway."""
+    # a divisor of zero would make every estimate infinite and trim the whole
+    # history on the first request
+    return max(0.1, _get_env_float("ARC3_TEXT_TOKEN_CHARS_MIN", 1.0))
+
+
+def _text_token_chars_max() -> float:
+    """Ceiling for the calibrated divisor.
+
+    Exposed because 3.3 is a fitted constant, not a property of the tokeniser:
+    it came from 22 games measuring 2.75-3.69, so it binds on the sparsest
+    boards by design. A run whose prompts are denser than that sample - a
+    different game set, a longer system prompt, another model - is being
+    trimmed harder than its own measurements justify, and raising this is how
+    that is tested.
+    """
+    # never below the floor: an inverted pair would clamp every measurement to
+    # the ceiling and silently invert the guard's meaning
+    return max(_text_token_chars_min(),
+               _get_env_float("ARC3_TEXT_TOKEN_CHARS_MAX", 3.3))
+
+
+def _image_token_estimate_enabled() -> bool:
+    """Count image parts by their real vision-token cost instead of by the
+    length of their base64 payload. Without this an image is charged roughly
+    one token per three base64 characters - thousands for a board render whose
+    true cost is a few hundred - so enabling images silently evicts real
+    history."""
+    return _get_env_bool("ARC3_IMAGE_TOKEN_ESTIMATE", True)
+
+
+def _png_dimensions(data_url: str) -> tuple[int, int] | None:
+    """Width/height from a base64 PNG data URL, read from the IHDR header
+    without decoding the image. Only the first 24 bytes are needed."""
+    marker = "base64,"
+    index = data_url.find(marker)
+    if index < 0:
+        return None
+    head = data_url[index + len(marker):index + len(marker) + 32]
+    if len(head) < 32:
+        return None
+    try:
+        raw = base64.b64decode(head, validate=True)
+    except Exception:
         return None
-    if start_action_num <= 0 or end_action_num <= 0:
+    if len(raw) < 24 or not raw.startswith(_PNG_SIGNATURE) or raw[12:16] != b"IHDR":
         return None
-    if start_action_num == end_action_num:
-        return f"{start_action_num}"
-    return f"{start_action_num}-{end_action_num}"
+    width = int.from_bytes(raw[16:20], "big")
+    height = int.from_bytes(raw[20:24], "big")
+    if width <= 0 or height <= 0:
+        return None
+    return width, height
+
+
+def _image_part_tokens(part: dict[str, Any]) -> int:
+    """Vision tokens for one image content part: one token per merged patch
+    plus the two sentinel tokens. ARC3_IMAGE_TOKENS_FLAT overrides the
+    computation (useful for non-Qwen backends) and is also the fallback when
+    the header cannot be read."""
+    flat = _get_env_int("ARC3_IMAGE_TOKENS_FLAT", 0)
+    if flat > 0:
+        return flat
+    url = ""
+    image_url = part.get("image_url")
+    if isinstance(image_url, dict):
+        url = str(image_url.get("url") or "")
+    dims = _png_dimensions(url) if url else None
+    if dims is None:
+        return _IMAGE_TOKENS_FALLBACK
+    width, height = dims
+    cells = -(-width // _VISION_PATCH_PIXELS) * -(-height // _VISION_PATCH_PIXELS)
+    return cells + _VISION_SENTINEL_TOKENS
+
+
+def _split_messages_for_estimate(
+    messages: list[dict[str, Any]],
+) -> tuple[list[dict[str, Any]], int]:
+    """Return (messages with image payloads replaced by a short placeholder,
+    total vision tokens). Originals are never mutated."""
+    image_tokens = 0
+    rewritten: list[dict[str, Any]] = []
+    for message in messages:
+        content = message.get("content")
+        if not isinstance(content, list):
+            rewritten.append(message)
+            continue
+        parts: list[Any] = []
+        for part in content:
+            if isinstance(part, dict) and part.get("type") == "image_url":
+                image_tokens += _image_part_tokens(part)
+                parts.append({"type": "image_url", "image_url": {"url": "<image>"}})
+            else:
+                parts.append(part)
+        rewritten.append({**message, "content": parts})
+    return rewritten, image_tokens
 
 
-def _estimate_tokens(value: Any) -> int:
+def _estimate_tokens(value: Any, chars_per_token: float | None = None) -> int:
     try:
-        rendered = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
+        # ensure_ascii=False is deliberate: with escaping ON, every non-ASCII
+        # character renders as a six-character \uXXXX sequence, so any prompt
+        # containing arrows, box-drawing glyphs or a multiplication sign is
+        # counted at up to 6x its real size. The trimmer decides when to evict
+        # in these units, so the inflation makes it discard history it did not
+        # need to - and the error grows with the share of non-ASCII text.
+        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
     except TypeError:
         rendered = str(value)
-    return max(1, (len(rendered) + 2) // 3)
+    chars = chars_per_token or _text_token_chars()
+    return max(1, int(len(rendered) / chars) + (1 if len(rendered) % chars else 0))
 
 
 def _host_accessible_base_url(base_url: str) -> str:
@@ -785,20 +3327,29 @@ def _append_request_snapshot(
     event: str | None = None,
     tool_choice: str | None = None,
     finish_reason: str | None = None,
+    served_by: str | None = None,
     analysis_step: int | None = None,
     action: int | None = None,
     request_index_within_turn: int | None = None,
+    usage: dict[str, Any] | None = None,
+    chat_template_kwargs: dict[str, Any] | None = None,
 ) -> None:
     payload = {
         "messages": messages,
         "tools": tools or [],
     }
+    if isinstance(chat_template_kwargs, dict) and chat_template_kwargs:
+        payload["chat_template_kwargs"] = dict(chat_template_kwargs)
+    if isinstance(usage, dict) and usage:
+        payload["usage"] = usage
     if event:
         payload["event"] = event
     if tool_choice:
         payload["tool_choice"] = tool_choice
     if finish_reason is not None:
         payload["finish_reason"] = str(finish_reason)
+    if served_by:
+        payload["served_by"] = str(served_by)
     if analysis_step is not None:
         payload["analysis_step"] = analysis_step
     if action is not None:
@@ -885,9 +3436,13 @@ def _extract_reasoning_text(message: dict[str, Any]) -> str:
 
 
 def _is_context_length_error(exc: BaseException) -> bool:
-    message = str(exc).lower()
+    message = str(exc).lower().replace("\u2019", "'")
     return (
         "maximum context length" in message
+        # SGLang reports an oversized prompt with this wording instead of
+        # "maximum context length". Route it through the same drain/retry path.
+        or "is longer than the model's context length" in message
+        or "context_length_exceeded" in message
         or "reduce the length of the input prompt" in message
         or "parameter=input_tokens" in message
         or '"param":"input_tokens"' in message
@@ -899,6 +3454,7 @@ class _ChatCompletionResult:
     message: dict[str, Any]
     finish_reason: str = ""
     usage: dict[str, Any] | None = None
+    served_by: str = ""
 
 
 class ToolAgent:
@@ -910,6 +3466,7 @@ class ToolAgent:
         model: str = _DEFAULT_ANALYZER_MODEL,
         timeout: Optional[float] = None,
         save_request_logs: bool = False,
+        dispatch_index: int = 0,
         api_key: str | None = None,
         base_url: str | None = None,
         provider: str | None = None,
@@ -932,6 +3489,10 @@ class ToolAgent:
         self._tool_steps = None if _LOCAL_ANALYZER_TOOL_STEPS <= 0 else max(1, _LOCAL_ANALYZER_TOOL_STEPS)
         self._python_timeout = min(30, max(1, _LOCAL_ANALYZER_TOOL_TIMEOUT))
         self._yield_seconds = None if _LOCAL_ANALYZER_YIELD_SECONDS <= 0 else float(_LOCAL_ANALYZER_YIELD_SECONDS)
+        self._yield_tokens = (
+            None if _LOCAL_ANALYZER_YIELD_TOKENS <= 0 else int(_LOCAL_ANALYZER_YIELD_TOKENS)
+        )
+        self._turn_generated_tokens = 0
         configured_max_output = _LOCAL_ANALYZER_MAX_OUTPUT
         self._max_output_tokens = None if configured_max_output <= 0 else max(1, configured_max_output)
         self._reply_reserve_tokens = self._max_output_tokens or 512
@@ -946,6 +3507,75 @@ class ToolAgent:
             1024,
             _LOCAL_ANALYZER_CONTEXT_WINDOW - self._reply_reserve_tokens - self._request_safety_margin_tokens,
         )
+        self._turns_without_wm_update = 0
+        self._pending_wm_rebuild_reason = ""
+        self._wm_absorbed_in_turn = False
+        self._auto_diff_prev_frame: Frame | None = None
+        self._pending_diff_image: dict[str, Any] | None = None
+        self._context_was_trimmed = False
+        self._has_evicted = False
+        self._actions_at_level_start = 0
+        self._tokens_at_level_start = 0
+        self._progress_pace = ProgressPace()
+        self._pace_last_completed_level = 0
+        # queue position spreads the untrimmed band so games do not tie
+        self._diag_name = f"g{int(dispatch_index):02d}"
+        self._game_elapsed_seconds: float | None = None
+        self._endgame_announced = False
+        self._priority_current = _PRIORITY_UNTRIMMED_BASE - max(0, int(dispatch_index))
+        self._priority_next = self._priority_current
+        self._kept_functions: dict[str, str] = {}
+        self._last_animation_chain: list[Any] = []
+        self._last_action_gameplay_changed: bool | None = None
+        self._pending_animation_image: dict[str, Any] | None = None
+        self._pending_animation_image_label: str = ""
+        self._pending_diff_image_label: str = ""
+        self._turn_diff_image: dict[str, Any] | None = None
+        self._turn_diff_image_label: str = ""
+        self._pending_gameover_images: list[tuple[str, dict[str, Any]]] = []
+        self._noop_guard_pre_hash: str = ""
+        # (interior board hash, action signature) already run in THIS snippet.
+        # Written in _record_guard_observations, read in _action_guard_hook,
+        # cleared before each sandbox call - three sites, one lifetime.
+        self._summary_context_tokens_since_attempt = 0
+        self._http_initial_grace_used = False
+        self._repeat_state_seen: set[tuple[str, str]] = set()
+        # (state_hash, action_sig) for each action the guard hook waved through,
+        # in execution order - the only place the intermediate board states of a
+        # batch are visible to the agent, since the solver reports outcomes per
+        # position but not the boards they ran against.
+        self._guard_hook_trace: list[tuple[str, str]] = []
+        self._rejected_section_updates: list[str] = []
+        # characters per token, measured from the last response; None until one
+        # arrives, and reported in the transcript whenever it changes
+        self._text_chars_per_token: float | None = None
+        # index into the reasoning-effort ladder; -1 means "server default"
+        self._reasoning_effort_rung: int = -1
+        self._guard_death_override_batch: bool = False
+        self._noop_guard_deaths_enabled: bool = _get_env_bool(
+            "ARC3_DEATH_REPEAT_GUARD", False
+        )
+        self._noop_repeat_guard: NoopRepeatGuard | None = (
+            NoopRepeatGuard()
+            if (
+                _get_env_bool("ARC3_NOOP_REPEAT_GUARD", False)
+                or self._noop_guard_deaths_enabled
+            )
+            else None
+        )
+        # only the no-op half is gated separately; the shared store is created
+        # when either half is enabled
+        self._noop_guard_noops_enabled: bool = _get_env_bool(
+            "ARC3_NOOP_REPEAT_GUARD", False
+        )
+        self._resume_after_yield: bool = False
+        self._resume_reason: str = ""
+        self._prev_level_start_frame: Frame | None = None
+        self._cur_level_start_frame: Frame | None = None
+        self._death_ledger_level: Any = None
+        self._death_ledger_attempts: list[dict[str, Any]] = []
+        self._death_ledger_index: dict[str, list[tuple[int, int]]] = {}
+        self._death_ledger_total: int = 0
         self._history_messages: list[dict[str, Any]] = []
         self._session_runtime_dir: Path | None = None
         self._session_total_tokens = 0
@@ -953,7 +3583,7 @@ class ToolAgent:
         self._step_env_callback: Callable[[dict[str, Any]], dict[str, Any]] | None = None
         self._current_valid_actions: list[str] = []
         self._last_step_summary: dict[str, Any] | None = None
-        self._last_action_result: dict[str, Any] | None = None
+        self._last_action_call_result: dict[str, Any] | None = None
         self._summarized_knowledge = _empty_world_model()
 
     def _headers(self) -> dict[str, str]:
@@ -976,12 +3606,20 @@ class ToolAgent:
         runtime_dir = state_path.parent
         if self._session_runtime_dir != runtime_dir:
             self._session_runtime_dir = runtime_dir
+            self._priority_level_count_warned = False
+            if _persistent_functions():
+                self._kept_functions = {}
+                self._retained_clear_notice = ""
             self._history_messages = []
             self._session_total_tokens = 0
             self._session_generated_tokens = 0
+            self._summary_context_tokens_since_attempt = 0
             self._last_step_summary = None
-            self._last_action_result = None
+            self._last_action_call_result = None
             self._summarized_knowledge = _empty_world_model()
+            self._turns_without_wm_update = 0
+            self._pending_wm_rebuild_reason = ""
+            self._pending_wm_revise_reason = ""
 
     @property
     def total_tokens(self) -> int:
@@ -991,7 +3629,19 @@ class ToolAgent:
     def generated_tokens(self) -> int:
         return max(0, int(self._session_generated_tokens))
 
-    def _accumulate_usage_tokens(self, usage: dict[str, Any] | None) -> None:
+    def _accumulate_usage_tokens(
+        self, usage: dict[str, Any] | None, *, count_toward_turn: bool = True
+    ) -> None:
+        """Add a response's usage to the counters.
+
+        count_toward_turn=False for generation that is not part of the turn's
+        deliberation - the rolling summary. Its tokens are real GPU time and
+        belong in the session and per-game totals, or a summary-heavy run looks
+        cheaper than it is and gets more compute than
+        max_generated_tokens_per_game intends. But the per-turn yield budget
+        measures how long the model has been working on THIS turn, and charging
+        it for a summary would shorten whichever turn happened to follow one.
+        """
         if not isinstance(usage, dict):
             return
         generated_token_count = 0
@@ -1003,6 +3653,8 @@ class ToolAgent:
             except (TypeError, ValueError):
                 continue
         self._session_generated_tokens += generated_token_count
+        if count_toward_turn:
+            self._turn_generated_tokens += generated_token_count
 
         total_tokens = usage.get("total_tokens")
         try:
@@ -1021,6 +3673,41 @@ class ToolAgent:
                 continue
         self._session_total_tokens += token_count
 
+    @staticmethod
+    def _turn_action_trace(executed_results: list[dict[str, Any]]) -> list[str]:
+        """Stitch per-call action traces into one sequentially numbered
+        turn-level trace; synthesize entries for calls without a trace."""
+        entries: list[str] = []
+        for item in executed_results:
+            item_trace = item.get("action_trace")
+            if isinstance(item_trace, list) and item_trace:
+                entries.extend(
+                    re.sub(r"^\d+:\s*", "", str(line)) for line in item_trace
+                )
+                continue
+            names = item.get("executed_actions")
+            if not isinstance(names, list) or not names:
+                display = str(item.get("action_display") or "").strip()
+                names = [display] if display else []
+            if item.get("game_over"):
+                verdict = "GAME OVER (attempt ended immediately after this action)"
+            elif item.get("run_complete") or item.get("done"):
+                verdict = "RUN COMPLETE"
+            elif item.get("level_completed"):
+                verdict = "LEVEL COMPLETED"
+            elif item.get("gameplay_changed") is False:
+                verdict = (
+                    "NO-OP (executed, no change inside the board area)"
+                    if _new_noop_wording()
+                    else "NO-OP (executed, changed nothing)"
+                )
+            elif item.get("gameplay_changed") is True:
+                verdict = "effect"
+            else:
+                verdict = "executed"
+            entries.extend(f"{name} -> {verdict}" for name in names)
+        return [f"{position}: {entry}" for position, entry in enumerate(entries, 1)]
+
     def _summarize_step_sequence(self, action_results: list[dict[str, Any]]) -> dict[str, Any] | None:
         if not action_results:
             return None
@@ -1063,58 +3750,161 @@ class ToolAgent:
             "level_transition": any(bool(item.get("level_completed")) for item in executed_results),
             "run_complete": any(bool(item.get("run_complete")) for item in executed_results),
             "game_over": any(bool(item.get("game_over")) for item in executed_results),
+            "game_over_trace": (
+                ToolAgent._turn_action_trace(executed_results)
+                if any(item.get("game_over") for item in executed_results)
+                else []
+            ),
+            "fatal_action": next(
+                (
+                    str(item.get("fatal_action") or item.get("action_display") or "")
+                    for item in executed_results
+                    if item.get("game_over")
+                ),
+                "",
+            ),
             "board_changed": any(bool(item.get("board_changed")) for item in executed_results),
+            # the interior-only flag, so the opener can say WHICH kind of change
+            # happened instead of hedging over the broader one
+            "gameplay_changed": any(
+                bool(item.get("gameplay_changed")) for item in executed_results
+            ),
             "stop_reason": last.get("stop_reason"),
+            "no_op_stops": [
+                {
+                    "action": item.get("no_op_action"),
+                    "index": item.get("no_op_action_index"),
+                    "skipped": item.get("skipped_actions") or [],
+                    "trace": item.get("action_trace") or [],
+                }
+                for item in executed_results
+                if item.get("stop_reason") == "no_op_action"
+            ],
+            # the animation of the last executed action, which is what
+            # last_animation_frames / last_animation_timeline will return when asked
+            "animation": (executed_results[-1].get("animation") if executed_results else None),
+            "animation_action": (
+                executed_results[-1].get("action_display") if executed_results else None
+            ),
+            "guard_stops": [
+                {
+                    "kind": item.get("stop_reason"),
+                    "action": item.get("blocked_action"),
+                    "detail": item.get("stop_detail"),
+                    "skipped": item.get("skipped_actions") or [],
+                    "trace": item.get("action_trace") or [],
+                }
+                # scans ALL results, not just executed ones: a refusal has
+                # executed=False by construction and is filtered out of
+                # executed_results before this point
+                for item in action_results
+                if item.get("stop_reason") in ("known_noop", "known_death", "stale_state")
+            ],
         }
 
-    def _describe_last_outcome(self, summary: dict[str, Any] | None) -> str:
-        if not summary:
-            return ""
-        span = _format_action_span(
-            summary.get("start_action_num"),
-            summary.get("end_action_num"),
-        )
-        count = summary.get("executed_count")
-        prefix = "Last executed sequence"
-        if span and count:
-            prefix = f"Actions {span} ({count} total)"
-        elif span:
-            prefix = f"Action span {span}"
-        elif count:
-            prefix = f"Last executed sequence ({count} total)"
+    # _describe_last_outcome removed. It built an outcome sentence and was
+    # called from nowhere - in this harness AND upstream - so neither its
+    # wording nor the fallback it carried ever reached a transcript. Two
+    # patches were spent revising text no model ever read. What it uniquely
+    # provided is now rendered inline in _build_user_prompt.
 
-        level = summary.get("level")
-        if summary.get("level_transition"):
-            level_text = f" to level {level}" if level is not None else ""
-            return f"{prefix} triggered a level transition{level_text}; re-ground on the new scene."
-        if summary.get("run_complete"):
-            return f"{prefix} completed the run."
-        if summary.get("game_over"):
-            return f"{prefix} reached GAME_OVER."
-
-        pieces = [prefix]
-        if summary.get("board_changed"):
-            pieces.append("produced a board change; verify that it affected gameplay objects rather than only HUD elements.")
-        else:
-            pieces.append("did not show a confirmed board change; treat this as weak evidence until verified.")
-        stop_reason = _normalize_summary_text(summary.get("stop_reason"))
-        if stop_reason:
-            pieces.append(f"stop_reason={stop_reason}.")
-        return " ".join(pieces)
-
-    def _update_summarized_knowledge_from_assistant(self, content: str) -> None:
-        note = _extract_scientist_note(content)
+    def _update_summarized_knowledge_from_assistant(self, content: str) -> list[str]:
+        if _memory_sections_disabled():
+            # nothing is carried or shown, so parsing would only build state
+            # that never reaches the model
+            return []
+        note = _extract_scientist_note(content, self._summarized_knowledge)
         if not note:
-            return
+            return []
+        updated: list[str] = []
+        rejected: list[str] = []
         for key, value in note.items():
-            if value:
-                self._summarized_knowledge[key] = value
+            if not value:
+                continue
+            if _is_degenerate_text(value):
+                # keep the previous version rather than truncating: a truncated
+                # degenerate section is still degenerate, and it would be
+                # re-injected into every later prompt
+                rejected.append(key)
+                continue
+            self._summarized_knowledge[key] = value
+            updated.append(key)
+        if rejected:
+            self._rejected_section_updates = sorted(set(rejected))
+            log.warning(
+                "rejected degenerate memory section update(s): %s",
+                ", ".join(self._rejected_section_updates),
+            )
+        if updated:
+            self._turns_without_wm_update = 0
+            self._wm_absorbed_in_turn = True
+            if "world_model" in updated:
+                self._pending_wm_rebuild_reason = ""
+            if "world_model" in updated or "current_plan" in updated:
+                self._pending_wm_revise_reason = ""
+        return updated
 
     def _update_summarized_knowledge_from_step_summary(self) -> None:
         summary = self._last_step_summary
         if not summary:
             return
-        if summary.get("level_transition") or summary.get("run_complete") or summary.get("game_over"):
+        wipe_reason = ""
+        revise_reason = ""
+        if summary.get("level_transition"):
+            # the priority estimate is per level, so both counts restart here
+            if _get_env_bool("ARC3_PRIORITY_PACE", False):
+                completed_level = max(1, int(summary.get("level") or 2) - 1)
+                if completed_level > self._pace_last_completed_level:
+                    self._progress_pace.record_completion(
+                        self._session_generated_tokens - self._tokens_at_level_start,
+                        _pace_reference_tokens(completed_level),
+                    )
+                    self._pace_last_completed_level = completed_level
+            self._actions_at_level_start = _priority_action_count(summary)
+            self._tokens_at_level_start = self._session_generated_tokens
+        if summary.get("level_transition") and _persistent_functions_scope() == "levelup":
+            # a helper can encode the old layout in its code even when it holds
+            # no stale data, so the level boundary is where it stops being safe
+            self._kept_functions = {}
+            if _persistent_functions():
+                self._retained_clear_notice = (
+                    "Your retained functions were cleared at this level transition. "
+                    "Redefine any helpers you still need after checking their assumptions "
+                    "against the new board."
+                )
+        elif (
+            summary.get("level_transition")
+            and _persistent_functions_scope() == "game"
+            and _persistent_functions()
+            and getattr(self, "_kept_functions", None)
+        ):
+            self._retained_clear_notice = (
+                "Your retained functions remain available from previous levels. "
+                "Check level-specific assumptions against the new board before reusing them."
+            )
+        if summary.get("level_transition"):
+            # read at call time (not a module constant) so this patch's hunks
+            # stay clear of the env-knob block that other patches extend
+            if _get_env_int("ARC3_WM_WIPE_ON_LEVEL", 1):
+                wipe_reason = "a level transition"
+            else:
+                revise_reason = "level"
+        elif summary.get("run_complete"):
+            wipe_reason = "run completion"
+        elif summary.get("game_over"):
+            if _WM_WIPE_ON_GAME_OVER:
+                wipe_reason = "a game over"
+            else:
+                revise_reason = "death"
+        if revise_reason:
+            self._pending_wm_revise_reason = revise_reason
+            self._wm_revise_hint_shown = False
+            # a newer keep-event supersedes any stale rebuild nag (and vice
+            # versa below): at most one pending nudge, always for the latest event
+            self._pending_wm_rebuild_reason = ""
+        if wipe_reason:
+            self._pending_wm_rebuild_reason = wipe_reason
+            self._pending_wm_revise_reason = ""
             for key in (
                 "world_model",
                 "goal_model",
@@ -1135,7 +3925,24 @@ class ToolAgent:
             ("Plan", self._summarized_knowledge.get("current_plan", "")),
             ("Cross-level notes", self._summarized_knowledge.get("cross_level_notes", "")),
         ]
+        if _memory_sections_disabled():
+            return []
+        rejected = list(getattr(self, "_rejected_section_updates", []))
+        self._rejected_section_updates = []
+        allowed = _memory_section_labels()
+        if allowed is not None:
+            entries = [(label, value) for label, value in entries if label in allowed]
         lines = [f"- {label}: {value}" for label, value in entries if value]
+        if rejected:
+            # without this the model sees its section unchanged and is likely to
+            # emit the same degenerate text again
+            lines.append(
+                "- NOTE: your last update to "
+                + ", ".join(f"`{name.replace('_', ' ')}`" for name in rejected)
+                + " was REJECTED as degenerate (highly repetitive or corrupted "
+                "text) and the previous version above was kept. Rewrite it "
+                "concisely in plain prose."
+            )
         if not lines:
             return []
         return [
@@ -1144,19 +3951,857 @@ class ToolAgent:
             "- Revise any item above immediately if `current_frame` or `history` contradicts it.",
         ]
 
+    def _prepare_auto_diff(
+        self,
+        current_frame: Frame | None,
+        previous_step_summary: dict[str, Any] | None,
+    ) -> list[str]:
+        """Track the frame shown at the previous analyzer turn and, when
+        enabled, produce the auto-injected frame-diff lines and/or the diff
+        image for the outgoing user message. Suppressed on the first turn and
+        on the turn after a game over (the auto-RESET makes that span
+        meaningless)."""
+        if not (_AUTO_FRAME_DIFF or _DIFF_IMAGE or _level_inventory_enabled()):
+            return []
+        # Imported lazily so this patch's hunks stay clear of the module
+        # import header, which other optional patches also modify.
+        from inference.agent.vision_context import diff_image_part
+        from inference.utils.frame_diff import compute_frame_diff
+        from inference.utils.grid_utils import ARC_COLOR_CHARS
+        from inference.utils.segmentation import segment_layer
+        self._pending_diff_image = None
+        self._pending_diff_image_label = ""
+        # a new opener supersedes the previous turn's diff whether or not it
+        # produces one of its own
+        self._turn_diff_image = None
+        self._turn_diff_image_label = ""
+        if current_frame is None:
+            return []
+        previous_frame = self._auto_diff_prev_frame
+        self._auto_diff_prev_frame = current_frame
+        if previous_frame is None:
+            self._cur_level_start_frame = current_frame
+            return []
+        prev_level = getattr(previous_frame, "level", None)
+        cur_level = getattr(current_frame, "level", None)
+        level_changed = (
+            prev_level is not None
+            and cur_level is not None
+            and prev_level != cur_level
+        )
+        if level_changed:
+            # shift the per-level start frames even when a game over follows:
+            # the current frame is this level's (near-)initial board
+            self._prev_level_start_frame = self._cur_level_start_frame
+            self._cur_level_start_frame = current_frame
+        if previous_step_summary and previous_step_summary.get("game_over"):
+            return []
+        if level_changed:
+            if not _level_inventory_enabled():
+                # still clear the diff image: a cross-level diff has no meaning
+                self._pending_diff_image = None
+                self._pending_diff_image_label = ""
+                return []
+            return self._level_transition_diff_lines(
+                current_frame, previous_step_summary
+            )
+        if not (_AUTO_FRAME_DIFF or _DIFF_IMAGE):
+            # reached only when the inventory knob alone kept us here
+            return []
+        prev_step = getattr(previous_frame, "step", None)
+        cur_step = getattr(current_frame, "step", None)
+        span = f"step {prev_step} -> {cur_step}"
+        if _DIFF_IMAGE:
+            part = diff_image_part(previous_frame, current_frame)
+            if part is not None and previous_frame.grid != current_frame.grid:
+                label = (
+                    f"Diff image ({span}): cells that changed since your previous "
+                    "analyzer turn in their new true color; unchanged cells dimmed "
+                    "to dark navy (an off-palette color)."
+                )
+                self._pending_diff_image = part
+                self._pending_diff_image_label = label
+                # kept beyond the single use above so a resumption can re-show
+                # it; replaced whenever a new opener produces a fresh diff, so
+                # a stale one from an earlier turn can never leak in
+                self._turn_diff_image = part
+                self._turn_diff_image_label = label
+        if not _AUTO_FRAME_DIFF:
+            return []
+        executed = 0
+        if previous_step_summary and not previous_step_summary.get("stale"):
+            # a stale summary describes an earlier exchange whose changes were
+            # already diffed then; nothing has executed since, so don't claim
+            # "no cell changed despite N executed actions"
+            try:
+                executed = int(previous_step_summary.get("executed_count") or 0)
+            except (TypeError, ValueError):
+                executed = 0
+        if previous_frame.grid == current_frame.grid:
+            if executed > 0:
+                return [
+                    f"Auto frame diff ({span}): no cell changed since your previous "
+                    f"analyzer turn despite {executed} executed action(s)."
+                ]
+            return []
+        prev_rows = len(previous_frame.grid)
+        cur_rows = len(current_frame.grid)
+        prev_cols = max((len(r) for r in previous_frame.grid), default=0)
+        cur_cols = max((len(r) for r in current_frame.grid), default=0)
+        if (prev_rows, prev_cols) != (cur_rows, cur_cols):
+            return [
+                f"Auto frame diff ({span}): grid shape changed "
+                f"({prev_rows}x{prev_cols} -> {cur_rows}x{cur_cols}); diff skipped."
+            ]
+        diff = compute_frame_diff(
+            previous_frame.grid,
+            current_frame.grid,
+            segment_layer(previous_frame.grid, ARC_COLOR_CHARS).get("nodes", []),
+            segment_layer(current_frame.grid, ARC_COLOR_CHARS).get("nodes", []),
+            max_group_match=_AUTO_FRAME_DIFF_MAX_GROUP,
+        )
+        rendered = _render_auto_frame_diff(
+            diff, _AUTO_FRAME_DIFF_BUDGET, total_cells=cur_rows * cur_cols
+        )
+        lines = [
+            f"Auto frame diff ({span}; covers everything since your previous "
+            "analyzer turn, not just the last action):"
+        ]
+        lines.extend(rendered)
+        return lines
+
+    def _death_ledger_record(self, history_entries: list["HistoryEntry"]) -> None:
+        entries = list(history_entries or [])
+        if len(entries) < 2 or str(entries[-1].action or "").upper() != "RESET":
+            return
+        death_idx = len(entries) - 2
+        death = entries[death_idx]
+        if death.frame is None or str(death.action or "").upper() == "RESET":
+            return
+        death_level = getattr(death.frame, "level", None)
+        if (
+            self._death_ledger_level is not None
+            and death_level is not None
+            and death_level != self._death_ledger_level
+        ):
+            self._death_ledger_attempts = []
+            self._death_ledger_index = {}
+        if death_level is not None:
+            self._death_ledger_level = death_level
+        start_idx = None
+        for idx in range(death_idx - 1, -1, -1):
+            entry = entries[idx]
+            if str(entry.action or "").upper() == "RESET":
+                start_idx = idx
+                break
+            level = (
+                getattr(entry.frame, "level", None)
+                if entry.frame is not None
+                else None
+            )
+            if (
+                death_level is not None
+                and level is not None
+                and level != death_level
+            ):
+                start_idx = idx + 1
+                break
+        partial = start_idx is None
+        if partial:
+            start_idx = 0
+        actions = [
+            str(entries[i].action) for i in range(start_idx + 1, death_idx + 1)
+        ]
+        frames = [entries[i].frame for i in range(start_idx, death_idx)]
+        trim = 0
+        for i in range(len(frames) - 1, -1, -1):
+            if frames[i] is None:
+                trim = i + 1
+                break
+        if trim:
+            frames = frames[trim:]
+            actions = actions[trim:]
+            partial = True
+        if not actions or len(frames) != len(actions):
+            return
+        for item in self._death_ledger_attempts:
+            if item["actions"] == actions:
+                self._death_ledger_total += 1
+                item["ids"].append(self._death_ledger_total)
+                return
+        self._death_ledger_total += 1
+        uidx = len(self._death_ledger_attempts)
+        hashes = [_interior_state_hash(f.grid) for f in frames]
+        self._death_ledger_attempts.append(
+            {
+                "actions": actions,
+                "hashes": hashes,
+                "ids": [self._death_ledger_total],
+                "partial": partial,
+            }
+        )
+        for pos, state_hash in enumerate(hashes):
+            self._death_ledger_index.setdefault(state_hash, []).append((uidx, pos))
+
+    def _death_ledger_payload(self) -> dict[str, Any]:
+        return {
+            "level": self._death_ledger_level,
+            "attempts": [
+                {
+                    "actions": list(item["actions"]),
+                    "state_hashes": list(item["hashes"]),
+                    "attempt_ids": list(item["ids"]),
+                    "partial": bool(item.get("partial")),
+                }
+                for item in self._death_ledger_attempts
+            ],
+        }
+
+    def _death_ledger_observe(
+        self,
+        history_entries: list["HistoryEntry"],
+        previous_step_summary: dict[str, Any] | None,
+    ) -> None:
+        """Record a death, if the previous sequence ended in one.
+
+        Called unconditionally rather than from the rendering path. The ledger
+        has TWO consumers - the advisory lines under ARC3_DEATH_LEDGER, and the
+        death guard's fatal lookup under ARC3_DEATH_REPEAT_GUARD - and the
+        recorder used to live inside the first. So enabling only the guard left
+        it consulting an index nothing ever wrote to: armed, checked, always
+        empty, every repeat of a fatal route allowed through. Observed as three
+        identical deaths in a row with the guard on.
+        """
+        if not (
+            previous_step_summary
+            and previous_step_summary.get("game_over")
+            and not previous_step_summary.get("stale")
+        ):
+            return
+        if not (
+            _get_env_bool("ARC3_DEATH_LEDGER", False)
+            or _get_env_bool("ARC3_DEATH_REPEAT_GUARD", False)
+        ):
+            # nothing reads it; skip the work rather than accumulate state
+            return
+        self._death_ledger_record(history_entries)
+
+    def _death_ledger_lines(
+        self,
+        current_frame: Frame | None,
+        history_entries: list["HistoryEntry"],
+        previous_step_summary: dict[str, Any] | None,
+    ) -> list[str]:
+        """Advisory display of recorded fatal continuations matching the
+        CURRENT interior state. Broad by design (bar/items may not be visible
+        in the interior), hence advisory phrasing; exact-sequence enforcement
+        is a separate concern."""
+        if current_frame is None:
+            return []
+        level = getattr(current_frame, "level", None)
+        if (
+            self._death_ledger_level is not None
+            and level is not None
+            and level != self._death_ledger_level
+        ):
+            self._death_ledger_attempts = []
+            self._death_ledger_index = {}
+            self._death_ledger_level = level
+        if not self._death_ledger_attempts:
+            return []
+        state_hash = _interior_state_hash(current_frame.grid)
+        matches: dict[int, int] = {}
+        for uidx, pos in self._death_ledger_index.get(state_hash, []):
+            matches[uidx] = max(matches.get(uidx, -1), pos)
+        if not matches:
+            return []
+        by_suffix: dict[tuple, list[int]] = {}
+        for uidx, pos in matches.items():
+            item = self._death_ledger_attempts[uidx]
+            suffix = tuple(item["actions"][pos:])
+            by_suffix.setdefault(suffix, []).extend(item["ids"])
+        ordered = sorted(by_suffix.items(), key=lambda kv: len(kv[0]))
+        total_deaths = sum(
+            len(item["ids"]) for item in self._death_ledger_attempts
+        )
+        lines = [
+            "Known fatal continuations FROM THE CURRENT BOARD STATE (this "
+            f"level has {total_deaths} failed attempt(s) recorded; the game "
+            "is deterministic):"
+        ]
+        for suffix, ids in ordered[:3]:
+            shown = ", ".join(suffix[:12]) + (", ..." if len(suffix) > 12 else "")
+            id_text = ", ".join(str(i) for i in sorted(set(ids)))
+            lines.append(
+                f"- {len(suffix)} action(s): {shown} -> GAME OVER "
+                f"(attempt(s) {id_text})"
+            )
+        if len(ordered) > 3:
+            lines.append(
+                f"- +{len(ordered) - 3} more matching fatal continuation(s)"
+            )
+        lines.append(
+            "Do NOT replay a listed continuation exactly - vary the route or "
+            "action at or before its first step. (Advisory: fatality can "
+            "depend on state not visible in the board interior, e.g. bar "
+            "level or held items; if you are certain the situation differs, "
+            "state why before proceeding.) The full ledger is available in "
+            "python as `death_ledger`."
+        )
+        return lines
+
+    def _level_transition_diff_lines(
+        self,
+        current_frame: Frame | None,
+        previous_step_summary: dict[str, Any] | None,
+    ) -> list[str]:
+        """First prompt of a new level: acknowledge the completing action and
+        render a START-vs-START inventory against the previous level instead
+        of the meaningless end-of-old-board vs new-board diff. The diff image
+        is suppressed for this prompt (no cross-level semantics)."""
+        from inference.utils.frame_diff import compute_frame_diff
+        from inference.utils.grid_utils import ARC_COLOR_CHARS
+        from inference.utils.segmentation import segment_layer
+
+        self._pending_diff_image = None
+        self._pending_diff_image_label = ""
+        lines: list[str] = []
+        final_action = ""
+        if previous_step_summary and not previous_step_summary.get("stale"):
+            actions = previous_step_summary.get("executed_actions")
+            if isinstance(actions, list) and actions:
+                final_action = str(actions[-1]).strip()
+        lines.append(
+            "Level completed - the board was replaced, so no object diff is "
+            "shown across the boundary"
+            + (f" (the final action {final_action!r} finished the level)" if final_action else "")
+            + "."
+        )
+        prev_start = self._prev_level_start_frame
+        cur_start = self._cur_level_start_frame
+        if prev_start is None or cur_start is None:
+            return lines
+        grid_a, grid_b = prev_start.grid, cur_start.grid
+        rows = len(grid_b)
+        cols = max((len(r) for r in grid_b), default=0)
+        if len(grid_a) != rows or max(
+            (len(r) for r in grid_a), default=0
+        ) != cols:
+            lines.append(
+                "New level inventory skipped (boards have different shapes)."
+            )
+            return lines
+        diff = compute_frame_diff(
+            grid_a,
+            grid_b,
+            segment_layer(grid_a, ARC_COLOR_CHARS).get("nodes", []),
+            segment_layer(grid_b, ARC_COLOR_CHARS).get("nodes", []),
+            max_group_match=_AUTO_FRAME_DIFF_MAX_GROUP,
+        )
+        lines.append(
+            "New level inventory (previous level START vs this level START; "
+            "read it as a TYPE comparison, not motion: 'moved' = object types "
+            "recurring from the previous level at a new spot, 'appeared' = "
+            "types NOT present on the previous level - likely new mechanics, "
+            "investigate these first, 'disappeared' = previous types absent "
+            "here; identical objects at identical positions are not listed):"
+        )
+        lines.extend(
+            _render_auto_frame_diff(
+                diff, _AUTO_FRAME_DIFF_BUDGET, total_cells=rows * cols
+            )
+        )
+        return lines
+
+    def _prepare_gameover_images(
+        self,
+        history_entries: list["HistoryEntry"],
+        previous_step_summary: dict[str, Any] | None = None,
+    ) -> None:
+        """Queue the death-turn visuals (ARC3_GAMEOVER_DIFF_IMAGE).
+
+        The auto-diff - and with it the diff image - is suppressed after a game
+        over because the auto-RESET makes that span meaningless, so the visual
+        slot is idle on exactly the turn where the death is invisible: the
+        board the model sees has already been reset. Two parts fill it:
+
+          1. the death frame itself (the budget bar in it is what the BAR RULE
+             asks about, and it exists nowhere else in the message), and
+          2. a change-highlight diff of the killing action (last alive frame ->
+             death frame), the spatial form of the textual 'Fatal step diff'.
+
+        Anchoring matches _game_over_diff_lines: history ends with the RESET
+        marker, entries[-2] is the death frame, entries[-3] the last alive one.
+        """
+        self._pending_gameover_images = []
+        if not _get_env_bool("ARC3_GAMEOVER_DIFF_IMAGE", False):
+            return
+        from inference.agent.vision_context import (
+            current_grid_image_part,
+            diff_image_part,
+        )
+
+        entries = list(history_entries or [])
+        if len(entries) < 3 or str(entries[-1].action or "").upper() != "RESET":
+            return
+        death = entries[-2]
+        last_alive = entries[-3]
+        if death.frame is None or str(death.action or "").upper() == "RESET":
+            return
+        death_part = current_grid_image_part(death.frame)
+        if death_part is not None:
+            self._pending_gameover_images.append(
+                (
+                    "Death frame image (the board at the moment the attempt "
+                    "ended, before the automatic reset): inspect the budget bar "
+                    "here for the BAR RULE.",
+                    death_part,
+                )
+            )
+        self._append_fatal_animation_image(previous_step_summary)
+        if last_alive.frame is None:
+            return
+        fatal_part = diff_image_part(last_alive.frame, death.frame)
+        if fatal_part is not None and last_alive.frame.grid != death.frame.grid:
+            self._pending_gameover_images.append(
+                (
+                    "Fatal step diff image (last alive frame -> death frame): "
+                    "cells changed by the killing action in their new true "
+                    "color; unchanged cells dimmed to dark navy (an off-palette "
+                    "color).",
+                    fatal_part,
+                )
+            )
+
+    def _game_over_diff_lines(
+        self,
+        history_entries: list["HistoryEntry"],
+        previous_step_summary: dict[str, Any] | None = None,
+    ) -> list[str]:
+        """After a game over (fresh rendering only), push two diffs over the
+        fatal sequence itself: sequence progress (frame before the sequence's
+        first action -> last alive frame, omitted when fewer than two actions
+        executed) and the fatal step (last alive -> death frame). Anchored by
+        the summary's executed_count - the turns before the fatal sequence
+        already delivered their own diffs while the model was alive."""
+        from inference.utils.frame_diff import compute_frame_diff
+        from inference.utils.grid_utils import ARC_COLOR_CHARS
+        from inference.utils.segmentation import segment_layer
+
+        entries = list(history_entries or [])
+        if len(entries) < 2 or str(entries[-1].action or "").upper() != "RESET":
+            return []
+        death_idx = len(entries) - 2
+        death = entries[death_idx]
+        if death.frame is None or str(death.action or "").upper() == "RESET":
+            return []
+        executed = 0
+        if previous_step_summary:
+            try:
+                executed = int(previous_step_summary.get("executed_count") or 0)
+            except (TypeError, ValueError):
+                executed = 0
+        if executed < 1:
+            executed = 1
+        start_idx = death_idx - executed
+        partial = start_idx < 0
+        if partial:
+            start_idx = 0
+        origin = "earliest retained frame" if partial else "sequence start"
+        death_level = getattr(death.frame, "level", None)
+        while start_idx < death_idx - 1:
+            frame = entries[start_idx].frame
+            level = getattr(frame, "level", None) if frame is not None else None
+            if (
+                frame is None
+                or (
+                    death_level is not None
+                    and level is not None
+                    and level != death_level
+                )
+            ):
+                # the sequence itself crossed a level-up (or a frame is
+                # missing); anchor at the first same-level frame within it
+                start_idx += 1
+                origin = "level start within the sequence"
+                continue
+            break
+        start = entries[start_idx]
+        last_alive = entries[death_idx - 1]
+        if start.frame is None or last_alive.frame is None:
+            return []
+
+        def _diff_lines(frame_a, frame_b, label: str) -> list[str]:
+            grid_a, grid_b = frame_a.grid, frame_b.grid
+            rows = len(grid_b)
+            cols = max((len(r) for r in grid_b), default=0)
+            if len(grid_a) != rows or max(
+                (len(r) for r in grid_a), default=0
+            ) != cols:
+                return [f"{label} skipped (grid shape changed)."]
+            diff = compute_frame_diff(
+                grid_a,
+                grid_b,
+                segment_layer(grid_a, ARC_COLOR_CHARS).get("nodes", []),
+                segment_layer(grid_b, ARC_COLOR_CHARS).get("nodes", []),
+                max_group_match=_AUTO_FRAME_DIFF_MAX_GROUP,
+            )
+            rendered = _render_auto_frame_diff(
+                diff, _AUTO_FRAME_DIFF_BUDGET, total_cells=rows * cols
+            )
+            return [label, *rendered]
+
+        out: list[str] = []
+        pre_death_actions = (death_idx - 1) - start_idx
+        if pre_death_actions >= 1:
+            out.extend(
+                _diff_lines(
+                    start.frame,
+                    last_alive.frame,
+                    f"Sequence progress diff ({origin} step "
+                    f"{getattr(start.frame, 'step', '?')} -> step "
+                    f"{getattr(last_alive.frame, 'step', '?')}, what the first "
+                    f"{pre_death_actions} action(s) of the fatal sequence "
+                    "changed - the reset reverted all of it):",
+                )
+            )
+        out.extend(
+            _diff_lines(
+                last_alive.frame,
+                death.frame,
+                f"Fatal step diff (step {getattr(last_alive.frame, 'step', '?')} "
+                f"-> step {getattr(death.frame, 'step', '?')}, what the killing "
+                f"action {str(death.action)!r} changed):",
+            )
+        )
+        return out
+
+    def _append_fatal_animation_image(self, summary: dict[str, Any] | None) -> None:
+        """An image of what the fatal action's animation hid, on a game over.
+
+        Gated on transient pixels alone, NOT on gameplay_changed: after a death
+        the board has already reset, so gameplay_changed compares the
+        pre-action board against the restarted level and is essentially always
+        true. The justification differs from the ordinary case too - there the
+        board shows nothing, here it shows a different board entirely, and
+        either way the fatal moment is unreachable.
+        """
+        if not _animation_enabled():
+            return
+        mode = animation_death_image_mode()
+        if mode == "off" or not summary:
+            return
+        animation = summary.get("animation") or {}
+        if not int(animation.get("transient_pixels") or 0):
+            return
+        chain = self._last_animation_chain or []
+        cells = _animation_transient_cells(chain)
+        if not cells:
+            return
+        if mode == "composite":
+            part = animation_composite_part(
+                chain, cells, border=_get_env_int("ARC3_NOOP_GUARD_BORDER", 4)
+            )
+            label = (
+                "Fatal action animation image: the cells that changed during the fatal "
+                "action's animation, each in the last colour it held before reverting; "
+                "everything else, including the border, is dimmed to dark navy (an "
+                "off-palette colour). This is a composite of all cells that changed "
+                "during the animation, not a single board state that the game "
+                "displayed. A cell is drawn nearer its true colour the longer it held "
+                "it, so a bright trail marks where something stayed and a uniformly "
+                "dim one marks something that passed through. Neither the death frame "
+                "nor the last alive frame contains these cells - they existed only "
+                "between the two."
+            )
+        else:
+            part, index = animation_peak_part(chain, cells)
+            label = (
+                f"Fatal action animation image (frame {index} of {len(chain) - 1}): the "
+                "board partway through the fatal action's animation, at the point where "
+                "the most cells were showing something they later reverted. This is a "
+                "real state the game displayed, between the last alive frame and the "
+                "death frame."
+            )
+        if part is not None:
+            self._pending_gameover_images.append((label, part))
+
+    def _prepare_animation_image(self, summary: dict[str, Any] | None) -> None:
+        """Attach an image of what the animation hid, when the board did not move.
+
+        Gated on gameplay_changed being false: that is exactly when the board
+        image the model receives is uninformative, because the action did
+        something and no frame it can reach shows what. On any other turn the
+        current board already carries the outcome.
+        """
+        if not _animation_enabled():
+            # ARC3_ANIMATION_IMAGE needs ARC3_ANIMATION: without it no chain is
+            # captured, so this would silently do nothing. Stated rather than
+            # left implicit - a knob that quietly has no effect is the failure
+            # mode that costs the most time to diagnose.
+            return
+        mode = animation_image_mode()
+        if mode == "off" or not summary:
+            return
+        if self._last_action_gameplay_changed is not False:
+            return
+        animation = summary.get("animation") or {}
+        if not int(animation.get("transient_pixels") or 0):
+            return
+        chain = self._last_animation_chain or []
+        cells = _animation_transient_cells(chain)
+        if not cells:
+            return
+        if mode == "composite":
+            part = animation_composite_part(
+                chain, cells, border=_get_env_int("ARC3_NOOP_GUARD_BORDER", 4)
+            )
+            label = (
+                "Animation image: the cells that changed during the animation, each in "
+                "the last colour it held before reverting; everything else, including "
+                "the border, is dimmed to dark navy (an off-palette colour). This is a "
+                "composite of all cells that changed during the animation, not a single "
+                "board state that the game displayed. A cell is drawn nearer its true "
+                "colour the longer it held it, so a bright trail marks where something "
+                "stayed and a uniformly dim one marks something that passed through. "
+                "The action left the board area unchanged, so none of these cells "
+                "appear in `current_frame`."
+            )
+        else:
+            part, index = animation_peak_part(chain, cells)
+            label = (
+                f"Animation image (frame {index} of {len(chain) - 1}): the board partway "
+                "through the animation, at the point where the most cells were showing "
+                "something they later reverted. This is a real state the game displayed, "
+                "but not one that survives into `current_frame`."
+            )
+        if part is not None:
+            self._pending_animation_image = part
+            self._pending_animation_image_label = label
+
+    def _retained_function_context(self) -> str:
+        """Optional prompt metadata must never interrupt game execution."""
+        if not _persistent_functions():
+            return ""
+        try:
+            notice = getattr(self, "_retained_clear_notice", "")
+            signatures = [function_signature(src) for src in self._kept_functions.values()]
+            parts = [notice] if notice else []
+            if signatures:
+                scope = {"levelup": "on this level", "turn": "in this turn", "game": "in this game"}[_persistent_functions_scope()]
+                parts.append(
+                    "Your retained functions from earlier Python calls " + scope + ": "
+                    + ", ".join(signatures) + ". These are your previous definitions, "
+                    "available to call directly. Their presence does not mean they are "
+                    "correct; revise them if new evidence contradicts their assumptions."
+                )
+            self._retained_clear_notice = ""
+            return "\n\n" + "\n".join(parts) if parts else ""
+        except Exception:
+            self._kept_functions = {}
+            self._retained_clear_notice = ""
+            return "\n\nYour retained functions were cleared after a retention error; redefine any you need."
+
+    def _retained_sources(self):
+        if not _persistent_functions():
+            return None
+        try:
+            return list(self._kept_functions.values())
+        except Exception:
+            self._kept_functions = {}
+            return []
+
+    def _record_retained_functions(self, sandbox_result, payload) -> None:
+        if not _persistent_functions():
+            return
+        try:
+            previous = self._kept_functions
+            kept = {}
+            signatures = {}
+            for item in sandbox_result.get("keepable_functions") or []:
+                name, source = item["name"], item["source"]
+                signature = function_signature(source)
+                if signature.split("(", 1)[0] != name:
+                    raise ValueError("invalid retained function metadata")
+                kept[name] = source
+                signatures[name] = signature
+            notes = []
+            scope = {"levelup": "on this level", "turn": "in this turn", "game": "in this game"}[_persistent_functions_scope()]
+            for name, source in kept.items():
+                if previous.get(name) != source:
+                    notes.append("Retained your function " + signatures[name] + " for later Python calls " + scope + ".")
+            rejected = sandbox_result.get("retention_rejected") or []
+            repair_hints = _get_env_bool("ARC3_PERSISTENT_FUNCTIONS_REPAIR_HINTS", False)
+            rejected_names = set()
+            seen_rejections = set()
+            for item in rejected:
+                name, reason = str(item["name"]), str(item["reason"])
+                hint = (str(item.get("hint") or "").strip() if repair_hints else
+                        "Pass snippet-specific data as arguments or redefine it if needed.")
+                rejection_key = (name, reason, hint)
+                if rejection_key in seen_rejections:
+                    continue
+                seen_rejections.add(rejection_key)
+                rejected_names.add(name)
+                if name == "all":
+                    notes.append("Your retained functions were cleared: " + reason + ".")
+                    continue
+                notes.append("Your function " + name + " was not retained: " + reason + "."
+                             + (" " + hint if hint else ""))
+            for name in previous.keys() - kept.keys() - rejected_names:
+                notes.append("Your previous function " + name + " is no longer retained; redefine it if needed.")
+            self._kept_functions = kept
+            if notes:
+                payload["function_retention"] = "\n".join(notes)
+        except Exception:
+            self._kept_functions = {}
+            payload["function_retention"] = "Your retained functions were cleared after a retention error; redefine any you need."
+
     def _build_user_message(self, user_prompt: str, current_frame: Frame | None) -> dict[str, Any]:
+        user_prompt += self._retained_function_context()
         image_part = current_grid_image_part(current_frame)
+        diff_part = self._pending_diff_image
+        diff_label = self._pending_diff_image_label
+        animation_part = self._pending_animation_image
+        animation_label = self._pending_animation_image_label
+        gameover_images = list(self._pending_gameover_images)
+        self._pending_diff_image = None
+        self._pending_diff_image_label = ""
+        self._pending_animation_image = None
+        self._pending_animation_image_label = ""
+        self._pending_gameover_images = []
+        if animation_part is not None:
+            # the animation image only ever fires when nothing inside the board
+            # moved, which is precisely when the diff image shows the ticking
+            # budget bar and nothing else - two images, one of them noise
+            diff_part = None
         if image_part is None:
             return {"role": "user", "content": user_prompt}
 
-        return {
-            "role": "user",
-            "content": [
-                {"type": "text", "text": f"{user_prompt}\n\nCurrent grid image:"},
-                image_part,
-            ],
-        }
+        content: list[dict[str, Any]] = [
+            {"type": "text", "text": f"{user_prompt}\n\nCurrent grid image (the board AFTER the automatic reset - this is what you act on now):" if gameover_images else f"{user_prompt}\n\nCurrent grid image:"},
+            image_part,
+        ]
+        if diff_part is not None:
+            content.append({"type": "text", "text": diff_label})
+            content.append(diff_part)
+        if animation_part is not None:
+            content.append({"type": "text", "text": animation_label})
+            content.append(animation_part)
+        for label, part in gameover_images:
+            content.append({"type": "text", "text": label})
+            content.append(part)
+        return {"role": "user", "content": content}
+
+
+    def _memory_section_guidance(self, block_shown: bool) -> list[str]:
+        """The section explanation, in upstream's slot just before the tool-call
+        format guidance.
+
+        Returns nothing when the mechanic is off. In 'full' mode the wording
+        depends on whether a stored block was shown: the REPLACES warning is
+        meaningless when there is nothing stored to replace, so a first turn or
+        a post-wipe turn gets the format glossary without it."""
+        if _memory_sections_disabled():
+            return []
+        if _wm_section_explanation_mode() == "upstream":
+            return [_memory_section_guidance_upstream()]
+        if not block_shown:
+            return [_memory_section_guidance_empty()]
+        if _memory_section_labels() is None:
+            return [_MEMORY_SECTION_GUIDANCE_FULL]
+        # a mode carrying a subset needs the subset described, not the fixed
+        # two-section text: `medium` carries three
+        return [
+            _memory_section_guidance_empty()
+            + " IMPORTANT: each labeled section you write REPLACES the stored version shown "
+            "under 'Working world model carried from earlier turns' - write the COMPLETE "
+            "updated section; any fact you omit is permanently lost."
+        ]
+
+    @staticmethod
+    def _manual_reset_context_line(summary: dict[str, Any]) -> str | None:
+        if not reset_exposed() or any(summary.get(flag) for flag in (
+            "stale", "game_over", "run_complete", "done", "level_transition", "level_completed",
+        )):
+            return None
+        actions = [str(name).strip().upper() for name in summary.get("executed_actions") or []]
+        if "RESET" not in actions:
+            return None
+        if actions == ["RESET"]:
+            return (
+                "You deliberately reset the current level. The current frame shows "
+                "the state returned by RESET."
+            )
+        return (
+            "You deliberately reset the current level, then executed the subsequent "
+            "actions listed above. The current frame shows their resulting state."
+        )
+
+    def _summary_turn_context_lines(self) -> list[str]:
+        """What just happened, for the summary request.
 
+        Deliberately not _build_user_prompt: that consumes _pending_diff_image,
+        _pending_animation_image and the wipe-nudge state, so calling it twice
+        would swallow images the real opener should carry. This reads
+        _last_step_summary and mutates nothing - the stale flag is set by tool
+        dispatch, never by building a prompt, so an extra read is safe.
+        """
+        summary = self._last_step_summary
+        if not summary or summary.get("stale"):
+            # nothing executed since the last opener; the block would say
+            # "Executed actions: none" and repeat an unchanged state line
+            return []
+        lines: list[str] = []
+        count = summary.get("executed_count")
+        try:
+            normalized = int(count) if count is not None else None
+        except (TypeError, ValueError):
+            normalized = None
+        if not normalized:
+            return []
+        label = "action" if normalized == 1 else "actions"
+        lines.append(f"The code executed {normalized} {label} in the previous sequence.")
+        rendered = summary.get("executed_actions") or []
+        if rendered:
+            prefix = (
+                "Executed actions:" if len(rendered) <= 10
+                else "Executed actions (first 10):"
+            )
+            lines.append(f"{prefix} {', '.join(str(a) for a in rendered[:10])}.")
+        changed = summary.get("gameplay_changed")
+        if _report_gameplay_changed():
+            if changed is True:
+                lines.append("That sequence changed cells inside the board area.")
+            elif changed is False and summary.get("board_changed"):
+                lines.append(
+                    "That sequence changed only cells near the edge - nothing inside "
+                    "the board area moved."
+                )
+            elif changed is False:
+                lines.append("That sequence produced no change anywhere on the board.")
+        for stop in (summary.get("no_op_stops") or [])[:3]:
+            lines.append(
+                f"A batched sequence was stopped by the no-op guard: "
+                f"{stop.get('action')!r} executed but changed nothing, and "
+                f"{stop.get('skipped', 0)} later action(s) were skipped."
+            )
+        for stop in (summary.get("guard_stops") or [])[:3]:
+            lines.append(
+                f"The sequence was stopped before {stop.get('action')!r}: "
+                f"{stop.get('reason', 'a guard refused it')}."
+            )
+        if summary.get("level_completed"):
+            lines.append("You have progressed to a new level!")
+        elif summary.get("game_over"):
+            lines.append("The attempt ended and the level was reset.")
+        else:
+            lines.append(self._manual_reset_context_line(summary) or "You are still on the same level.")
+        step = summary.get("action_num")
+        level = summary.get("level")
+        if step is not None and level is not None:
+            lines.append(f"Current state: step {step}, level {level}.")
+        return lines
 
     def _build_user_prompt(
         self,
@@ -1183,7 +4828,9 @@ class ToolAgent:
             default=current_level,
         )
         lines: list[str] = []
-        if previous_step_summary:
+        if previous_step_summary and previous_step_summary.get("stale"):
+            lines.extend(_stale_summary_lines(previous_step_summary))
+        elif previous_step_summary:
             count = previous_step_summary.get("executed_count")
             try:
                 normalized_count = int(count) if count is not None else None
@@ -1200,18 +4847,236 @@ class ToolAgent:
                 lines.append(f"{action_prefix} {', '.join(rendered_actions[:10])}.")
             else:
                 lines.append("Executed actions: none.")
+            if _report_gameplay_changed() and normalized_count:
+                # Whether the sequence MOVED anything. The harness computes this
+                # for every action - it drives the batch stop - and until now
+                # said nothing about it: `_describe_last_outcome` produces the
+                # sentence but is called nowhere, so neither its wording nor the
+                # older one it replaced has ever reached a transcript. On a
+                # single-action turn no per-action trace is built either, so the
+                # model had no account of the outcome from any source.
+                border = _get_env_int("ARC3_NOOP_GUARD_BORDER", 4)
+                if previous_step_summary.get("gameplay_changed"):
+                    lines.append("That sequence changed cells inside the board area.")
+                elif previous_step_summary.get("board_changed"):
+                    lines.append(
+                        f"That sequence changed only cells within {border} of the edge "
+                        "(usually a timer or remaining-steps bar) - nothing inside the "
+                        "board area moved."
+                    )
+                else:
+                    lines.append("That sequence produced no change anywhere on the board.")
+            if _animation_enabled() and not previous_step_summary.get("game_over"):
+                animation_line = describe_animation(
+                    previous_step_summary.get("animation"),
+                    action_display=previous_step_summary.get("animation_action"),
+                    # the animated action's own flag, not the sequence's: an
+                    # inert SPACE at the end of a batch that moved other things
+                    # still hid everything it did
+                    gameplay_changed=self._last_action_gameplay_changed,
+                    timeline=_animation_timeline_enabled(),
+                )
+                if animation_line:
+                    lines.append(animation_line)
+                self._prepare_animation_image(previous_step_summary)
+            no_op_stops = previous_step_summary.get("no_op_stops") or []
+            for stop in no_op_stops[:3]:
+                skipped = stop.get("skipped") or []
+                skipped_text = (
+                    f" The remaining {len(skipped)} action(s) {skipped!r} were skipped."
+                    if skipped
+                    else ""
+                )
+                lines.append(
+                    f"Note: a batched sequence was stopped by the no-op guard: action "
+                    f"{stop.get('action')!r} (position {stop.get('index')} in that batch) "
+                    + (f"executed but produced no change inside the board area.{skipped_text} "
+                       if _new_noop_wording() else
+                       f"executed but changed nothing outside the border.{skipped_text} ")
+                    +
+                    "That action IS included in the executed list above, so it is the "
+                    f"LAST of them - only the first {max(0, int(stop.get('index') or 1) - 1)} "
+                    "executed action(s) moved anything. Do not count submitted or executed "
+                    "actions to update your position; re-locate from current_frame."
+                )
+                trace = stop.get("trace") or []
+                if trace:
+                    lines.append("Per-action trace: " + "; ".join(str(t) for t in trace) + ".")
+            if len(no_op_stops) > 3:
+                lines.append(
+                    f"({len(no_op_stops) - 3} further no-op guard stop(s) occurred this sequence.)"
+                )
+            guard_stops = previous_step_summary.get("guard_stops") or []
+            for stop in guard_stops[:3]:
+                skipped = stop.get("skipped") or []
+                skipped_text = (
+                    f" The remaining {len(skipped)} action(s) ({skipped!r}) were never executed."
+                    if skipped
+                    else ""
+                )
+                if stop.get("kind") == "stale_state":
+                    # the exception said this in the turn where it happened;
+                    # restated here because the plan being void is worth
+                    # carrying into the turn that has to replace it
+                    lines.append(
+                        f"Note: the sequence was stopped before {stop.get('action')!r} - an "
+                        + ("earlier action in that snippet produced no change inside the "
+                           "board area, so the positions and "
+                           if _new_noop_wording() else
+                           "earlier action in that snippet changed nothing, so the positions and ")
+                        +
+                        f"paths it had computed may have been wrong.{skipped_text} Re-ground on "
+                        "the current board before planning again."
+                    )
+                elif stop.get("kind") == "known_death":
+                    lines.append(
+                        f"Note: the sequence was stopped before {stop.get('action')!r} - that "
+                        "action ended a previous attempt from that exact board state, so it was "
+                        f"NOT executed and the attempt is intact.{skipped_text} Plan a different "
+                        "move, or repeat it deliberately if you now believe it is safe."
+                    )
+                else:
+                    lines.append(
+                        f"Note: the sequence was stopped before {stop.get('action')!r} - that "
+                        + ("action already produced no change inside the board area in that "
+                           "exact board state, so it was "
+                           if _new_noop_wording() else
+                           "action already changed nothing in that exact board state, so it was ")
+                        +
+                        f"NOT executed and no action was spent.{skipped_text}"
+                    )
+                trace = stop.get("trace") or []
+                if trace:
+                    lines.append("Per-action trace: " + "; ".join(str(t) for t in trace) + ".")
+            if len(guard_stops) > 3:
+                lines.append(
+                    f"({len(guard_stops) - 3} further guard stop(s) occurred this sequence.)"
+                )
             if previous_step_summary.get("run_complete"):
                 lines.append("You have completed the run!")
             elif previous_step_summary.get("level_transition"):
-                lines.append("You have progressed to a new level!")
+                lines.append(
+                    LEVEL_START_USER_PROMPT
+                    if _level_transfer_guidance_enabled()
+                    else "You have progressed to a new level!"
+                )
             else:
-                lines.append("You are still on the same level.")
+                lines.append(
+                    self._manual_reset_context_line(previous_step_summary)
+                    or "You are still on the same level."
+                )
             if previous_step_summary.get("game_over"):
-                lines.append("The game is over.")
+                lines.append(
+                    "GAME OVER occurred during the previous sequence. This means the "
+                    "attempt FAILED (it does NOT mean the run is finished). The game has "
+                    "already been automatically RESET: the current level restarted from "
+                    "its initial state and previously completed levels are kept. A game "
+                    "over is caused either by the specific action taken (e.g., moving "
+                    "onto a hazard cell) or by a step/time budget running out, which is "
+                    "usually indicated by a bar at the grid border that shrinks with each "
+                    "action. Before acting again, diagnose the cause by inspecting the "
+                    "board right before the death. `history[-1].action` is 'RESET', "
+                    "`history[-2].action` is the fatal action, `history[-2].frame` is the "
+                    "death frame (the board after the fatal action), and "
+                    "`history[-3].frame` is the board immediately before "
+                    "the fatal action — check what cell the fatal move targeted there, and "
+                    "inspect the budget bar in the death frame. BAR RULE: the bar "
+                    "shrinking with each action is NORMAL — a slightly shorter bar is "
+                    "NOT evidence of the cause of death. The budget caused the death "
+                    "ONLY if the bar is fully (or almost fully) depleted in the death "
+                    "frame (`history[-2].frame`). If it IS depleted: you must reach the "
+                    "goal in fewer actions, or look for a way to restore the bar (e.g., "
+                    "collecting an item that refills it). If the bar clearly had budget "
+                    "remaining: the fatal action itself caused the death (hazard or "
+                    "forbidden move) — do NOT retry the same path; a DIFFERENT approach "
+                    "must be tried. Record the diagnosed cause in your world model."
+                )
+                if _animation_enabled():
+                    # after the BAR RULE, which tells the model how to read
+                    # history[-2] and history[-3]; this says what those two
+                    # frames cannot contain
+                    death_animation_line = describe_animation(
+                        previous_step_summary.get("animation"),
+                        action_display=previous_step_summary.get("animation_action"),
+                        # the animated action's own flag, not the sequence's: an
+                    # inert SPACE at the end of a batch that moved other things
+                    # still hid everything it did
+                    gameplay_changed=self._last_action_gameplay_changed,
+                        game_over=True,
+                        timeline=_animation_timeline_enabled(),
+                    )
+                    if death_animation_line:
+                        lines.append(death_animation_line)
+                fatal_action = str(previous_step_summary.get("fatal_action") or "")
+                if fatal_action:
+                    lines.append(
+                        f"The game over occurred immediately after action {fatal_action!r}. "
+                        "Apply the BAR RULE above to decide: bar fully depleted in the "
+                        "death frame means budget death (shorter route or bar restore "
+                        "needed); bar with budget remaining means this action itself was "
+                        "fatal and a DIFFERENT approach is required."
+                    )
+                game_over_trace = previous_step_summary.get("game_over_trace") or []
+                if game_over_trace:
+                    lines.append(
+                        "Per-action trace of the previous sequence: "
+                        + "; ".join(str(t) for t in game_over_trace)
+                        + "."
+                    )
+                self._prepare_gameover_images(
+                    history_entries, previous_step_summary
+                )
+                if _get_env_bool("ARC3_GAMEOVER_DIFF", False):
+                    # read at call time so this patch's hunks stay clear of the
+                    # env-knob block that other patches extend
+                    lines.extend(
+                        self._game_over_diff_lines(
+                            history_entries, previous_step_summary
+                        )
+                    )
+                lines.append(
+                    "Do NOT re-submit this same series of moves: the game is "
+                    "deterministic, so replaying the sequence that just killed you "
+                    "will kill you again. Change the plan BEFORE the fatal point - a "
+                    "different route, action, or timing. If you believe the same "
+                    "moves 'should' work, that belief is exactly what this death "
+                    "falsified; record the correction in your world model instead "
+                    "of retesting it."
+                )
         elif (current_frame is not None and current_frame.step > 0) or action_num > 0:
             lines.append("No previous action sequence was captured.")
         else:
             lines.append("No previous sequence has been executed yet.")
+        if self._pending_wm_rebuild_reason and _wm_rebuild_nudge_enabled():
+            lines.append(
+                f"IMPORTANT: your memory sections were reset by {self._pending_wm_rebuild_reason} "
+                "(cross-level notes were kept). Rebuild them NOW: write fresh "
+                # naming a section a restricted mode drops would ask for a
+                # rebuild that is parsed and then thrown away
+                f"{_section_list_phrase(_advertised_section_labels())} sections for this "
+                "level in your next message, before or alongside your tool call."
+            )
+        elif (
+            _wm_revise_nudges_enabled()
+            and getattr(self, "_pending_wm_revise_reason", "") in ("level", "death")
+        ):
+            _revise_reason = getattr(self, "_pending_wm_revise_reason", "")
+            lines.append(
+                _wm_revise_hint(
+                    _revise_reason,
+                    getattr(self, "_wm_revise_hint_shown", False),
+                )
+            )
+            self._wm_revise_hint_shown = True
+        elif _WM_NUDGE_TURNS > 0 and self._turns_without_wm_update >= _WM_NUDGE_TURNS:
+            lines.append(
+                f"You have not updated any memory section for "
+                f"{self._turns_without_wm_update} analysis steps. Review what recent "
+                "action traces, stops, and observations taught you, and write updated "
+                "sections now - knowledge that is not written down is eventually lost "
+                "to context eviction."
+            )
+        lines.extend(self._prepare_auto_diff(current_frame, previous_step_summary))
         state_line = f"Current state: step {current_step}, level {current_level}"
         if observed_max_level > current_level:
             state_line += f" out of observed max level {observed_max_level} so far"
@@ -1220,21 +5085,65 @@ class ToolAgent:
             [
                 state_line,
                 f"Valid actions right now: {_format_valid_action_line(valid_actions)}.",
-                "Only tool: `python`. It receives `current_frame`, `previous_frame`, `history`, `transitions`, `last_transition`, `valid_actions`, `last_action_result`, and `action(actions)`.",
+            ]
+        )
+        # recording is independent of display: the guard needs the ledger even
+        # when the advisory lines are switched off
+        self._death_ledger_observe(history_entries, previous_step_summary)
+        if _get_env_bool("ARC3_DEATH_LEDGER", False):
+            # read at call time so this patch's hunks stay clear of the
+            # env-knob block that other patches extend; placed directly after
+            # the valid-actions line so the fatal continuations sit adjacent
+            # to the action choice they constrain
+            lines.extend(
+                self._death_ledger_lines(
+                    current_frame, history_entries, previous_step_summary
+                )
+            )
+        lines.extend(
+            [
+                "Only tool: `python`. It receives `current_frame`, `previous_frame`, `history`, `transitions`, `last_transition`, `valid_actions`, `last_action_call_result`, `frame_diff(before, after)`, and `action(actions)`.",
                 "Only letter-coded board views and lightweight metadata are exposed; raw numeric color IDs are not available.",
                 "Keep tool output compact: use `current_frame.segmentation` as the primary view, and `current_frame.ascii` only for a small specific region; never print full boards.",
                 "For the most recent change, compare `previous_frame` to `current_frame`, or `last_transition.before_frame` to `last_transition.after_frame`; `history[-1].frame` is the current frame, not the previous one.",
-                "Use Python to inspect the evidence, refine that world model from the newest history, and search or score candidate actions or short sequences against the current goal as you currently understand it.",
-                "Maintain a compact working world model of what the current level seems to contain, what actions appear to do, what the goal seems to be, what is still uncertain, and what plan currently looks best.",
-                "Below you are provided with the current world model from the previous turn. The default behavior is to copy it and add or remove things based on the evidence that you gathered. BEFORE EXECUTING NEW ACTIONS YOU MUST ALWAYS GIVE THE REVISED VERSION OF THE WORLD MODEL.",
             ]
         )
-        lines.append(
-            "You may call `action(actions)` more than once in one Python snippet if your search or control loop needs it, "
-            "but stop immediately if a result reports `game_over`, `run_complete`, `level_completed`, or `done`."
-        )
-        lines.extend(self._summarized_knowledge_lines())
-        lines.append("end of world model. ")
+        # Upstream states four things about the world model here, and with
+        # ARC3_MEMORY_SECTIONS=off three of them are false or unanswerable: the
+        # block they refer to is suppressed, so "Below you are provided with the
+        # current world model" is followed by nothing, and the model is told in
+        # capitals to always give a revised version with nowhere to put it.
+        # Observed as a drop in replies carrying any assistant text at all,
+        # 76% to 43% - a broken instruction, not an absent one, which is why
+        # `off` was never a clean control.
+        #
+        # The first sentence loses only its middle clause: "from the newest
+        # history" attaches to inspecting the evidence just as well, so the
+        # remainder still says use Python, inspect, and score against the goal.
+        if _memory_sections_disabled():
+            lines.append(
+                "Use Python to inspect the evidence from the newest history, and search or score candidate actions or short sequences against the current goal as you currently understand it."
+            )
+        else:
+            lines.extend(
+                [
+                    "Use Python to inspect the evidence, refine that world model from the newest history, and search or score candidate actions or short sequences against the current goal as you currently understand it.",
+                    "Maintain a compact working world model of what the current level seems to contain, what actions appear to do, what the goal seems to be, what is still uncertain, and what plan currently looks best.",
+                    "Below you are provided with the current world model from the previous turn. The default behavior is to copy it and add or remove things based on the evidence that you gathered. BEFORE EXECUTING NEW ACTIONS YOU MUST ALWAYS GIVE THE REVISED VERSION OF THE WORLD MODEL.",
+                ]
+            )
+        if not _dedupe_multicall_line():
+            # kept mid-opener only for the upstream wording; the surviving copy
+            # in the closing block is where a note about calling action() twice
+            # belongs, next to the instruction to call it at all
+            lines.append(
+                "You may call `action(actions)` more than once in one Python snippet if your search or control loop needs it, "
+                "but stop immediately if a result reports `game_over`, `run_complete`, `level_completed`, or `done`."
+            )
+        knowledge_lines = self._summarized_knowledge_lines()
+        if knowledge_lines:
+            lines.extend(knowledge_lines)
+            lines.append("end of world model. ")
         if action_num == 0:
             lines.append(
                 "Ground yourself in `current_frame` before acting, but start with a compact structural summary rather than restating the full frame."
@@ -1247,12 +5156,21 @@ class ToolAgent:
             [
                 "When ready, call `action(actions)` from inside the `python` tool with the best valid action or ordered batch selected by your code. If your code has found a reliable short sequence, prefer batching it in one call.",
                 "You may call `action(actions)` more than once in one Python snippet if your search or control loop needs it.",
-                "If you include assistant text before a tool call, keep it short and use it to update the world model. Helpful optional prefixes are `World model:`, `Goal model:`, `Action model:`, `Recent findings:`, `Open questions:`, `Plan:`, and `Cross-level notes:`.",
+                *self._memory_section_guidance(bool(knowledge_lines)),
                 TOOL_CALL_FORMAT_GUIDANCE,
             ]
         )
         if "MOUSE" in _normalize_valid_actions(valid_actions):
             lines.append("If you use MOUSE, include integer row and col arguments.")
+        if _opener_commit_hint():
+            # last line of the opener, where the resumption prompt sits relative
+            # to the reply it influences
+            lines.append(
+                "When your evidence supports a choice, call `action(actions)` with the "
+                "sequence it supports. Probe with a single action when you genuinely "
+                "need to distinguish two possibilities; otherwise batch the route you "
+                "have."
+            )
         return "\n".join(lines)
 
     def _tools(self, state_path: Path) -> list[dict[str, Any]]:
@@ -1279,35 +5197,105 @@ class ToolAgent:
             }
         ]
 
+    def _harness_template_kwargs(self) -> dict[str, Any]:
+        """Chat-template kwargs the HARNESS sets per request.
+
+        Shared by the request builder and the request log so the log records
+        what was actually sent rather than a reconstruction - which matters
+        for the effort ladder, where the value changes mid-run, and for
+        preserve_thinking, where the question is whether it reaches the server
+        at all."""
+        kwargs: dict[str, Any] = {}
+        preserve_thinking = _preserve_thinking_kwarg()
+        if preserve_thinking is not None:
+            kwargs["preserve_thinking"] = preserve_thinking
+        ladder = _reasoning_effort_ladder()
+        rung = getattr(self, "_reasoning_effort_rung", -1)
+        if ladder and 0 <= rung < len(ladder):
+            kwargs["reasoning_effort"] = ladder[rung]
+        return kwargs
+
     def _chat_completion(
         self,
         messages: list[dict[str, Any]],
         *,
         tools: list[dict[str, Any]] | None,
         request_timeout_seconds: float | None = None,
+        max_tokens: int | None = None,
+        enable_thinking: bool | None = None,
     ) -> _ChatCompletionResult:
         payload = build_chat_payload(
             provider=self._model.provider,
             model=self._model.model_id,
             messages=messages,
-            max_tokens=self._max_output_tokens,
+            max_tokens=self._max_output_tokens if max_tokens is None else max_tokens,
             temperature=_LOCAL_ANALYZER_TEMPERATURE,
             top_p=_LOCAL_ANALYZER_TOP_P,
             top_k=_LOCAL_ANALYZER_TOP_K,
-            thinking=bool(_LOCAL_ANALYZER_ENABLE_THINKING),
+            thinking=(
+                bool(_LOCAL_ANALYZER_ENABLE_THINKING)
+                if enable_thinking is None
+                else enable_thinking
+            ),
             tools=tools,
             tool_choice=_request_tool_choice(tools),
             seed=_LOCAL_ANALYZER_SEED,
         )
+        provider_prefs = _openrouter_provider_prefs(self._model.provider)
+        if provider_prefs:
+            payload["provider"] = provider_prefs
+        # merge rather than assign: the payload builder may already have set
+        # chat_template_kwargs (enable_thinking), and replacing the dict would
+        # drop it - as a request-level dict can also drop a server-side default
+        template_kwargs = dict(payload.get("chat_template_kwargs") or {})
+        template_kwargs.update(self._harness_template_kwargs())
+        if enable_thinking is not None:
+            # after the merge: _harness_template_kwargs does not set
+            # enable_thinking today, but an override that a later edit could
+            # silently undo is the bug that keeps recurring here
+            template_kwargs["enable_thinking"] = bool(enable_thinking)
+        if template_kwargs:
+            payload["chat_template_kwargs"] = template_kwargs
+        payload["messages"] = _strip_control_keys(
+            _apply_summary_visibility(
+                payload.get("messages") or [],
+                evicted=getattr(self, "_has_evicted", False),
+            )
+        )
         def post_chat(request_payload: dict[str, Any]) -> requests.Response:
             return requests.post(
                 f"{self._model.base_url.rstrip('/')}/chat/completions",
                 headers=self._headers(),
                 json=request_payload,
-                timeout=request_timeout_seconds if request_timeout_seconds is not None else self._timeout,
+                # (connect, read): one scalar gives the connect phase the whole
+                # read budget, so an unreachable host stalls a single attempt
+                # for the rest of the game
+                timeout=(
+                    _CONNECT_TIMEOUT_SECONDS,
+                    request_timeout_seconds if request_timeout_seconds is not None else self._timeout,
+                ),
+                stream=bool(request_payload.get("stream")),
             )
 
-        response = post_chat(payload)
+        # Per agent, not per process: every game meets the cold server on its
+        # own first request, so one shared flag would grant the grace to
+        # whichever game happened to go first and let the rest fail fast.
+        initial_grace = 0.0
+        if not self._http_initial_grace_used:
+            initial_grace = _env_float("ARC3_HTTP_RETRY_INITIAL_SECONDS", 0.0)
+        _diag(self, "request")
+        try:
+            response = _post_with_retries(
+                lambda: post_chat(payload),
+                retries=_env_int("ARC3_HTTP_RETRIES", 3),
+                base_seconds=_env_float("ARC3_HTTP_RETRY_BASE_SECONDS", 5.0),
+                max_seconds=_env_float("ARC3_HTTP_RETRY_MAX_SECONDS", 5.0),
+                initial_seconds=initial_grace,
+            )
+        finally:
+            # consumed whether the call succeeded or failed: the grace covers
+            # the server coming up, and by now it either did or will not
+            self._http_initial_grace_used = True
         try:
             response.raise_for_status()
         except requests.HTTPError as exc:
@@ -1322,22 +5310,63 @@ class ToolAgent:
             if detail:
                 message += f" | response: {detail}"
             raise requests.RequestException(message)
-        payload = response.json()
+        if payload.get("stream"):
+            # decode_unicode=True decodes each transport chunk INDEPENDENTLY, so a
+            # multi-byte UTF-8 sequence straddling a chunk boundary is split into
+            # two broken halves - an em-dash (E2 80 94) becomes "\u00e2" plus
+            # fragments. Most characters survive, which is why short probes look
+            # clean and only long streamed replies are corrupted. The assembler
+            # already decodes bytes itself, and a complete SSE line is always a
+            # whole UTF-8 sequence, so hand it raw bytes instead.
+            payload = assemble_streamed_chat_response(
+                response.iter_lines(decode_unicode=False)
+            )
+        else:
+            payload = response.json()
         choices = payload.get("choices", [])
         if not choices:
-            raise requests.RequestException("server returned no choices")
+            detail = payload.get("error") or {
+                k: v for k, v in payload.items() if k not in ("choices",)
+            }
+            raise requests.RequestException(
+                f"server returned no choices | body: {str(detail)[:600]}"
+            )
         choice = choices[0]
         return _ChatCompletionResult(
-            message=choice.get("message", {}),
+            message=_extract_leaked_tool_calls(choice.get("message", {})),
             finish_reason=str(choice.get("finish_reason", "") or ""),
             usage=payload.get("usage"),
+            served_by=str(payload.get("provider", "") or ""),
         )
 
     def _trim_tool_text(self, text: str) -> tuple[str, bool]:
-        if len(text) <= self._tool_output_chars:
+        """Keep both ends and drop the middle.
+
+        Truncating the tail loses the half that matters. An `error` field ends
+        with `SomeError: <the explanation>` after the traceback, so a deep
+        traceback pushed the explanation out entirely. And a snippet's `stdout`
+        typically prints the state, acts, then prints the state again - so the
+        tail holds the result and the head holds the setup.
+
+        Weighted toward the HEAD: a snippet's output is mostly sequential
+        reasoning that reads from the top, so two thirds there. The remaining
+        third is ample for what has to survive at the end - the longest refusal
+        message measures ~550 characters and a user-frame traceback ~250, well
+        inside a 1365-character tail at the default budget.
+        """
+        limit = self._tool_output_chars
+        if len(text) <= limit:
             return text, False
-        omitted = len(text) - self._tool_output_chars
-        return f"{text[:self._tool_output_chars]}\n... [truncated {omitted} chars]", True
+        omitted = len(text) - limit
+        if not _get_env_bool("ARC3_MIDDLE_TRUNCATION", False):
+            return f"{text[:limit]}\n... [truncated {omitted} chars]", True
+        tail = limit // 3
+        head = limit - tail
+        return (
+            f"{text[:head]}\n... [truncated {omitted} chars from the middle] ...\n"
+            f"{text[-tail:]}",
+            True,
+        )
 
     def _summarize_planned_actions(self, value: Any) -> Any:
         if isinstance(value, dict):
@@ -1366,6 +5395,9 @@ class ToolAgent:
         for field in truncate_fields:
             value = result.get(field)
             if isinstance(value, str):
+                value, suppressed_rows = _suppress_board_dumps(value)
+                if suppressed_rows:
+                    result["board_print_suppressed_rows"] = suppressed_rows
                 result[field], field_truncated = self._trim_tool_text(value)
                 truncated = truncated or field_truncated
         if truncated:
@@ -1413,13 +5445,230 @@ class ToolAgent:
             raise TypeError(f"Action {index} must be a string or a dict.")
         return normalized
 
+    def _fatal_prefix_match(
+        self, state_hash: str, remaining_sigs: list[str]
+    ) -> tuple[int, str] | None:
+        """Does a recorded fatal sequence continue from here?
+
+        Returns ``(offset, action)`` when executing ``remaining_sigs`` from
+        this board would replay a recorded attempt into its death - offset 0
+        meaning the very next action is the fatal one. Returns None otherwise.
+
+        Generalizes the single-action check: an attempt recorded as actions
+        ``A[0..n-1]`` with pre-action hashes ``H[0..n-1]`` is fatal from here
+        if some ``H[k]`` equals this board AND ``A[k:]`` is a prefix of what
+        the model is about to run. The environment is deterministic and a
+        death resets to level start, so replaying ``A[k:]`` from ``H[k]``
+        reproduces ``H[k+1] ...`` and ends the attempt.
+
+        Catching it at the FIRST position of such a batch is what makes the
+        block worth having: the per-position check alone would let the harmless
+        prefix run and refuse only the final move, spending those actions to
+        arrive somewhere the model has already been. A model that genuinely
+        wants the prefix can submit it without the fatal tail, which no longer
+        matches and is not blocked.
+        """
+        index = getattr(self, "_death_ledger_index", None)
+        attempts = getattr(self, "_death_ledger_attempts", None)
+        if not index or not attempts or not remaining_sigs:
+            return None
+        for attempt_idx, position in index.get(state_hash, ()):  # type: ignore[union-attr]
+            try:
+                actions = attempts[attempt_idx]["actions"]
+            except (IndexError, KeyError, TypeError):
+                continue
+            tail = [action_signature(a) for a in actions[position:]]
+            if not tail or len(tail) > len(remaining_sigs):
+                continue
+            if remaining_sigs[: len(tail)] == tail:
+                return len(tail) - 1, tail[-1]
+        return None
+
+    def _action_guard_hook(
+        self,
+        grid: Any,
+        action_name: str,
+        action_data: dict[str, Any],
+        level: Any = None,
+        remaining: list[tuple[str, dict[str, Any]]] | None = None,
+    ) -> tuple[str, str] | None:
+        """Called by the solver before each action of a batch. Returns
+        ``(stop_reason, detail)`` to refuse, or None to proceed."""
+        guard = self._noop_repeat_guard
+        if guard is None or not grid:
+            return None
+        if not _guards_active(level):
+            # level 1 is where the model probes hardest and where an action is
+            # worth least, so the repeat guards can be told to start later. The
+            # guard still WATCHES - note_level and the state hash below run
+            # either way - so that by the time it starts refusing it already
+            # knows what has been tried.
+            guard.note_level(level)
+            self._noop_guard_pre_hash = _interior_state_hash(
+                [list(row) for row in grid]
+            )
+            return None
+        rows = [list(row) for row in grid]
+        state_hash = _interior_state_hash(rows)
+        guard.note_level(level)
+        # the solver passes ENGINE names (ACTION2); every other consumer - the
+        # death ledger, the transcript, the model itself - speaks model names
+        # (DOWN), so translate here or the guard both mismatches the ledger and
+        # reports "ACTION2" to a model that has never seen that label
+        action_sig = action_signature(
+            {"action": to_model_action(action_name), **(action_data or {})}
+        )
+        self._noop_guard_pre_hash = state_hash
+        if self._noop_guard_deaths_enabled and not self._guard_death_override_batch:
+            # with lookahead off, only THIS action is considered, so a match
+            # can be found solely at offset 0 - the per-position behaviour
+            window = (
+                (remaining or [(action_name, action_data)])
+                if _fatal_lookahead_enabled()
+                else [(action_name, action_data)]
+            )
+            remaining_sigs = [
+                action_signature({"action": to_model_action(name), **(data or {})})
+                for name, data in window
+            ]
+            match = self._fatal_prefix_match(state_hash, remaining_sigs)
+            if guard.should_block_death(
+                state_hash, action_sig, lambda _h, _a: match is not None
+            ):
+                offset, fatal_action = match  # type: ignore[misc]
+                if offset == 0:
+                    detail = (
+                        f"{fatal_action} ended a previous attempt on this level from this "
+                        "exact board state, so it was NOT executed."
+                    )
+                else:
+                    detail = (
+                        "this sequence replays a path that ended a previous attempt: "
+                        f"{fatal_action} is fatal {offset} action(s) from here, so NOTHING "
+                        "was executed and no actions were spent. Submit a different "
+                        "continuation, or the same prefix without the fatal tail."
+                    )
+                return (
+                    "known_death",
+                    detail
+                    + " This refusal is not catchable and ended the snippet, so anything "
+                    "after the call did not run. Issuing the same action again as the "
+                    "very next action will override the block and execute the action. "
+                    "That is only worth doing if you have a specific reason to believe "
+                    "it will not end the attempt this time. Otherwise, plan a different "
+                    "continuation from this state.",
+                )
+            # the override fired: let the rest of THIS call run without further
+            # death checks, or the model would have to insist again at every
+            # position of a sequence it has already deliberately confirmed
+            if match is not None:
+                self._guard_death_override_batch = True
+        if (
+            self._noop_guard_noops_enabled
+            and not (reset_exposed() and action_sig == "RESET")
+            and guard.should_block(state_hash, action_sig)
+        ):
+            return (
+                "known_noop",
+                (f"{action_sig} already produced no change inside the board area in this "
+                   "exact board state "
+                   if _new_noop_wording() else
+                   f"{action_sig} already changed nothing in this exact board state ")
+                +
+                "earlier on this level, so it was NOT executed and no action was "
+                "spent. This refusal is not catchable and ended the snippet, so "
+                "anything after the call did not run - a retry loop cannot probe "
+                "past it. It applies to that one action from this one board state; "
+                "it is not a constraint on sequences or batch size. Issuing the same "
+                "action again as the very next action will override the block and "
+                "execute the action. That is only worth doing if you have a specific "
+                "reason to believe it will behave differently now. Otherwise, plan a "
+                "different continuation from this state.",
+            )
+        if (
+            _repeat_state_guard_enabled()
+            and (state_hash, action_sig) in self._repeat_state_seen
+        ):
+            # after known_noop deliberately: a repeated no-op is better reported
+            # as a no-op, which also keeps its override intact. Reaching here
+            # means the action DID change the board and still returned to a
+            # state this snippet has already acted from.
+            return (
+                "repeated_action_in_state",
+                f"{action_sig} was already run from this exact board state earlier "
+                "in this snippet, so what it does here is known and it was NOT "
+                # not "no action budget was spent": by construction this fires
+                # after the snippet has already spent budget on the actions that
+                # got it here, and the model reads that phrase as covering the
+                # whole snippet
+                "executed. A sequence that returns to "
+                # examples, not a diagnosis: the harness knows the board
+                # repeated, not why, and stating the cause would have the model
+                # looking for a wrapping cursor it may not have
+                "a board it has already acted from is not advancing (e.g. a cursor "
+                "that wraps around, or a loop stepping the wrong way past its "
+                "target). "
+                "Nothing after the call ran. Re-read `current_frame`, work out why "
+                "the sequence is not converging, and write a different snippet - "
+                "this block is per snippet, so the next one starts with no history.",
+            )
+        guard.note_executed(state_hash, action_sig)
+        self._guard_hook_trace.append((state_hash, action_sig))
+        return None
+
+    def _record_guard_observations(
+        self,
+        guard: NoopRepeatGuard,
+        raw_payload: dict[str, Any],
+        compact_payload: dict[str, Any],
+    ) -> None:
+        """Record every executed position of this call, not just the first.
+
+        The guard hook fires once per position immediately before that action
+        runs, so `_guard_hook_trace` holds the board hash each action actually
+        executed against - including the intermediate states of a batch, which
+        appear nowhere in the returned payload. The solver reports outcomes as
+        `gameplay_changed_per_action`, index-aligned with `executed_actions`
+        and therefore with the trace; a single-action call has no such list, so
+        the aggregate verdict is used instead.
+
+        Zipping to the shorter of the two is deliberate: a batch stopped early
+        (blocked, fatal, invalid) leaves one more trace entry than there are
+        outcomes, and an unmatched entry must not be recorded against a verdict
+        that belongs to a different action.
+        """
+        trace = list(self._guard_hook_trace)
+        self._guard_hook_trace = []
+        if not trace:
+            return
+        verdicts = raw_payload.get("gameplay_changed_per_action")
+        if not isinstance(verdicts, list) or not verdicts:
+            if len(trace) != 1 or not compact_payload.get("executed"):
+                return
+            verdicts = [compact_payload.get("gameplay_changed")]
+        for (state_hash, action_sig), changed in zip(trace, verdicts):
+            if reset_exposed() and action_sig == "RESET":
+                # Retain its position in the trace so following verdicts stay
+                # aligned, but do not learn that a bar-only reset is ineffective.
+                continue
+            # unconditional: known_noop reaches a repeated no-op first, so
+            # filtering on `changed` here would buy nothing
+            self._repeat_state_seen.add((state_hash, action_sig))
+            guard.observe(state_hash, action_sig, gameplay_changed=changed)
+
     def _compact_action_result(self, payload: dict[str, Any]) -> dict[str, Any]:
         compact = {
             "executed": bool(payload.get("executed")),
             "action_num": payload.get("action_num"),
             "level": payload.get("level"),
             "score": payload.get("score"),
-            "reward": payload.get("reward"),
+            # `reward` is not carried: it is (levels_completed - previous) /
+            # number_of_levels, so it reads 0.0 on every action that does not
+            # finish a level. A model seeing 0.0 on almost every action can
+            # reasonably take it for a verdict on the action, which it is not,
+            # and `level_completed` in this same dict already says the thing it
+            # was trying to say. The solver still computes it for the viewer
+            # and the raw payload.
             "state": payload.get("state"),
             "valid_actions": payload.get("valid_actions", []),
             "board_changed": bool(payload.get("board_changed")),
@@ -1429,6 +5678,8 @@ class ToolAgent:
             "run_complete": bool(payload.get("run_complete")),
             "action_display": payload.get("action_display") or payload.get("action_name"),
         }
+        if "gameplay_changed" in payload:
+            compact["gameplay_changed"] = bool(payload.get("gameplay_changed"))
         executed_actions = payload.get("executed_actions")
         if isinstance(executed_actions, list) and executed_actions:
             compact["executed_actions"] = [str(action).strip() for action in executed_actions if str(action).strip()]
@@ -1439,6 +5690,11 @@ class ToolAgent:
             compact["requested_count"] = payload.get("requested_count", batch_size)
             compact["executed_count"] = payload.get("executed_count", batch_size)
             compact["stopped_early"] = bool(payload.get("stopped_early"))
+        # blocked_action names the refused action for the next turn's opener;
+        # without it the model was told "the sequence was stopped before None"
+        for passthrough_key in ("no_op_action_index", "no_op_action", "skipped_actions", "gameplay_changed_per_action", "action_trace", "no_op", "fatal_action", "blocked_action", "action_echo", "animation"):
+            if passthrough_key in payload:
+                compact[passthrough_key] = payload.get(passthrough_key)
         if payload.get("stop_reason"):
             compact["stop_reason"] = payload.get("stop_reason")
         if payload.get("stop_detail"):
@@ -1446,6 +5702,14 @@ class ToolAgent:
         for timing_key in ("run_elapsed_seconds", "time_remaining_seconds"):
             if timing_key in payload:
                 compact[timing_key] = payload.get(timing_key)
+        # the session's own clock, which is also what decides when the game
+        # stops - so the endgame phase cannot drift away from the deadline it
+        # is meant to anticipate. It starts at session construction, so a game
+        # waiting on the gate is already ageing, which is the same convention
+        # max_runtime_s_per_game uses.
+        elapsed = payload.get("run_elapsed_seconds")
+        if isinstance(elapsed, (int, float)):
+            self._game_elapsed_seconds = float(elapsed)
         if payload.get("error"):
             compact["error"] = payload.get("error")
         return compact
@@ -1466,7 +5730,7 @@ class ToolAgent:
         def _serialized_runtime_state(
             *,
             next_valid_actions: list[str] | None = None,
-            last_action_result: dict[str, Any] | None = None,
+            last_action_call_result: dict[str, Any] | None = None,
         ) -> dict[str, Any]:
             refreshed_frame, refreshed_history = load_runtime_state(state_path)
             current_frame_payload = _ascii_frame_view_payload(refreshed_frame)
@@ -1475,15 +5739,20 @@ class ToolAgent:
             else:
                 sanitized_actions = list(valid_actions)
             persisted_action_result = (
-                last_action_result
-                if isinstance(last_action_result, dict)
-                else self._last_action_result
+                last_action_call_result
+                if isinstance(last_action_call_result, dict)
+                else self._last_action_call_result
             )
             return {
                 "current_frame": current_frame_payload,
                 "history": _ascii_history_view_payload(refreshed_history),
                 "valid_actions": sanitized_actions,
-                "last_action_result": (
+                "death_ledger": (
+                    self._death_ledger_payload()
+                    if _get_env_bool("ARC3_DEATH_LEDGER", False)
+                    else None
+                ),
+                "last_action_call_result": (
                     dict(persisted_action_result)
                     if isinstance(persisted_action_result, dict)
                     else {}
@@ -1491,9 +5760,24 @@ class ToolAgent:
             }
 
         terminal_action_result: dict[str, Any] | None = None
+        snippet_executed_actions = 0
 
-        def _handle_action(actions: list[dict[str, Any]]) -> dict[str, Any]:
-            nonlocal terminal_action_result
+        def _handle_animation(request: dict[str, Any]) -> dict[str, Any]:
+            if self._step_env_callback is None:
+                return {"error": "Animation frames are not available in this session."}
+            raw = self._step_env_callback({"query": "animation"})
+            record = raw.get("record") if isinstance(raw, dict) else None
+            record = record if isinstance(record, dict) else None
+            if str(request.get("kind") or "") == "frames":
+                return build_animation_frames(record)
+            if not _animation_timeline_enabled():
+                return {}
+            return build_animation_view(record)
+
+        def _handle_action(
+            actions: list[dict[str, Any]], *, stale_after: str | None = None
+        ) -> dict[str, Any]:
+            nonlocal terminal_action_result, snippet_executed_actions
             if self._step_env_callback is None:
                 raise RuntimeError("action(actions) is not available in this session.")
             normalized_actions = self._normalize_python_actions(actions)
@@ -1504,7 +5788,6 @@ class ToolAgent:
                     "action_num": terminal_action_result.get("action_num"),
                     "level": terminal_action_result.get("level"),
                     "score": terminal_action_result.get("score"),
-                    "reward": 0.0,
                     "state": terminal_action_result.get("state"),
                     "valid_actions": [],
                     "board_changed": False,
@@ -1518,37 +5801,107 @@ class ToolAgent:
                     "stop_reason": f"previous_{reason}",
                     "stop_detail": _terminal_action_stop_detail(reason),
                 }
-                self._last_action_result = dict(compact_payload)
+                self._last_action_call_result = dict(compact_payload)
                 return {
                     "action_result": compact_payload,
                     "state": _serialized_runtime_state(
                         next_valid_actions=[],
-                        last_action_result=compact_payload,
+                        last_action_call_result=compact_payload,
                     ),
                 }
-            raw_payload = self._step_env_callback({"actions": normalized_actions})
+            if reset_exposed():
+                reset_positions = [
+                    index for index, item in enumerate(normalized_actions)
+                    if to_engine_action(item.get("action")) == "RESET"
+                ]
+                if reset_positions and (snippet_executed_actions or reset_positions != [0]):
+                    detail = (
+                        "RESET must be the first executed game action in a Python snippet. "
+                        "This call executed no actions. Start a new snippet with RESET "
+                        "as its first action; inspection or computation may come before it."
+                    )
+                    refusal = {
+                        "executed": False,
+                        "stop_reason": "reset_not_first",
+                        "stop_detail": detail,
+                        "error": detail,
+                        "valid_actions": list(self._current_valid_actions),
+                    }
+                    self._last_action_call_result = dict(refusal)
+                    return {
+                        "action_result": refusal,
+                        "state": _serialized_runtime_state(last_action_call_result=refusal),
+                    }
+            guard = self._noop_repeat_guard
+            self._guard_hook_trace = []
+            self._guard_death_override_batch = False
+            # No pre-check here: the solver's per-position hook covers every
+            # action of every call, including the first. Checking in both
+            # places consumed the override slot twice - the pre-check armed it,
+            # the retry cleared it, and the hook then blocked the very action
+            # the model had been invited to repeat.
+            step_arguments: dict[str, Any] = {"actions": normalized_actions}
+            if stale_after:
+                # forwarded as an argument so the solver can weigh it AFTER its
+                # own per-position guards, which name a specific action and
+                # carry an override
+                step_arguments["stale_after"] = stale_after
+            raw_payload = self._step_env_callback(step_arguments)
             if not isinstance(raw_payload, dict):
                 raise RuntimeError("action(actions) did not return a JSON-like payload.")
+            if reset_exposed() and raw_payload.get("executed"):
+                # Use actual execution, not submitted batch size: a refused
+                # call does not consume the snippet's first-action allowance.
+                snippet_executed_actions += max(1, int(raw_payload.get("executed_count") or 1))
+            if _animation_enabled():
+                # Held on the agent, not routed through the sandbox: the chain
+                # is dropped by compaction so the raw grids never become text,
+                # which also meant it never reached the summary. It is only
+                # ever used to render an image here, so it has no business
+                # crossing the pipe twice.
+                self._last_animation_chain = raw_payload.get("animation_chain") or []
+                # Whether the LAST executed action moved anything, which is not
+                # the same as whether the sequence did: a batch of UP, DOWN,
+                # LEFT, SPACE reports gameplay_changed for the batch, so an
+                # inert SPACE at the end was hidden behind three moves that
+                # were not. The image exists for the action that animated, so
+                # it is that action's flag that decides.
+                per_action = raw_payload.get("gameplay_changed_per_action")
+                if isinstance(per_action, list) and per_action:
+                    self._last_action_gameplay_changed = bool(per_action[-1])
+                else:
+                    self._last_action_gameplay_changed = raw_payload.get(
+                        "gameplay_changed"
+                    )
             compact_payload = self._compact_action_result(raw_payload)
+            if guard is not None and normalized_actions:
+                self._record_guard_observations(guard, raw_payload, compact_payload)
             next_valid_actions = raw_payload.get("valid_actions")
             if isinstance(next_valid_actions, list):
                 self._current_valid_actions = _normalize_valid_actions(next_valid_actions)
             if compact_payload.get("executed") and _terminal_action_reason(compact_payload):
                 terminal_action_result = compact_payload
-            self._last_action_result = dict(compact_payload)
+            self._last_action_call_result = dict(compact_payload)
             return {
                 "action_result": compact_payload,
                 "state": _serialized_runtime_state(
                     next_valid_actions=next_valid_actions if isinstance(next_valid_actions, list) else None,
-                    last_action_result=compact_payload,
+                    last_action_call_result=compact_payload,
                 ),
             }
 
+        # per snippet: a repeat across snippets is the model choosing, not a
+        # loop failing to converge
+        self._repeat_state_seen.clear()
         sandbox_result = run_sandboxed_python(
             code=code,
             timeout_seconds=self._python_timeout,
             initial_state=_serialized_runtime_state(),
             action_handler=_handle_action,
+            animation_handler=_handle_animation if _animation_enabled() else None,
+            kept_functions=self._retained_sources(),
+            retain_imports=_get_env_bool("ARC3_PERSISTENT_FUNCTIONS_IMPORTS", False),
+            repair_hints=_get_env_bool("ARC3_PERSISTENT_FUNCTIONS_REPAIR_HINTS", False),
         )
 
         action_results = [
@@ -1557,6 +5910,7 @@ class ToolAgent:
             if isinstance(item, dict)
         ]
         payload: dict[str, Any] = {"tool": "python"}
+        self._record_retained_functions(sandbox_result, payload)
         rendered_stdout = str(sandbox_result.get("stdout", "") or "")
         rendered_error = str(sandbox_result.get("error", "") or "")
         if rendered_error:
@@ -1575,16 +5929,28 @@ class ToolAgent:
                 else:
                     payload["result"] = {
                         "action_calls": len(action_results),
-                        "last_action_result": action_results[-1],
+                        "last_action_call_result": action_results[-1],
                     }
 
         step_executed = any(bool(item.get("executed")) for item in action_results)
+        made_progress = any(
+            bool(item.get("gameplay_changed"))
+            or bool(item.get("level_completed"))
+            or bool(item.get("game_over"))
+            or bool(item.get("run_complete"))
+            for item in action_results
+        )
         if step_executed:
             self._last_step_summary = self._summarize_step_sequence(action_results)
             self._update_summarized_knowledge_from_step_summary()
+        elif self._last_step_summary and not self._last_step_summary.get("stale"):
+            # carried over across an exchange that executed nothing: mark it so
+            # the next prompt reminds instead of re-narrating it as fresh
+            self._last_step_summary = {**self._last_step_summary, "stale": True}
         return _ToolDispatchResult(
             self._render_tool_payload(payload, truncate_fields=("stdout", "error", "result")),
             step_executed=step_executed,
+            made_progress=made_progress,
         )
 
     def _dispatch_tool(self, state_path: Path, name: str, arguments: dict[str, Any]) -> _ToolDispatchResult:
@@ -1593,22 +5959,284 @@ class ToolAgent:
             return self._run_python_tool(state_path, arguments)
         return _ToolDispatchResult(json.dumps({"error": f"Unknown tool: {name}"}, indent=2))
 
+    def _calibrate_from_usage(
+        self,
+        messages: list[dict[str, Any]],
+        tools: list[dict[str, Any]] | None,
+        usage: dict[str, Any] | None,
+    ) -> str:
+        """Update the characters-per-token divisor from a served request.
+
+        The images are subtracted using the SAME formula that will re-add them,
+        so the two cancel and the divisor measures text alone. Clamped because
+        an under-count risks context overflow, which is the dangerous
+        direction; a bad measurement should cost accuracy, not a rejected
+        request."""
+        if not _calibrate_text_tokens() or not isinstance(usage, dict):
+            return ""
+        try:
+            prompt_tokens = int(usage.get("prompt_tokens") or 0)
+        except (TypeError, ValueError):
+            return ""
+        if prompt_tokens <= 0:
+            return ""
+        scrubbed, image_tokens = _split_messages_for_estimate(list(messages))
+        payload: dict[str, Any] = {"messages": scrubbed}
+        if tools:
+            payload["tools"] = tools
+            payload["tool_choice"] = _request_tool_choice(tools)
+        try:
+            rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
+        except TypeError:
+            return ""
+        text_tokens = prompt_tokens - image_tokens
+        if text_tokens <= 0 or not rendered:
+            return ""
+        measured = len(rendered) / text_tokens
+        clamped = min(_text_token_chars_max(), max(_text_token_chars_min(), measured))
+        previous = self._text_chars_per_token
+        self._text_chars_per_token = clamped
+        if previous is None or abs(clamped - previous) >= 0.05:
+            note = "" if clamped == measured else f" (clamped from {measured:.2f})"
+            # returned rather than written here: append_transcript is a closure
+            # inside analyze(), not reachable from a method
+            return (
+                f"text_token_calibration: {clamped:.2f} chars/token{note} "
+                f"[prompt_tokens {prompt_tokens}, images {image_tokens}, "
+                f"text chars {len(rendered)}]"
+            )
+        return ""
+
+    def _append_context_message(
+        self, messages: list[dict[str, Any]], message: dict[str, Any]
+    ) -> None:
+        """Append and count new context once, before any history can be evicted.
+
+        Use the context estimator for all roles, including reasoning and image
+        tokens. Subtract the empty request envelope; static system/tool schema
+        tokens and re-sent history are not new context. The summary exchange
+        itself bypasses this helper and starts a fresh interval.
+        """
+        messages.append(message)
+        if _summary_interval_tokens() <= 0:
+            return
+        added = max(
+            0,
+            self._estimate_request_input_tokens([message])
+            - self._estimate_request_input_tokens([]),
+        )
+        self._summary_context_tokens_since_attempt = (
+            getattr(self, "_summary_context_tokens_since_attempt", 0) + added
+        )
+
+    def _maybe_append_rolling_summary(
+        self, append_transcript, tools: list[dict[str, Any]] | None,
+        *, request_timeout_seconds: float | None = None,
+    ) -> None:
+        """Attempt a summary after another interval of newly added context.
+
+        Called at turn start, which is the only clean boundary: nothing is in
+        flight, and a yield resumption arrives here too so no separate handling
+        is needed. Crossing the threshold is not urgent - the summary is not
+        needed until eviction reaches it - so overshooting by a turn costs
+        nothing and interrupting an investigation would.
+
+        The request carries the current history unchanged, so it is a
+        continuation rather than a prefix: both the KV prefix and the recurrent
+        state stay warm. Both messages are tagged, and the trimmer's
+        summary-aware drain navigates by that tag.
+        """
+        interval = _summary_interval_tokens()
+        if interval <= 0:
+            return
+        history = self._history_messages
+        if not history:
+            return
+
+        since = getattr(self, "_summary_context_tokens_since_attempt", 0)
+        if since < interval:
+            return
+        # before the instruction: context first, then the ask, so the ask is
+        # still the last thing read. Stored exactly as sent - a request message
+        # that differs from the one in history would diverge from the cached
+        # prefix at that point and cost a prefill on every later turn.
+        context_lines = self._summary_turn_context_lines() if _summary_turn_context() else []
+        request_text = (
+            "\n".join([*context_lines, "", SUMMARY_REQUEST_PROMPT])
+            if context_lines
+            else SUMMARY_REQUEST_PROMPT
+        )
+        request = _mark_control_message(
+            {"role": "user", "content": request_text},
+            _SUMMARY_CONTROL_KIND,
+        )
+        messages = [
+            {"role": "system", "content": self._system_prompt},
+            *history,
+            request,
+        ]
+        # Every attempt starts a new interval, including HTTP errors, empty
+        # replies and rejected tool calls. Reset before dispatch so all exit
+        # paths share that behavior; the summary exchange is excluded from this clock.
+        self._summary_context_tokens_since_attempt = 0
+        try:
+            result = self._chat_completion(
+                messages,
+                tools=tools,
+                max_tokens=_summary_max_gen_tokens(self._max_output_tokens),
+                enable_thinking=_summary_enable_thinking(),
+                request_timeout_seconds=request_timeout_seconds,
+            )
+        except BaseException as exc:  # a failed summary must not end the turn
+            log.warning("rolling summary request failed: %s", exc)
+            return
+        # before any rejection path: the tokens were generated whatever the
+        # reply turned out to be, and a summary that is thrown away still cost
+        # the time
+        self._accumulate_usage_tokens(
+            getattr(result, "usage", None), count_toward_turn=False
+        )
+        dropped_call = ""
+        message = result.message or {}
+        text = _normalize_message_content(message.get("content", "")).strip()
+        if message.get("tool_calls"):
+            # A reply that calls a tool may still have written a real summary
+            # first - the instruction not to act is advisory, and the model
+            # sometimes summarises and then reaches for the tool anyway. Keeping
+            # a usable one is better than asking again for something it has
+            # already produced, so the text is judged on its length: real
+            # summaries here run 4,000 to 6,000 characters, while a preamble
+            # introducing a tool call runs to tens.
+            #
+            # The tool call itself is dropped either way. It was never going to
+            # execute: this request is outside the turn loop, with no sandbox
+            # and no action budget attached.
+            names = ", ".join(
+                str((c.get("function") or {}).get("name", "?"))
+                for c in message.get("tool_calls") or []
+                if isinstance(c, dict)
+            )
+            if len(text) < _SUMMARY_MIN_USABLE_CHARS:
+                log.warning(
+                    "rolling summary tried to call %s with only %d chars of text; "
+                    "skipping",
+                    names or "a tool", len(text),
+                )
+                if text:
+                    append_transcript(
+                        "ROLLING SUMMARY REJECTED",
+                        f"[tried to call {names}, text too short to use]\n{text}",
+                    )
+                return
+            log.warning(
+                "rolling summary tried to call %s but wrote %d chars first; "
+                "keeping the text, dropping the call",
+                names or "a tool", len(text),
+            )
+            dropped_call = names or "a tool"
+        if not text:
+            # A reply that deliberated to the output limit without reaching the
+            # content wants a different fix from one that simply said nothing -
+            # a larger max_output or a shorter interval - so the two are logged
+            # apart rather than both as "empty".
+            reasoning_chars = len(_extract_reasoning_text(message))
+            if reasoning_chars:
+                log.warning(
+                    "rolling summary produced %d chars of reasoning and no summary "
+                    "(finish_reason=%s); skipping",
+                    reasoning_chars,
+                    str(getattr(result, "finish_reason", "") or "?"),
+                )
+            else:
+                log.warning("rolling summary came back empty; skipping")
+            return
+        truncated = str(getattr(result, "finish_reason", "")).strip() == "length"
+        if truncated:
+            # Kept rather than rejected: a summary cut at the output limit still
+            # covers the early material, and the alternative is no summary at
+            # all, which means the same span is evicted with nothing standing in
+            # for it. But the tail is where "what I was in the middle of" lives,
+            # so the model is told the text is incomplete rather than being left
+            # to infer it from a sentence that stops.
+            log.warning(
+                "rolling summary hit the output limit at %d chars of summary, after "
+                "%d chars of reasoning; keeping it, marked as incomplete",
+                len(text),
+                len(_extract_reasoning_text(message)),
+            )
+            text = f"{text}\n[This summary was cut off at the output limit and is incomplete.]"
+        reply = _mark_control_message(
+            {"role": "assistant", "content": text}, _SUMMARY_CONTROL_KIND
+        )
+        if _summary_replaces_history():
+            # only now, with a usable summary in hand - a failed or rejected one
+            # returned above and left history untouched
+            replaced_count = len(self._history_messages)
+            self._history_messages = [request, reply]
+            # this discards history without going through the trimmer, so the
+            # prefix is invalid and the flag has to be set by hand - otherwise
+            # the next request keeps a priority that assumes a warm cache
+            self._note_history_evicted()
+        else:
+            self._history_messages.append(request)
+            self._history_messages.append(reply)
+            replaced_count = 0
+        reasoning_chars = len(_extract_reasoning_text(message))
+        append_transcript(
+            "ANALYZER STATUS",
+            f"rolling_summary: {len(text)} chars of summary after "
+            f"{reasoning_chars} chars of reasoning, {since} new context tokens since the previous attempt"
+            + (" [TRUNCATED]" if truncated else "")
+            + (f" [tool call dropped: {dropped_call}]" if dropped_call else "")
+            + (f" [replaced {replaced_count} messages of history]"
+               if replaced_count else ""),
+        )
+        # the text itself, so a run can be read back and judged rather than only
+        # counted - this is the artefact the whole mechanism produces. Salvaged
+        # ones get their own section name so they can be grepped and read apart
+        # from the clean ones, which is the comparison that says whether a reply
+        # that also reached for a tool wrote a worse summary.
+        append_transcript(
+            "ROLLING SUMMARY SALVAGED" if dropped_call else "ROLLING SUMMARY",
+            text,
+        )
+
     def _estimate_request_input_tokens(
         self,
         messages: list[dict[str, Any]],
         *,
         tools: list[dict[str, Any]] | None = None,
     ) -> int:
+        messages = _apply_summary_visibility(
+            messages, evicted=getattr(self, "_has_evicted", False)
+        )
+        image_tokens = 0
+        if _image_token_estimate_enabled():
+            messages, image_tokens = _split_messages_for_estimate(messages)
         payload: dict[str, Any] = {"messages": messages}
         if tools:
             payload["tools"] = tools
             payload["tool_choice"] = _request_tool_choice(tools)
-        return _estimate_tokens(payload)
+        return _estimate_tokens(payload, self._text_chars_per_token) + image_tokens
 
     def _drop_oldest_history_block(self, history: list[dict[str, Any]], *, preserve_recent: int) -> bool:
         removable = len(history) - preserve_recent
         if removable <= 0:
             return False
+        # Never drop the last user message. Dropping one and draining to the
+        # next is fine while another exists; when none does, the drain runs to
+        # preserve_recent and _drop_until_first_user_message then empties the
+        # history outright - the request becomes the system prompt alone and the
+        # server answers "No user query found in messages".
+        #
+        # Reachable mid-turn, where the list ends with a tool result rather than
+        # the opener, so preserve_recent protects the tool message and the
+        # opener is the first thing popped. Seen in a live request log as a
+        # 1-message request at request_index_within_turn=2. Pre-existing and
+        # independent of the summary drain, which only makes it easier to reach
+        # by evicting further in one step.
+        if not any(str(m.get("role", "")).strip() == "user" for m in history[1:]):
+            return False
         first = history.pop(0)
         first_role = str(first.get("role", "")).strip()
         if first_role in {"assistant", "tool"}:
@@ -1617,6 +6245,13 @@ class ToolAgent:
             return True
         while history and history[0].get("role") == "tool" and len(history) > preserve_recent:
             history.pop(0)
+        # Only navigate by summaries when there is one left to land on, and only
+        # one that is a REQUEST (a user message) and sits within the removable
+        # range. Checking merely that a summary exists somewhere let the loop
+        # skip every remaining user message when the last one was already at the
+        # head, stripping history down to preserve_recent with no user message
+        # in it at all - which the chat template rejects outright:
+        # "No user query found in messages", HTTP 400, turn lost.
         while history and history[0].get("role") != "user" and len(history) > preserve_recent:
             history.pop(0)
         return True
@@ -1626,9 +6261,28 @@ class ToolAgent:
         messages: list[dict[str, Any]],
         *,
         max_turns: int,
+        drain_turns: int = 0,
+        force_drain: bool = False,
     ) -> list[dict[str, Any]]:
-        if max_turns <= 0 or not messages:
+        if not messages:
             return []
+        if max_turns <= 0:
+            # explicit opt-out: leave the token budget as the only limit
+            return list(messages)
+
+        drain_turns = min(max(0, drain_turns), max(0, max_turns - 1))
+        if drain_turns:
+            present = sum(
+                1
+                for message in messages
+                if str(message.get("role", "")).strip() == "assistant"
+            )
+            # trigger at the cap unless this commit has already invalidated the
+            # prefix for token reasons, in which case take the deep cut now
+            if present > max_turns or force_drain:
+                max_turns = max_turns - drain_turns
+            elif not force_drain:
+                return list(messages)
 
         kept_reversed: list[dict[str, Any]] = []
         assistant_turns = 0
@@ -1650,14 +6304,34 @@ class ToolAgent:
             trimmed.pop(0)
         return trimmed
 
+    def _note_history_evicted(self) -> None:
+        # The scheduler consumes the first flag at the next handover. The
+        # second stays set so summaries remain visible after history is lost.
+        self._context_was_trimmed = True
+        self._has_evicted = True
+
     def _persistent_history_messages(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
+        original_history = messages[1:]
+        if _any_control_pruning_enabled():
+            messages = _prune_control_messages(messages)
+        keep_images = _history_image_keep()
+        if keep_images is not None:
+            # after pruning, so a dropped resumption opener cannot consume a
+            # retention slot
+            messages = _apply_history_image_window(messages, keep_images)
+        before_trim = len(messages)
         trimmed = self._trim_messages_for_context(messages, tools=tools)
         if not trimmed:
+            if original_history:
+                self._note_history_evicted()
             return []
+        token_trim_dropped = len(trimmed) < before_trim
         trimmed_history = trimmed[1:]
         history = self._keep_recent_history_turns(
             trimmed_history,
-            max_turns=_PERSISTENT_HISTORY_ASSISTANT_TURNS,
+            max_turns=_persistent_history_assistant_turns(),
+            drain_turns=_history_turn_drain(),
+            force_drain=token_trim_dropped and _history_drain_coalesce(),
         )
         if (
             history
@@ -1667,7 +6341,115 @@ class ToolAgent:
             previous_message = trimmed_history[len(trimmed_history) - len(history) - 1]
             if str(previous_message.get("role", "")).strip() == "user":
                 history = [previous_message, *history]
-        return self._drop_until_first_user_message(history)
+        history = self._drop_until_first_user_message(history)
+        # Include turn-cap eviction, control pruning and image removal, which
+        # can change the cached prefix without the token trimmer dropping any
+        # messages. Image removal can leave the message count unchanged.
+        if history != original_history:
+            self._note_history_evicted()
+        return history
+
+    def _priority_total_levels(self) -> int:
+        """Read public game metadata through the existing session callback.
+
+        No new solver state or prompt fields. A custom callback, old object,
+        or unavailable metadata can fall back without interrupting a game.
+        """
+        try:
+            callback = getattr(self, "_step_env_callback", None)
+            session = getattr(callback, "__self__", None)
+            game = getattr(session, "game", None)
+            count = parse_total_levels(getattr(game, "number_of_levels", None))
+        except Exception:
+            count = None
+        if count is not None:
+            return count
+        if not getattr(self, "_priority_level_count_warned", False):
+            self._priority_level_count_warned = True
+            log.warning("priority: total level count unavailable or invalid; using 10")
+        return 10
+
+    def _maybe_handover(self) -> None:
+        """Re-price this game, and yield the slot if a gate is running.
+
+        Called right after the trimmer and before the request goes out, which is
+        where the prefix has just been invalidated. Consumes the trim flag: it
+        describes the trim that produced THIS request, and leaving it set would
+        keep the game demoted long after the prefix had been rebuilt.
+        """
+        if not self._context_was_trimmed:
+            return
+        self._context_was_trimmed = False
+        summary = self._last_step_summary or {}
+        level = int(summary.get("level") or 1)
+        actions = max(
+            0, _priority_action_count(summary) - self._actions_at_level_start
+        )
+        # getattr: an agent unpickled from before this field existed would
+        # otherwise raise here on every turn
+        tokens = max(
+            0,
+            self._session_generated_tokens
+            - getattr(self, "_tokens_at_level_start", 0),
+        )
+        gate = _priority_gate()
+        snapshot = None
+        options = _priority_level_options()
+        total_levels = (
+            self._priority_total_levels()
+            if options["normalize_score"] or options["tail_lookup"] else None
+        )
+        if gate is not None and _priority_refresh_enabled():
+            pace = self._progress_pace.cost_multiplier() if _get_env_bool("ARC3_PRIORITY_PACE", False) else 1.0
+            snapshot = PrioritySnapshot(level, actions, tokens, pace, total_levels)
+            estimate = gate._snapshot_priority(snapshot, time.monotonic())
+        else:
+            estimate = _game_priority(
+                level, actions, tokens, endgame=self._in_endgame(),
+                total_levels=total_levels,
+            )
+        # this request pays the prefill, so it drops to the bottom band; the one
+        # after it, with the prefix rebuilt, returns to the warm band
+        self._priority_current = estimate
+        self._priority_next = estimate + _PRIORITY_BAND
+        if gate is not None:
+            _diag(self, "gate")
+            gate.handover(estimate, snapshot)
+
+    def _in_endgame(self) -> bool:
+        """Whether this game has been running long enough to switch phase.
+
+        Read from the session's run_elapsed_seconds rather than a clock of its
+        own: that is the timer max_runtime_s_per_game uses to end the game, so
+        the two cannot disagree about how old a game is.
+        """
+        minutes = _endgame_start_minutes()
+        if minutes <= 0:
+            return False
+        elapsed = getattr(self, "_game_elapsed_seconds", None)
+        if elapsed is None:
+            return False          # no step has reported the clock yet
+        if elapsed < minutes * 60.0:
+            return False
+        if not getattr(self, "_endgame_announced", False):
+            self._endgame_announced = True
+            log.warning(
+                "endgame phase after %.0f minutes: ranking on what completing "
+                "the current level would bank, with no tail and no hazard",
+                elapsed / 60.0,
+            )
+        return True
+
+    def _note_request_succeeded(self) -> None:
+        """Promote to the warm band once a request has actually landed.
+
+        Called on a completed response rather than on dispatch: a retry after a
+        connection failure has rebuilt nothing, so the game stays in the bottom
+        band until one lands.
+        """
+        nxt = getattr(self, "_priority_next", None)
+        if nxt is not None:
+            self._priority_current = nxt
 
     def _trim_messages_for_context(
         self,
@@ -1676,6 +6458,7 @@ class ToolAgent:
         tools: list[dict[str, Any]] | None = None,
         preserve_recent: int = 1,
         extra_safety_tokens: int = 0,
+        force_drain: bool = False,
     ) -> list[dict[str, Any]]:
         if not messages:
             return []
@@ -1683,10 +6466,83 @@ class ToolAgent:
         history = list(messages[1:])
         preserve_recent = max(0, preserve_recent)
         budget_tokens = max(1, self._context_budget_tokens - max(0, extra_safety_tokens))
-        while history and self._estimate_request_input_tokens([system_message, *history], tools=tools) > budget_tokens:
+        # capped at half the budget: a mis-set knob should cost history depth,
+        # not erase history
+        drain_tokens = min(_context_drain_tokens(), budget_tokens // 2)
+        target_tokens = max(1, budget_tokens - drain_tokens) if drain_tokens else budget_tokens
+        floor_mode = bool(drain_tokens) and _context_drain_floor_mode()
+        # Normally eviction starts only once the estimate exceeds the budget.
+        # After the server has rejected a request the estimate is known to be
+        # wrong, so waiting for it to admit the request is too big would drop
+        # nothing at all; force_drain enters the loop already draining and lets
+        # the existing target - budget minus one drain, computed against the
+        # real budget - decide where to stop.
+        draining = bool(force_drain and drain_tokens)
+        def _head_is_summary() -> bool:
+            """A summary REQUEST at the head, not its reply.
+
+            Landing on the assistant reply leaves history starting with an
+            assistant message, which _drop_until_first_user_message then strips -
+            so the exit would drop the very thing it stopped for.
+            """
+            return bool(
+                history
+                and _message_is_summary(history[0])
+                and str(history[0].get("role", "")).strip() == "user"
+            )
+
+        # `draining` means two different things. On the normal path it is the
+        # hysteresis phase - we crossed the budget and are heading for the
+        # target - and stopping early there is exactly the point. Under
+        # force_drain it means the server rejected the request and the estimate
+        # cannot be trusted, so a summary-shaped head says nothing about whether
+        # the request now fits; stopping early there risks another rejection and
+        # another round trip.
+        def _stop_early(used_tokens: int) -> bool:
+            return (
+                _drain_stop_at_summaries()
+                and draining
+                and not force_drain
+                and used_tokens <= budget_tokens
+                and _head_is_summary()
+            )
+
+        while history:
+            used = self._estimate_request_input_tokens([system_message, *history], tools=tools)
+            over_budget = used > budget_tokens
+            if not over_budget and not draining:
+                break
+            if floor_mode and not over_budget:
+                # the mandatory phase is done; keep draining only while the NEXT
+                # drop would still leave us at or above the target. Looking
+                # ahead rather than dropping and checking is what puts a real
+                # floor under the retained history.
+                if _stop_early(used):
+                    break
+                probe = list(history)
+                if not self._drop_oldest_history_block(probe, preserve_recent=preserve_recent):
+                    break
+                after = self._estimate_request_input_tokens(
+                    [system_message, *probe], tools=tools
+                )
+                if after < target_tokens:
+                    break
+                history[:] = probe
+                continue
+            if not floor_mode and used <= (target_tokens if draining else budget_tokens):
+                break
+            if _stop_early(used):
+                break
+            draining = True
             if not self._drop_oldest_history_block(history, preserve_recent=preserve_recent):
                 break
         history = self._drop_until_first_user_message(history)
+        # A dropped message invalidates the server's prefix for this
+        # conversation, so the next request has to be prefilled from the
+        # divergence point regardless. That makes it the one moment when
+        # yielding the slot to another game costs nothing.
+        if len(history) != len(messages) - 1:
+            self._note_history_evicted()
         return [system_message, *history]
 
     def _force_reduce_messages(
@@ -1701,6 +6557,7 @@ class ToolAgent:
         history = list(messages[1:])
         if not self._drop_oldest_history_block(history, preserve_recent=max(0, preserve_recent)):
             return list(messages)
+        self._note_history_evicted()
         return [system_message, *history]
 
     def analyze(
@@ -1719,6 +6576,10 @@ class ToolAgent:
             return None
         self._ensure_session(state_path)
         self._step_env_callback = step_env
+        if self._noop_repeat_guard is not None:
+            solver_obj = getattr(step_env, "__self__", None)
+            if solver_obj is not None:
+                solver_obj.action_guard_hook = self._action_guard_hook
         self._current_valid_actions = _normalize_valid_actions(valid_actions)
 
         analyzer_log = transcript_path or (state_path.parent / f"{state_path.stem}_analyzer.txt")
@@ -1732,6 +6593,7 @@ class ToolAgent:
             previous_step_summary=self._last_step_summary,
         )
         display_action_num = _display_action_number(action_num)
+        self._wm_absorbed_in_turn = False
 
         with open(analyzer_log, "a", encoding="utf-8") as f:
             step_label = f"analysis_step={analysis_step} | " if analysis_step is not None else ""
@@ -1749,22 +6611,108 @@ class ToolAgent:
                 transcript_updated("".join(transcript_parts))
 
         append_transcript("SYSTEM PROMPT", self._system_prompt)
-        append_transcript("USER PROMPT", user_prompt)
 
         previous_history_messages = list(self._history_messages)
+        # Mirrors the history snapshot: a turn abandoned on a request failure is
+        # rolled back, and the per-game token allowance has to roll back with it
+        # or the limit stops measuring the game and starts measuring how flaky
+        # the endpoint was. The tokens were really generated - the server spent
+        # them - but the point of the limit is that the same game gets the same
+        # budget whatever the infrastructure did.
+        tokens_at_turn_start = self._session_generated_tokens
+        # The tools go with it, even though the summary must not call one. The
+        # template renders their definitions into a system block near the start
+        # of the prompt, so a request without them diverges from the first
+        # tokens and prefills the whole context: measured at 15% shared prefix
+        # against 99% with them, and as prefix reuse falling from 75% to 45%
+        # across a run. The instruction not to act, and the harness rejecting a
+        # reply that calls a tool, are what keep it from acting - not withholding
+        # the definitions.
+        self._maybe_append_rolling_summary(
+            append_transcript, self._tools(state_path),
+            request_timeout_seconds=request_timeout_seconds,
+        )
         preserve_history = True
-        messages: list[dict[str, Any]] = self._trim_messages_for_context(
-            [{"role": "system", "content": self._system_prompt}, *self._history_messages, self._build_user_message(user_prompt, current_frame)],
+        resuming_after_yield = bool(getattr(self, "_resume_after_yield", False))
+        resume_reason = str(getattr(self, "_resume_reason", "") or "yield_bare")
+        self._resume_after_yield = False
+        self._resume_reason = ""
+        if not resuming_after_yield and _persistent_functions_scope() == "turn":
+            # A resumption is the same turn, so helpers survive it either way.
+            if _persistent_functions() and getattr(self, "_kept_functions", None):
+                self._retained_clear_notice = (
+                    "Your retained functions were cleared at the start of this new analyzer turn; "
+                    "redefine any you still need."
+                )
+            self._kept_functions = {}
+        resume_mode = _yield_resume_prompt_mode() if resuming_after_yield else "full"
+        # state_only always carries the board - that IS the mode - so the
+        # image knob does not apply to it
+        resume_frame = (
+            current_frame
+            if not resuming_after_yield
+            or resume_mode == "state_only"
+            or _resume_prompt_images_enabled()
+            else None
+        )
+        if (
+            resuming_after_yield
+            and resume_mode in ("state_only", "short")
+            and _resume_diff_image_enabled()
+            and self._turn_diff_image is not None
+        ):
+            # _build_user_message consumes the pending slot, so re-arm it from
+            # the turn cache for this one message
+            self._pending_diff_image = self._turn_diff_image
+            self._pending_diff_image_label = self._turn_diff_image_label
+        if resume_mode == "state_only":
+            opening_message = self._build_user_message(
+                _yield_resume_prompt(resume_reason) + "\n\n" + _STATE_ONLY_CAPTION,
+                current_frame,
+            )
+        elif resume_mode == "short":
+            opening_message = self._build_user_message(
+                _yield_resume_prompt(resume_reason), resume_frame
+            )
+        else:
+            opening_message = self._build_user_message(
+                user_prompt, resume_frame if resuming_after_yield else current_frame
+            )
+        # log what was actually SENT, not the argument that was passed in: on a
+        # resumed invocation the short/state_only substitution happens here, so
+        # transcribing `user_prompt` would report a full opener the model never
+        # saw - and at a 60s yield budget that is most turns
+        append_transcript("USER PROMPT", _message_display_text(opening_message))
+        if resuming_after_yield:
+            # Tag EVERY resumption message, not just the short form. A yield
+            # leaves this turn's original opener in history, so the message
+            # appended here is a duplicate restatement of unchanged state
+            # (same step, level, board, diffs, ledger - a no-execution yield
+            # changes nothing). It is needed for the request that resumes the
+            # turn and is pure duplication afterwards, which is precisely what
+            # ARC3_PRUNE_CONTROL_CONTEXT exists to drop.
+            _mark_control_message(opening_message, "resume")
+        messages: list[dict[str, Any]] = [
+            {"role": "system", "content": self._system_prompt}, *self._history_messages,
+        ]
+        self._append_context_message(messages, opening_message)
+        messages = self._trim_messages_for_context(
+            messages,
             tools=self._tools(state_path),
             preserve_recent=1,
         )
         step_executed = False
+        turn_tool_call_count = 0
         captured_reasoning = ""
         latest_request_messages: list[dict[str, Any]] | None = None
         latest_request_tools: list[dict[str, Any]] | None = None
         latest_request_tool_choice: str | None = None
         latest_request_index = 0
         turn_started_at = time.monotonic()
+        # per TURN, not per session: the budget is how much the model may
+        # generate before the turn ends, and a resumption continues the same
+        # turn, so this resets where the clock does
+        self._turn_generated_tokens = 0
         yielded_control_reason: str | None = None
 
         def control_yield_reason() -> str | None:
@@ -1776,6 +6724,11 @@ class ToolAgent:
                     log.warning("analyzer stop check failed at action %d: %s", display_action_num, exc)
             if self._yield_seconds is not None and (time.monotonic() - turn_started_at) >= self._yield_seconds:
                 return "turn_time_budget"
+            if (
+                self._yield_tokens is not None
+                and self._turn_generated_tokens >= self._yield_tokens
+            ):
+                return "turn_token_budget"
             return None
 
         try:
@@ -1788,6 +6741,7 @@ class ToolAgent:
                 tools = self._tools(state_path)
                 tool_choice = _request_tool_choice(tools)
                 messages = self._trim_messages_for_context(messages, tools=tools)
+                self._maybe_handover()
                 latest_request_messages = json.loads(json.dumps(messages))
                 latest_request_tools = json.loads(json.dumps(tools))
                 latest_request_tool_choice = tool_choice
@@ -1818,9 +6772,17 @@ class ToolAgent:
                             analysis_step=analysis_step,
                             action=display_action_num,
                             request_index_within_turn=latest_request_index,
+                            chat_template_kwargs=self._harness_template_kwargs(),
                         )
                     result = self._chat_completion(messages, **request_kwargs)
+                    _calibration_note = self._calibrate_from_usage(
+                        messages, request_kwargs.get("tools"), result.usage
+                    )
+                    if _calibration_note:
+                        append_transcript("ANALYZER STATUS", _calibration_note)
                     self._accumulate_usage_tokens(result.usage)
+                    self._note_request_succeeded()
+                    _diag(self, "local")
                     if self._save_request_logs:
                         _append_request_snapshot(
                             _resolve_request_log_path(state_path),
@@ -1832,14 +6794,25 @@ class ToolAgent:
                             action=display_action_num,
                             request_index_within_turn=latest_request_index,
                             finish_reason=result.finish_reason,
+                            served_by=result.served_by,
+                            usage=result.usage,
+                            chat_template_kwargs=self._harness_template_kwargs(),
                         )
                 except requests.RequestException as exc:
                     if not _is_context_length_error(exc):
                         raise
+                    # A rejected request means the estimate was wrong, and 512
+                    # tokens only helps when it was wrong by less than that -
+                    # otherwise nothing is dropped and the fallback removes one
+                    # block per rejected round trip. force_drain evicts down to
+                    # the drain target instead, in one step, which is what the
+                    # drain is for: land clear of the ceiling rather than just
+                    # under it.
                     trimmed_messages = self._trim_messages_for_context(
                         messages,
                         tools=tools,
                         extra_safety_tokens=_CONTEXT_OVERFLOW_RETRY_TRIM_TOKENS,
+                        force_drain=True,
                     )
                     if trimmed_messages == messages:
                         trimmed_messages = self._force_reduce_messages(messages)
@@ -1851,6 +6824,21 @@ class ToolAgent:
                     )
                     messages = trimmed_messages
                     continue
+                truncated_tool_call = False
+                if str(result.finish_reason or "").strip() == "length":
+                    ladder = _reasoning_effort_ladder()
+                    if ladder and self._reasoning_effort_rung < len(ladder) - 1:
+                        self._reasoning_effort_rung += 1
+                        log.warning(
+                            "generation hit the output ceiling; reducing reasoning_effort to %r",
+                            ladder[self._reasoning_effort_rung],
+                        )
+                        append_transcript(
+                            "ANALYZER STATUS",
+                            "reasoning_effort_reduced: generation was truncated at the output "
+                            f"limit; subsequent requests use reasoning_effort="
+                            f"{ladder[self._reasoning_effort_rung]!r} until an action executes.",
+                        )
                 raw_reasoning = _extract_reasoning_text(result.message)
                 raw_content = _normalize_message_content(result.message.get("content", ""))
                 tool_calls = json.loads(json.dumps(result.message.get("tool_calls") or []))
@@ -1871,8 +6859,35 @@ class ToolAgent:
                             json.loads(raw_arguments)
                         except json.JSONDecodeError as exc:
                             malformed_argument_errors.append(f"{tool_name}: invalid JSON arguments ({exc})")
+                if malformed_argument_errors:
+                    # Drop ALL calls from this reply rather than the bad one.
+                    #
+                    # An unparseable arguments string means the reply was cut
+                    # off mid-call (finish_reason=length is the usual cause),
+                    # so anything after the truncation point is missing and an
+                    # earlier "valid" call in the same message expresses only
+                    # part of an intent - dispatching half a plan is worse than
+                    # dispatching none.
+                    #
+                    # Critically, the message must also never reach history:
+                    # servers re-parse tool calls in the incoming message list
+                    # when rendering the chat template, so one malformed call
+                    # committed to history makes EVERY later request fail at
+                    # render time, before inference - an unbreakable loop that
+                    # the HTTP retry cannot help with. Content and reasoning
+                    # are kept; only the calls are dropped, which routes this
+                    # into the existing no-tool-call path.
+                    log.warning(
+                        "dropping %d malformed tool call(s) (finish_reason=%s): %s",
+                        len(tool_calls),
+                        result.finish_reason or "?",
+                        "; ".join(malformed_argument_errors),
+                    )
+                    tool_calls = []
+                    truncated_tool_call = str(result.finish_reason or "").strip() == "length"
                 response_meta = _format_model_response_meta(
                     finish_reason=result.finish_reason,
+                    served_by=result.served_by,
                     reasoning=reasoning,
                     content=content,
                     tool_calls=tool_calls,
@@ -1889,7 +6904,8 @@ class ToolAgent:
                 if reasoning:
                     captured_reasoning = reasoning
                     append_transcript("THINKING", reasoning)
-                    assistant_message["reasoning"] = reasoning
+                    for _reasoning_key in _reasoning_history_keys():
+                        assistant_message[_reasoning_key] = reasoning
 
                 if not tool_calls:
                     if content:
@@ -1900,11 +6916,87 @@ class ToolAgent:
                         assistant_message["content"] = None
 
                     if content or reasoning:
-                        messages.append(assistant_message)
+                        if (
+                            not content
+                            and reasoning
+                            and _is_degenerate_text(reasoning)
+                        ):
+                            # A reasoning-only reply that is one character
+                            # repeated to the output cap is a serving fault, not
+                            # thinking: observed as exactly 6144 "!" from the
+                            # first token, with content_chars 0 and no tool
+                            # call, under speculative decoding. Keeping it cost
+                            # far more than the lost turn - 12k characters of
+                            # noise pushed the trimmer from 31 history messages
+                            # to 12, evicting real history, and the model then
+                            # concluded "my replies keep failing to emit the
+                            # tool call" and started writing shorter answers to
+                            # fix a formatting error it had not made.
+                            # transcript AND log: the transcript records it
+                            # where it happened, but a serving fault should be
+                            # visible while the run is going rather than only
+                            # on a later grep
+                            log.warning(
+                                "degenerate reasoning dropped: %d chars, no content "
+                                "and no tool call - the reply was a repetition loop. "
+                                "This is a serving fault, not the model reasoning.",
+                                len(reasoning),
+                            )
+                            append_transcript(
+                                "ANALYZER STATUS",
+                                "degenerate_reasoning_dropped: the reply was a "
+                                f"repetition loop ({len(reasoning)} chars) with no "
+                                "content or tool call, and was replaced in context.",
+                            )
+                            assistant_message = _mark_control_message(
+                                {"role": "assistant", "content": _DEGENERATE_CONTENT_STUB},
+                                "stub",
+                            )
+                        elif content and _is_degenerate_text(content):
+                            # a repetition loop feeds on its own output, so the
+                            # garbage must not reach the next request's context
+                            append_transcript(
+                                "ANALYZER STATUS",
+                                "degenerate_content_stubbed: assistant text was highly "
+                                "repetitive and was replaced in context.",
+                            )
+                            assistant_message = {
+                                "role": "assistant",
+                                "content": _DEGENERATE_CONTENT_STUB,
+                            }
+                        elif (
+                            not content
+                            and reasoning
+                            and _stub_dead_reasoning_enabled()
+                        ):
+                            # reasoning-only reply: nothing to continue from,
+                            # and re-sending it biases the retry toward the
+                            # same dead end. Transcript keeps the full text.
+                            assistant_message = _mark_control_message(
+                                {"role": "assistant", "content": _DEAD_REASONING_STUB},
+                                "stub",
+                            )
+                        self._append_context_message(messages, assistant_message)
                     yielded_control_reason = control_yield_reason()
                     if yielded_control_reason is not None:
                         break
                     followup_prefix = "You have not acted yet. Investigate first. "
+                    if truncated_tool_call:
+                        # the only signal the model gets that LENGTH, not
+                        # formatting, is why its call vanished
+                        followup_prefix = (
+                            "Your previous reply was CUT OFF before the tool call finished, so the "
+                            "call was incomplete and could not be executed. Keep your reasoning "
+                            "much shorter this time and emit the `python` tool call early enough "
+                            "that it completes within the output limit. Prefer one small snippet "
+                            "over a long analysis. "
+                        )
+                    elif malformed_argument_errors:
+                        followup_prefix = (
+                            "Your previous tool call had arguments that were not valid JSON, so it "
+                            "was not executed. Emit exactly one `python` tool call whose arguments "
+                            "are a well-formed JSON object. "
+                        )
                     if tool_call_markup_in_text:
                         followup_prefix = (
                             "You did not call a tool. We detected `<tool_call>` markup inside your reasoning or assistant text, "
@@ -1912,10 +7004,31 @@ class ToolAgent:
                             "Emit exactly one `python` tool call directly as your next response. "
                             "Do not place `<tool_call>` markup inside reasoning, explanation, or notes. "
                         )
+                    # the clause names where transient facts SHOULD go, so it
+                    # holds only while that section is carried; a mode that
+                    # drops it would be pointing the model nowhere
+                    _transient_clause = (
+                        " \u2014 current positions and this turn's events go in "
+                        "`Recent findings:`. "
+                        if "Recent findings" in _advertised_section_labels()
+                        # "only" wants something to contrast against, and the
+                        # clause that supplied it names a section this mode has
+                        # dropped; keep the contrast, drop the destination
+                        else ", not current positions or what just happened. "
+                    )
                     followup_prompt = (
                         f"{followup_prefix}"
                         "Then investigate and revise your working world model of what the level contains, what actions appear to do, what the current goal seems to be, and what plan looks best. "
-                        "If helpful, include short world-model update lines such as `World model:`, `Goal model:`, `Action model:`, `Recent findings:`, `Open questions:`, `Plan:`, or `Cross-level notes:`. "
+                        # the same narrowing the opener applies: advertising a section a
+                        # restricted mode drops invites the model to write
+                        # where it will never read it back
+                        f"If helpful, include memory update lines such as "
+                        f"{_section_list_phrase(_advertised_section_labels(), 'or')}. "
+                        "`World model:` holds durable environment rules"
+                        + (" only" if "Recent findings" in _advertised_section_labels() else "")
+                        + f"{_transient_clause}"
+                        "Each section you write replaces the stored one entirely, "
+                        "so restate what is still true, not only what is new. "
                         "Call the `python` tool with code that inspects `current_frame`, `previous_frame`, `last_transition`, `history`, or `valid_actions` -- use `current_frame.segmentation` as the primary view, and `.ascii` only for a small specific region -- "
                         "compare `previous_frame` to `current_frame` for the most recent change, "
                         "derives a compact board summary, programs a small search or scorer over candidate actions or short sequences, "
@@ -1923,7 +7036,12 @@ class ToolAgent:
                         f"{TOOL_CALL_FORMAT_GUIDANCE}"
                     )
                     append_transcript("USER PROMPT", followup_prompt)
-                    messages.append({"role": "user", "content": followup_prompt})
+                    self._append_context_message(
+                        messages,
+                        _mark_control_message(
+                            {"role": "user", "content": followup_prompt}, "nudge"
+                        )
+                    )
                     continue
 
                 if content:
@@ -1931,7 +7049,8 @@ class ToolAgent:
                     append_transcript("ASSISTANT", content)
                     assistant_message["content"] = content
                 assistant_message["tool_calls"] = tool_calls
-                messages.append(assistant_message)
+                turn_tool_call_count += len(tool_calls)
+                self._append_context_message(messages, assistant_message)
 
                 for tool_index, tool_call in enumerate(tool_calls):
                     function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
@@ -1954,8 +7073,17 @@ class ToolAgent:
                     dispatch = self._dispatch_tool(state_path, tool_name, arguments)
                     if dispatch.step_executed:
                         step_executed = True
+                        if self._reasoning_effort_rung >= 0 and dispatch.made_progress:
+                            # the turn achieved something, so restore the server
+                            # default; an inert action leaves the reduction in
+                            # place rather than inviting another truncation
+                            log.info(
+                                "action changed the board; restoring default reasoning_effort"
+                            )
+                            self._reasoning_effort_rung = -1
                     append_transcript(f"TOOL RESULT: {tool_name}", _render_tool_result_display(dispatch.content))
-                    messages.append(
+                    self._append_context_message(
+                        messages,
                         {
                             "role": "tool",
                             "tool_call_id": tool_call.get("id", ""),
@@ -1977,8 +7105,44 @@ class ToolAgent:
                     break
 
         except requests.RequestException as exc:
+            if _is_read_timeout(exc) and _yield_on_timeout_enabled():
+                # Keep this turn's completed exchanges and hand control back to
+                # the solver, which re-invokes on the same game state.
+                append_transcript(
+                    "ANALYZER STATUS",
+                    f"request_timeout (yielding, history preserved): {exc}",
+                )
+                log.warning(
+                    "analyzer request timed out at action %d; yielding with %d tool call(s) preserved: %s",
+                    display_action_num,
+                    turn_tool_call_count,
+                    exc,
+                )
+                self._resume_after_yield = True
+                self._resume_reason = "timeout"
+                if latest_request_messages is not None:
+                    _write_prompt_log_snapshot(
+                        prompt_log,
+                        model_id=self._model.model_id,
+                        base_url=self._model.base_url,
+                        display_action_num=display_action_num,
+                        analysis_step=analysis_step,
+                        request_index=latest_request_index,
+                        messages=latest_request_messages,
+                        tools=latest_request_tools,
+                        tool_choice=latest_request_tool_choice,
+                        transcript="".join(transcript_parts),
+                    )
+                # preserve_history stays True, so the finally block commits
+                # this turn's exchanges exactly as a normal yield would.
+                return AnalyzerTurnResult(
+                    step_executed=False,
+                    reasoning=captured_reasoning,
+                    yielded_control=True,
+                )
             append_transcript("ANALYZER STATUS", f"request_error: {exc}")
             preserve_history = False
+            self._session_generated_tokens = tokens_at_turn_start
             if latest_request_messages is not None:
                 _write_prompt_log_snapshot(
                     prompt_log,
@@ -1997,6 +7161,7 @@ class ToolAgent:
         except Exception as exc:
             append_transcript("ANALYZER STATUS", f"error: {exc}")
             preserve_history = False
+            self._session_generated_tokens = tokens_at_turn_start
             if latest_request_messages is not None:
                 _write_prompt_log_snapshot(
                     prompt_log,
@@ -2013,10 +7178,15 @@ class ToolAgent:
             log.warning("analyzer failed at action %d: %s", display_action_num, exc)
             return None
         finally:
+            if step_executed and not self._wm_absorbed_in_turn:
+                self._turns_without_wm_update += 1
             if preserve_history:
                 self._history_messages = self._persistent_history_messages(messages, tools=self._tools(state_path))
             else:
                 self._history_messages = previous_history_messages
+            solver_obj = getattr(self._step_env_callback, "__self__", None)
+            if solver_obj is not None and hasattr(solver_obj, "action_guard_hook"):
+                solver_obj.action_guard_hook = None
             self._step_env_callback = None
             self._current_valid_actions = []
 
@@ -2036,6 +7206,7 @@ class ToolAgent:
             f"request_safety_margin_tokens: {self._request_safety_margin_tokens}\n"
             f"tool_output_tokens: {self._tool_output_tokens}\n"
             f"yield_seconds: {self._yield_seconds if self._yield_seconds is not None else 'disabled'}\n"
+            f"yield_tokens: {self._yield_tokens if self._yield_tokens is not None else 'disabled'}\n"
             f"available_tools: python\n"
             f"python_timeout_seconds: {self._python_timeout}\n"
             f"history_messages: {len(self._history_messages)}\n"
@@ -2056,6 +7227,31 @@ class ToolAgent:
                 tool_choice=latest_request_tool_choice,
                 transcript="".join(transcript_parts),
             )
+        # An invocation that executed nothing will be re-invoked by the solver
+        # on the same game state, so whatever it appends next is a duplicate
+        # restatement. This covers BOTH non-execution exits: the time budget
+        # (yielded_control_reason set) and the tool-step budget (loop condition
+        # went false, no yield reason). Execution and yielding are mutually
+        # exclusive, so `not step_executed` is the whole condition.
+        self._resume_after_yield = not step_executed
+        if yielded_control_reason is not None:
+            self._resume_reason = (
+                "yield_tools" if turn_tool_call_count > 0 else "yield_bare"
+            )
+        else:
+            self._resume_reason = (
+                "tool_steps" if turn_tool_call_count > 0 else "yield_bare"
+            )
+        if not step_executed and turn_tool_call_count == 0 and self._last_step_summary:
+            # no dispatch ran, so _dispatch_tool never marked the summary
+            # stale; without this the next opener re-narrates an older
+            # execution as if it had just happened (or claims nothing has run
+            # yet) directly beneath this turn's own reasoning.
+            self._last_step_summary = {
+                **self._last_step_summary,
+                "stale": True,
+                "stale_reason": "no_tool_call",
+            }
         return AnalyzerTurnResult(
             step_executed=step_executed,
             reasoning=captured_reasoning,
diff --git a/ARC3-Inference/inference/agent/vision_context.py b/ARC3-Inference/inference/agent/vision_context.py
index bb8232f..e3dadff 100644
--- a/ARC3-Inference/inference/agent/vision_context.py
+++ b/ARC3-Inference/inference/agent/vision_context.py
@@ -3,6 +3,7 @@ from __future__ import annotations
 
 import base64
 import io
+from collections import defaultdict
 import os
 from typing import Any
 
@@ -39,6 +40,24 @@ def current_grid_image_enabled() -> bool:
     return multimodal_context() == "current_grid"
 
 
+_RESAMPLE_FILTERS = {
+    "nearest": Image.Resampling.NEAREST,
+    "bilinear": Image.Resampling.BILINEAR,
+    "bicubic": Image.Resampling.BICUBIC,
+    "lanczos": Image.Resampling.LANCZOS,
+}
+
+
+def multimodal_resample() -> Image.Resampling:
+    """Upscale filter for grid images. Nearest neighbor (the default) keeps
+    cells as flat exact-palette blocks; interpolating filters produce softer
+    edges that sit closer to a vision encoder's natural-image training
+    distribution, which has measurably helped on some models. Model-dependent:
+    test against the model actually served."""
+    raw = os.environ.get("MULTIMODAL_RESAMPLE", "").strip().lower()
+    return _RESAMPLE_FILTERS.get(raw, Image.Resampling.NEAREST)
+
+
 def current_grid_image_upscale() -> int:
     raw = os.environ.get("MULTIMODAL_UPSCALE", "").strip()
     if not raw:
@@ -63,7 +82,7 @@ def frame_to_png_data_url(frame: Frame, *, upscale: int | None = None) -> str:
             value = row[col_idx] if col_idx < len(row) else 0
             pixels[col_idx, row_idx] = ARC_COLOR_MAP.get(int(value), ARC_COLOR_MAP[0])
     if scale > 1:
-        image = image.resize((cols * scale, rows * scale), Image.Resampling.NEAREST)
+        image = image.resize((cols * scale, rows * scale), multimodal_resample())
 
     buffer = io.BytesIO()
     image.save(buffer, format="PNG")
@@ -80,3 +99,234 @@ def current_grid_image_part(frame: Frame | None) -> dict[str, Any] | None:
             "url": frame_to_png_data_url(frame),
         },
     }
+
+
+# Off-palette neutral for unchanged cells in diff images. The ARC palette's
+# indices 0-5 span a white-to-black grayscale ramp, so no gray (or black, or
+# white) can be used without colliding with real cell values; a dark
+# desaturated navy is unambiguous.
+DIFF_UNCHANGED_RGB: tuple[int, int, int] = (16, 24, 48)
+
+
+_ANIMATION_IMAGE_MODES = ("off", "peak", "composite")
+
+
+def _animation_image_mode(name: str) -> str:
+    """One of off, peak, composite. Anything else raises.
+
+    These two are the only enum knobs among a set of otherwise boolean ARC3_*
+    variables, so `=1` is the natural thing to write - and silently meant off,
+    which cost a whole run to notice. Failing at startup is better than a
+    feature that is configured and quietly absent.
+    """
+    mode = os.environ.get(name, "off").strip().lower()
+    if mode not in _ANIMATION_IMAGE_MODES:
+        raise ValueError(
+            f"{name}={mode!r} is not valid. Use one of "
+            f"{', '.join(_ANIMATION_IMAGE_MODES)} - this knob is not a boolean."
+        )
+    return mode
+
+
+def animation_death_image_mode() -> str:
+    """off | peak | composite, for the fatal action of a game over.
+
+    Separate from ARC3_ANIMATION_IMAGE because the two fire on different
+    conditions and a game-over turn already carries two images, so a third is
+    a cost worth deciding on its own.
+    """
+    return _animation_image_mode("ARC3_ANIMATION_DEATH_IMAGE")
+
+
+def animation_image_mode() -> str:
+    """off | peak | composite.
+
+    Only ever attached when the action left the board area unchanged, which is
+    exactly when the board image the model receives is uninformative: the
+    action did something, and no frame it can reach shows what.
+    """
+    return _animation_image_mode("ARC3_ANIMATION_IMAGE")
+
+
+def animation_composite_part(
+    chain: list[tuple[tuple[int, ...], ...]],
+    transient_cells: list[tuple[int, int]],
+    *,
+    border: int,
+) -> dict[str, Any] | None:
+    """Every transient cell in the last colour it held before reverting.
+
+    Not a board state: cells from different moments composited onto one grid.
+    The alternative - a single real frame - misses a sweep that touches
+    different regions at different times, which has no peak moment.
+
+    Everything else is dimmed to the diff image's off-palette navy, so the
+    image carries only what no reachable frame contains. That includes cells
+    that changed just once (visible in current_frame anyway) and the border,
+    where the budget bar ticks on every action and would otherwise be the
+    brightest thing in the picture.
+    """
+    if not chain or not transient_cells:
+        return None
+    if not current_grid_image_enabled():
+        return None
+    final = chain[-1]
+    rows = len(final)
+    cols = max((len(row) for row in final), default=0)
+    if rows <= 0 or cols <= 0:
+        return None
+
+    wanted = set(transient_cells)
+    last_seen: dict[tuple[int, int], int] = {}
+    held: dict[tuple[int, int], int] = defaultdict(int)
+    for grid in chain[:-1]:
+        for row_idx, row in enumerate(grid):
+            final_row = final[row_idx] if row_idx < len(final) else ()
+            for col_idx, value in enumerate(row):
+                cell = (row_idx, col_idx)
+                if cell not in wanted:
+                    continue
+                if row_idx < border or row_idx >= rows - border:
+                    continue
+                if col_idx < border or col_idx >= cols - border:
+                    continue
+                final_value = final_row[col_idx] if col_idx < len(final_row) else None
+                if value == final_value:
+                    continue
+                value = int(value)
+                if last_seen.get(cell) != value:
+                    # a cell that flickers between colours restarts the count,
+                    # so the duration belongs to the colour actually painted
+                    held[cell] = 0
+                # later frames overwrite earlier ones, so this ends up holding
+                # the LAST non-final colour the cell showed
+                last_seen[cell] = value
+                held[cell] += 1
+    if not last_seen:
+        return None
+
+    # Brightness carries DURATION: a cell blended toward its true colour in
+    # proportion to how many frames it held it. This separates two mechanics
+    # that are otherwise identical in a flat composite - a 4x4 block falling
+    # down a column touches each cell for a frame or two and renders uniformly
+    # dim, while a column filling from the top holds its highest cell for the
+    # whole animation and its lowest for a moment, so it renders as a gradient
+    # pointing back at the source.
+    span = max(1, len(chain) - 1)
+    image = Image.new("RGB", (cols, rows), DIFF_UNCHANGED_RGB)
+    pixels = image.load()
+    for (row_idx, col_idx), value in last_seen.items():
+        colour = ARC_COLOR_MAP.get(value, ARC_COLOR_MAP[0])
+        # floored so a single-frame cell stays legible rather than vanishing
+        weight = max(0.35, min(1.0, held[(row_idx, col_idx)] / span))
+        pixels[col_idx, row_idx] = tuple(
+            int(round(base + weight * (target - base)))
+            for base, target in zip(DIFF_UNCHANGED_RGB, colour)
+        )
+    scale = current_grid_image_upscale()
+    if scale > 1:
+        image = image.resize((cols * scale, rows * scale), multimodal_resample())
+    buffer = io.BytesIO()
+    image.save(buffer, format="PNG")
+    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
+    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}
+
+
+def animation_peak_part(
+    chain: list[tuple[tuple[int, ...], ...]],
+    transient_cells: list[tuple[int, int]],
+) -> tuple[dict[str, Any] | None, int]:
+    """The real frame showing the most transient cells at once, and its index.
+
+    Unlike the composite this is a state the game actually displayed, so
+    nothing in it is fabricated - at the cost of missing anything that was
+    hidden at a different moment.
+    """
+    if not chain or not transient_cells or len(chain) <= 2:
+        return None, -1
+    if not current_grid_image_enabled():
+        return None, -1
+    final = chain[-1]
+    wanted = set(transient_cells)
+    best_index, best_count = -1, -1
+    for index, grid in enumerate(chain[1:]):
+        count = 0
+        for row_idx, col_idx in wanted:
+            row = grid[row_idx] if row_idx < len(grid) else ()
+            final_row = final[row_idx] if row_idx < len(final) else ()
+            value = row[col_idx] if col_idx < len(row) else None
+            final_value = final_row[col_idx] if col_idx < len(final_row) else None
+            if value != final_value:
+                count += 1
+        if count > best_count:
+            best_index, best_count = index, count
+    if best_index < 0 or best_count <= 0:
+        return None, -1
+
+    grid = chain[1:][best_index]
+    rows = len(grid)
+    cols = max((len(row) for row in grid), default=0)
+    image = Image.new("RGB", (cols, rows), ARC_COLOR_MAP[0])
+    pixels = image.load()
+    for row_idx, row in enumerate(grid):
+        for col_idx in range(cols):
+            value = row[col_idx] if col_idx < len(row) else 0
+            pixels[col_idx, row_idx] = ARC_COLOR_MAP.get(int(value), ARC_COLOR_MAP[0])
+    scale = current_grid_image_upscale()
+    if scale > 1:
+        image = image.resize((cols * scale, rows * scale), multimodal_resample())
+    buffer = io.BytesIO()
+    image.save(buffer, format="PNG")
+    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
+    return (
+        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
+        best_index,
+    )
+
+
+def diff_image_part(
+    previous_frame: Frame | None, current_frame: Frame | None
+) -> dict[str, Any] | None:
+    """Render a change-highlight image: cells that differ between the two
+    frames appear in their (new) true palette color; unchanged cells are
+    dimmed to an off-palette dark navy. Returns None when disabled, frames
+    are missing, or shapes differ (a full redraw makes a diff meaningless)."""
+    if previous_frame is None or current_frame is None:
+        return None
+    if not current_grid_image_enabled():
+        return None
+    prev_grid, cur_grid = previous_frame.grid, current_frame.grid
+    rows = len(cur_grid)
+    cols = max((len(row) for row in cur_grid), default=0)
+    if rows <= 0 or cols <= 0:
+        return None
+    if len(prev_grid) != rows or max(
+        (len(row) for row in prev_grid), default=0
+    ) != cols:
+        return None
+
+    scale = current_grid_image_upscale()
+    image = Image.new("RGB", (cols, rows), DIFF_UNCHANGED_RGB)
+    pixels = image.load()
+    for row_idx in range(rows):
+        prev_row = prev_grid[row_idx]
+        cur_row = cur_grid[row_idx]
+        for col_idx in range(cols):
+            prev_value = prev_row[col_idx] if col_idx < len(prev_row) else 0
+            cur_value = cur_row[col_idx] if col_idx < len(cur_row) else 0
+            if prev_value != cur_value:
+                pixels[col_idx, row_idx] = ARC_COLOR_MAP.get(
+                    int(cur_value), ARC_COLOR_MAP[0]
+                )
+    if scale > 1:
+        image = image.resize((cols * scale, rows * scale), multimodal_resample())
+
+    buffer = io.BytesIO()
+    image.save(buffer, format="PNG")
+    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
+    return {
+        "type": "image_url",
+        "image_url": {
+            "url": f"data:image/png;base64,{encoded}",
+        },
+    }
diff --git a/ARC3-Inference/inference/framework/run.py b/ARC3-Inference/inference/framework/run.py
index 4d4accf..fc7d80c 100644
--- a/ARC3-Inference/inference/framework/run.py
+++ b/ARC3-Inference/inference/framework/run.py
@@ -419,6 +419,7 @@ def _make_solver(
         analyzer_timeout=getattr(args, "analyzer_timeout", 120),
         max_actions_per_game=args.max_actions,
         max_runtime_s_per_game=max_runtime_minutes_per_game * 60.0,
+        max_generated_tokens_per_game=args.max_generated_tokens_per_game,
         concurrency=effective_concurrency,
         save_request_logs=bool(args.analyzer_save_request_logs),
         start_local_server=start_local_server,
@@ -1222,6 +1223,15 @@ def main() -> None:
     parser.add_argument("--experiment-dir", dest="experiment_dir", default="")
     parser.add_argument("--max-actions", type=int, default=None)
     parser.add_argument("--max-runtime-minutes", type=float, default=None)
+    parser.add_argument(
+        "--max-generated-tokens-per-game",
+        type=int,
+        default=None,
+        help="Give up on a game once it has generated this many tokens. A "
+             "wall-clock limit means something different on every backend and "
+             "moves hour to hour on a hosted API; this is what makes an "
+             "offline run comparable with a Kaggle one.",
+    )
     parser.add_argument("--max-experiment-runtime-minutes", type=float, default=None)
     parser.add_argument("--max-experiment-runtime-hours", type=float, default=None)
     parser.add_argument("--n-passes", dest="n_passes", type=int, default=1)
diff --git a/ARC3-Inference/inference/framework/solver.py b/ARC3-Inference/inference/framework/solver.py
index 0b69d62..e40976c 100644
--- a/ARC3-Inference/inference/framework/solver.py
+++ b/ARC3-Inference/inference/framework/solver.py
@@ -7,6 +7,7 @@ import contextlib
 import copy
 import functools
 import html
+import logging
 import json
 import os
 import re
@@ -23,7 +24,13 @@ import arcengine
 import taaf.game
 from taaf.solver import Solver
 
+from inference.utils.animation import (
+    collapse_frames,
+    normalize_frames,
+    summarize_animation,
+)
 from inference.agent.action_names import (
+    reset_exposed,
     to_engine_action,
     to_model_action,
     to_model_actions,
@@ -34,7 +41,11 @@ from inference.agent.runtime_state import (
     RUNTIME_STATE_FILENAME,
     write_runtime_state,
 )
-from inference.agent.tool_agent import ToolAgent
+from inference.agent.tool_agent import (
+    _PRIORITY_UNTRIMMED_BASE,
+    ToolAgent,
+    _priority_gate,
+)
 from inference.framework.kaggle import (
     DEFAULT_QWEN_MODEL_DATASET_SOURCE,
     DEFAULT_SERVED_MODEL_NAME,
@@ -53,10 +64,242 @@ from inference.utils.viewer_artifacts import (
     reset_raw_events_sidecar,
 )
 
+log = logging.getLogger(__name__)
+
 AnalyzerFactory = Callable[[taaf.game.Game, int], Any]
 
-ANALYZER_RETRY_BACKOFF_SECONDS = 1.0
+def _env_number(name: str, default: float, cast=float) -> float:
+    """A numeric env override that cannot take the run down at import time."""
+    try:
+        return cast(os.environ.get(name, "") or default)
+    except (TypeError, ValueError):
+        return cast(default)
+
+
+ANALYZER_RETRY_BACKOFF_SECONDS = max(
+    0.0, _env_number("ARC3_ANALYZER_RETRY_BACKOFF_SECONDS", 10.0)
+)
+# A request that fails outright - a dead endpoint, a refused connection - is
+# reported as retryable and the solver goes back to the top of the loop. With
+# no limit that is an infinite spin: nothing monitors the local server, so if
+# it exits mid-run every game retries once a second until its runtime budget is
+# gone, and the run finishes reporting whatever each game had reached with no
+# crash marker anywhere. Giving up after a run of consecutive failures makes
+# that visible - the run ends in seconds rather than at the wall clock, and the
+# games carry a note saying why. 0 restores the unlimited behaviour.
+# Negative means keep retrying; 0 means give up on the first failure. No
+# max(0, ...) clamp here - it would flatten a negative to zero and turn
+# "never give up" into "give up at once".
+# Kaggle requires an action within 15 minutes of the run starting. SGLang
+# needs about 10.5 of those to load weights - 6 on the main model, 3 more on the
+# MTP draft head, which is a second full pass over the same 206 shards - and the
+# notebook blocks on /health before playing anything, so a slow load spends the
+# whole allowance before a single action is issued and the submission scores
+# zero. With this set, the first N games to start execute one RESET straight
+# against the engine before touching the analyzer, which satisfies the clock
+# while the server is still loading; the HTTP retry loop then covers the rest of
+# the wait. 0 disables.
+WARMUP_ACTION_GAMES = _env_number("ARC3_WARMUP_ACTION_GAMES", 0, int)
+
+
+MAX_CONSECUTIVE_ANALYZER_FAILURES = _env_number("ARC3_MAX_ANALYZER_FAILURES", 10, int)
 DEFAULT_CANCEL_DRAIN_TIMEOUT_SECONDS = 120.0
+
+# No-op guard: width of the border frame treated as HUD (progress bars,
+# timers). 0 disables the guard. Simple v1 heuristic; see step_env.
+# arcengine renders a frame after every internal step(), so one action can come
+# back as a short animation whose intermediate frames the harness has always
+# discarded. Off by default: new information for the model, not a fix.
+ANIMATION_ENABLED = os.environ.get("ARC3_ANIMATION", "0").strip().lower() in (
+    "1",
+    "true",
+    "yes",
+    "on",
+)
+NOOP_GUARD_BORDER_WIDTH = max(0, int(os.environ.get("ARC3_NOOP_GUARD_BORDER", "4")))
+# Print one line per action(...) call into the snippet's stdout. Costs a line of
+# tool output per call, which is charged against tool_output_tokens.
+# The patch-110 rewordings, behind one switch. They replaced "changed nothing"
+# with "no change inside the board area" across every guard message, because the
+# old phrasing was false in the case that matters - the HUD usually DID change.
+# The reasoning still looks right, but they shipped alongside the outcome line
+# in a run that lost 43% of the previous competition score, so the two need to
+# be separable before either can be judged. Off restores the wording of the run
+# that scored 4.05.
+NEW_NOOP_WORDING = os.environ.get(
+    "ARC3_NEW_CHANGED_PROMPTS", "0"
+).strip().lower() in ("1", "true", "yes", "on")
+ACTION_ECHO = os.environ.get("ARC3_ACTION_ECHO", "0").strip().lower() in (
+    "1",
+    "true",
+    "yes",
+    "on",
+)
+# When an action changes nothing, the sandbox arms a per-snippet latch and the
+# NEXT action(...) call is refused. Off restores the older behaviour of letting
+# the snippet keep acting on a state its own beliefs no longer describe.
+STALE_STATE_BLOCK = os.environ.get(
+    "ARC3_STALE_STATE_BLOCK", "0"
+).strip().lower() in ("1", "true", "yes", "on")
+
+
+# Stops a batch at the first action that changes nothing inside the board, so a
+# wrong movement model costs one action rather than the whole sequence. It used
+# to be implied by ARC3_NOOP_GUARD_BORDER being positive, which conflated the
+# crop width with whether the guard runs at all - the border is also what
+# gameplay_changed means, and that is wanted independently. Off by default:
+# the guard converts actions into turns, which is a good trade only when tokens
+# are cheap relative to the action budget.
+# Level 1 pays the smallest share of the score - one part in N - and it is
+# where the model knows least, so it probes hardest: repeating an action to see
+# whether the repeat differs, retrying something that killed it to learn what
+# killed it. The guards read that as waste. Set to 2 to let level 1 run
+# unguarded and start guarding at level 2, where an action is worth more and
+# the model has a working model of the game.
+#
+# The batch no-op stop is deliberately NOT covered: it costs one action of a
+# planned sequence rather than refusing a deliberate probe, and a wrong
+# movement model is exactly as expensive on level 1 as anywhere. Nor is the
+# terminal latch, which stops a snippet acting after a level-up or a death -
+# that is a board that no longer exists, not a judgement about the action.
+GUARDS_FROM_LEVEL = max(1, int(os.environ.get("ARC3_GUARDS_FROM_LEVEL", "1") or 1))
+
+
+def _guards_active(level: Any) -> bool:
+    """Whether the per-action guards apply at this level."""
+    try:
+        return int(level) >= GUARDS_FROM_LEVEL
+    except (TypeError, ValueError):
+        return True  # unknown level: guard rather than not
+
+
+BATCH_NOOP_BLOCK = os.environ.get(
+    "ARC3_BATCH_NOOP_BLOCK", "0"
+).strip().lower() in ("1", "true", "yes", "on")
+
+
+_BLOCKED_VERDICTS = {
+    "known_noop": "BLOCKED (known no-op in this exact board state; never executed)",
+    "known_death": "BLOCKED (ended a previous attempt from this exact board state; never executed)",
+}
+
+
+def _action_verdict(item: dict) -> str:
+    """The one-word account of what an executed action did.
+
+    Shared with the per-action trace so the echo introduces no second
+    vocabulary for the model to learn."""
+    if item.get("game_over"):
+        return "GAME OVER (attempt ended immediately after this action)"
+    if item.get("run_complete") or item.get("done"):
+        return "RUN COMPLETE"
+    if item.get("level_completed"):
+        return "LEVEL COMPLETED"
+    changed = item.get("gameplay_changed")
+    if changed is False:
+        return (
+            "NO-OP (executed, no change inside the board area)"
+            if NEW_NOOP_WORDING
+            else "NO-OP (executed, changed nothing)"
+        )
+    if changed is True:
+        return "effect"
+    return "executed"
+
+
+def _build_action_echo(
+    requested_displays: list[str],
+    executed_payloads: list[dict],
+    stop_reason: str | None,
+) -> str:
+    """One line per action(...) call, for the snippet's own stdout.
+
+    A single action produces no per-action trace at all - the trace is built
+    only for batches, game overs and refusals - so the snippet's only account
+    of it arrives a turn later in the opener. Inside a loop that is too late to
+    branch on. This fills the gap at the moment of the call, in the trace's own
+    words, without repeating the trace for batches that already get one.
+    """
+    if not requested_displays:
+        return ""
+    if len(requested_displays) == 1:
+        if executed_payloads:
+            return f"[action] {requested_displays[0]} -> {_action_verdict(executed_payloads[0])}"
+        reason = f" | stop_reason: {stop_reason}" if stop_reason else ""
+        return f"[action] {requested_displays[0]} -> not executed{reason}"
+    ran = ", ".join(
+        str(item.get("action_display") or item.get("action_name") or "?")
+        for item in executed_payloads
+    )
+    line = (
+        f"[action] batch of {len(requested_displays)} actions, "
+        f"{len(executed_payloads)} executed"
+    )
+    if ran:
+        line += f": {ran}"
+    if stop_reason:
+        line += f" | stop_reason: {stop_reason}"
+    return line
+
+
+def _build_action_trace(
+    requested_displays: list[str],
+    executed_payloads: list[dict],
+    blocked_kind: str | None = None,
+) -> list[str]:
+    """One self-describing line per submitted action: index, name, verdict.
+
+    A guard refusal is called out at its own position rather than lumped in
+    with the generic SKIPPED verdict: the prompt tells the model to treat this
+    trace as the authoritative account of what happened, so a refusal that
+    reads as "skipped" would send it looking for an upstream cause that does
+    not exist. Positions AFTER the refusal are genuinely skipped and keep the
+    plain verdict.
+    """
+    trace: list[str] = []
+    blocked_position = len(executed_payloads) + 1 if blocked_kind else None
+    for position, name in enumerate(requested_displays, 1):
+        if blocked_position is not None and position == blocked_position:
+            trace.append(
+                f"{position}: {name} -> "
+                + _BLOCKED_VERDICTS.get(blocked_kind or "", "BLOCKED (never executed)")
+            )
+            continue
+        if position <= len(executed_payloads):
+            verdict = _action_verdict(executed_payloads[position - 1])
+        else:
+            verdict = "SKIPPED (never executed)"
+        trace.append(f"{position}: {name} -> {verdict}")
+    return trace
+
+
+def _interior_gameplay_changed(
+    previous_grid: list[list[int]] | None,
+    new_grid: list[list[int]] | None,
+    *,
+    border: int = NOOP_GUARD_BORDER_WIDTH,
+) -> bool:
+    """True when any cell outside the border frame differs between grids.
+
+    Falls back to a plain inequality check (treat as changed) when the guard
+    is disabled, grids are missing, shapes differ, or the border swallows the
+    whole grid — the guard must fail open.
+    """
+    if border <= 0 or previous_grid is None or new_grid is None:
+        return previous_grid != new_grid
+    rows = len(previous_grid)
+    cols = len(previous_grid[0]) if rows else 0
+    if rows != len(new_grid) or any(len(r) != cols for r in (previous_grid[0:1] + new_grid[0:1])):
+        return True
+    if rows <= 2 * border or cols <= 2 * border:
+        return previous_grid != new_grid
+    for r in range(border, rows - border):
+        prev_row = previous_grid[r]
+        new_row = new_grid[r]
+        if prev_row[border:cols - border] != new_row[border:cols - border]:
+            return True
+    return False
+
 _LOCAL_SERVER_PROCESS_ENV_KEYS = (
     "LOCAL_ANALYZER_API_KEY",
     "OPENAI_API_KEY",
@@ -90,6 +333,24 @@ def artifact_stem(value: str) -> str:
     return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
 
 
+def _animation_chain(
+    previous_grid: tuple[tuple[int, ...], ...],
+    state: taaf.game.GameState | None,
+) -> list[tuple[tuple[int, ...], ...]]:
+    """The pre-action board, then each distinct state the action passed through.
+
+    `state.frame` is only `raw.frame[-1]`; the rest of that list is the
+    animation. Runs of identical grids collapse here rather than at render
+    time, because a single action can return dozens of frames and they stay in
+    memory until the next action replaces them.
+    """
+    if state is None:
+        return []
+    return collapse_frames(
+        previous_grid, normalize_frames(getattr(state.raw, "frame", None))
+    )
+
+
 def _grid_from_state(state: taaf.game.GameState | None) -> tuple[tuple[int, ...], ...]:
     if state is None:
         return ()
@@ -113,7 +374,7 @@ def _engine_action_names(game: taaf.game.Game) -> list[str]:
             name = arcengine.GameAction.from_id(int(action_id)).name
         except Exception:
             continue
-        if name == "RESET":
+        if name == "RESET" and not reset_exposed():
             continue
         if name not in names:
             names.append(name)
@@ -127,6 +388,46 @@ def _model_mouse_action_data(
     return {"row": int(data.get("y", 0)), "col": int(data.get("x", 0))}
 
 
+def _guard_action_data(action: Any) -> dict[str, Any]:
+    """Action data in the row/col form the guard signature expects.
+
+    ACTION6 carries x/y from the engine; `_format_action_display` already
+    translates it the same way, so the ledger and the guard agree on what
+    identifies a click."""
+    data = dict(getattr(action, "data", None) or {})
+    if getattr(getattr(action, "id", None), "name", "") == "ACTION6":
+        return _model_mouse_action_data(data)
+    return data
+
+
+def _stale_repeat(stale_after: str, action: Any) -> bool:
+    """Is this the same action that just changed nothing?
+
+    Compared on the DISPLAY form, which carries mouse coordinates - a click
+    somewhere else after a no-op click is a different action, and the repeat
+    guard's override is meaningful for it.
+    """
+    return str(stale_after).strip() == _format_action_display(
+        action.id.name, dict(action.data)
+    ).strip()
+
+
+def _stale_detail(action: Any, stale_after: str) -> str:
+    return (
+        f"{_format_action_display(action.id.name, dict(action.data))} was not "
+        f"executed. The action immediately before it, {stale_after}, was a "
+        + ("no-op - it produced no change inside the board area - so positions "
+           if NEW_NOOP_WORDING else
+           "no-op - it changed nothing outside the board border - so positions ")
+        +
+        "and paths computed since then may be wrong, which is why execution "
+        "has been halted here. Re-read current_frame and last_action_call_result, "
+        "work out why nothing moved, then act on what you find. "
+        "(StaleStateActionError can be caught if a snippet has good reason to "
+        "keep acting through no-ops.)"
+    )
+
+
 def _format_action_display(
     action_name: str, action_data: dict[str, Any] | None = None
 ) -> str:
@@ -182,6 +483,14 @@ class _HarnessGameSession:
     analysis_step: int = 0
     last_engine_action: str | None = None
     token_baseline: int = 0
+    # token_baseline is a cursor: it is dragged forward after every step so the
+    # engine can be told what that step cost, and so cannot answer "how much
+    # has this game spent". This one is captured once and never moved.
+    game_token_baseline: int = 0
+    # The last executed action's animation, or None. One record, not a
+    # history: the model reads it via animation() in the turn that follows,
+    # and older animations are almost never what it is asking about.
+    animation_record: dict[str, Any] | None = None
     _viewer_events_flushed: int = field(default=0, init=False, repr=False)
 
     def current_frame(self) -> Frame:
@@ -243,6 +552,30 @@ class _HarnessGameSession:
             return None
         return max(0.1, min(candidates))
 
+    def token_limit_reached(self) -> bool:
+        """Whether this game has generated its allowance.
+
+        Read from the analyzer rather than summed per step: should_stop is
+        passed into the analyzer and runs at every loop boundary, so reading a
+        counter that only advances when an action executes would leave a game
+        that deliberates without acting running past its allowance. The
+        analyzer's accumulator advances on every response.
+
+        Measured against a baseline captured at game start, which is 0 for the
+        per-game analyzer the solver builds but need not be if an
+        analyzer_factory hands back a shared one.
+        """
+        # getattr, not attribute access: the notebook unpickles a solver whose
+        # __dict__ was captured before this field existed, and __setstate__ does
+        # a plain dict update - it applies no dataclass defaults. A bare access
+        # would raise AttributeError here on the first turn of every game, since
+        # should_stop() runs this every time.
+        limit = getattr(self.solver, "max_generated_tokens_per_game", None)
+        if limit is None or limit <= 0:
+            return False
+        spent = _analyzer_reported_tokens(self.analyzer) - self.game_token_baseline
+        return spent >= limit
+
     def should_stop(self) -> bool:
         run = self.game.game_run
         if run is None or run.state != "playing":
@@ -258,8 +591,15 @@ class _HarnessGameSession:
             and self.action_count >= self.solver.max_actions_per_game
         ):
             return True
+        if self.token_limit_reached():
+            return True
         return False
 
+    def _queue_position(self) -> int:
+        """Position in the dispatch order, used only to break ties in the
+        untrimmed priority band so games do not all queue at one value."""
+        return max(0, int(self.game_index)) * 100 + max(0, int(self.pass_index))
+
     def play(self) -> None:
         run = self.game.game_run
         assert run is not None, "TAAF starts games before invoking the solver."
@@ -267,12 +607,51 @@ class _HarnessGameSession:
         self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
         self.transcript_path.touch(exist_ok=True)
         self.token_baseline = _analyzer_reported_tokens(self.analyzer)
-        self.seed_initial_history()
-        self.write_runtime_state()
-        self._append_initial_viewer_event()
-        self.write_viewer_payload()
+        self.game_token_baseline = self.token_baseline
+        # A game that has not started yet sits in the top band, so every game
+        # gets going in queue order before any that has already trimmed comes
+        # back for a second turn.
+        gate = _priority_gate()
+        if gate is not None:
+            deadline = (
+                self.started_at + self.solver.max_runtime_s_per_game
+                if self.solver.max_runtime_s_per_game is not None else None
+            )
+            soft_remaining = self.solver.soft_time_remaining_seconds()
+            if soft_remaining is not None:
+                soft_deadline = time.monotonic() + soft_remaining
+                deadline = min(deadline, soft_deadline) if deadline is not None else soft_deadline
+            gate.configure_clock(self.started_at, deadline)
+            gate.acquire(_PRIORITY_UNTRIMMED_BASE - self._queue_position())
+        try:
+            self.seed_initial_history()
+            self.write_runtime_state()
+            self._append_initial_viewer_event()
+            self.write_viewer_payload()
+            self._play_inner()
+        finally:
+            if gate is not None:
+                gate.release()
+
+    def _play_inner(self) -> None:
+        run = self.game.game_run
+        if self.solver._claim_warmup_slot():
+            # Straight to the engine - _execute_auto_reset takes the same path
+            # and needs no analyzer, so this works with the endpoint still
+            # loading. One action off this game's budget buys the whole
+            # submission a deadline it would otherwise miss.
+            # warning, not info: _configure_logging only runs under run.py, so
+            # the notebook has no handler and falls back to logging's lastResort,
+            # which emits WARNING and above. An info line would vanish exactly
+            # where this message matters most.
+            log.warning(
+                "%s: warmup RESET issued before the analyzer is reachable",
+                run.game_id,
+            )
+            self._execute_auto_reset()
         try:
             retry_analysis_step: int | None = None
+            consecutive_failures = 0
             while not self.should_stop():
                 if (
                     _is_engine_game_over(self.game)
@@ -310,15 +689,42 @@ class _HarnessGameSession:
                 if result is None:
                     raise RuntimeError("Analyzer did not return a result.")
                 if result.retryable_failure:
+                    consecutive_failures += 1
+                    if (
+                        MAX_CONSECUTIVE_ANALYZER_FAILURES >= 0
+                        and consecutive_failures > MAX_CONSECUTIVE_ANALYZER_FAILURES
+                    ):
+                        log.error(
+                            "%s: analyzer failed %d times in a row; giving up on this "
+                            "game. The endpoint is probably down - nothing here can "
+                            "restart it.",
+                            run.game_id,
+                            consecutive_failures,
+                        )
+                        run.solver_note = (
+                            f"analyzer failed {consecutive_failures} times consecutively"
+                        )
+                        break
                     retry_analysis_step = analysis_step
                     if self.should_stop():
                         break
                     time.sleep(ANALYZER_RETRY_BACKOFF_SECONDS)
                     continue
 
+                # a turn that came back at all clears the run: the limit is for
+                # an endpoint that has stopped answering, not for a game that
+                # hits an occasional blip
+                consecutive_failures = 0
                 retry_analysis_step = None
                 if getattr(result, "yielded_control", False):
-                    retry_analysis_step = analysis_step
+                    # Reuse the step id only for a pure thinking continuation.
+                    # If this invocation already executed actions, the resumed
+                    # invocation opens with a fresh batch-summary prompt - a new
+                    # logical turn - and must get a fresh analysis_step, or the
+                    # viewer merges two turns into one card (and sums their
+                    # token usage).
+                    if not result.step_executed:
+                        retry_analysis_step = analysis_step
                     continue
                 if not result.step_executed:
                     continue
@@ -586,6 +992,14 @@ class _HarnessGameSession:
         }
 
     def step_env(self, arguments: dict[str, Any]) -> dict[str, Any]:
+        if str(arguments.get("query") or "").strip() == "animation":
+            # Routed through step_env so no analyzer-protocol signature has to
+            # change. Executes nothing and spends no action budget.
+            return {
+                "executed": False,
+                "query": "animation",
+                "record": self.animation_record if ANIMATION_ENABLED else None,
+            }
         requested_actions, error = self._normalize_actions(arguments)
         if error is not None or requested_actions is None:
             return self._error_payload(error or "Could not parse action request.")
@@ -601,10 +1015,84 @@ class _HarnessGameSession:
             for action in requested_actions
         ]
 
+        guard_hook = getattr(self, "action_guard_hook", None)
+        blocked_detail: str | None = None
+        blocked_kind: str | None = None
+
+        # Checked AFTER the per-position guards below, so a known death or a
+        # known no-op takes precedence: both name a specific action and offer an
+        # override, which is more actionable than a general staleness warning.
+        stale_after = (
+            arguments.get("stale_after")
+            if STALE_STATE_BLOCK and _guards_active(_level_number(self.game))
+            else None
+        )
+
         for batch_index, action in enumerate(requested_actions, start=1):
             if self.should_stop():
                 stop_reason = "stopped"
                 break
+            if (
+                stale_after
+                and batch_index == 1
+                and _stale_repeat(stale_after, action)
+            ):
+                # the model is re-issuing the action that just changed nothing;
+                # the staleness message is the correct one here, so take it
+                # before the repeat guard can claim the turn
+                blocked_kind = "stale_state"
+                blocked_detail = _stale_detail(action, stale_after)
+                stop_reason = blocked_kind
+                break
+            if guard_hook is not None:
+                # Per-position check: the hash is recomputed from the CURRENT
+                # board, so position N of a batch is judged against the state
+                # its predecessors actually produced - not the state the batch
+                # started in. This is what lets a fatal move be caught no
+                # matter which route reached it.
+                # The engine stores mouse coordinates as x/y; every consumer
+                # that identifies an action - the ledger, the action display,
+                # the guard signature - works in row/col. Passing action.data
+                # raw made every click share the signature "MOUSE" with no
+                # coordinates, so one no-op click poisoned clicking ANYWHERE
+                # from that board state. On a mouse-only game that is terminal:
+                # observed as 175 replies and one executed action in a run.
+                verdict = guard_hook(
+                    _grid_from_state(self.game.current_state),
+                    action.id.name,
+                    _guard_action_data(action),
+                    _level_number(self.game),
+                    [
+                        (later.id.name, _guard_action_data(later))
+                        for later in requested_actions[batch_index - 1:]
+                    ],
+                )
+                if verdict:
+                    blocked_kind, blocked_detail = verdict
+                    stop_reason = blocked_kind
+                    break
+            if stale_after and batch_index == 1:
+                # Ordering against the repeat guards above:
+                #
+                # Different action - the repeat guard wins. It names the action
+                # and its override is meaningful: the model may have reason to
+                # believe circumstances changed since that earlier visit.
+                #
+                # SAME action as the one that just no-opped - this message wins,
+                # and the check is placed before the guards for that case (see
+                # the early branch above). Telling a model that just watched X
+                # do nothing to "issue it again and it will execute" is advice
+                # to repeat a failure; worse, the guard's own override is
+                # already satisfied by this call, so without the swap the action
+                # would simply run with no message at all.
+                blocked_kind = "stale_state"
+                blocked_detail = _stale_detail(action, stale_after)
+                stop_reason = blocked_kind
+                break
+            # anything that reaches execution clears the flag: it can only
+            # happen when the check above declined to fire, and leaving a stale
+            # marker set behind an executed action would be a latent surprise
+            stale_after = None
             if action.id.value not in self.game.current_state.available_actions:
                 message = f"{_format_action_display(action.id.name, dict(action.data))} is not valid right now."
                 if executed_payloads:
@@ -636,8 +1124,35 @@ class _HarnessGameSession:
             if payload.get("level_completed"):
                 stop_reason = "level_completed"
                 break
+            if (
+                BATCH_NOOP_BLOCK
+                and NOOP_GUARD_BORDER_WIDTH > 0
+                and batch_size >= 2
+                and not payload.get("gameplay_changed", True)
+            ):
+                stop_reason = "no_op_action"
+                break
 
         if not executed_payloads:
+            if blocked_detail is not None:
+                # Nothing ran at all: report the refusal itself rather than a
+                # generic error, so the sandbox can raise the right exception.
+                return {
+                    **self._error_payload(blocked_detail),
+                    "executed": False,
+                    "stop_reason": blocked_kind,
+                    "stop_detail": blocked_detail,
+                    "requested_count": batch_size,
+                    "executed_count": 0,
+                    "stopped_early": True,
+                    "requested_actions": requested_displays,
+                    "skipped_actions": list(requested_displays),
+                    # the refused action is position 1 here, since nothing ran.
+                    # Omitting it left the next turn's opener saying "stopped
+                    # before None" - the later assignment for the partial-batch
+                    # case is never reached on this path.
+                    "blocked_action": requested_displays[0] if requested_displays else None,
+                }
             return self._error_payload("No action was executed.")
 
         final_payload = dict(executed_payloads[-1])
@@ -657,12 +1172,64 @@ class _HarnessGameSession:
         final_payload["stopped_early"] = len(executed_payloads) < batch_size
         if stop_reason is not None:
             final_payload["stop_reason"] = stop_reason
+        if final_payload["stopped_early"]:
+            final_payload["skipped_actions"] = requested_displays[len(executed_payloads):]
+        for item in executed_payloads:
+            if item.get("game_over"):
+                final_payload["fatal_action"] = str(item.get("action_display") or "")
+                break
+        if batch_size > 1 or final_payload.get("game_over") or blocked_kind:
+            final_payload["action_trace"] = _build_action_trace(
+                requested_displays, executed_payloads, blocked_kind
+            )
+            final_payload["gameplay_changed_per_action"] = [
+                bool(item.get("gameplay_changed", True)) for item in executed_payloads
+            ]
+        if ACTION_ECHO:
+            echo = _build_action_echo(requested_displays, executed_payloads, stop_reason)
+            if echo:
+                # consumed by the sandbox, which prints it and pops it: leaving
+                # it in the returned dict shows the model the same line twice,
+                # once on stdout and once in the payload it reads back
+                final_payload["action_echo"] = echo
+        if blocked_detail is not None:
+            final_payload["stop_detail"] = blocked_detail
+            final_payload["blocked_action"] = requested_displays[len(executed_payloads)]
+        if stop_reason == "no_op_action":
+            noop_index = len(executed_payloads)
+            final_payload["no_op_action_index"] = noop_index
+            final_payload["no_op_action"] = final_payload["executed_actions"][-1]
+            skipped = requested_displays[noop_index:]
+            final_payload["stop_detail"] = (
+                f"Action {noop_index} of {batch_size} "
+                + (
+                    f"({final_payload['executed_actions'][-1]!r}) EXECUTED but produced no "
+                    f"change inside the board area - only the outer {NOOP_GUARD_BORDER_WIDTH} "
+                    "cells, where a timer or remaining-steps bar usually sits, could have moved. "
+                    if NEW_NOOP_WORDING else
+                    f"({final_payload['executed_actions'][-1]!r}) EXECUTED but changed nothing "
+                    f"outside the {NOOP_GUARD_BORDER_WIDTH}-cell border (treated as HUD). "
+                )
+                +
+                f"Actions {noop_index + 1}-{batch_size} ({skipped!r}) were SKIPPED and did "
+                "not execute. Indices are 1-based positions in your submitted list. "
+                f"Net effect: only the first {max(0, noop_index - 1)} executed action(s) "
+                "changed the board; the final executed action moved nothing. Do NOT "
+                "update your position estimate by counting submitted moves - re-locate "
+                "the player from current_frame before planning. "
+                "The per-action verdicts are listed in action_trace. "
+                "This stop is about one action from one board state, not about batch "
+                "size - batching is still the right approach, and the stop is what keeps "
+                "a wrong movement model from costing the whole sequence."
+            )
         self.write_viewer_payload()
         return final_payload
 
     def _execute_auto_reset(self) -> None:
         action = arcengine.ActionInput(id=arcengine.GameAction.RESET, data={})
-        self._execute_action(action, batch_index=1, batch_size=1, generated_tokens=0)
+        self._execute_action(
+            action, batch_index=1, batch_size=1, generated_tokens=0, automatic=True
+        )
 
     def _execute_action(
         self,
@@ -672,6 +1239,7 @@ class _HarnessGameSession:
         batch_size: int,
         generated_tokens: int | None = None,
         flush_viewer_payload: bool = True,
+        automatic: bool = False,
     ) -> dict[str, Any]:
         previous_grid = _grid_from_state(self.game.current_state)
         previous_completed = int(self.game.current_state.levels_completed)
@@ -690,17 +1258,40 @@ class _HarnessGameSession:
             step=self.action_count,
             level=_level_number(self.game),
         )
-        self.history_entries.append(
-            HistoryEntry(action=action_display, frame=current_frame)
-        )
-        self.write_runtime_state()
 
         completed = int(new_state.levels_completed)
         reward = float(completed - previous_completed) / max(
             1.0, float(self.game.number_of_levels)
         )
         raw_state = new_state.raw.state
-        board_changed = previous_grid != _grid_from_state(new_state)
+        new_grid = _grid_from_state(new_state)
+        board_changed = previous_grid != new_grid
+        gameplay_changed = _interior_gameplay_changed(previous_grid, new_grid)
+        no_op = gameplay_changed is False
+        animation_summary = None
+        if ANIMATION_ENABLED and action.id.name != "RESET":
+            # The auto-reset after a death runs through here too. Letting it
+            # touch the record would clear the fatal action's animation before
+            # the model ever reads it - and a death is the case where those
+            # frames matter most, since the board it now sees is the restarted
+            # level and the death exists nowhere else.
+            chain = _animation_chain(previous_grid, new_state)
+            animation_summary = summarize_animation(chain)
+            # ONE record, replaced by every executed action - animated or not.
+            # A no-op that animated is the case this exists for, and the guards
+            # refuse the NEXT action without executing it, so the record still
+            # describes the animated one when the model reads the opener.
+            self.animation_record = (
+                {
+                    "action_display": action_display,
+                    "chain": chain,
+                    "summary": animation_summary,
+                    "step": self.action_count,
+                    "level": _level_number(self.game),
+                }
+                if animation_summary
+                else None
+            )
         level_completed = bool(
             new_state.just_won_level and raw_state != arcengine.GameState.WIN
         )
@@ -713,6 +1304,8 @@ class _HarnessGameSession:
             "state": raw_state.name,
             "valid_actions": to_model_actions(_engine_action_names(self.game)),
             "board_changed": board_changed,
+            "gameplay_changed": gameplay_changed,
+            "no_op": no_op,
             "done": raw_state == arcengine.GameState.WIN,
             "level_completed": level_completed,
             "game_over": raw_state == arcengine.GameState.GAME_OVER,
@@ -728,6 +1321,25 @@ class _HarnessGameSession:
             "batch_size": batch_size,
             **self.timing_payload(),
         }
+        if animation_summary:
+            payload["animation"] = animation_summary
+            # the raw chain, for rendering an image in the opener. Dropped by
+            # _compact_action_result, so it never reaches the model as text.
+            payload["animation_chain"] = chain
+        # Store the individual action before step_env folds results into a batch.
+        # Keep only compact metadata, without grids, animation chains, or clocks.
+        payload["automatic"] = automatic
+        result_keys = (
+            "executed", "action_num", "level", "score", "state", "valid_actions",
+            "board_changed", "gameplay_changed", "no_op", "done", "level_completed",
+            "game_over", "run_complete", "action_name", "action_data", "action_display",
+            "animation", "automatic",
+        )
+        individual_result = copy.deepcopy({k: payload[k] for k in result_keys if k in payload})
+        self.history_entries.append(
+            HistoryEntry(action=action_display, frame=current_frame, result=individual_result)
+        )
+        self.write_runtime_state()
         self._append_action_viewer_event(payload, current_frame)
         if flush_viewer_payload:
             self.write_viewer_payload()
@@ -743,6 +1355,12 @@ class HarnessSolver(Solver):
     analyzer_timeout: float | None = 120.0
     max_actions_per_game: int | None = None
     max_runtime_s_per_game: float | None = None
+    # Generated tokens a game may spend before it gives up, counted from the
+    # analyzer's own accumulator so it measures what the model produced rather
+    # than what was billed. A wall-clock limit means something different on
+    # every backend - and on a hosted API it moves hour to hour - so this is
+    # what makes an offline run comparable with a Kaggle one. None disables.
+    max_generated_tokens_per_game: int | None = None
     concurrency: int = 16
     save_request_logs: bool = False
     start_local_server: bool = False
@@ -816,6 +1434,7 @@ class HarnessSolver(Solver):
         state.pop("_local_servers", None)
         state.pop("_local_server_original_env", None)
         state.pop("_worker_pool", None)
+        state.pop("_warmup_lock", None)
         return state
 
     def __setstate__(self, state: dict[str, Any]) -> None:
@@ -829,6 +1448,7 @@ class HarnessSolver(Solver):
         self._local_servers = []
         self._local_server_original_env = {}
         self._worker_pool = None
+        self._warmup_lock = threading.Lock()
 
     def __deepcopy__(self, memo: dict[int, Any]) -> "HarnessSolver":
         cls = type(self)
@@ -845,6 +1465,8 @@ class HarnessSolver(Solver):
                 object.__setattr__(new, key, {})
             elif key == "_worker_pool":
                 object.__setattr__(new, key, None)
+            elif key == "_warmup_lock":
+                object.__setattr__(new, key, threading.Lock())
             else:
                 object.__setattr__(new, key, copy.deepcopy(value, memo))
         return new
@@ -879,6 +1501,12 @@ class HarnessSolver(Solver):
         )
 
     def _setup(self) -> None:
+        # Created here rather than in __init__ because the notebook unpickles a
+        # solver whose __dict__ predates these attributes, and __setstate__
+        # applies no dataclass defaults. _setup runs after unpickling, so the
+        # attributes always exist by the time a game asks for a slot.
+        self._warmup_remaining = WARMUP_ACTION_GAMES
+        self._warmup_lock = threading.Lock()
         if self.start_local_server:
             self._start_local_servers()
         self._worker_pool = ThreadPoolExecutor(
@@ -1178,6 +1806,23 @@ class HarnessSolver(Solver):
             return None
         return self._local_servers[int(game_index) % len(self._local_servers)]
 
+    def _claim_warmup_slot(self) -> bool:
+        """Whether this game should spend one action before the analyzer is up.
+
+        Claimed by whichever runners start first rather than by index: task
+        creation order and semaphore fairness both happen to give indices 0..N,
+        but neither is guaranteed by the API, and the cost of being wrong is a
+        game that never fires the action the deadline depends on.
+        """
+        lock = getattr(self, "_warmup_lock", None)
+        if lock is None:
+            return False
+        with lock:
+            if getattr(self, "_warmup_remaining", 0) <= 0:
+                return False
+            self._warmup_remaining -= 1
+            return True
+
     def _make_analyzer(
         self,
         game: taaf.game.Game,
@@ -1190,6 +1835,9 @@ class HarnessSolver(Solver):
             model=self.model,
             timeout=self.analyzer_timeout,
             save_request_logs=self.save_request_logs,
+            # queue position, so the untrimmed priority band has a deterministic
+            # order instead of every game tying at the same value
+            dispatch_index=index,
             api_key=(
                 local_server.api_key
                 if local_server is not None
diff --git a/ARC3-Inference/inference/utils/animation.py b/ARC3-Inference/inference/utils/animation.py
new file mode 100644
index 0000000..d2a1484
--- /dev/null
+++ b/ARC3-Inference/inference/utils/animation.py
@@ -0,0 +1,333 @@
+"""Animation awareness.
+
+``arcengine`` renders a frame after every internal ``step()``, so a single
+action can come back as a short animation. ``GameState.frame`` is only the last
+of those, and the harness consumed nothing else - so every intermediate frame
+was discarded before the agent could see it. On some games the whole effect of
+an action lives there: the board before and the board after are identical, and
+reading the final frame cannot possibly reveal what happened.
+
+Two shapes occur. Where cells change and change BACK, the information is
+genuinely hidden and worth spending prompt tokens on. Where every cell changes
+at most once, the frames are motion interpolation toward a state the final
+frame already shows, and there is nothing to find - those animations are
+reported nowhere, because a line about them would cost tokens to say that
+nothing is missing.
+"""
+from __future__ import annotations
+
+from collections import defaultdict
+from typing import Any, Sequence
+
+from inference.utils.grid_utils import ARC_COLOR_CHARS, format_grid_ascii
+
+Grid = tuple[tuple[int, ...], ...]
+
+# Retrieval budget.
+#
+# Why a diff timeline rather than the frames themselves: one 64x64 grid renders
+# as roughly 4,100 characters, and a single action can return dozens of frames.
+# Whole frames are never affordable against a ~1,024-token tool budget, so
+# retrieval collapses identical frames and reports only the cells that changed
+# between the rest. A single frame can still be read verbatim, cropped.
+# No step limit: the timeline crosses the pipe as data and the model prints
+# only what it chooses, so dropping steps served a prompt budget that no longer
+# applies. It also cost a real misreading - a step count reported as omitted
+# was read as "those steps had no changes", the exact inverse.
+ANIMATION_MAX_CELLS_PER_STEP = 24
+ANIMATION_MAX_TOTAL_CELLS = 80
+
+
+def normalize_frames(raw_frames: Any) -> list[Grid]:
+    """Engine frame list to hashable, comparable grids."""
+    frames: list[Grid] = []
+    for frame in raw_frames or ():
+        rows = frame.tolist() if hasattr(frame, "tolist") else frame
+        frames.append(tuple(tuple(int(cell) for cell in row) for row in rows or ()))
+    return frames
+
+
+def collapse_frames(before: Grid | None, frames: Sequence[Grid]) -> list[Grid]:
+    """The pre-action board, then each distinct state the action passed through.
+
+    Runs of identical grids collapse to one entry; non-adjacent repeats stay
+    separate, because a grid at positions 2 and 7 with something between them
+    is two distinct moments and the timeline should show both transitions.
+
+    ``before`` is prepended and is NOT optional. A cell showing X before the
+    action, Y during it and X again at the end arrives from the engine as
+    [Y, Y, X] - one transition, which would not register as transient. With the
+    board before it is [X, Y, Y, X], which gives two. Without this the very
+    case the feature exists for is invisible.
+    """
+    collapsed: list[Grid] = []
+    if before is not None:
+        collapsed.append(before)
+    for grid in frames:
+        if collapsed and collapsed[-1] == grid:
+            continue
+        collapsed.append(grid)
+    return collapsed
+
+
+def _transient_cells(chain: Sequence[Grid]) -> list[tuple[int, int]]:
+    """Cells whose value changes two or more times across the chain.
+
+    A cell that changes once and stays changed is visible in the final frame,
+    so the agent loses nothing by never seeing the frames between. A cell that
+    changes twice has shown something that is no longer there - a rejected
+    click, a consumed attempt, a sprite crossing a background cell. Those are
+    exactly the pixels no frame the agent can reach will ever contain.
+    """
+    changes: defaultdict[tuple[int, int], int] = defaultdict(int)
+    for before, after in zip(chain, chain[1:]):
+        rows = max(len(before), len(after))
+        for row in range(rows):
+            before_row = before[row] if row < len(before) else ()
+            after_row = after[row] if row < len(after) else ()
+            for col in range(max(len(before_row), len(after_row))):
+                old = before_row[col] if col < len(before_row) else None
+                new = after_row[col] if col < len(after_row) else None
+                if old != new:
+                    changes[(row, col)] += 1
+    return sorted(cell for cell, count in changes.items() if count >= 2)
+
+
+def summarize_animation(chain: Sequence[Grid]) -> dict[str, Any] | None:
+    """Compact metadata for one action, or ``None`` if it did not animate.
+
+    ``None`` for the single-frame case means ordinary actions carry no extra
+    tokens at all - the key is simply absent from the action result.
+
+    ``frames`` counts engine frames after collapsing and excludes the prepended
+    pre-action board, so it answers "how many distinct states did this action
+    pass through" rather than describing the internal representation.
+    """
+    if len(chain) <= 2:
+        return None
+    transient = _transient_cells(chain)
+    summary: dict[str, Any] = {
+        "frames": len(chain) - 1,
+        "transient_pixels": len(transient),
+    }
+    if transient:
+        rows = [cell[0] for cell in transient]
+        cols = [cell[1] for cell in transient]
+        # Inclusive [top, left, bottom, right]: enough to point at the region
+        # without spending tokens on a coordinate list.
+        summary["transient_bbox"] = [min(rows), min(cols), max(rows), max(cols)]
+    return summary
+
+
+def _bbox_phrase(bbox: Sequence[int] | None) -> str:
+    """Where the transient cells are, collapsing degenerate ranges.
+
+    A single cell reads as "at (20,30)", a single row or column as a span in
+    the other axis only; "rows 20-20, cols 30-30" is noise the model has to
+    parse before noticing it means one cell.
+    """
+    if not bbox:
+        return ""
+    top, left, bottom, right = bbox
+    if top == bottom and left == right:
+        return f" at ({top},{left})"
+    if top == bottom:
+        return f" in row {top}, cols {left}-{right}"
+    if left == right:
+        return f" in col {left}, rows {top}-{bottom}"
+    return f" around rows {top}-{bottom}, cols {left}-{right}"
+
+
+def describe_animation(
+    summary: dict[str, Any] | None,
+    *,
+    action_display: str | None,
+    gameplay_changed: bool | None,
+    game_over: bool = False,
+    timeline: bool = True,
+) -> str:
+    """The opener line, or "" when there is nothing worth saying.
+
+    Silent unless cells changed and changed back. Where every cell changed at
+    most once the final frame already shows the outcome, and a line saying so
+    would spend tokens to report that nothing is missing.
+    """
+    if not summary:
+        return ""
+    transient = int(summary.get("transient_pixels") or 0)
+    if transient <= 0:
+        return ""
+    frames = summary.get("frames")
+    cells = "1 cell changed and changed back" if transient == 1 else (
+        f"{transient} cells changed and changed back"
+    )
+    where = _bbox_phrase(summary.get("transient_bbox"))
+    name = f" (`{action_display}`)" if action_display else ""
+    # point at whichever accessor is actually available
+    where_to_look = (
+        "`last_animation_timeline`"
+        if timeline
+        else "`last_animation_frames`"
+    )
+
+    if game_over:
+        # Follows the BAR RULE block, which has just walked the model through
+        # history[-3] and history[-2]. This says what those two frames cannot
+        # contain, which is precisely what "transient" means.
+        return (
+            f"That fatal action animated over {frames} frames, and inside it "
+            f"{cells}{where}. Those cells are in neither "
+            f"`history[-3].frame` nor `history[-2].frame` - read {where_to_look} to see "
+            "what happened between them."
+        )
+    if gameplay_changed is False:
+        return (
+            f"The last executed action{name} animated over {frames} frames and left the "
+            f"board area unchanged, but it was not a no-op: {cells}{where}. "
+            "Whatever this action did is visible only there - "
+            f"read {where_to_look} before concluding it failed."
+        )
+    return (
+        f"The last executed action{name} animated over {frames} frames. "
+        f"{cells[0].upper()}{cells[1:]}{where}. {where_to_look} shows them."
+    )
+
+
+def _color_char(value: int | None) -> str:
+    if value is None:
+        return "?"
+    return ARC_COLOR_CHARS[max(0, min(15, int(value)))]
+
+
+def _cell_at(grid: Grid, row: int, col: int) -> int | None:
+    if row >= len(grid):
+        return None
+    line = grid[row]
+    return line[col] if col < len(line) else None
+
+
+def _diff_cells(before: Grid, after: Grid) -> list[tuple[int, int, int | None, int | None]]:
+    rows = max(len(before), len(after))
+    changes: list[tuple[int, int, int | None, int | None]] = []
+    for row in range(rows):
+        cols = max(
+            len(before[row]) if row < len(before) else 0,
+            len(after[row]) if row < len(after) else 0,
+        )
+        for col in range(cols):
+            old = _cell_at(before, row, col)
+            new = _cell_at(after, row, col)
+            if old != new:
+                changes.append((row, col, old, new))
+    return changes
+
+
+def _bbox_text(cells: Sequence[tuple[int, int, Any, Any]]) -> str:
+    rows = [cell[0] for cell in cells]
+    cols = [cell[1] for cell in cells]
+    return f"rows {min(rows)}-{max(rows)}, cols {min(cols)}-{max(cols)}"
+
+
+def _format_changes(
+    cells: Sequence[tuple[int, int, int | None, int | None]], budget: int
+) -> list[str]:
+    """Changed cells grouped by colour transition, one compact line each.
+
+    Strings rather than nested coordinate lists because the caller renders with
+    json.dumps(indent=2), which would put every [row, col] pair on four lines.
+    """
+    grouped: dict[str, list[str]] = {}
+    for row, col, old, new in cells[:budget]:
+        key = f"{_color_char(old)}>{_color_char(new)}"
+        grouped.setdefault(key, []).append(f"({row},{col})")
+    lines = [f"{key} @ {' '.join(points)}" for key, points in grouped.items()]
+    if len(cells) > budget:
+        lines.append(f"... {len(cells) - budget} further changed cells omitted")
+    return lines
+
+
+def build_animation_frames(record: dict[str, Any] | None) -> list[dict[str, Any]]:
+    """Every distinct state the action passed through, as FrameView payloads.
+
+    The pre-action board is dropped here: the model already has it as
+    `previous_frame`, and including it would make index 0 mean something
+    different from the step numbers in the timeline.
+
+    Sends the grid as well as the ascii so `segmentation` works on the far
+    side. That doubles the payload, but it crosses a pipe rather than the
+    prompt - what reaches the model is only what its own code prints.
+    """
+    if not record:
+        return []
+    chain = list(record.get("chain") or ())
+    if len(chain) <= 2:
+        return []
+    step = record.get("step") or 0
+    level = record.get("level") or 0
+    return [
+        {
+            "ascii": format_grid_ascii(grid),
+            "step": step,
+            "level": level,
+            "shape": (len(grid), max((len(row) for row in grid), default=0)),
+            "grid": [list(row) for row in grid],
+        }
+        for grid in chain[1:]
+    ]
+
+
+def build_animation_view(record: dict[str, Any] | None) -> dict[str, Any]:
+    """The diff timeline: which cells changed at each step.
+
+    The cheap first look. Reading a frame verbatim is no longer this
+    function's job - `last_transition.frames()` hands back FrameView objects
+    and the model crops their ascii the same way it crops current_frame, with
+    its own helpers rather than a region parameter it has to guess.
+    """
+    empty = {
+        "error": (
+            "No animation is available: the last executed action returned a single frame."
+        )
+    }
+    if not record:
+        return empty
+    chain: list[Grid] = list(record.get("chain") or ())
+    if len(chain) <= 2:
+        return empty
+
+    header: dict[str, Any] = {
+        "action": record.get("action_display"),
+        # excludes the prepended pre-action board, matching the metadata
+        "frames": len(chain) - 1,
+    }
+
+    steps: list[dict[str, Any]] = []
+    for position in range(1, len(chain)):
+        cells = _diff_cells(chain[position - 1], chain[position])
+        if cells:
+            steps.append({"_position": position, "_cells": cells})
+
+    rendered: list[dict[str, Any]] = []
+    remaining = ANIMATION_MAX_TOTAL_CELLS
+    for step in steps:
+        cells = step["_cells"]
+        entry: dict[str, Any] = {
+            "step": step["_position"],
+            "changed": len(cells),
+            "bbox": _bbox_text(cells),
+        }
+        budget = min(ANIMATION_MAX_CELLS_PER_STEP, max(0, remaining))
+        if len(cells) > ANIMATION_MAX_CELLS_PER_STEP or budget == 0:
+            # Too broad to enumerate usefully: a colour-transition census says
+            # more per token than an arbitrarily truncated cell list.
+            census: dict[str, int] = {}
+            for _, _, old, new in cells:
+                key = f"{_color_char(old)}>{_color_char(new)}"
+                census[key] = census.get(key, 0) + 1
+            entry["transitions"] = census
+        else:
+            entry["changes"] = _format_changes(cells, budget)
+            remaining -= min(len(cells), budget)
+        rendered.append(entry)
+
+    return {**header, "steps": rendered}
diff --git a/ARC3-Inference/inference/utils/frame_diff.py b/ARC3-Inference/inference/utils/frame_diff.py
new file mode 100644
index 0000000..a49b067
--- /dev/null
+++ b/ARC3-Inference/inference/utils/frame_diff.py
@@ -0,0 +1,307 @@
+"""Object-level frame diff core, shared by the sandbox runtime and the host.
+
+This module must stay dependency-free (stdlib only, no imports): its source is
+injected verbatim into the sandbox bootstrap template (see
+python_tool_sandbox.py), which runs under ``python -I -S`` and cannot import
+project modules. The host imports it normally for the auto frame-diff feature.
+"""
+
+
+def _node_topleft(node):
+    points = node.get("boundary") or []
+    if not points:
+        return (0, 0)
+    return (min(p[0] for p in points), min(p[1] for p in points))
+
+def _hungarian(cost):
+    # Minimum-cost assignment (Jonker-Volgonant style with potentials).
+    # cost: n x m matrix with n <= m. Returns list row -> column.
+    n = len(cost)
+    m = len(cost[0]) if n else 0
+    INF = float("inf")
+    u = [0.0] * (n + 1)
+    v = [0.0] * (m + 1)
+    p = [0] * (m + 1)      # p[j] = row assigned to column j (1-based; 0 = none)
+    way = [0] * (m + 1)
+    for i in range(1, n + 1):
+        p[0] = i
+        j0 = 0
+        minv = [INF] * (m + 1)
+        used = [False] * (m + 1)
+        while True:
+            used[j0] = True
+            i0 = p[j0]
+            delta = INF
+            j1 = 0
+            for j in range(1, m + 1):
+                if used[j]:
+                    continue
+                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
+                if cur < minv[j]:
+                    minv[j] = cur
+                    way[j] = j0
+                if minv[j] < delta:
+                    delta = minv[j]
+                    j1 = j
+            for j in range(m + 1):
+                if used[j]:
+                    u[p[j]] += delta
+                    v[j] -= delta
+                else:
+                    minv[j] -= delta
+            j0 = j1
+            if p[j0] == 0:
+                break
+        while True:
+            j1 = way[j0]
+            p[j0] = p[j1]
+            j0 = j1
+            if j0 == 0:
+                break
+    assignment = [-1] * n
+    for j in range(1, m + 1):
+        if p[j]:
+            assignment[p[j] - 1] = j - 1
+    return assignment
+
+def _node_bbox(node):
+    points = node.get("boundary") or []
+    if not points:
+        return (0, 0, 0, 0)
+    rows = [p[0] for p in points]; cols = [p[1] for p in points]
+    return (min(rows), min(cols), max(rows), max(cols))
+
+def _node_center2(node):
+    # bbox center in half-cell units (ints): (r0+r1, c0+c1)
+    points = node.get("boundary") or []
+    if not points:
+        return (0, 0)
+    rows = [p[0] for p in points]; cols = [p[1] for p in points]
+    return (min(rows) + max(rows), min(cols) + max(cols))
+
+def compute_frame_diff(grid_a, grid_b, nodes_a, nodes_b, max_group_match=None):
+    # Object-level diff between two same-shaped grids with precomputed
+    # segmentation nodes. Returns only what changed; unchanged objects are
+    # omitted. When max_group_match is set, identical-hash groups larger than
+    # it skip pairwise matching and report plain appeared/disappeared entries
+    # (guards against O(n^3) assignment on e.g. hundreds of identical pixels).
+    changed = set()
+    for r, (row_a, row_b) in enumerate(zip(grid_a, grid_b)):
+        if row_a == row_b:
+            continue
+        for c, (va, vb) in enumerate(zip(row_a, row_b)):
+            if va != vb:
+                changed.add((r, c))
+    if not changed:
+        return {"changed_cell_count": 0, "moved": [], "rotated": [],
+                "appeared": [], "disappeared": [],
+                "changed_color": [], "resized": []}
+
+    def touched_nodes(nodes):
+        touched = []
+        for node in nodes:
+            points = node.get("boundary") or []
+            if not points:
+                continue
+            r0 = min(p[0] for p in points); r1 = max(p[0] for p in points)
+            c0 = min(p[1] for p in points); c1 = max(p[1] for p in points)
+            # dilate by 1 so objects that shrank/grew still register as
+            # touched by the cells they vacated/claimed
+            if any(r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1 for (r, c) in changed):
+                touched.append(node)
+        return touched
+
+    before_nodes = touched_nodes(nodes_a)
+    after_nodes = touched_nodes(nodes_b)
+
+    moved, appeared, disappeared = [], [], []
+    rotated = []
+
+    def _classify_match(node_a, node_b):
+        rot_a = node_a.get("rotation")
+        rot_b = node_b.get("rotation")
+        symmetry = node_a.get("rotational_symmetry") or 1
+        rotated_by = None
+        if rot_a is not None and rot_b is not None:
+            modulus = 180 if symmetry == 2 else 360
+            delta = (rot_b - rot_a) % modulus
+            if delta:
+                rotated_by = delta
+        pa, pb = _node_topleft(node_a), _node_topleft(node_b)
+        base_entry = {"color": node_a.get("color"),
+                      "pixels": node_a.get("pixels"),
+                      "hash": node_a.get("hash")}
+        if rotated_by is not None:
+            ca, cb = _node_center2(node_a), _node_center2(node_b)
+            in_place = (abs(ca[0] - cb[0]) <= 2 and abs(ca[1] - cb[1]) <= 2)
+            base_entry["rotated_by"] = rotated_by
+            if in_place:
+                base_entry["at"] = list(pb)
+                rotated.append(base_entry)
+            else:
+                base_entry["from"] = list(pa); base_entry["to"] = list(pb)
+                moved.append(base_entry)
+        elif pa != pb:
+            base_entry["from"] = list(pa); base_entry["to"] = list(pb)
+            moved.append(base_entry)
+        # same hash, same position, same rotation: unchanged -> omit
+
+    groups_before = {}
+    for node in before_nodes:
+        groups_before.setdefault(node.get("hash"), []).append(node)
+    groups_after = {}
+    for node in after_nodes:
+        groups_after.setdefault(node.get("hash"), []).append(node)
+
+    for key, group_a in groups_before.items():
+        group_b = groups_after.pop(key, [])
+        if (
+            max_group_match is not None
+            and max(len(group_a), len(group_b)) > max_group_match
+        ):
+            # too many identical objects to match affordably
+            for node_a in group_a:
+                disappeared.append({"color": node_a.get("color"),
+                                    "pixels": node_a.get("pixels"),
+                                    "hash": node_a.get("hash"),
+                                    "shape_hash": node_a.get("shape_hash"),
+                                    "at": list(_node_topleft(node_a)),
+                                    "bbox": list(_node_bbox(node_a))})
+            for node_b in group_b:
+                appeared.append({"color": node_b.get("color"),
+                                 "pixels": node_b.get("pixels"),
+                                 "hash": node_b.get("hash"),
+                                 "shape_hash": node_b.get("shape_hash"),
+                                 "at": list(_node_topleft(node_b)),
+                                 "bbox": list(_node_bbox(node_b))})
+            continue
+        if not group_b:
+            for node_a in group_a:
+                disappeared.append({"color": node_a.get("color"),
+                                    "pixels": node_a.get("pixels"),
+                                    "hash": node_a.get("hash"),
+                                    "shape_hash": node_a.get("shape_hash"),
+                                    "at": list(_node_topleft(node_a)),
+                                    "bbox": list(_node_bbox(node_a))})
+            continue
+        # optimal assignment over bbox-center distances; rows = smaller side
+        transposed = len(group_a) > len(group_b)
+        rows, cols = (group_b, group_a) if transposed else (group_a, group_b)
+        cost = []
+        for row_node in rows:
+            cr = _node_center2(row_node)
+            cost.append([
+                abs(cr[0] - _node_center2(col_node)[0])
+                + abs(cr[1] - _node_center2(col_node)[1])
+                for col_node in cols
+            ])
+        assignment = _hungarian(cost)
+        matched_cols = set()
+        for row_index, col_index in enumerate(assignment):
+            if col_index < 0:
+                continue
+            matched_cols.add(col_index)
+            if transposed:
+                _classify_match(cols[col_index], rows[row_index])
+            else:
+                _classify_match(rows[row_index], cols[col_index])
+        if transposed:
+            for col_index, node_a in enumerate(cols):
+                if col_index not in matched_cols:
+                    disappeared.append({"color": node_a.get("color"),
+                                        "pixels": node_a.get("pixels"),
+                                        "hash": node_a.get("hash"),
+                                    "shape_hash": node_a.get("shape_hash"),
+                                        "at": list(_node_topleft(node_a)),
+                                        "bbox": list(_node_bbox(node_a))})
+        else:
+            for col_index, node_b in enumerate(cols):
+                if col_index not in matched_cols:
+                    appeared.append({"color": node_b.get("color"),
+                                     "pixels": node_b.get("pixels"),
+                                     "hash": node_b.get("hash"),
+                             "shape_hash": node_b.get("shape_hash"),
+                                     "at": list(_node_topleft(node_b)),
+                                     "bbox": list(_node_bbox(node_b))})
+            continue
+        # transposed: every after-node (a row) got matched; nothing appeared
+    for group_b in groups_after.values():
+        for node_b in group_b:
+            appeared.append({"color": node_b.get("color"),
+                             "pixels": node_b.get("pixels"),
+                             "hash": node_b.get("hash"),
+                             "shape_hash": node_b.get("shape_hash"),
+                             "at": list(_node_topleft(node_b)),
+                             "bbox": list(_node_bbox(node_b))})
+    # Phase A - changed_color: EXACT pairs only. Same position, byte-
+    # identical shape (color-independent pose signature), different
+    # color. This asserts "these exact cells changed color" and nothing
+    # more; fuzzy transformation claims are deliberately not made.
+    changed_color = []
+    used_gone, used_new = set(), set()
+    for gi, gone in enumerate(disappeared):
+        for ni, new in enumerate(appeared):
+            if ni in used_new:
+                continue
+            if (gone["at"] == new["at"]
+                    and gone.get("shape_hash")
+                    and gone.get("shape_hash") == new.get("shape_hash")
+                    and gone["color"] != new["color"]):
+                changed_color.append({"at": gone["at"],
+                                      "pixels": gone["pixels"],
+                                      "before_color": gone["color"],
+                                      "after_color": new["color"],
+                                      "before_hash": gone.get("hash"),
+                                      "after_hash": new.get("hash")})
+                used_gone.add(gi); used_new.add(ni)
+                break
+    disappeared = [entry for index, entry in enumerate(disappeared) if index not in used_gone]
+    appeared = [entry for index, entry in enumerate(appeared) if index not in used_new]
+
+    # Phase B - resized: same-color pairs whose bboxes substantially
+    # overlap (overlap area / smaller bbox area >= 0.5); greedy
+    # one-to-one by descending overlap then size similarity.
+    resized = []
+
+    def _bbox_area(b):
+        return max(0, b[2] - b[0] + 1) * max(0, b[3] - b[1] + 1)
+
+    def _overlap_ratio(b1, b2):
+        inter_r = min(b1[2], b2[2]) - max(b1[0], b2[0]) + 1
+        inter_c = min(b1[3], b2[3]) - max(b1[1], b2[1]) + 1
+        if inter_r <= 0 or inter_c <= 0:
+            return 0.0
+        smaller = min(_bbox_area(b1), _bbox_area(b2))
+        return (inter_r * inter_c) / smaller if smaller else 0.0
+
+    candidates = []
+    for gi, gone in enumerate(disappeared):
+        for ni, new in enumerate(appeared):
+            if gone["color"] != new["color"]:
+                continue
+            bg = gone.get("bbox") or (0, 0, -1, -1)
+            bn = new.get("bbox") or (0, 0, -1, -1)
+            ratio = _overlap_ratio(bg, bn)
+            if ratio >= 0.5:
+                area_g, area_n = _bbox_area(bg), _bbox_area(bn)
+                similarity = (min(area_g, area_n) / max(area_g, area_n)) if max(area_g, area_n) else 0.0
+                candidates.append((ratio, similarity, gi, ni))
+    candidates.sort(key=lambda item: (-item[0], -item[1]))
+    used_gone, used_new = set(), set()
+    for ratio, similarity, gi, ni in candidates:
+        if gi in used_gone or ni in used_new:
+            continue
+        used_gone.add(gi); used_new.add(ni)
+        gone, new = disappeared[gi], appeared[ni]
+        resized.append({"at_before": gone["at"], "at_after": new["at"],
+                        "color": gone["color"],
+                        "before_pixels": gone["pixels"],
+                        "after_pixels": new["pixels"],
+                        "before_hash": gone.get("hash"),
+                        "after_hash": new.get("hash")})
+    disappeared = [entry for index, entry in enumerate(disappeared) if index not in used_gone]
+    appeared = [entry for index, entry in enumerate(appeared) if index not in used_new]
+    return {"changed_cell_count": len(changed), "moved": moved,
+            "rotated": rotated,
+            "appeared": appeared, "disappeared": disappeared,
+            "changed_color": changed_color, "resized": resized}
diff --git a/ARC3-Inference/inference/utils/openai_compat.py b/ARC3-Inference/inference/utils/openai_compat.py
index 157e113..2fadb51 100644
--- a/ARC3-Inference/inference/utils/openai_compat.py
+++ b/ARC3-Inference/inference/utils/openai_compat.py
@@ -1,6 +1,8 @@
 """Helpers for provider-specific OpenAI-compatible requests."""
 from __future__ import annotations
 
+import json
+
 from typing import Any
 
 
@@ -46,22 +48,46 @@ def build_chat_payload(
     tools: list[dict[str, Any]] | None = None,
     tool_choice: str | None = None,
     seed: int | None = None,
+    priority: int | None = None,
+    stream: bool = False,
 ) -> dict[str, Any]:
     payload: dict[str, Any] = {
         "model": model,
         "messages": messages,
-        "stream": False,
+        "stream": bool(stream),
         "temperature": temperature,
         "top_p": top_p,
     }
     if max_tokens is not None:
         payload["max_tokens"] = max_tokens
+    if priority is not None:
+        # scheduling hint, ignored by servers that do not implement it
+        payload["priority"] = int(priority)
     if tools:
         payload["tools"] = tools
         if tool_choice:
             payload["tool_choice"] = tool_choice
 
     normalized = normalize_provider(provider)
+    if stream:
+        payload["stream_options"] = {"include_usage": True}
+    if normalized == "openrouter":
+        import os
+        if not payload["stream"] and os.environ.get(
+            "OPENROUTER_STREAM", ""
+        ).strip().lower() not in ("0", "false", "no", "off"):
+            payload["stream"] = True
+        if payload["stream"]:
+            payload["stream_options"] = {"include_usage": True}
+        payload["reasoning"] = {"enabled": bool(thinking)}
+        _order = os.environ.get("OPENROUTER_PROVIDER_ORDER", "").strip()
+        if _order:
+            payload["provider"] = {
+                "order": [s.strip() for s in _order.split(",") if s.strip()],
+                "allow_fallbacks": os.environ.get(
+                    "OPENROUTER_ALLOW_FALLBACKS", ""
+                ).strip().lower() in ("1", "true", "yes"),
+            }
     if normalized == "vllm":
         if top_k > 0:
             payload["top_k"] = top_k
@@ -70,3 +96,101 @@ def build_chat_payload(
             payload["seed"] = seed
 
     return payload
+
+
+def assemble_streamed_chat_response(lines) -> dict[str, Any]:
+    """Consume an SSE chat-completion stream and rebuild the non-streaming
+    response shape: {"choices": [{"message", "finish_reason"}], "usage"}.
+
+    Tolerates keep-alive comments (": ..."), event: lines, usage-only chunks,
+    and mid-stream error events. If an error arrives and no assistant content
+    was produced, returns {"error": ..., "choices": []} so callers surface it.
+    """
+    content_parts: list[str] = []
+    reasoning_parts: list[str] = []
+    tool_calls_by_index: dict[int, dict[str, Any]] = {}
+    finish_reason = ""
+    usage: dict[str, Any] | None = None
+    error: Any = None
+    provider_name = [""]
+
+    for raw_line in lines:
+        if raw_line is None:
+            continue
+        line = raw_line.decode("utf-8", "replace") if isinstance(raw_line, bytes) else str(raw_line)
+        line = line.strip()
+        if not line or line.startswith(":") or line.startswith("event:"):
+            continue
+        if line.startswith("data:"):
+            line = line[5:].strip()
+        if line == "[DONE]":
+            break
+        try:
+            chunk = json.loads(line)
+        except json.JSONDecodeError:
+            continue
+        if not isinstance(chunk, dict):
+            continue
+        if chunk.get("error"):
+            error = chunk["error"]
+            continue
+        if isinstance(chunk.get("usage"), dict):
+            usage = chunk["usage"]
+        if chunk.get("provider") and not provider_name[0]:
+            provider_name[0] = str(chunk["provider"])
+        for choice in chunk.get("choices") or []:
+            if not isinstance(choice, dict):
+                continue
+            if choice.get("finish_reason"):
+                finish_reason = str(choice["finish_reason"])
+            delta = choice.get("delta") or {}
+            if not isinstance(delta, dict):
+                continue
+            if isinstance(delta.get("content"), str):
+                content_parts.append(delta["content"])
+            if isinstance(delta.get("reasoning"), str):
+                reasoning_parts.append(delta["reasoning"])
+            elif isinstance(delta.get("reasoning_content"), str):
+                reasoning_parts.append(delta["reasoning_content"])
+            elif isinstance(delta.get("reasoning_details"), list):
+                for detail in delta["reasoning_details"]:
+                    if isinstance(detail, dict) and isinstance(detail.get("text"), str):
+                        reasoning_parts.append(detail["text"])
+            for fragment in delta.get("tool_calls") or []:
+                if not isinstance(fragment, dict):
+                    continue
+                index = int(fragment.get("index", 0) or 0)
+                entry = tool_calls_by_index.setdefault(
+                    index,
+                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
+                )
+                if fragment.get("id"):
+                    entry["id"] = str(fragment["id"])
+                if fragment.get("type"):
+                    entry["type"] = str(fragment["type"])
+                function = fragment.get("function") or {}
+                if isinstance(function, dict):
+                    if function.get("name"):
+                        entry["function"]["name"] = str(function["name"])
+                    if isinstance(function.get("arguments"), str):
+                        entry["function"]["arguments"] += function["arguments"]
+
+    produced = bool(content_parts or reasoning_parts or tool_calls_by_index)
+    if error is not None and not produced:
+        return {"error": error, "choices": []}
+
+    message: dict[str, Any] = {"role": "assistant", "content": "".join(content_parts)}
+    if reasoning_parts:
+        message["reasoning"] = "".join(reasoning_parts)
+    if tool_calls_by_index:
+        message["tool_calls"] = [tool_calls_by_index[i] for i in sorted(tool_calls_by_index)]
+    result: dict[str, Any] = {
+        "choices": [{"message": message, "finish_reason": finish_reason}],
+    }
+    if usage is not None:
+        result["usage"] = usage
+    if error is not None:
+        result["error"] = error
+    if provider_name[0]:
+        result["provider"] = provider_name[0]
+    return result
diff --git a/ARC3-Inference/inference/utils/retained_functions.py b/ARC3-Inference/inference/utils/retained_functions.py
new file mode 100644
index 0000000..66aa5a2
--- /dev/null
+++ b/ARC3-Inference/inference/utils/retained_functions.py
@@ -0,0 +1,301 @@
+"""Conservative source-only function retention; no snippet state is serialized.
+
+Also embedded into the isolated Python runner. Keep this module stdlib-only.
+"""
+import ast
+import dis
+import sys
+import types
+
+
+def _import_statements(nodes):
+    """One explicit absolute import per binding; never replay wildcard imports."""
+    statements = {}
+    for node in nodes:
+        if isinstance(node, ast.Import):
+            for alias in node.names:
+                binding = alias.asname or alias.name.split('.')[0]
+                statements[binding] = ast.unparse(ast.Import(names=[alias]))
+        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
+            for alias in node.names:
+                if alias.name == '*':
+                    raise ValueError('wildcard imports are not retained; use explicit imports inside the function')
+                statements[alias.asname or alias.name] = ast.unparse(
+                    ast.ImportFrom(module=node.module, names=[alias], level=0))
+        else:
+            raise ValueError('only explicit absolute imports can accompany retained functions')
+    return statements
+
+
+def _source_parts(source, retain_imports=True):
+    tree = ast.parse(source, '<retained>', 'exec')
+    if (not tree.body or not isinstance(tree.body[-1], ast.FunctionDef)
+            or (not retain_imports and len(tree.body) != 1)):
+        raise ValueError('only top-level function definitions can be retained')
+    imports = _import_statements(tree.body[:-1])
+    return tree.body[-1], imports
+
+
+def function_signature(source):
+    node, _ = _source_parts(source)
+    return node.name + '(' + ast.unparse(node.args) + ')'
+
+
+def _function_record(source, retain_imports=False):
+    node, imports = _source_parts(source, retain_imports)
+    if getattr(node, "type_params", []):
+        raise ValueError("generic type parameters are not retained")
+    if node.decorator_list:
+        raise ValueError('decorated functions are not retained')
+    args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
+    args += [a for a in (node.args.vararg, node.args.kwarg) if a is not None]
+    if node.returns is not None or any(a.annotation is not None for a in args):
+        raise ValueError('use an unannotated definition for retention')
+    # Restoration must never evaluate model calls or snapshot a frame via defaults.
+    for default in [*node.args.defaults, *node.args.kw_defaults]:
+        if default is not None:
+            try:
+                ast.literal_eval(default)
+            except Exception:
+                raise ValueError('defaults must be literals; pass changing data as arguments') from None
+    tree = ast.Module(body=[node], type_ignores=[])
+    code = next(c for c in compile(tree, '<retained>', 'exec').co_consts
+                if isinstance(c, types.CodeType))
+    reads = set()
+    pending = [code]
+    while pending:
+        current = pending.pop()
+        for instruction in dis.get_instructions(current):
+            if instruction.opname in ('LOAD_GLOBAL', 'LOAD_NAME'):
+                reads.add(instruction.argval)
+            elif instruction.opname in ('STORE_GLOBAL', 'DELETE_GLOBAL'):
+                raise ValueError('functions that assign global variables are not retained')
+        pending.extend(c for c in current.co_consts if isinstance(c, types.CodeType))
+    return node.name, source, reads - imports.keys()
+
+
+def _rejection(name, reason, hint=None, **metadata):
+    result = {'name': name, 'reason': reason, **metadata}
+    if hint:
+        result['hint'] = hint
+    return result
+
+
+def _validation_hint(reason):
+    if 'decorat' in reason:
+        return 'Use a plain top-level def without decorators if you want to reuse it.'
+    if 'annotat' in reason:
+        return 'Remove parameter and return annotations if you want to retain this function.'
+    if 'default' in reason:
+        return 'Use literal defaults and pass changing values explicitly as arguments.'
+    if 'global variables' in reason:
+        return 'Return updated values instead of assigning global variables.'
+    if 'import' in reason:
+        return 'Put explicit, sandbox-supported imports inside the function.'
+    return 'Use a plain top-level def if you want to retain this function.'
+
+
+def prepare_functions(sources, available, *, retain_imports=False):
+    """Validate sources without executing them; drop invalid dependency chains."""
+    records, rejected = {}, []
+    for source in sources:
+        name = '<unknown>'
+        try:
+            parsed = ast.parse(source)
+            if (parsed.body and isinstance(parsed.body[-1], ast.FunctionDef)
+                    and (retain_imports or len(parsed.body) == 1)):
+                name = parsed.body[-1].name
+                records.pop(name, None)
+            name, source, reads = _function_record(source, retain_imports)
+            _, imports = _source_parts(source)
+            if imports.keys() & set(available):
+                raise ValueError('import alias conflicts with a provided global or builtin')
+            records[name] = (source, reads)
+        except Exception as exc:
+            reason = str(exc)[:180]
+            rejected.append(_rejection(name, reason, _validation_hint(reason)))
+    # Import bundles must not overwrite another retained function at restore.
+    function_names = set(records)
+    for name, (source, _) in list(records.items()):
+        _, imports = _source_parts(source)
+        if imports.keys() & function_names:
+            del records[name]
+            rejected.append(_rejection(name, 'import alias conflicts with a retained function',
+                                       'Choose a different import alias or function name.'))
+    # A wrapper must not survive its unavailable dependency. Re-run to a fixed point.
+    while True:
+        dropped = []
+        for name, (source, reads) in records.items():
+            missing = reads - set(available) - records.keys()
+            if missing:
+                dropped.append(name)
+                names = ', '.join(sorted(missing))
+                rejected.append(_rejection(
+                    name, 'unavailable dependencies: ' + names,
+                    'Pass ' + names + ' as arguments, or put their imports inside the function.',
+                    missing_dependencies=sorted(missing)))
+        if not dropped:
+            break
+        for name in dropped:
+            del records[name]
+    return {name: value[0] for name, value in records.items()}, rejected
+
+
+def restore_functions(sources, namespace, available, *, retain_imports=False,
+                      repair_hints=False):
+    """Restore definitions and explicit imports through the sandbox importer.
+
+    Never replay assignments, expressions, decorators, or previous actions.
+    """
+    kept, rejected = prepare_functions(sources, available, retain_imports=retain_imports)
+    failed = set()
+    for name, source in kept.items():
+        before = dict(namespace)
+        try:
+            if name in available:
+                raise ValueError('name conflicts with a provided global or builtin')
+            exec(compile(source, '<retained>', 'exec'), namespace, namespace)
+        except Exception:
+            # A multi-import bundle can fail partway through. Restore bindings
+            # so one failed helper does not replace another helper's globals.
+            namespace.clear()
+            namespace.update(before)
+            failed.add(name)
+            reason = ('could not restore this definition or its imports' if repair_hints else
+                      'could not restore this definition; redefine it if needed')
+            rejected.append(_rejection(name, reason,
+                                       'Check its imports are supported by the sandbox, then redefine it if needed.'))
+    survivors, dependencies = prepare_functions(
+        [source for name, source in kept.items() if name not in failed], available,
+        retain_imports=retain_imports)
+    rejected.extend(dependencies)
+    for name in kept.keys() - survivors.keys():
+        if name not in available:
+            namespace.pop(name, None)
+    return survivors, rejected
+
+
+def _matching_imports(nodes, namespace):
+    """Recognize live import bindings without importing or evaluating anything.
+
+    Only module bindings and callable/immutable exports are eligible. Ordinary
+    snippet assignments and mutable imported data are never serialized. Looking
+    in module dictionaries avoids invoking __getattr__ during bookkeeping.
+    """
+    matches, declared = {}, {}
+    for node in nodes:
+        if not isinstance(node, (ast.Import, ast.ImportFrom)):
+            continue
+        try:
+            statements = _import_statements([node])
+        except ValueError:
+            continue
+        declared.update(statements)
+        for binding, statement in statements.items():
+            single = ast.parse(statement).body[0]
+            alias = single.names[0]
+            if isinstance(single, ast.Import):
+                # Even unaliased dotted imports must already have executed.
+                if alias.name not in sys.modules:
+                    continue
+                module_name = alias.name if alias.asname else alias.name.split('.')[0]
+                expected = sys.modules.get(module_name)
+            else:
+                module = sys.modules.get(single.module)
+                expected = vars(module).get(alias.name) if type(module) is types.ModuleType else None
+            eligible = type(expected) in (
+                types.ModuleType, types.FunctionType, types.BuiltinFunctionType, types.MethodType,
+                type, str, bytes, int, float, complex, bool,
+            )
+            if eligible and binding in namespace and namespace[binding] is expected:
+                matches[binding] = statement
+    return matches, declared
+
+
+def collect_functions(code_text, previous, namespace, available, *, retain_imports=False,
+                      repair_hints=False):
+    """Collect from the actual namespace, including after partial execution.
+
+    A later def may never have run. Match the live function's source location
+    rather than choosing the last definition in the submitted snippet.
+    """
+    candidates = {}
+    rejected = []
+    try:
+        tree = ast.parse(code_text, '<python_tool>')
+    except SyntaxError:
+        # Parsing failed before execution; restored functions are still valid.
+        tree = ast.Module(body=[], type_ignores=[])
+    fresh = {}
+    for node in tree.body:
+        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
+            fresh.setdefault(node.name, []).append(node)
+    import_nodes = []
+    for source in previous.values():
+        # Old source-only records remain valid; no migration is necessary.
+        import_nodes.extend(ast.parse(source).body[:-1])
+    import_nodes.extend(tree.body)
+    imports, declared_imports = (_matching_imports(import_nodes, namespace)
+                                if retain_imports or repair_hints else ({}, {}))
+    for name in dict.fromkeys([*previous, *fresh]):
+        value = namespace.get(name)
+        if not repair_hints and (name in available or type(value) is not types.FunctionType
+                                 or value.__name__ != name):
+            rejected.append(_rejection(name, 'definition did not execute, was removed/replaced, or conflicts with a provided name'))
+            continue
+        if name in available:
+            rejected.append(_rejection(name, 'name conflicts with a provided global or builtin',
+                                       'Choose a different function name.'))
+            continue
+        if name not in namespace:
+            reason = 'function was removed' if name in previous else 'definition was not reached or was removed'
+            rejected.append(_rejection(name, reason, 'Define it before calling it if it is still needed.'))
+            continue
+        if type(value) is not types.FunctionType or value.__name__ != name:
+            rejected.append(_rejection(name, 'function name was replaced by another value',
+                                       'Use a separate variable name; redefine the function only if needed.'))
+            continue
+        if name in previous and value.__code__.co_filename == '<retained>':
+            # Preserve the restored definition when execution stopped before a
+            # replacement (or while evaluating that replacement's defaults).
+            candidates[name] = previous[name]
+            continue
+        node = next((node for node in fresh.get(name, [])
+                     if isinstance(node, ast.FunctionDef) and not node.decorator_list
+                     and value.__code__.co_filename == '<python_tool>'
+                     and value.__code__.co_firstlineno == node.lineno), None)
+        if node is None:
+            rejected.append(_rejection(name, 'definition was decorated, replaced outside a top-level def, or is unsupported',
+                                       'Use a plain top-level def without decorators if you want to reuse it.'))
+            continue
+        candidates[name] = ast.get_source_segment(code_text, node)
+    # Rebuild every bundle from the LIVE namespace. An old helper must not
+    # silently restore an obsolete import when that alias was rebound/deleted.
+    for name, source in (list(candidates.items()) if retain_imports else []):
+        try:
+            node, _ = _source_parts(source)
+            definition = ast.get_source_segment(source, node)
+            _, _, reads = _function_record(definition)
+            statements = [imports[n] for n in sorted(reads & imports.keys()) if n not in available]
+            candidates[name] = '\n'.join([*statements, definition])
+        except Exception:
+            # Validation below produces the specific rejection and prunes callers.
+            pass
+    kept, invalid = prepare_functions(candidates.values(), available, retain_imports=retain_imports)
+    for item in invalid:
+        missing = item.get('missing_dependencies', [])
+        if not missing:
+            continue
+        hints = []
+        missing_imports = [n for n in missing if n in declared_imports]
+        missing_helpers = [n for n in missing if n in candidates and n not in declared_imports]
+        missing_data = sorted(set(missing) - set(missing_imports) - set(missing_helpers))
+        if missing_imports:
+            statements = '; '.join(declared_imports[n] for n in missing_imports)
+            hints.append('Put `' + statements + '` inside this function if you want to reuse it.')
+        if missing_helpers:
+            hints.append('Make the helper dependencies retainable too: ' + ', '.join(missing_helpers) + '.')
+        if missing_data:
+            hints.append('Pass ' + ', '.join(missing_data) + ' as arguments; snippet variables are not retained.')
+        item['hint'] = ' '.join(hints)
+    return kept, rejected + invalid
diff --git a/ARC3-Inference/inference/utils/segmentation.py b/ARC3-Inference/inference/utils/segmentation.py
index 20f10ed..33f8848 100644
--- a/ARC3-Inference/inference/utils/segmentation.py
+++ b/ARC3-Inference/inference/utils/segmentation.py
@@ -62,17 +62,61 @@ def _corner_points(contour):
     return corners
 
 
-def _object_hash(cells, color):
-    """Translation-invariant signature of an object: its color plus its cell shape,
-    normalized so the top-left of its bounding box is the origin. Same shape + color
-    => same hash regardless of position, so objects can be matched across frames."""
+def _normalize_cells(cells):
     min_r = min(r for r, _ in cells)
     min_c = min(c for _, c in cells)
-    norm = sorted((r - min_r, c - min_c) for r, c in cells)
-    payload = repr((color, norm)).encode()
+    return sorted((r - min_r, c - min_c) for r, c in cells)
+
+
+def _rotate_cells_90(norm_cells):
+    """Rotate a normalized cell set 90 degrees clockwise; returns normalized."""
+    max_r = max(r for r, _ in norm_cells)
+    return sorted((c, max_r - r) for r, c in norm_cells)
+
+
+def _cells_hash(norm_cells, color):
+    payload = repr((color, norm_cells)).encode()
     return hashlib.sha1(payload).hexdigest()[:16]
 
 
+def _object_signature(cells, color):
+    """Rotation-canonical identity of an object.
+
+    Hashes the shape in all four 90-degree rotations (extend the transform list
+    with reflections for full D4 canonicalization later) and takes the lowest
+    hash as the canonical id. Returns (canonical_hash, rotation, symmetry,
+    pose_hash) where:
+      - canonical_hash: identical for all rotated copies of the same shape+color.
+      - rotation: degrees CLOCKWISE to rotate the canonical form to obtain the
+        observed object. None for 4-fold symmetric shapes (all rotations equal);
+        reduced mod 180 (i.e. 0 or 90) for 2-fold symmetric shapes.
+      - symmetry: rotational symmetry order (1, 2, or 4).
+      - pose_hash: the classic rotation-SENSITIVE hash of the observed pose.
+    """
+    norm = _normalize_cells(cells)
+    variants = [norm]
+    for _ in range(3):
+        variants.append(_rotate_cells_90(variants[-1]))
+    hashes = [_cells_hash(v, color) for v in variants]
+    # The reference orientation is picked from the COLOUR-FREE hash. Choosing it
+    # from the colour-bearing one made the argmin depend on the colour, so the
+    # same shape in the same orientation reported a different rotation for a red
+    # object than for a blue one - and a model comparing rotations across
+    # objects was reading noise.
+    shape_hashes = [_cells_hash(v, "__shape__") for v in variants]
+    k = shape_hashes.index(min(shape_hashes))
+    canonical = hashes[k]
+    distinct = len(set(shape_hashes))
+    symmetry = {1: 4, 2: 2, 4: 1}.get(distinct, 1)
+    rotation = ((4 - k) % 4) * 90  # so observed = canonical rotated this much CW
+    if symmetry == 4:
+        rotation = None
+    elif symmetry == 2:
+        rotation = rotation % 180
+    shape_hash = _cells_hash(norm, "__shape__")
+    return canonical, rotation, symmetry, hashes[0], shape_hash
+
+
 def segment_layer(layer, color_chars):
     """Segment one frame layer into connected-component nodes.
 
@@ -189,11 +233,16 @@ def segment_layer(layer, color_chars):
         comp = components[cid]
         color = color_chars[max(0, min(15, comp["value"]))]
         boundary = _corner_points(_trace_outer_contour(comp["cells"], comp["start"]))
+        signature = _object_signature(comp["cells"], color)
         nodes.append(
             {
                 "id": cid,
                 "color": color,
-                "hash": _object_hash(comp["cells"], color),
+                "hash": signature[0],
+                "rotation": signature[1],
+                "rotational_symmetry": signature[2],
+                "pose_hash": signature[3],
+                "shape_hash": signature[4],
                 "pixels": len(comp["cells"]),
                 "boundary": [[r, c] for r, c in boundary],
                 "children": children[cid],

import json
import os
import pickle
import subprocess
import sys
import sysconfig
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlopen

# Kaggle might cancel the run in no action is done in some timeframe.
# To prevent this, harness will start up even if backend is not yet ready,
# and the first started game will perform a single reset action.
NOTEBOOK_START_TIME = time.time()
SERVER_STARTUP_TIMEOUT = 12 * 60
os.environ['ARC3_WARMUP_ACTION_GAMES'] = '1'
os.environ['ARC3_HTTP_RETRY_INITIAL_SECONDS'] = '900'

# model and dataset paths
WHEELHOUSE_DIR    = '/kaggle/input/datasets/dfranzen/pennyroyal-v253'
MODEL_DIR         = '/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1'
DRAFT_MODEL_DIR   = '/kaggle/input/models/dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/default/1'
ORIG_BUNDLE_DIR   = '/kaggle/input/datasets/dfranzen/taaf-kaggle-source-bundle-copy'
SERVED_MODEL_NAME = 'flashnext'

# Apply harness patch
BUNDLE_DIR = Path('/kaggle/taaf-kaggle-source-share')
!rm -Rf $BUNDLE_DIR
!cp -a $ORIG_BUNDLE_DIR $BUNDLE_DIR
subprocess.run(["git", "apply", "--include=ARC3-Inference/*", "-v", "/kaggle/harness-changes.patch"], cwd=f"{BUNDLE_DIR}/src", check=True)
print('harness patch applied successfully')

# True only inside a real competition rerun; switches diagnostics + soft deadline.
TRUE_SUBMISSION = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}
NOTEBOOK_START_EPOCH = time.time()

# Non-interactive matplotlib backend: diagnostics render plots with no display attached.
os.environ["MPLBACKEND"] = "Agg"
# Marks the run as a (real or emulated) submission so the framework + solver can adjust.
os.environ["TAAF_RUN_AS_SUBMISSION"] = "1" if TRUE_SUBMISSION else "0"
# In submission, disable the periodic JSON/HTML diagnostics writes and per-frame logging.
os.environ["TAAF_MINIMAL_DIAGNOSTICS"] = "1" if TRUE_SUBMISSION else "0"
# Pin arc_agi's cached level_reset_only before its client is built (RESET keeps the level).
os.environ["ONLY_RESET_LEVELS"] = "true"

# Everything the run produces is written here.
WORKING_DIR = Path("/kaggle/working")
WORKING_DIR.mkdir(parents=True, exist_ok=True)
print(f"taaf.kaggle: TRUE_SUBMISSION={TRUE_SUBMISSION}")
SERVED_MODEL_HOST = '127.0.0.1'
SERVED_MODEL_PORT = 8001
SERVER_BASE_URL = f'http://{SERVED_MODEL_HOST}:{SERVED_MODEL_PORT}/v1'
USE_PRIORITY_SCHEDULING = True

setup_env = {
    ########## transformers config ##########
    'USE_TF': '0',
    'TRANSFORMERS_NO_TF': '1',
    'TRANSFORMERS_NO_TORCHVISION': '1',

    ########## serving config ##########
    'LOCAL_ANALYZER_BASE_URL': SERVER_BASE_URL,
    'OPENAI_BASE_URL': SERVER_BASE_URL,
    'LOCAL_ANALYZER_PROVIDER': 'vllm',
    'OPENAI_PROVIDER': 'vllm',
    'LOCAL_ANALYZER_MODEL_ID': SERVED_MODEL_NAME,
    'INFERENCE_ANALYZER_MODEL': SERVED_MODEL_NAME,
    'LOCAL_ANALYZER_APP_NAME': 'ARC3 Agent Harness',
    
    ########## analyzer config ##########
    'ARC3_REASONING_HISTORY_KEY': 'reasoning_content',
    'LOCAL_ANALYZER_ENABLE_THINKING': 'true',
    'LOCAL_ANALYZER_TOOL_STEPS': '0',
    'LOCAL_ANALYZER_TOOL_TIMEOUT': '30',
    'LOCAL_ANALYZER_YIELD_SECONDS': '0',
    'LOCAL_ANALYZER_YIELD_TOKENS': '2048',
    'LOCAL_ANALYZER_TEMPERATURE': '0.7',
    'LOCAL_ANALYZER_TOP_P': '0.95',
    'LOCAL_ANALYZER_TOP_K': '20',

    ########## context pruning ##########
    'LOCAL_ANALYZER_CONTEXT_WINDOW': (80+12)*1024,  # WAVE-1 E2 (control (116+12)*1024)
    'LOCAL_ANALYZER_MAX_OUTPUT': 12*1024,
    'ARC3_CONTEXT_DRAIN_TOKENS': 40*1024,  # WAVE-1 E2 (control 58*1024)
    'ARC3_HISTORY_ASSISTANT_TURNS': 150,
    'ARC3_HISTORY_TURN_DRAIN': 30,
    'ARC3_HISTORY_DRAIN_COALESCE': '1',

    ########## tool output ##########
    'LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS': '3072',
    'ARC3_MIDDLE_TRUNCATION': '1',
    'ARC3_ACTION_ECHO': '1',
    
    ########## prompt changes ##########
    'ARC3_EXPLAIN_GAMEPLAY_CHANGED': '1',
    'ARC3_NEW_CHANGED_PROMPTS': '1',
    'ARC3_LEVEL_TRANSFER_GUIDANCE': '1',
    'ARC3_ACTION_INFO': '1',
    'ARC3_DEDUPE_MULTICALL_LINE': '1',
    'ARC3_FRAME_DIFF_HINT': '1',

    ########## vision & animations ##########
    'MULTIMODAL_CONTEXT': 'current_grid',
    'MULTIMODAL_UPSCALE': '10',
    'ARC3_DIFF_IMAGE': '1',
    'ARC3_GAMEOVER_DIFF_IMAGE': '1',
    'ARC3_ANIMATION': '1',

    ########## guards ##########
    'ARC3_GUARDS_FROM_LEVEL': 2,
    'ARC3_BATCH_NOOP_BLOCK': '1',
    'ARC3_STALE_STATE_BLOCK': '1',

    ########## world model ##########
    'ARC3_MEMORY_SECTIONS': 'off',

    ########## persistent functions ##########
    'ARC3_PERSISTENT_FUNCTIONS': '1',
    'ARC3_PERSISTENT_FUNCTIONS_SCOPE': 'game',
    'ARC3_PERSISTENT_FUNCTIONS_IMPORTS': '1',

    ########## controls ##########
    'EXPOSE_UNDO': 'on',
}

if USE_PRIORITY_SCHEDULING: setup_env.update({
    'ARC3_MAX_ACTIVE_STREAMS': 14,  # WAVE-1 E2 (control 10)
    'ARC3_PRIORITY_REFRESH_QUEUE': '1',
    'ARC3_PRIORITY_PACE': '0',
    'ARC3_PRIORITY_TAIL_FADE': '1',
    'ARC3_PRIORITY_TAIL_FADE_FRACTION': '0.2',
    'ARC3_PRIORITY_TAIL_LOOKUP': 'remaining',
    'ARC3_PRIORITY_SCORE_NORMALIZATION': '1',
})

os.environ.update({k: str(v) for k,v in setup_env.items()})

# M2 B1: install verified source before importing the harness.
import base64, hashlib, json, sys, zlib
from pathlib import Path
if any(n == "inference" or n.startswith("inference.") for n in sys.modules):
    raise RuntimeError("M2 B1 overlay must be installed before harness imports")
import os
os.environ["LOCAL_ANALYZER_TOOL_TIMEOUT"] = "60"  # B1: harness reads it at import; ceiling raised to 60 s
_m2_files = json.loads(zlib.decompress(base64.b64decode('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')))
_m2_root = BUNDLE_DIR / "src" / "ARC3-Inference"
for _m2_rel, _m2_item in _m2_files.items():
    _m2_path = _m2_root / _m2_rel
    _m2_hash = hashlib.sha256(_m2_path.read_bytes()).hexdigest()
    if _m2_hash not in (_m2_item["before"], _m2_item["after"]):
        raise RuntimeError("Unexpected donor source: " + _m2_rel)
for _m2_rel, _m2_item in _m2_files.items():
    _m2_path = _m2_root / _m2_rel
    _m2_tmp = _m2_path.with_suffix(".m2-tmp")
    _m2_tmp.write_text(_m2_item["source"], encoding="utf-8")
    _m2_tmp.replace(_m2_path)
    assert hashlib.sha256(_m2_path.read_bytes()).hexdigest() == _m2_item["after"]
print("M2 B1 installed", {k: v["after"] for k, v in _m2_files.items()})


import os
import threading
import time

from concurrent.futures import as_completed, ThreadPoolExecutor

def precache(*paths, times=1, delay=0, threads=16) -> None:
    time.sleep(delay)
    print(f'Precaching: Started...')
    start_time = time.time()

    CHUNK_SIZE = 32 * 1024 * 1024  # 128 MB

    chunks = []  # list of (path, offset, length)
    total_size = 0
    file_count = 0

    def _add_file(i: int, path: str) -> None:
        nonlocal total_size, file_count
        size = os.path.getsize(path)
        total_size += size
        file_count += 1
        for offset in range(0, max(size, 1), CHUNK_SIZE):
            length = min(CHUNK_SIZE, size - offset)
            chunks.append((i, path, offset, length))

    for i, path in enumerate(paths):
        try:
            if os.path.isfile(path):
                _add_file(i, path)
                continue
            for root, _, files in os.walk(path):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    if os.path.isfile(file_path):
                        _add_file(i, file_path)
        except: pass

    chunks.sort()

    _local = threading.local()

    def _read_chunk(chunk: tuple) -> None:
        try:
            buf = _local.buf
        except AttributeError:
            _local.buf = buf = bytearray(CHUNK_SIZE)
        i, path, offset, _ = chunk
        try:
            with open(path, 'rb') as f:
                f.seek(offset)
                remaining = CHUNK_SIZE
                for i in range(1024):
                    n = f.readinto(buf)
                    if not n:
                        break
                    remaining -= n
                    if remaining <= 0: break
        except: pass

    with ThreadPoolExecutor(max_workers=threads) as pool:
        for _ in range(times):
            list(pool.map(_read_chunk, chunks))

    elapsed = time.time() - start_time
    print(f'Precaching: loaded {file_count} files ({total_size / 1e9:.2f} GB) in {elapsed:.2f} seconds.\n')

precache_wheels_thread = threading.Thread(target=precache, args=(WHEELHOUSE_DIR,))
precache_wheels_thread.start()
precache_wheels_thread.join()

precache_model_thread = threading.Thread(target=precache, args=(MODEL_DIR, DRAFT_MODEL_DIR), kwargs={'delay': 60, 'threads': 3})

# Install the ARC runtime from the bundled competition wheels.
# Quiet: stdout is discarded; stderr (and a non-zero exit) still surface real failures.
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--no-index",
        "--no-warn-conflicts",
        "--disable-pip-version-check",
        "--find-links",
        "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels",
        "arc-agi",
    ],
    stdout=subprocess.DEVNULL,
)

# Each bundled repo exposes its importable tree at <repo>/src or <repo>.
def _source_path_entries(bundle_dir: Path) -> list:
    entries = []
    for repo in sorted((bundle_dir / "src").iterdir(), reverse=True):
        for candidate in (repo / "src", repo):
            if candidate.is_dir():
                entries.append(candidate)
    return entries


# Environment handed to each setup command (paths + any keys it has persisted).
def _command_env() -> dict:
    env = os.environ.copy()
    # "$PYTHON" in a command resolves to this notebook's interpreter.
    env["PYTHON"] = sys.executable
    # Absolute path to the mounted source bundle.
    env["TAAF_KAGGLE_BUNDLE_DIR"] = str(BUNDLE_DIR)
    # The writable /kaggle/working directory.
    env["TAAF_KAGGLE_WORKING_DIR"] = str(WORKING_DIR)
    # A command writes a JSON object here to persist env keys to later commands + the run.
    return env


# Make the bundled repos importable here (sys.path) and in child processes (.pth).
source_entries = _source_path_entries(BUNDLE_DIR)
for entry in source_entries:
    sys.path.insert(0, str(entry))
pth_path = Path(sysconfig.get_paths()["purelib"]) / "taaf_kaggle_sources.pth"
pth_path.write_text("".join(f"{entry}\n" for entry in source_entries))
print(f"taaf.kaggle: wrote {pth_path} ({len(source_entries)} source roots)")

# Paste this entire file into the launcher cell.
# Required notebook variables: MODEL_DIR, DRAFT_MODEL_DIR, WHEELHOUSE_DIR,
# WORKING_DIR, SERVED_MODEL_NAME, SERVED_MODEL_PORT.
# MODEL_DIR: Intel AutoRound INT4 target (BF16 PLE or albucino FP8 PLE).
# DRAFT_MODEL_DIR: downloaded albucino runtime/mtp-int4-g32, or its dataset root.
import os, sys, glob, time, shutil, subprocess, urllib.request, urllib.error
import copy, hashlib, json, re, shlex, socket, zipfile
from pathlib import Path

PREFIX = "/tmp/sgl-intel"       # Separate from the previous fork's /tmp/sgl venv.
LOG = str(Path(WORKING_DIR) / "serve.log")
CFG = dict(
    CTX=(80+12+8)*1024,  # WAVE-1 E2 (control (116+12+8)*1024)
    MEMFRAC=0.96,
    MAXREQ=14,  # WAVE-1 E2 (control 10)
    CUDAGRAPH_MAXBS=14,  # WAVE-1 E2 (control 10)
    MAMBA_CACHE=84,  # WAVE-1 E2 (control 60; 6 slots per stream)
    CHUNK=8192,
    MAX_PREFILL=16384,
    SPEC=True,
    FRSPEC=True,
    AUTOTUNE=True,
    # Preserve the original Kaggle checkpoint prefetch/cache behavior.
    PREFETCH_CHECKPOINTS=True,
    DROP_CACHE_AFTER_LOAD=True,
    LINEAR_BACKEND="flashinfer",
    KVDTYPE="fp8_e4m3",
    SSM_DTYPE="bfloat16",
    MAMBA_RADIX="extra_buffer",
    GDN_MTP_CACHE_MODE="none",
    SERVED_NAME=SERVED_MODEL_NAME,
    SPEC_ACCEPT_SINGLE=1.0,
    SPEC_ACCEPT_ACC=1.0,
    SPEC_STEPS=3,
)
VENV = f"{PREFIX}/venv"
PYTHON = f"{VENV}/bin/python"
SGLANG = f"{VENV}/bin/sglang"
TOKEN_MAP_SHA = "becfa41d394b86c26c632bea8f3c6ea64bbb76d7b238d8673c06afae21269f25"
TOKENIZER_SHA = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
EXPERT_TARGET = r"re:^mtp\.layers\.0\.mlp\.experts\.[0-9]+\.(gate_proj|up_proj|down_proj)$"
DENSE_IGNORE = r"re:^(?!mtp\.layers\.0\.mlp\.experts(?:\.|$)).*"


def run(cmd, *, check=True, env=None):
    cmd = list(map(str, cmd))
    print("$", shlex.join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if result.stdout.strip(): print(result.stdout.strip()[-6000:])
    if result.stderr.strip() and result.returncode: print(result.stderr.strip()[-6000:])
    if check and result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {shlex.join(cmd)}")
    return result


def find_unique(root, name, *, directory=False, required=True):
    root = Path(root)
    direct = root if root.name == name else root / name
    valid = lambda p: p.is_dir() if directory else p.is_file()
    if valid(direct): return direct
    hits = sorted(p for p in root.rglob(name) if valid(p))
    if len(hits) > 1:
        raise RuntimeError(f"Multiple {name} matches under {root}; use a more specific directory: {hits}")
    if hits: return hits[0]
    if required: raise FileNotFoundError(f"{name} not found under {root}")
    return None


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""): h.update(block)
    return h.hexdigest()


def indexed_shards(root):
    root = Path(root)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    names = sorted(set(index["weight_map"].values()))
    if not names: raise RuntimeError(f"Empty weight index: {root}")
    for name in names:
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise RuntimeError(f"Invalid shard path: {name}")
        p = root / rel
        if not p.is_file() or p.stat().st_size < 8:
            raise RuntimeError(f"Missing/empty shard: {p}")
    return index, names


def prepare_draft_view(target, source_root, work_root):
    """Adapt metadata only; link original dataset weights without copying them."""
    target, source_root, work_root = map(Path, (target, source_root, work_root))
    configs = [source_root / "config.json"] if (source_root / "config.json").is_file() else []
    configs += sorted(p for p in source_root.rglob("config.json") if p not in configs)
    candidates = []
    for path in configs:
        cfg = json.loads(path.read_text())
        q = cfg.get("quantization_config", {})
        if q.get("quant_method") == "compressed-tensors" and "mtp_routed_experts" in q.get("config_groups", {}):
            candidates.append((path.parent, cfg))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one INT4 g32 MTP checkpoint under DRAFT_MODEL_DIR={source_root}; found {len(candidates)}")
    source, cfg = candidates[0]
    index, shards = indexed_shards(source)
    if not any(k.startswith("mtp.layers.0.mlp.experts.") for k in index["weight_map"]):
        raise RuntimeError("Draft index does not contain the expected MTP experts")
    cfg = copy.deepcopy(cfg)
    q = cfg["quantization_config"]
    group = q["config_groups"]["mtp_routed_experts"]
    w = group["weights"]
    if (w["num_bits"], w["group_size"], w["symmetric"]) != (4, 32, True):
        raise RuntimeError("Expected symmetric INT4 group32 draft experts")
    if group["targets"] not in (["RoutedExperts"], [EXPERT_TARGET]):
        raise RuntimeError(f"Unexpected draft target rules: {group['targets']}")
    if q.get("ignore", []) not in ([], [DENSE_IGNORE]):
        raise RuntimeError("Unexpected draft ignore rules")
    group["targets"] = [EXPERT_TARGET]
    q["ignore"] = [DENSE_IGNORE]
    # Content/path-specific directory makes notebook reruns idempotent without
    # overwriting an unrelated view. We never modify the read-only datasets.
    identity = json.dumps({"source": str(source.resolve()), "target": str(target.resolve()), "config": cfg}, sort_keys=True)
    suffix = hashlib.sha256(identity.encode()).hexdigest()[:16]
    view = work_root / f"draft-view-{suffix}"
    view.mkdir(parents=True, exist_ok=True)
    config_path = view / "config.json"
    config_text = json.dumps(cfg, indent=2) + "\n"
    if config_path.exists() or config_path.is_symlink():
        if config_path.is_symlink() or config_path.read_text() != config_text:
            raise RuntimeError(f"Unexpected existing draft-view config: {config_path}")
    else:
        config_path.write_text(config_text)
    links = {name: source / name for name in shards}
    links["model.safetensors.index.json"] = source / "model.safetensors.index.json"
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json", "generation_config.json"):
        if (target / name).is_file(): links[name] = target / name
    for name, src in links.items():
        dest = view / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            if not dest.is_symlink() or dest.resolve() != src.resolve():
                raise RuntimeError(f"Unexpected existing draft-view file: {dest}")
        else:
            dest.symlink_to(src.resolve())
    print(f"Draft source: {source}\nSGLang draft view: {view} ({len(shards)} linked weight shards)")
    return view


# ---- validate notebook inputs / identify the offline bundle ----
MODEL_DIR = str(Path(MODEL_DIR).expanduser().resolve())
assert Path(MODEL_DIR).is_dir(), f"model dir not found: {MODEL_DIR}"
assert CFG["CUDAGRAPH_MAXBS"] >= CFG["MAXREQ"], "CUDAGRAPH_MAXBS must cover MAXREQ"
assert CFG["CHUNK"] > 0 and CFG["CHUNK"] % 64 == 0
# Refuse to start a second server on an occupied port; do not kill other jobs.
with socket.socket() as probe:
    if probe.connect_ex(("127.0.0.1", int(SERVED_MODEL_PORT))) == 0:
        raise RuntimeError(f"Port {SERVED_MODEL_PORT} is occupied; stop the existing server before rerunning")
Path(PREFIX).mkdir(parents=True, exist_ok=True)
Path(LOG).parent.mkdir(parents=True, exist_ok=True)
WHEELS = find_unique(WHEELHOUSE_DIR, "wheels", directory=True)
LOCK = find_unique(WHEELHOUSE_DIR, "requirements.lock")
sglang_wheels = sorted(WHEELS.glob("sglang-*.whl"))
assert len(sglang_wheels) == 1, f"Expected one SGLang wheel, found {sglang_wheels}"
wheel = sglang_wheels[0]
install_marker = Path(PREFIX) / "installed-bundle.txt"

# ---- one-time offline install, isolated from the old mratsim environment ----
install_env = dict(os.environ, UV_OFFLINE="1", UV_PYTHON_DOWNLOADS="never", PYTHONNOUSERSITE="1")
install_env.pop("PYTHONPATH", None)
if not Path(SGLANG).exists() or not install_marker.is_file():
    print("=== installing Pennyroyal (offline) ===")
    # Bootstrap uv directly from its wheel; no global pip changes/ensurepip.
    uv_wheels = sorted(WHEELS.glob("uv-*.whl"))
    assert len(uv_wheels) == 1, "Expected one bundled uv wheel"
    uv_binary = Path(PREFIX) / "bin/uv"
    uv_binary.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(uv_wheels[0]) as archive:
        members = [n for n in archive.namelist() if Path(n).name == "uv" and not n.endswith("/")]
        assert len(members) == 1, "Cannot identify the Linux uv executable in its wheel"
        binary = archive.read(members[0])
        assert binary.startswith(b"\x7fELF"), "Bundled uv executable is not a Linux ELF binary"
        uv_binary.write_bytes(binary)
    uv_binary.chmod(0o755)
    uv = [str(uv_binary)]
    if not Path(PYTHON).exists():
        run(uv + ["venv", "--python", sys.executable, VENV], env=install_env)
    install = uv + ["pip", "install", "--python", PYTHON, "--no-index", "--find-links", str(WHEELS)]
    run(install + ["-r", str(LOCK)], env=install_env)
    run(install + ["--reinstall", "--no-deps", str(wheel)], env=install_env)
    install_marker.write_text('install')
else:
    print(f"Matching offline bundle already installed at {VENV}")

# ---- bundled CUDA toolkit and linker symlinks ----
cu13 = sorted(Path(VENV).glob("lib/python*/site-packages/nvidia/cu13"))
if not cu13 or not (cu13[0] / "bin/nvcc").is_file():
    raise RuntimeError("Bundled CUDA 13 nvcc missing; check that the complete Pennyroyal wheelhouse was uploaded")
CUDA_HOME = str(cu13[0])
libdir = Path(CUDA_HOME) / "lib"
lib64 = Path(CUDA_HOME) / "lib64"
if not lib64.exists() and not lib64.is_symlink(): lib64.symlink_to("lib")
for so in sorted(libdir.glob("*.so.*")):
    link = libdir / re.sub(r"\.so\..*$", ".so", so.name)
    if not link.exists() and not link.is_symlink(): link.symlink_to(so.name)
assert (libdir / "libcudart.so").exists(), "libcudart.so missing: JIT linker cannot find CUDA runtime"
if not (libdir / "libcuda.so").exists():
    candidates = []
    for directory in ("/usr/local/nvidia/lib64", "/usr/local/nvidia/lib", "/usr/lib/x86_64-linux-gnu", "/usr/lib64"):
        candidates += sorted(p for p in Path(directory).glob("libcuda.so*") if p.is_file())
    if not candidates:
        code = 'import torch; torch.cuda.init(); print(open("/proc/self/maps").read())'
        maps = run([PYTHON, "-c", code], env=install_env).stdout
        candidates = [Path(p) for p in re.findall(r"(/[^\s]*libcuda\.so[^\s]*)", maps)]
    if not candidates: raise RuntimeError("GPU driver libcuda.so not found")
    link = libdir / "libcuda.so"
    if link.is_symlink(): link.unlink()
    link.symlink_to(candidates[0].resolve())
CXX = next((c for c in ("g++-15", "g++-14", "g++-13", "g++-12", "g++-11", "g++") if shutil.which(c)), None)
assert CXX, "Host C++ compiler not found"
CC = CXX.replace("g++", "gcc")
assert shutil.which(CC), f"Matching C compiler missing: {CC}"

# ---- runtime environment ----
env = dict(os.environ)
env.pop("PYTHONPATH", None)
cache = f"{PREFIX}/cache"
for name in ("huggingface", "torch", "torchinductor", "triton", "flashinfer", "cuda", "sglang/jit"):
    Path(cache, name).mkdir(parents=True, exist_ok=True)
env.update({
    "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1", "UV_OFFLINE": "1", "UV_PYTHON_DOWNLOADS": "never",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "CUDA_HOME": CUDA_HOME, "CUDACXX": f"{CUDA_HOME}/bin/nvcc",
    "PATH": f"{CUDA_HOME}/bin:{VENV}/bin:" + env.get("PATH", ""),
    "LD_LIBRARY_PATH": f"{CUDA_HOME}/lib64:{CUDA_HOME}/lib:" + env.get("LD_LIBRARY_PATH", ""),
    "CC": CC, "CXX": CXX, "CUDAHOSTCXX": CXX,
    "TORCH_CUDA_ARCH_LIST": "12.0",
    "HF_HOME": f"{cache}/huggingface", "XDG_CACHE_HOME": cache,
    "TORCH_HOME": f"{cache}/torch", "CUDA_CACHE_PATH": f"{cache}/cuda",
    "TORCHINDUCTOR_CACHE_DIR": f"{cache}/torchinductor", "TRITON_CACHE_DIR": f"{cache}/triton",
    "FLASHINFER_WORKSPACE_BASE": f"{cache}/flashinfer",
    "SGLANG_CACHE_DIR": f"{cache}/sglang", "SGLANG_JIT_CACHE_DIR": f"{cache}/sglang/jit",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
    "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "TOKENIZERS_PARALLELISM": "false",
    "MAX_JOBS": "8", "CMAKE_BUILD_PARALLEL_LEVEL": "8",
    "FLASHINFER_NINJA_JOBS": "8", "FLASHINFER_NVCC_THREADS": "2", "TORCHINDUCTOR_COMPILE_THREADS": "8",
    "SGLANG_ENABLE_SM120_LOWM_BF16_GEMM": "1", "SGLANG_SM120_ONLINE_MXFP8": "0",
    "SGLANG_SM120_LOWM_FP8_WEIGHT": "0", "SGLANG_SM120_LM_HEAD_FP8": "0",
    "SGLANG_MAMBA_CONV_DTYPE": "bfloat16", "SGLANG_NUMA_BIND_V2": "false",
    "SGLANG_MM_PREPROCESS_DEVICE": "cpu", "NUMPY_MADVISE_HUGEPAGE": "0",
})
run([str(Path(CUDA_HOME) / "bin/nvcc"), "--version"], env=env)
print("Host compiler:", CXX)
# Report the optional later caching patch. This cell does not modify wheel code.
run([PYTHON, "-c", '''import pathlib, sglang, torch
print("sglang", sglang.__version__, "| torch", torch.__version__)
assert torch.cuda.is_available(), "CUDA unavailable"
print("GPU:", torch.cuda.get_device_name(0), "capability:", torch.cuda.get_device_capability(0))
'''], env=env)

# ---- validate target and create the read-only draft's SGLang view ----
target_cfg = json.loads(Path(MODEL_DIR, "config.json").read_text())
q = target_cfg.get("quantization_config", {})
assert q.get("quant_method") == "auto-round" and q.get("bits") == 4, "MODEL_DIR must be the Intel/albucino AutoRound INT4 checkpoint"
_, target_shards = indexed_shards(MODEL_DIR)
print("Target:", MODEL_DIR, "| shards:", len(target_shards), "| PLE dtype:", target_cfg.get("text_config", {}).get("ple_embedding_dtype", "checkpoint default (typically BF16)"))
DRAFT_VIEW = None
if CFG["SPEC"]:
    assert Path(DRAFT_MODEL_DIR).is_dir(), f"draft dir not found: {DRAFT_MODEL_DIR}"
    DRAFT_VIEW = prepare_draft_view(MODEL_DIR, DRAFT_MODEL_DIR, PREFIX)

# ---- assemble Pennyroyal / AutoRound launch arguments ----
graph_bs = sorted({1, 2, 4, 7, 8, 9, 10, 12, CFG["MAXREQ"], CFG["CUDAGRAPH_MAXBS"]})  # WAVE-1 E2: +12
graph_bs = [n for n in graph_bs if n <= CFG["CUDAGRAPH_MAXBS"]]
args = [SGLANG, "serve", "--model-path", MODEL_DIR, "--load-format", "safetensors",
    "--model-loader-extra-config", '{"enable_multithread_load":false}',
    "--served-model-name", str(CFG["SERVED_NAME"]), "--host", "127.0.0.1",
    "--port", str(SERVED_MODEL_PORT), "--tensor-parallel-size", "1",
    "--dtype", "bfloat16", "--quantization", "auto-round",
    "--kv-cache-dtype", CFG["KVDTYPE"], "--mem-fraction-static", str(CFG["MEMFRAC"]),
    "--context-length", str(CFG["CTX"]), "--page-size", "64",
    "--max-running-requests", str(CFG["MAXREQ"]), "--chunked-prefill-size", str(CFG["CHUNK"]),
    "--max-prefill-tokens", str(CFG["MAX_PREFILL"]),
    "--cuda-graph-max-bs-decode", str(CFG["CUDAGRAPH_MAXBS"]),
    "--cuda-graph-bs-decode", *map(str, graph_bs),
    "--mamba-ssm-dtype", CFG["SSM_DTYPE"], "--max-mamba-cache-size", str(CFG["MAMBA_CACHE"]),
    "--mamba-radix-cache-strategy", CFG["MAMBA_RADIX"], "--mamba-track-interval", "64",
    "--mamba-backend", CFG["LINEAR_BACKEND"],
    "--linear-attn-decode-backend", CFG["LINEAR_BACKEND"],
    "--linear-attn-prefill-backend", CFG["LINEAR_BACKEND"],
    "--moe-runner-backend", "auto", "--ple-offload-embedding", "--trust-remote-code",
    "--mm-feature-transport", "cpu", "--image-processor-backend", "pil",
    "--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder",
    "--default-chat-template-kwargs", '{"preserve_thinking":true}',
    "--watchdog-timeout", "1800", "--schedule-policy", "lpm", "--warmups", "structured_output",
    "--enable-cache-report", "--enable-metrics", "--enable-request-time-stats-logging"]
if CFG["PREFETCH_CHECKPOINTS"]: args += ["--weight-loader-prefetch-checkpoints"]
if CFG["DROP_CACHE_AFTER_LOAD"]: args += ["--weight-loader-drop-cache-after-load"]
tmpl = Path(MODEL_DIR, "chat_template.jinja")
if tmpl.is_file(): args += ["--chat-template", str(tmpl)]
if not CFG["AUTOTUNE"]: args += ["--disable-flashinfer-autotune"]
if CFG["GDN_MTP_CACHE_MODE"]: args += ["--gdn-mtp-cache-mode", CFG["GDN_MTP_CACHE_MODE"]]
if CFG["SPEC"]:
    args += ["--speculative-algorithm", "NEXTN", "--speculative-num-steps", str(CFG["SPEC_STEPS"]),
        "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", str(CFG["SPEC_STEPS"]+1),
        "--speculative-draft-model-path", str(DRAFT_VIEW),
        "--speculative-draft-model-quantization", "compressed-tensors", "--speculative-moe-runner-backend", "auto",
        "--speculative-draft-kv-cache-dtype", CFG["KVDTYPE"],
        "--speculative-accept-threshold-single", str(CFG["SPEC_ACCEPT_SINGLE"]),
        "--speculative-accept-threshold-acc", str(CFG["SPEC_ACCEPT_ACC"])]
    if CFG["FRSPEC"]:
        tok = find_unique(WHEELHOUSE_DIR, "hot_tokens_64k.pt", required=False)
        if tok is None: tok = find_unique(WHEELHOUSE_DIR, "flash-next-64k.pt")
        assert sha256(tok) == TOKEN_MAP_SHA, "Wrong FR-Spec map; use the map from the pinned Pennyroyal bundle"
        assert sha256(Path(MODEL_DIR, "tokenizer.json")) == TOKENIZER_SHA, "Tokenizer differs from the FR-Spec map; set CFG['FRSPEC']=False"
        args += ["--speculative-token-map", str(tok)]

# ---- launch detached and wait for health ----
precache_model_thread.start()
print("\n" + shlex.join(args) + "\n")
log_offset = Path(LOG).stat().st_size if Path(LOG).exists() else 0
with open(LOG, "ab", buffering=0) as logf:
    proc = subprocess.Popen(args, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
print(f"pid {proc.pid} -> {LOG}")

def show_log_tail():
    with open(LOG, "rb") as f:
        f.seek(0, 2)
        f.seek(max(log_offset, f.tell()-20000))
        print(f.read().decode(errors="replace"))

url, t0 = f"http://127.0.0.1:{SERVED_MODEL_PORT}/health", time.time()
last_report = -30
while time.time() - NOTEBOOK_START_TIME < SERVER_STARTUP_TIMEOUT:
    if proc.poll() is not None:
        show_log_tail()
        break
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            if r.status == 200:
                print(f"\nREADY after {int(time.time()-t0)}s -> {url.replace('/health','/v1')}")
                break
    except Exception:
        pass
    elapsed = int(time.time()-t0)
    if elapsed-last_report >= 30:
        print(f"  {elapsed}s ...", flush=True); last_report=elapsed
    time.sleep(3)
else:
    print(f"\nDEADLINE at {int(time.time()-NOTEBOOK_START_TIME)}s from notebook start; "
          f"releasing the benchmark with the server still loading")
if proc.poll() is not None:  # WAVE-1: a dead server must end the run, not burn the cap
    raise RuntimeError(f"SGLang server exited during startup (code {proc.returncode}); see log tail above")


# Restore the deployment target and record the real submission state on it.
with open(BUNDLE_DIR / "deploy_target.pkl", "rb") as file:
    target = pickle.load(file)
target.actual_run_as_submission = TRUE_SUBMISSION
target.is_competition_rerun = TRUE_SUBMISSION

# Restore the benchmark and point its outputs at the Kaggle working dir.
with open(BUNDLE_DIR / "benchmark_initial.pkl", "rb") as file:
    bm = pickle.load(file)
bm.job_dir = WORKING_DIR

# Make one-off changes to `bm`, `bm.games`, or `bm.solver` here before the run starts.
bm.solver.max_runtime_s_per_game = 48*60 #7920.0
bm.solver.analyzer_timeout = 900.0
bm.solver.concurrency = 10
bm.solver.max_actions_per_game = None
bm.solver.save_request_logs = True
bm.n_passes = 1
bm.game_weights = None

demo_excluded_games = []  # CONTROL: all 25 public games

if TRUE_SUBMISSION:  # overrides for contest run
    bm.solver.save_request_logs = False
    bm.n_passes = 1
    if USE_PRIORITY_SCHEDULING:
        bm.solver.concurrency = 120  # game count should be 110
        bm.solver.max_runtime_s_per_game = 532*60
else:
    if USE_PRIORITY_SCHEDULING:
        bm.solver.concurrency = (25 - len(demo_excluded_games)) * bm.n_passes
        bm.solver.max_runtime_s_per_game = 532*60 * bm.solver.concurrency // 110  # CONTROL: hidden-equivalent share = 7254 s
    if float(getattr(target, 'max_runtime_s', 0.0) or 0.0) != 32400.0:
        raise RuntimeError(
            f'Expected the 32400-second notebook budget, got {target.max_runtime_s!r}.'
        )

# show config
print('benchmark.solver:', bm.solver)

# --- system census, one line every 10 minutes -------------------------------
# Everything reads as used/total. RAM is total - MemAvailable, which runs higher
# than the "used" column in `free -h`: that one excludes reclaimable page cache,
# and MemAvailable is the figure that actually predicts an OOM. Disk likewise
# counts the root reserve as unavailable, matching df.
#
# A daemon thread rather than a hook in the run loop: the point of watching a
# long run is to see what happens when it stops behaving, and a print driven by
# the harness stops with it. This keeps reporting through a deadlock.
import os, subprocess, threading, sys

def _sysline(paths=("/", "/kaggle/working")):
    GB = 1024 ** 3
    out = []
    mem = {}
    try:
        for line in open("/proc/meminfo"):
            k, _, rest = line.partition(":")
            p = rest.split()
            if p:
                mem[k] = int(p[0]) * 1024
    except OSError:
        pass
    if mem.get("MemTotal") and mem.get("MemAvailable") is not None:
        out.append(f"RAM {(mem['MemTotal']-mem['MemAvailable'])/GB:.1f}/{mem['MemTotal']/GB:.1f}")
    if mem.get("SwapTotal"):
        out.append(f"swap {(mem['SwapTotal']-mem['SwapFree'])/GB:.1f}/{mem['SwapTotal']/GB:.1f}")
    else:
        out.append("swap none")
    st = {}
    try:
        for line in open("/proc/self/status"):
            k, _, rest = line.partition(":")
            p = rest.split()
            if k in ("VmRSS", "Threads") and p:
                st[k] = int(p[0])
    except OSError:
        pass
    proc = []
    if "VmRSS" in st:
        proc.append(f"{st['VmRSS']*1024/GB:.1f} GB")
    proc.append(f"{st.get('Threads', threading.active_count())} threads")
    try:
        proc.append(f"{len(os.listdir('/proc/self/fd'))} fds")
    except OSError:
        pass
    out.append("proc " + ", ".join(proc))
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5, check=False)
        gpus = []
        for i, line in enumerate(r.stdout.splitlines()):
            p = [x.strip() for x in line.split(",")]
            if len(p) == 2:
                gpus.append(f"GPU{i} {float(p[0])/1024:.1f}/{float(p[1])/1024:.1f}")
        out.append(" ".join(gpus) if gpus else "GPU unavailable")
    except Exception:
        out.append("GPU unavailable")          # nvidia-smi hangs on a wedged GPU
    disks = []
    for path in paths:
        try:
            s = os.statvfs(path)
        except OSError:
            continue
        total = s.f_blocks * s.f_frsize
        disks.append(f"{path} {(total - s.f_bavail*s.f_frsize)/GB:.1f}/{total/GB:.1f}")
    if disks:
        out.append("disk " + ", ".join(disks))
    return "[sys] " + " | ".join(out)


def start_monitor(interval_minutes=10.0, paths=("/", "/kaggle/working")):
    stop = threading.Event()
    def loop():
        while not stop.is_set():
            try:
                print(_sysline(paths), flush=True)
            except Exception as exc:
                print(f"[sys] census failed: {exc!r}", flush=True)
            stop.wait(interval_minutes * 60.0)
    threading.Thread(target=loop, name="system-stats", daemon=True).start()
    return stop

if not TRUE_SUBMISSION:
    _stop_monitor = start_monitor(2)
    print(_sysline())

print('Starting benchmark...')

# Build the live competition game list from the gateway's available environments.
def _competition_games():
    import arc_agi

    import taaf.game_api

    spec = taaf.game_api.ArcadeSpec(
        operation_mode=arc_agi.OperationMode.COMPETITION,
        arc_base_url=os.environ["ARC_BASE_URL"],
        environments_dir="",
    )
    arcade = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.COMPETITION,
        arc_base_url=spec.arc_base_url,
        environments_dir="",
    )
    game_ids = [env_info.game_id for env_info in arcade.available_environments]
    if not game_ids:
        raise RuntimeError("Competition Arcade exposed zero environments.")
    return [taaf.game_api.GameAPI(env_name=game_id, arcade_spec=spec) for game_id in game_ids]


# Build the offline game list from the competition's bundled environment files.
def _offline_games(env_dir: str):
    import arc_agi

    import taaf.game_api

    spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
    game_ids = [env_info.game_id for env_info in arcade.available_environments]
    if not game_ids:
        raise RuntimeError(f"No offline environments found under {env_dir}.")
    game_ids = [game_id for game_id in game_ids if not any(game_id.startswith(g) for g in demo_excluded_games)]
    return [taaf.game_api.GameAPI(env_name=game_id, arcade_spec=spec) for game_id in game_ids]


# The gateway can take a while to come up; poll until it answers.
def _wait_for_gateway(base_url: str, timeout_s: float = 600.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with urlopen(f"{base_url}api/games", timeout=10) as response:
                if response.status < 500:
                    return
        except Exception as exc:
            last_error = repr(exc)
        time.sleep(5)
    raise RuntimeError(f"Kaggle gateway did not become ready: {last_error}")


# Print the run preamble and persist the launcher's git status for diagnostics.
#print((BUNDLE_DIR / "preamble.txt").read_text())
(WORKING_DIR / "git_status.txt").write_text((BUNDLE_DIR / "git_status.txt").read_text())

# arc_agi reads RECORDINGS_DIR and ARC_API_KEY from env (ArcadeSpec carries neither); operation
# mode, environments dir, and base url are all passed explicitly via the spec, so no env is needed.
os.environ.setdefault("RECORDINGS_DIR", str(WORKING_DIR / "server_recording"))

if TRUE_SUBMISSION:
    # Real submission: play the live competition Arcade served by the Kaggle gateway.
    os.environ.setdefault("ARC_API_KEY", "test-key-123")
    os.environ.setdefault("ARC_BASE_URL", "http://gateway:8001/")
    # The gateway boots asynchronously; wait before swapping in its game list.
    _wait_for_gateway(os.environ["ARC_BASE_URL"])
    bm.games = _competition_games()
else:
    # Interactive run: play the bundled competition environments offline (no gateway).
    # The competition's environment files ship alongside the wheelhouse in the competition dataset.
    competition_env_files = str(Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels").parent / "environment_files")
    bm.games = _offline_games(competition_env_files)

# Outside a real submission, stop ~10 min before the wall-clock budget for a graceful exit.
soft_end = None
if not TRUE_SUBMISSION:
    budget = float(getattr(target, "max_runtime_s", 0.0) or 0.0)
    if budget > 0:
        soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(seconds=budget - min(600.0, budget / 2))

# Play the benchmark; teardown commands run even if the run raises.
await bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=TRUE_SUBMISSION)
if not TRUE_SUBMISSION:
    # An offline run isn't scored, but Kaggle still expects a submission.parquet output.
    import pandas as pd

    pd.DataFrame(
        [["1_0", "1", True, 1]],
        columns=["row_id", "game_id", "end_of_game", "score"],
    ).to_parquet(WORKING_DIR / "submission.parquet", index=False)

from html import escape

from IPython.display import HTML, display

diagnostics_html = WORKING_DIR / "diagnostics.html"
if diagnostics_html.is_file():
    # Isolate the full document in an iframe so its styles don't leak into the notebook.
    display(
        HTML(
            f'<iframe srcdoc="{escape(diagnostics_html.read_text(), quote=True)}" '
            'width="100%" height="900" style="border:0"></iframe>'
        )
    )
else:
    print("No diagnostics.html — minimal diagnostics (real submission) suppresses it.")