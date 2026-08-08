# Prefix-24 machinery versus human annotation

The human annotation is SHA-256
`a4617e804ff7ba872b46ca11af244665416416ffafd566f31545aa182c33f878` and describes the
engine's zero-based level index 2: the third playable level, with public `levels_completed == 2`.
The compared public frame is observation 24, SHA-256
`bc5d4572d2261968eef20bdda33ce658724757cbc3fe3dc8a3a5daea897ac879`.

| Human annotation | Prefix-only machinery output | Comparison |
|---|---|---|
| Player | Controlled track 11, bbox `[30,32,24,29)`, with all four 5-cell action displacements | Semantic match |
| Current key | Mutable workspace track 22, bbox `[55,61,3,9)` | Semantic match |
| Goal with lock | Invariant target track 19, bbox `[51,54,55,58)` | Semantic match |
| Rotate key | Patch at `[10,49]`; three prior public contacts support one clockwise quarter-turn; two contacts required | Causal match |
| Color-change key | Five disconnected component tracks 37–41, union bbox `[46,49,30,33)` | Geometry only; no grouping, role, or effect |
| Refill action bar | Outer/component tracks 9 and 10, bbox `[31,34,20,23)` | Geometry only; no role or effect |
| Blue move-until-wall objects | Static blue tracks 1 and 2 at `[4,5,54,59)` and `[5,10,8,9)` | Geometry only; no pairing or effect |
| Action bar | Changing green track 25 and changing remainder track 34 | Change measured; no budget or reset rule |
| Empty bar resets level | No prefix-supported rule | Missing |

The machinery therefore returns a scene with many of the same visual regions, but not a comparable
semantic game description. It assigns decision-relevant meaning to the player, workspace, target,
and rotation trigger. The compact model request omitted the other annotated mechanics entirely.

Scorer-only replay after this audit identified the large v11 transition at action 11: attempts fell
from 3 to 2, the action-bar budget refilled from 0 to 42, and the player returned to spawn. It was an
empty-bar reset, not contact with a changer as the original run log reported. The same correction
applies to the action-11 transition in v7.

This changes the interpretation of feedback-planning v11. The run still shows that E4B did not route
to the explicitly supplied rotation trigger. It does **not** show failure despite a complete or
sufficient causal state: the causal state was partial. A fair next planning ceiling should first
compile the other public-visible interactables as unresolved hypotheses or learned affordances.
