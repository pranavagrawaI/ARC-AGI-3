## Turn 0 (step 0, level 0)
State: start of level 0. Playfield: big room of 3, solid wall block rows 25-49 cols ~13-33 splitting it, goal 12-pad at rows 45-46 cols 34-38, 9-item clusters above pad and in side rooms, plus-sprite at (31-33, 20-22).
Built world_model v2 (rigid sprite, walkable={3,12}), BFS over sprite-center positions -> 26-action plan DOWN x14 + RIGHT x12 ending in pad contact; replayed through predict(): 25x "move", last "level_up".
Known: batching closed (unmeasured). Submit ONE action: DOWN, expect "move".
Carry-forward: plan in self.v.plan; pad contact = win is H4, unconfirmed.

