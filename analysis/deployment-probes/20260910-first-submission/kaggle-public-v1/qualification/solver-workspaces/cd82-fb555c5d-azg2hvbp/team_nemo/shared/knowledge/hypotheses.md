## Hypotheses (all untested)
- H1: arrow keys move one of the objects (player); predict = no-op is wrong for at least one direction.
- H2: The f regions are collectibles/targets and 0 boxes are pits or rooms.
- H3: SPACEBAR interacts with the object the player is on/adjacent to.
- H4: The win condition involves reaching/matching a marked region.


| H5 | arrows erase one boundary wall cell (UP cleared [63,63]) | untested-repeat | test: UP again (does [63,62] clear? or same cell again?), then DOWN/LEFT/RIGHT to map |
| H1 | arrows move a player object | contradicted (no visible mover at step 1; may be hidden) | re-evaluate once wall-erase pattern is known |


| H5 | each UP press erodes one bottom-wall cell, sequentially | weakened (2nd UP blocked, no 2nd cell) | test DOWN then LEFT |
| H6 | invisible player at bottom edge; UP steps it up leaving 5; blocked by hidden wall | untested | DOWN should move it back down (grid may refill or not) |
| H7 | UP digs once then is exhausted/blocked by rule | untested | LEFT/DOWN behavior decides |


| H5 | each press erodes next boundary cell | contradicted (2nd UP was blocked) | - |
| H8 | player at bottom row; arrows carve adjacent wall cell | partial (UP carved 63, DOWN carved 62) | test LEFT: expect carve at 61 or 64? |
| H9 | carver cursor moves along bottom row | untested | test LEFT/RIGHT |


| H11 (RIGHT = exact inverse of LEFT) | CONFIRMED | turn5: blob region identical to pre-LEFT |
| H12 (blob shape depends on position, deforms against fixed 0-walls) | untested | shape 114@cols25-38 vs 128@cols14-30 |
| H13 (verticals also move blob) | untested | test UP/DOWN |
| H14 (bottom-row carve position is direction/cursor dependent: DOWN->63, LEFT->61, RIGHT->none) | untested | re-test after blob returns to start |

