# Hypotheses
h1: game is click-only (actions RESET/CLICK).
h2: rings (c2/c4 nested, c11, c14) are objectives; c13 dots and c3 bars are counters/markers.
h3: clicking a marker changes it or a ring.
Evidence: step 0 grid measurement. Tested: none yet.


h4: CLICK advances a floor counter (3->4 at right end; previously nothing). Clicks seem to be positional-independent? (9,9) changed (63,63).
   Evidence step 1. Open: does each click always flip exactly one more floor cell to 4? From left to right?
h5: tick bars (c3) and +shapes (c13) are score/counters or targets.
h6 (weakened): rings are objectives — untested.


h4 (stronger): each CLICK flips floor cells at row 63, right end extending leftward. Growth: step1 -> 1 new cell; step2 -> 2 new cells. Candidate rules: grows by n on click n (triangular), or grows randomly, or grows by (n-? ) — test with click 3.
h5: ticks (+bars c3) and +shapes (c13) untested. Rings untested.


h7: clicking an object erodes a DIFFERENT object (the c14 ring chunk + c13 dot (10,31)) — some association by position/index.
h8: floor growth amounts 1,2,1,1,1,2: still unexplained.


h7 (revised): the c14 chunk erosion at step 6 was NOT caused by clicking the bar. Clicking (41,24) hit the bar; the bar did not change; the erosion was a coincidental or separate mechanism.
h9: Clicks on any non-floor object (ring, bar, +dot) only grow the floor — the objects are inert decoration. The ONLY dynamic is the floor run.
- This would mean the game is: grow the floor run to complete the level? Or the floor run is just a timer/counter and the objects are static.
- Open: what is the win condition? Is it filling the floor? Or does the floor run need to reach a specific length?


h10: the step-6 erosion was a one-shot event tied to some condition satisfied only at that moment (e.g. first time a bar was clicked, or a fixed step number).
h11: the game objective may involve the c14 ring + c13 dots: the erosion removed the "top cap" of the c14 ring around (10,28-32). The c13 dot (10,31) vanished. Perhaps the objective is to erode all such caps/dots (a "clear the markers" game) and the floor run is a step counter/HUD.
Open: what erodes next? Which regions are "caps"? The c14 ring still has caps at (20-22, 38-46).

