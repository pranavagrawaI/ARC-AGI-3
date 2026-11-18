# packet-live-probe-v4 result

## Validity

Valid. 24 of 24 completed; model `qwen/qwen3.8-27b`; zero reasoning tokens; no
tool calls; every packet had a null pending objection; every W0 packet had an
empty `actor_claims`.

## Observation

| boundary | v2 (w10) | v3 (w10 + rationales) | **W0** (0 + rationales) | **W3** (3 + rationales) |
|---|---:|---:|---:|---:|
| critic:3 | 2/4 | 3/4 | 2/4 | 3/4 |
| critic:5 | 1/4 | 2/4 | 1/4 | 1/4 |
| critic:7 | 0/4 | 1/4 | 1/4 | 2/4 |
| **total objective_gap** | 3/12 | 6/12 | **4/12** | **6/12** |
| valid and bound | 10/12 | 10/12 | **12/12** | **12/12** |
| mechanics | 0 | 1 | 1 | 0 |
| reported prompt tokens | 4,352-5,122 | 4,705-5,613 | **1,712-1,849** | **2,771-3,090** |
| cited a rationale / cited the window | — | 9 / 0 | 10 / 0 | 11 / 0 |

Prediction "neither condition below v3's 6": **W0 failed (4), W3 held (6).**
W0 tokens under 2,200: held. Valid ≥10 of 12 in each: held, at 12 of 12 both.
Mechanics ≤2 in each: held.

Verbatim questions are appended below.

## Interpretation

By the pre-registered map: W0 < 6 and W3 ≥ 6 → **PACKET_CLAIMS = 3**. Applied
to v9 and re-pinned; 174 of 174 local tests pass.

Two honest qualifications. First, 4 against 6 of 12 is inside any reasonable
noise band for twelve samples; the map committed to acting on it and I have,
but the trace does not *establish* that the window carries information. What is
consistent across four arms is the citation pattern — the critic has now cited
a rationale 30 times and the code window 0 times — and the schema pattern:
window 10 gave 2 invalid responses in each of two runs, windows 3 and 0 gave
none in 24. Second, the objective-gap rate has plateaued at 6 of 12 across v3
and W3; two packet changes did not move it. What remains is not "is the bar the
goal?" being missed — at critic:7 with two affordable actions the critic
increasingly asks about the live commitment's scope or the budget, which the
rule cannot credit. The rule was right for v2's failure and is now the limit.

Net against v8 at the same boundaries: the packet the critic will actually read
is 2.6-6.7% of the v8 prompt (fixture estimate calibrated at 1.13-1.22x), with
every response valid in the last 24 calls.

## Permitted next action

Adopt window 3 (done). The next step per the map is a controlled Kaggle card
for v9 versus v8. Not launched; needs its own card and explicit scope.

## Verbatim

Condition W0:

- critic:3 s1 (other, turn-6): Given clicks always advance the row 63 bar, does the specific click location (object vs. background) actually constrain *which* cells fill?
- critic:3 s2 (objective_gap, turn-6): If background clicks fill the bar, does the level end automatically or require a separate trigger?
- critic:3 s3 (empty): 
- critic:3 s4 (objective_gap, turn-6): If clicking distinct targets (object at step 5, background at step 6) fills the same bar cells, what observed feature distinguishes the target needed to finish from these generic fills?
- critic:5 s1 (objective_gap, turn-9): Which specific observed state change constitutes level completion, rather than just further bar progression?
- critic:5 s2 (other, turn-9): With ~6 actions left, what single observation confirms if the remaining bar space requires more than 6 clicks to fill?
- critic:5 s3 (other, turn-9): At 39s/action with only 6 affordable actions left, will the predicted row-63 shift occur in time, or must I pivot to a cheaper probe?
- critic:5 s4 (other, turn-5): If the background click is just a timer, what non-click event should trigger the expected move?
- critic:7 s1 (mechanics, turn-13): How do you distinguish a +2 bar change caused by a special target from two sequential clicks if the tool executes clicks differently than predicted?
- critic:7 s2 (other, turn-13): If the orange plus click also shifts the bar by exactly one cell, why would you conclude the targets are scenery rather than merely non-primary interaction points?
- critic:7 s3 (objective_gap, turn-13): Which single observation connects the observed bar-cell changes to actual level completion?
- critic:7 s4 (empty): 

Condition W3:

- critic:3 s1 (objective_gap, turn-6): What specific grid completion condition or target state must be achieved to finish level 0, and how do current bar fills count toward it?
- critic:3 s2 (other, turn-6): Do you know the total number of 4s required for level completion?
- critic:3 s3 (objective_gap, turn-6): If background clicks also advance the bar, what observed feature makes level completion conditional rather than just continuing to click the background?
- critic:3 s4 (objective_gap, turn-6): Could 58 clicks at 37.8s each finish the bar before the 373s wall ends?
- critic:5 s1 (empty): 
- critic:5 s2 (other, turn-9): Do steps 4 and 8 both fill 2 cells? Does any 1-cell step ever fill the bar by 2?
- critic:5 s3 (objective_gap, turn-9): Will the bar ever stop filling and trigger the expected move, or does it keep advancing until the level boundary is reached?
- critic:5 s4 (other, turn-9): Is the bar filling on every click regardless of the specific location?
- critic:7 s1 (objective_gap, turn-13): If filling the 64-cell bar is not the win condition, what specific observable state change indicates level completion?
- critic:7 s2 (other, turn-13): With only ~2 affordable actions left, does testing a second target now justify the cost, or should you commit to filling the bar?
- critic:7 s3 (objective_gap, turn-13): If the 10-step batch filled ~5 cells at 2 per click, what evidence connects this specific bar rate to level completion?
- critic:7 s4 (other, turn-13): With only two affordable actions left, will spending another on a click actually settle whether targets are special before the wall clock expires?