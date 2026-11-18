# packet-live-probe-v2 result

## Validity

Valid. 16 of 16 completed; model `qwen/qwen3.8-27b` on every call; zero reasoning
tokens; no tool calls; every condition-A packet had a null pending objection;
reported prompt tokens 1.13-1.22x the corrected fixture estimate (band 0.8-1.3).

## Observation

Prediction A: at least 8 of 12 objective_gap. **Observed 3 of 12.** Negative.

| boundary | objective_gap | mechanics | other | invalid | empty |
|---|---:|---:|---:|---:|---:|
| critic:3 | 2 | 0 | 2 | 0 | 0 |
| critic:5 | 1 | 0 | 3 | 0 | 0 |
| critic:7 | 0 | 0 | 1 | 2 | 1 |

Valid and bound: 10 of 12 (predicted ≥10). Mechanics by rule: 0 (predicted ≤2).

Verbatim, condition A:

- c3 s1 (other): What observation distinguishes clicking empty background from clicking shapes if both only increment the bar?
- c3 s2 (other): If the bar needs 58 more clicks and the budget only allows 9, what specific feature of the grid enables faster filling?
- c3 s3 (objective_gap): Does clicking the 58 remaining bar-3 cells turn them to 4 and end the level?
- c3 s4 (objective_gap): Does the bar's leftward filling stop at a specific target column, or does it require filling the entire 64-cell row to end the level?
- c5 s1 (other): Does a failed z-value prediction still count as a correct prediction for earning batch permission, or does only the 'expect' label matter?
- c5 s2 (objective_gap): With 43+ clicks remaining but only 6 affordable before the wall clock ends, what plan achieves level completion without the bar-filling loop?
- c5 s3 (other): Does the wall-clock budget allow enough single-click submissions to earn batch permission before time ends?
- c5 s4 (other): Which observed feature, beyond periodicity, proves step 9 gains only one bar cell rather than two?
- c7 s1: invalid (question with empty ref)
- c7 s2: empty
- c7 s3: invalid (question with empty ref)
- c7 s4 (other): With only two affordable actions left, will continuing to click the background to fill the bar be the final commitment if clicking the object yields no distinct change?

Prediction B: at most 1 of 4 re-raises by the rule "objective_gap naming bar, fill,
row 63, or 64". **Observed 2 of 4** (s2, s3) — negative by rule. The same two
also satisfied the card's "affordability" clause, which the card did not make
exclusive: a card defect. Verbatim:

- B s1 (objective_gap): If the level is truly un-solvable by simple clicking, why has no failure signal (error, state change) appeared after 17 identical-impact actions?
- B s2 (objective_gap, rule re-raise): If only two actions remain, does committing to the remaining 46 bar clicks risk exhausting the wall clock without confirming bar completion?
- B s3 (objective_gap, rule re-raise): With two affordable actions and a 1-to-64 fill rate, which immediate action best tests for a shortcut before spending the budget?
- B s4 (other): What specific observation would make you stop clicking the background to fill the bar instead of testing another reversible mechanic?

## Interpretation

A is negative and the cause is visible in the packet. At critic:5 the ten-claim
window (assistant messages 23-32) is transition dumps and the Actor's fight to
earn batch permission; two of the four questions are about batch-permission
rules — drift into *host* mechanics that the v8 critic never showed. The
objective-level claim ("fill the bar — click background center", turn 6) had
scrolled out of the window. The result map prescribes adding recent
commitments' rationales; done in v9 and tested in packet-live-probe-v3.

B by the pre-registered rule is negative. By reading, none of the four re-raises
"is the bar the goal?"; they question the probe's affordability and its
falsification condition, which is what a critic should ask after a `probe`
closure. The rule was too loose to distinguish those. Left open: not enough to
justify a mechanical reopen cooldown, not enough to rule it out.

A behaviour v8 never showed: six of the fourteen valid questions across A and B
are affordability questions grounded in the packet's budget block. That is the
question the S5I5 run needed and never received.

## Permitted next action

A negative → enrich the packet with recent commitments and retest under a new
card (packet-live-probe-v3). No Kaggle run.
