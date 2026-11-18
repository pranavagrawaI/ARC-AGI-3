# m2-evidence-smoke-1 result (2026-10-08)

Invalidity: none. COMPLETE; "M2 evidence installed" at t=8 s; solver config max_runtime_s_per_game=870, concurrency=3, priority gate 10.
GPU charge: 1,624 s (Kaggle quota delta 35,159.9 → 36,783.9 s, no other kernel active). The collector's 9,001 s was a polling gap; collect.py now caps charges at the hard cap.

Observation:
| game | state | levels | score | tokens | entries | revisions | re-shows |
|---|---|---|---|---|---|---|---|
| lf52 | gave_up at cap | 2 | 5.5 | 79.8K | 2 | 0 | 2 |
| s5i5 | gave_up at cap | 2 | 8.3 | 90.8K | 3 | 1 | 2 |
| ft09 | gave_up at cap | 4 | 47.6 | 85.7K | 2 | 0 | 4 |
No overlay exception. One rejected call (s5i5: `revises` pointed at a nonexistent id); the model recovered and its E3 revises E2.
The python tracebacks (4 per game) are ordinary harness errors (KeyError, StaleState, TerminalState, AttributeError on TransitionView.step).

Defect found: 6 of 7 entries have interpretation == "level". The model passed scope positionally as the 5th argument,
which is the interpretation slot, so the revisable "what remains open" field was erased.
Fix (after smoke, before pair freeze): record_evidence is keyword-only and gives a guiding TypeError; the prompt shows the keyword form.
Gate check `positional_rejected_with_guidance` added. Re-qualified, and the installer plus gate were re-run on the real Kaggle bundle.
The fixed build has NOT been GPU-smoked. A second smoke would have eaten most of the follow-up allowance;
the pair's delivery counter (games with entries) reports uptake, and a model that cannot adapt shows up as not-delivered.

Usage pattern: entries are mostly retrospective level-up lessons written at the first step of the next level
(5 of 7), not records made after failed tests while stuck. s5i5 E2 is the exception: it recorded "no config reaches the crosshair"
mid-level, then E3 revised it to solved. Whether stuck periods get records is a pair-1 question.

Not a comparison: lf52 and s5i5 each reached 2 levels in 870 s, where control r1 reached 1 each in 7,254 s. That is n=1 under
different load (3 games) and is not evidence of benefit.

Per card: operational pass AND ≥1 entry → launch pair-1, with the keyword-only fix recorded in pair-1's card before launch.
